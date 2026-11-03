//! Stateful supervisor with shared in-memory key-value store
//!
//! This module provides a parallel implementation of the supervisor tree with built-in
//! shared state via `WorkerContext`. Workers receive the context as a parameter in their
//! factory functions, allowing them to share data through a concurrent in-memory store.
//!
//! # Choosing Between Stateful and Regular Supervisors
//!
//! Use [`StatefulSupervisorSpec`] when:
//! - Workers need to share state (counters, caches, configuration)
//! - You need coordination between workers (flags, semaphores)
//! - State should survive worker restarts
//! - You want built-in concurrency-safe storage without external dependencies
//!
//! Use [`SupervisorSpec`](crate::SupervisorSpec) when:
//! - Workers are stateless or manage their own state independently
//! - No data sharing is required between workers
//! - You want minimal overhead (no shared context)
//! - Workers communicate through channels or external systems
//!
//! # Key Differences
//!
//! | Feature | `StatefulSupervisorSpec` | `SupervisorSpec` |
//! |---------|--------------------------|------------------|
//! | Worker Factory | `Fn(Arc<WorkerContext>) -> W` | `Fn() -> W` |
//! | Shared State | ✅ Built-in `WorkerContext` | ❌ None |
//! | Use Case | Coordinated workers | Independent workers |
//! | Overhead | Slightly higher (context management) | Minimal |
//!
//! # Example: When to Use Stateful
//!
//! ```rust,no_run
//! use ash_flare::{StatefulSupervisorSpec, StatefulSupervisorHandle, Worker, WorkerContext};
//! use async_trait::async_trait;
//! use std::sync::Arc;
//!
//! #[derive(Debug)]
//! struct CounterWorker {
//!     id: String,
//!     context: Arc<WorkerContext>,
//! }
//!
//! #[async_trait]
//! impl Worker for CounterWorker {
//!     type Error = std::io::Error;
//!     
//!     async fn run(&mut self) -> Result<(), Self::Error> {
//!         // Workers can share and update counters
//!         self.context.update("global_count", |v| {
//!             let count = v.and_then(|v| v.as_u64()).unwrap_or(0);
//!             Some(serde_json::json!(count + 1))
//!         });
//!         Ok(())
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() {
//! let spec = StatefulSupervisorSpec::new("counter-supervisor")
//!     .with_worker("counter-1", |ctx| CounterWorker {
//!         id: "counter-1".to_owned(),
//!         context: ctx
//!     }, ash_flare::RestartPolicy::Permanent);
//!
//! let handle = StatefulSupervisorHandle::start(spec);
//! // Workers share state through the context
//! # handle.shutdown().await.ok();
//! # }
//! ```

use crate::restart::{
    BackoffTracker, RestartBackoff, RestartIntensity, RestartPolicy, RestartStrategy,
    RestartTracker,
};
use crate::supervisor::RuntimeControl;
use crate::supervisor_common::{
    LinkedInit, RestartDecision, WorkerTermination, decide_restart, run_worker,
};
use crate::types::{ChildExitReason, ChildId, ChildInfo, ChildType, ShutdownSignal, WorkerContext};
use crate::worker::Worker;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

// ============================================================================
// Worker Specification (Stateful)
// ============================================================================

/// Specification for creating and restarting a stateful worker
pub(crate) struct StatefulWorkerSpec<W: Worker> {
    pub id: ChildId,
    pub worker_factory: Arc<dyn Fn(Arc<WorkerContext>, ShutdownSignal) -> W + Send + Sync>,
    pub restart_policy: RestartPolicy,
    pub context: Arc<WorkerContext>,
}

impl<W: Worker> Clone for StatefulWorkerSpec<W> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            worker_factory: Arc::clone(&self.worker_factory),
            restart_policy: self.restart_policy,
            context: Arc::clone(&self.context),
        }
    }
}

impl<W: Worker> StatefulWorkerSpec<W> {
    pub(crate) fn new(
        id: impl Into<String>,
        factory: impl Fn(Arc<WorkerContext>) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
        context: Arc<WorkerContext>,
    ) -> Self {
        Self::with_signal(
            id,
            move |ctx, _signal| factory(ctx),
            restart_policy,
            context,
        )
    }

    /// Builds a spec whose factory also receives the worker's [`ShutdownSignal`].
    pub(crate) fn with_signal(
        id: impl Into<String>,
        factory: impl Fn(Arc<WorkerContext>, ShutdownSignal) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
        context: Arc<WorkerContext>,
    ) -> Self {
        Self {
            id: id.into(),
            worker_factory: Arc::new(factory),
            restart_policy,
            context,
        }
    }

    pub(crate) fn create_worker(&self, signal: ShutdownSignal) -> W {
        (self.worker_factory)(Arc::clone(&self.context), signal)
    }
}

impl<W: Worker> fmt::Debug for StatefulWorkerSpec<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StatefulWorkerSpec")
            .field("id", &self.id)
            .field("restart_policy", &self.restart_policy)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Worker Process (Stateful)
// ============================================================================

/// Running stateful worker process with its specification and task handle
pub(crate) struct StatefulWorkerProcess<W: Worker> {
    pub spec: StatefulWorkerSpec<W>,
    pub handle: Option<JoinHandle<()>>,
    /// Incarnation number, unique within the owning supervisor. Termination
    /// reports carry it so reports from replaced instances can be ignored.
    pub generation: u64,
    /// Triggers cooperative shutdown; dropping it also releases the worker.
    shutdown_tx: tokio::sync::watch::Sender<bool>,
}

impl<W: Worker> StatefulWorkerProcess<W> {
    pub(crate) fn spawn<Cmd>(
        spec: StatefulWorkerSpec<W>,
        supervisor_name: String,
        control_tx: mpsc::UnboundedSender<Cmd>,
        generation: u64,
    ) -> Self
    where
        Cmd: From<WorkerTermination> + Send + 'static,
    {
        Self::spawn_inner(spec, supervisor_name, control_tx, generation, None)
    }

    /// Spawns a worker with linked initialization handshake
    pub(crate) fn spawn_with_link<Cmd>(
        spec: StatefulWorkerSpec<W>,
        supervisor_name: String,
        control_tx: mpsc::UnboundedSender<Cmd>,
        generation: u64,
        init_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    ) -> Self
    where
        Cmd: From<WorkerTermination> + Send + 'static,
    {
        Self::spawn_inner(spec, supervisor_name, control_tx, generation, Some(init_tx))
    }

    fn spawn_inner<Cmd>(
        spec: StatefulWorkerSpec<W>,
        supervisor_name: String,
        control_tx: mpsc::UnboundedSender<Cmd>,
        generation: u64,
        init_tx: Option<tokio::sync::oneshot::Sender<Result<(), String>>>,
    ) -> Self
    where
        Cmd: From<WorkerTermination> + Send + 'static,
    {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let worker = spec.create_worker(ShutdownSignal::new(shutdown_rx.clone()));
        let worker_id = spec.id.clone();
        let signal = ShutdownSignal::new(shutdown_rx);

        let handle = tokio::spawn(async move {
            run_worker(
                supervisor_name,
                worker_id,
                generation,
                worker,
                control_tx,
                init_tx,
                signal,
            )
            .await;
        });

        Self {
            spec,
            handle: Some(handle),
            generation,
            shutdown_tx,
        }
    }

    /// Stops the worker, giving it `timeout` to wind down cooperatively before
    /// aborting the task.
    ///
    /// Returns `true` if the worker stopped on its own (so its `shutdown` hook
    /// ran), `false` if it had to be aborted.
    pub(crate) async fn stop(&mut self, timeout: std::time::Duration) -> bool {
        let Some(mut handle) = self.handle.take() else {
            return true;
        };

        // Ask the worker to stop. A send error means the task is already gone.
        let _notified = self.shutdown_tx.send(true);

        if timeout.is_zero() {
            handle.abort();
            let _join_result = handle.await;
            return false;
        }

        match tokio::time::timeout(timeout, &mut handle).await {
            Ok(_joined) => true,
            Err(_elapsed) => {
                tracing::warn!(
                    worker = %self.spec.id,
                    timeout_ms = %timeout.as_millis(),
                    "worker did not stop within shutdown timeout, aborting"
                );
                handle.abort();
                let _join_result = handle.await;
                false
            }
        }
    }
}

impl<W: Worker> Drop for StatefulWorkerProcess<W> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

// ============================================================================
// Supervisor Specification (Stateful)
// ============================================================================

/// Specification for a child (either worker or supervisor)
pub(crate) enum StatefulChildSpec<W: Worker> {
    Worker(StatefulWorkerSpec<W>),
    Supervisor(Arc<StatefulSupervisorSpec<W>>),
}

impl<W: Worker> Clone for StatefulChildSpec<W> {
    fn clone(&self) -> Self {
        match self {
            StatefulChildSpec::Worker(w) => StatefulChildSpec::Worker(w.clone()),
            StatefulChildSpec::Supervisor(s) => StatefulChildSpec::Supervisor(Arc::clone(s)),
        }
    }
}

/// Describes a stateful supervisor and its children in a tree structure.
pub struct StatefulSupervisorSpec<W: Worker> {
    pub(crate) name: String,
    pub(crate) children: Vec<StatefulChildSpec<W>>,
    pub(crate) restart_strategy: RestartStrategy,
    pub(crate) restart_intensity: RestartIntensity,
    pub(crate) restart_backoff: RestartBackoff,
    pub(crate) shutdown_timeout: std::time::Duration,
    pub(crate) context: Arc<WorkerContext>,
}

impl<W: Worker> Clone for StatefulSupervisorSpec<W> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            children: self.children.clone(),
            restart_strategy: self.restart_strategy,
            restart_intensity: self.restart_intensity,
            restart_backoff: self.restart_backoff,
            shutdown_timeout: self.shutdown_timeout,
            context: Arc::clone(&self.context),
        }
    }
}

// Only the Python bindings validate ids at registration time; the Rust builder
// leaves duplicate detection to `start_child`.
#[cfg(feature = "python")]
impl<W: Worker> StatefulChildSpec<W> {
    /// Identifier this child will be registered under.
    pub(crate) fn id(&self) -> &str {
        match self {
            StatefulChildSpec::Worker(w) => &w.id,
            StatefulChildSpec::Supervisor(s) => &s.name,
        }
    }
}

impl<W: Worker> StatefulSupervisorSpec<W> {
    /// Returns true if a child with `id` is already registered.
    #[cfg(feature = "python")]
    pub(crate) fn has_child(&self, id: &str) -> bool {
        self.children.iter().any(|c| c.id() == id)
    }

    /// Creates a new stateful supervisor specification with the provided name.
    /// Automatically initializes an empty `WorkerContext` for sharing state between workers.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            children: Vec::new(),
            restart_strategy: RestartStrategy::default(),
            restart_intensity: RestartIntensity::default(),
            restart_backoff: RestartBackoff::default(),
            shutdown_timeout: crate::supervisor::DEFAULT_SHUTDOWN_TIMEOUT,
            context: Arc::new(WorkerContext::new()),
        }
    }

    /// Sets the restart strategy for this supervisor.
    #[must_use]
    pub fn with_restart_strategy(mut self, strategy: RestartStrategy) -> Self {
        self.restart_strategy = strategy;
        self
    }

    /// Sets the restart intensity for this supervisor.
    #[must_use]
    pub fn with_restart_intensity(mut self, intensity: RestartIntensity) -> Self {
        self.restart_intensity = intensity;
        self
    }

    /// Sets the backoff applied between a child terminating and being restarted.
    ///
    /// Defaults to an exponential backoff of 100ms doubling up to 30s. Use
    /// [`RestartBackoff::none`] to restart immediately, at the cost of allowing a
    /// crash-looping child to saturate a CPU core.
    #[must_use]
    pub fn with_restart_backoff(mut self, backoff: RestartBackoff) -> Self {
        self.restart_backoff = backoff;
        self
    }

    /// Sets a fixed delay between a child terminating and being restarted.
    ///
    /// Convenience wrapper over [`StatefulSupervisorSpec::with_restart_backoff`]
    /// for a constant delay; `delay` is used as both the initial and maximum delay.
    #[must_use]
    pub fn with_restart_delay(self, delay: std::time::Duration) -> Self {
        self.with_restart_backoff(RestartBackoff::exponential(delay, delay))
    }

    /// Sets how long a worker may take to stop cooperatively before its task is
    /// aborted.
    ///
    /// On termination each worker is sent a [`ShutdownSignal`]; a worker that
    /// returns from `run` within this window also gets its
    /// [`shutdown`](Worker::shutdown) hook run. A worker that ignores the signal
    /// is aborted and does **not** get the hook. Defaults to 5 seconds;
    /// `Duration::ZERO` aborts immediately.
    #[must_use]
    pub fn with_shutdown_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.shutdown_timeout = timeout;
        self
    }

    /// Adds a stateful worker whose factory also receives the worker's
    /// [`ShutdownSignal`], so it can observe cooperative shutdown.
    #[must_use]
    pub fn with_worker_signal(
        mut self,
        id: impl Into<String>,
        factory: impl Fn(Arc<WorkerContext>, ShutdownSignal) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
    ) -> Self {
        let context = Arc::clone(&self.context);
        self.children
            .push(StatefulChildSpec::Worker(StatefulWorkerSpec::with_signal(
                id,
                factory,
                restart_policy,
                context,
            )));
        self
    }

    /// Adds a stateful worker child to this supervisor specification.
    /// The factory function receives a `WorkerContext` parameter for accessing shared state.
    #[must_use]
    pub fn with_worker(
        mut self,
        id: impl Into<String>,
        factory: impl Fn(Arc<WorkerContext>) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
    ) -> Self {
        self.children
            .push(StatefulChildSpec::Worker(StatefulWorkerSpec::new(
                id,
                factory,
                restart_policy,
                Arc::clone(&self.context),
            )));
        self
    }

    /// Adds a nested stateful supervisor child to this supervisor specification.
    #[must_use]
    pub fn with_supervisor(mut self, supervisor: StatefulSupervisorSpec<W>) -> Self {
        self.children
            .push(StatefulChildSpec::Supervisor(Arc::new(supervisor)));
        self
    }

    /// Returns a reference to the `WorkerContext` for this supervisor tree.
    #[must_use]
    pub fn context(&self) -> &Arc<WorkerContext> {
        &self.context
    }
}

// ============================================================================
// Child Management (Stateful)
// ============================================================================

/// Represents either a worker or a nested supervisor in the supervision tree
pub(crate) enum StatefulChild<W: Worker> {
    Worker(StatefulWorkerProcess<W>),
    Supervisor {
        handle: StatefulSupervisorHandle<W>,
        spec: Arc<StatefulSupervisorSpec<W>>,
        /// Incarnation number of this nested supervisor.
        generation: u64,
    },
}

impl<W: Worker> StatefulChild<W> {
    #[inline]
    pub fn id(&self) -> &str {
        match self {
            StatefulChild::Worker(w) => &w.spec.id,
            StatefulChild::Supervisor { spec, .. } => &spec.name,
        }
    }

    /// Incarnation number of the currently running instance of this child.
    #[inline]
    pub fn generation(&self) -> u64 {
        match self {
            StatefulChild::Worker(w) => w.generation,
            StatefulChild::Supervisor { generation, .. } => *generation,
        }
    }

    #[inline]
    pub fn child_type(&self) -> ChildType {
        match self {
            StatefulChild::Worker(_) => ChildType::Worker,
            StatefulChild::Supervisor { .. } => ChildType::Supervisor,
        }
    }

    #[inline]
    #[allow(clippy::unnecessary_wraps)]
    pub fn restart_policy(&self) -> Option<RestartPolicy> {
        match self {
            StatefulChild::Worker(w) => Some(w.spec.restart_policy),
            StatefulChild::Supervisor { .. } => Some(RestartPolicy::Permanent),
        }
    }

    /// Policy as seen by the shared restart decision: `None` for nested
    /// supervisors, which are always treated as permanent.
    #[inline]
    pub fn restart_policy_for_decision(&self) -> Option<RestartPolicy> {
        match self {
            StatefulChild::Worker(w) => Some(w.spec.restart_policy),
            StatefulChild::Supervisor { .. } => None,
        }
    }

    /// Stops the child, allowing `timeout` for a worker to wind down
    /// cooperatively before its task is aborted.
    pub async fn shutdown(&mut self, timeout: std::time::Duration) {
        match self {
            StatefulChild::Worker(w) => {
                let _graceful = w.stop(timeout).await;
            }
            StatefulChild::Supervisor { handle, .. } => {
                // Waits until the nested tree has fully stopped, so a restart
                // never runs the old and the new subtree side by side.
                let _shutdown_result = handle.shutdown().await;
            }
        }
    }
}

// ============================================================================
// Supervisor Error (Stateful)
// ============================================================================

/// Errors returned by stateful supervisor operations.
#[derive(Debug)]
pub enum StatefulSupervisorError {
    /// Supervisor has no children
    NoChildren(String),
    /// All children have failed
    AllChildrenFailed(String),
    /// Supervisor is shutting down
    ShuttingDown(String),
    /// Child with this ID already exists
    ChildAlreadyExists(String),
    /// Child with this ID not found
    ChildNotFound(String),
    /// Child initialization failed
    InitializationFailed {
        /// ID of the child that failed to initialize
        child_id: String,
        /// Reason for initialization failure
        reason: String,
    },
    /// Child initialization timed out
    InitializationTimeout {
        /// ID of the child that timed out
        child_id: String,
        /// Duration after which timeout occurred
        timeout: std::time::Duration,
    },
}

impl fmt::Display for StatefulSupervisorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatefulSupervisorError::NoChildren(name) => {
                write!(f, "stateful supervisor '{name}' has no children")
            }
            StatefulSupervisorError::AllChildrenFailed(name) => {
                write!(
                    f,
                    "all children failed for stateful supervisor '{name}' - restart intensity limit exceeded"
                )
            }
            StatefulSupervisorError::ShuttingDown(name) => {
                write!(
                    f,
                    "stateful supervisor '{name}' is shutting down - operation not permitted"
                )
            }
            StatefulSupervisorError::ChildAlreadyExists(id) => {
                write!(
                    f,
                    "child with id '{id}' already exists - use a unique identifier"
                )
            }
            StatefulSupervisorError::ChildNotFound(id) => {
                write!(
                    f,
                    "child with id '{id}' not found - it may have already terminated"
                )
            }
            StatefulSupervisorError::InitializationFailed { child_id, reason } => {
                write!(f, "child '{child_id}' initialization failed: {reason}")
            }
            StatefulSupervisorError::InitializationTimeout { child_id, timeout } => {
                write!(
                    f,
                    "child '{child_id}' initialization timed out after {timeout:?}"
                )
            }
        }
    }
}

impl std::error::Error for StatefulSupervisorError {}

// ============================================================================
// Supervisor Runtime (Stateful)
// ============================================================================

/// Internal commands sent to stateful supervisor runtime
pub(crate) enum StatefulSupervisorCommand<W: Worker> {
    StartChild {
        spec: StatefulWorkerSpec<W>,
        respond_to: oneshot::Sender<Result<ChildId, StatefulSupervisorError>>,
    },
    StartChildLinked {
        spec: StatefulWorkerSpec<W>,
        timeout: std::time::Duration,
        respond_to: oneshot::Sender<Result<ChildId, StatefulSupervisorError>>,
    },
    TerminateChild {
        id: ChildId,
        respond_to: oneshot::Sender<Result<(), StatefulSupervisorError>>,
    },
    WhichChildren {
        respond_to: oneshot::Sender<Result<Vec<ChildInfo>, StatefulSupervisorError>>,
    },
    GetRestartStrategy {
        respond_to: oneshot::Sender<RestartStrategy>,
    },
    GetUptime {
        respond_to: oneshot::Sender<u64>,
    },
    ChildTerminated {
        id: ChildId,
        reason: ChildExitReason,
        /// Incarnation that terminated; stale reports are ignored.
        generation: u64,
    },
    /// Fires once a restart backoff has elapsed. Restarts are deferred through
    /// the command queue so the supervisor keeps serving commands while a child
    /// is backing off.
    RestartChild {
        id: ChildId,
        strategy: RestartStrategy,
        /// Incarnation whose failure scheduled this restart. If the child has
        /// been replaced meanwhile (for example by a sibling's `OneForAll`
        /// round) the restart is skipped.
        generation: u64,
    },
    /// Outcome of a linked child's `initialize`, delivered once it completes or
    /// times out. Waiting for it happens off the command loop.
    LinkedInitDone {
        id: ChildId,
        generation: u64,
        outcome: LinkedInit,
    },
    Shutdown,
}

/// A linked child whose `initialize` has not finished yet. It joins the
/// children list only once it reports success.
struct PendingLinked<W: Worker> {
    worker: StatefulWorkerProcess<W>,
    respond_to: oneshot::Sender<Result<ChildId, StatefulSupervisorError>>,
    /// A termination reported before the init outcome was processed (for
    /// example `run` returning immediately). Applied once the child is admitted.
    early_exit: Option<ChildExitReason>,
}

impl<W: Worker> From<WorkerTermination> for StatefulSupervisorCommand<W> {
    fn from(term: WorkerTermination) -> Self {
        StatefulSupervisorCommand::ChildTerminated {
            id: term.id,
            reason: term.reason,
            generation: term.generation,
        }
    }
}

/// Internal state machine that manages stateful supervisor lifecycle and child processes
pub(crate) struct StatefulSupervisorRuntime<W: Worker> {
    name: String,
    children: Vec<StatefulChild<W>>,
    control_rx: mpsc::UnboundedReceiver<StatefulSupervisorCommand<W>>,
    control_tx: mpsc::UnboundedSender<StatefulSupervisorCommand<W>>,
    restart_strategy: RestartStrategy,
    restart_tracker: RestartTracker,
    backoff_tracker: BackoffTracker,
    shutdown_timeout: std::time::Duration,
    created_at: std::time::Instant,
    /// Source of incarnation numbers for children of this supervisor.
    next_generation: u64,
    /// Linked children still running `initialize`.
    pending_linked: HashMap<ChildId, PendingLinked<W>>,
}

impl<W: Worker> StatefulSupervisorRuntime<W> {
    pub(crate) fn new(
        spec: StatefulSupervisorSpec<W>,
        control_rx: mpsc::UnboundedReceiver<StatefulSupervisorCommand<W>>,
        control_tx: mpsc::UnboundedSender<StatefulSupervisorCommand<W>>,
    ) -> Self {
        let mut children = Vec::with_capacity(spec.children.len());
        let mut next_generation = 0_u64;

        for child_spec in spec.children {
            next_generation = next_generation.saturating_add(1);
            match child_spec {
                StatefulChildSpec::Worker(worker_spec) => {
                    let worker = StatefulWorkerProcess::spawn(
                        worker_spec,
                        spec.name.clone(),
                        control_tx.clone(),
                        next_generation,
                    );
                    children.push(StatefulChild::Worker(worker));
                }
                StatefulChildSpec::Supervisor(supervisor_spec) => {
                    let supervisor = StatefulSupervisorHandle::start_supervised(
                        (*supervisor_spec).clone(),
                        control_tx.clone(),
                        next_generation,
                    );
                    children.push(StatefulChild::Supervisor {
                        handle: supervisor,
                        spec: Arc::clone(&supervisor_spec),
                        generation: next_generation,
                    });
                }
            }
        }

        Self {
            name: spec.name,
            children,
            control_rx,
            control_tx,
            restart_strategy: spec.restart_strategy,
            restart_tracker: RestartTracker::new(spec.restart_intensity),
            backoff_tracker: BackoffTracker::new(spec.restart_backoff),
            shutdown_timeout: spec.shutdown_timeout,
            created_at: std::time::Instant::now(),
            next_generation,
            pending_linked: HashMap::new(),
        }
    }

    /// Hands out the next incarnation number.
    fn next_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.saturating_add(1);
        self.next_generation
    }

    /// Whether `id` is taken by a running child or one still initializing.
    fn id_in_use(&self, id: &str) -> bool {
        self.pending_linked.contains_key(id) || self.children.iter().any(|c| c.id() == id)
    }

    /// Serves commands until the supervisor terminates, and reports *why* it
    /// terminated so a parent supervisor can apply its own policy.
    pub(crate) async fn run(mut self) -> ChildExitReason {
        while let Some(command) = self.control_rx.recv().await {
            match command {
                StatefulSupervisorCommand::StartChild { spec, respond_to } => {
                    let result = self.handle_start_child(spec);
                    let _send = respond_to.send(result);
                }
                StatefulSupervisorCommand::StartChildLinked {
                    spec,
                    timeout,
                    respond_to,
                } => {
                    self.handle_start_child_linked(spec, timeout, respond_to);
                }
                StatefulSupervisorCommand::LinkedInitDone {
                    id,
                    generation,
                    outcome,
                } => {
                    if self.handle_linked_init_done(id, generation, outcome).await
                        == RuntimeControl::Stop
                    {
                        return ChildExitReason::Abnormal;
                    }
                }
                StatefulSupervisorCommand::TerminateChild { id, respond_to } => {
                    let result = self.handle_terminate_child(&id).await;
                    let _send = respond_to.send(result);
                }
                StatefulSupervisorCommand::WhichChildren { respond_to } => {
                    let result = self.handle_which_children();
                    let _send = respond_to.send(result);
                }
                StatefulSupervisorCommand::GetRestartStrategy { respond_to } => {
                    let _send = respond_to.send(self.restart_strategy);
                }
                StatefulSupervisorCommand::GetUptime { respond_to } => {
                    let uptime = self.created_at.elapsed().as_secs();
                    let _send = respond_to.send(uptime);
                }
                StatefulSupervisorCommand::ChildTerminated {
                    id,
                    reason,
                    generation,
                } => {
                    if self.handle_child_terminated(id, reason, generation).await
                        == RuntimeControl::Stop
                    {
                        // Restart intensity exceeded: children are already shut
                        // down. Terminate the supervisor itself, reporting an
                        // abnormal exit so a parent supervisor observes the
                        // failure and applies its own policy, as OTP does.
                        return ChildExitReason::Abnormal;
                    }
                }
                StatefulSupervisorCommand::RestartChild {
                    id,
                    strategy,
                    generation,
                } => {
                    self.handle_restart_child(&id, strategy, generation).await;
                }
                StatefulSupervisorCommand::Shutdown => {
                    self.shutdown_children().await;
                    return ChildExitReason::Shutdown;
                }
            }
        }

        self.shutdown_children().await;
        ChildExitReason::Shutdown
    }

    fn handle_start_child(
        &mut self,
        spec: StatefulWorkerSpec<W>,
    ) -> Result<ChildId, StatefulSupervisorError> {
        // Check if child with same ID already exists
        if self.id_in_use(&spec.id) {
            return Err(StatefulSupervisorError::ChildAlreadyExists(spec.id.clone()));
        }

        let id = spec.id.clone();
        let generation = self.next_generation();
        let worker = StatefulWorkerProcess::spawn(
            spec,
            self.name.clone(),
            self.control_tx.clone(),
            generation,
        );

        self.children.push(StatefulChild::Worker(worker));
        tracing::debug!(
            supervisor = %self.name,
            child = %id,
            "dynamically started child"
        );

        Ok(id)
    }

    /// Starts a linked child without blocking the command loop.
    ///
    /// The worker is spawned straight away; a helper task waits (up to
    /// `timeout`) for its `initialize` and reports back through
    /// [`StatefulSupervisorCommand::LinkedInitDone`]. Meanwhile the supervisor keeps
    /// serving commands and handling other children's failures.
    fn handle_start_child_linked(
        &mut self,
        spec: StatefulWorkerSpec<W>,
        timeout: std::time::Duration,
        respond_to: oneshot::Sender<Result<ChildId, StatefulSupervisorError>>,
    ) {
        if self.id_in_use(&spec.id) {
            let _send = respond_to.send(Err(StatefulSupervisorError::ChildAlreadyExists(
                spec.id.clone(),
            )));
            return;
        }

        let id = spec.id.clone();
        let generation = self.next_generation();
        let (init_tx, init_rx) = oneshot::channel();

        let worker = StatefulWorkerProcess::spawn_with_link(
            spec,
            self.name.clone(),
            self.control_tx.clone(),
            generation,
            init_tx,
        );

        let control_tx = self.control_tx.clone();
        let waiter_id = id.clone();
        tokio::spawn(async move {
            let outcome = match tokio::time::timeout(timeout, init_rx).await {
                Ok(Ok(Ok(()))) => LinkedInit::Ready,
                Ok(Ok(Err(reason))) => LinkedInit::Failed(reason),
                Ok(Err(_closed)) => LinkedInit::Vanished,
                Err(_elapsed) => LinkedInit::TimedOut(timeout),
            };
            let _send = control_tx.send(StatefulSupervisorCommand::LinkedInitDone {
                id: waiter_id,
                generation,
                outcome,
            });
        });

        self.pending_linked.insert(
            id,
            PendingLinked {
                worker,
                respond_to,
                early_exit: None,
            },
        );
    }

    /// Admits a linked child whose `initialize` succeeded, or discards it.
    ///
    /// Init failures do not trigger restart policies: the caller of
    /// `start_child_linked` gets the error instead.
    async fn handle_linked_init_done(
        &mut self,
        id: ChildId,
        generation: u64,
        outcome: LinkedInit,
    ) -> RuntimeControl {
        let is_current = self
            .pending_linked
            .get(&id)
            .is_some_and(|pending| pending.worker.generation == generation);
        if !is_current {
            // Stopped by a shutdown while it was initializing.
            return RuntimeControl::Continue;
        }
        let Some(pending) = self.pending_linked.remove(&id) else {
            return RuntimeControl::Continue;
        };

        let error = match outcome {
            LinkedInit::Ready => {
                tracing::debug!(
                    supervisor = %self.name,
                    child = %id,
                    "linked child started successfully"
                );
                self.children.push(StatefulChild::Worker(pending.worker));
                let _send = pending.respond_to.send(Ok(id.clone()));
                return match pending.early_exit {
                    Some(reason) => self.handle_child_terminated(id, reason, generation).await,
                    None => RuntimeControl::Continue,
                };
            }
            LinkedInit::Failed(reason) => StatefulSupervisorError::InitializationFailed {
                child_id: id.clone(),
                reason,
            },
            LinkedInit::Vanished => StatefulSupervisorError::InitializationFailed {
                child_id: id.clone(),
                reason: "worker exited before reporting initialization".to_owned(),
            },
            LinkedInit::TimedOut(timeout) => StatefulSupervisorError::InitializationTimeout {
                child_id: id.clone(),
                timeout,
            },
        };

        tracing::error!(
            supervisor = %self.name,
            child = %id,
            error = %error,
            "linked child failed to start"
        );
        // Dropping the process aborts the task if it is still running.
        drop(pending.worker);
        let _send = pending.respond_to.send(Err(error));
        RuntimeControl::Continue
    }

    async fn handle_terminate_child(&mut self, id: &str) -> Result<(), StatefulSupervisorError> {
        let position = self
            .children
            .iter()
            .position(|c| c.id() == id)
            .ok_or_else(|| StatefulSupervisorError::ChildNotFound(id.to_owned()))?;

        let mut child = self.children.remove(position);
        child.shutdown(self.shutdown_timeout).await;
        self.backoff_tracker.reset_child(id);

        tracing::debug!(
            supervisor = %self.name,
            child = %id,
            "terminated child"
        );
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn handle_which_children(&self) -> Result<Vec<ChildInfo>, StatefulSupervisorError> {
        let info = self
            .children
            .iter()
            .map(|child| ChildInfo {
                id: child.id().to_owned(),
                child_type: child.child_type(),
                restart_policy: child.restart_policy(),
            })
            .collect();

        Ok(info)
    }

    #[allow(clippy::indexing_slicing)]
    async fn handle_child_terminated(
        &mut self,
        id: ChildId,
        reason: ChildExitReason,
        generation: u64,
    ) -> RuntimeControl {
        if let Some(pending) = self.pending_linked.get_mut(&id) {
            if pending.worker.generation == generation {
                // Still initializing: decide once the init outcome arrives.
                pending.early_exit = Some(reason);
            }
            return RuntimeControl::Continue;
        }

        tracing::debug!(
            supervisor = %self.name,
            child = %id,
            reason = ?reason,
            "child terminated"
        );

        let Some(position) = self.children.iter().position(|c| c.id() == id) else {
            tracing::warn!(
                supervisor = %self.name,
                child = %id,
                "terminated child not found in list"
            );
            return RuntimeControl::Continue;
        };

        if self.children[position].generation() != generation {
            // A report from an instance that has already been replaced, e.g. a
            // sibling that failed in the same instant under `OneForAll` and was
            // restarted by the first failure's round. Acting on it would run a
            // second, spurious restart round and burn restart intensity.
            tracing::debug!(
                supervisor = %self.name,
                child = %id,
                "ignoring termination of a replaced instance"
            );
            return RuntimeControl::Continue;
        }

        // Supervision semantics live in one shared place; this runtime only
        // carries out the decision.
        let decision = decide_restart(
            self.children[position].restart_policy_for_decision(),
            reason,
            self.restart_strategy,
            &mut self.restart_tracker,
            &mut self.backoff_tracker,
            &id,
        );

        match decision {
            RestartDecision::Ignore => {
                // The supervisor stopped this child itself, as part of a restart
                // round or a shutdown. Nothing to do; the child list already
                // reflects whatever that operation intended.
                RuntimeControl::Continue
            }
            RestartDecision::Drop => {
                tracing::debug!(
                    supervisor = %self.name,
                    child = %id,
                    policy = ?self.children[position].restart_policy(),
                    reason = ?reason,
                    "not restarting child"
                );
                self.children.remove(position);
                RuntimeControl::Continue
            }
            RestartDecision::Escalate => {
                tracing::error!(
                    supervisor = %self.name,
                    "restart intensity exceeded, shutting down"
                );
                self.shutdown_children().await;
                RuntimeControl::Stop
            }
            RestartDecision::Restart { delay, strategy } => {
                if delay.is_zero() {
                    self.apply_restart(position, strategy).await;
                } else {
                    self.defer_restart(id, delay, strategy, generation);
                }

                RuntimeControl::Continue
            }
        }
    }

    /// Schedules a restart for after `delay` without blocking the command loop.
    ///
    /// Sleeping inline would stall `Shutdown`, `which_children` and every other
    /// command for the whole backoff - up to 30s with the default settings.
    fn defer_restart(
        &self,
        id: ChildId,
        delay: std::time::Duration,
        strategy: RestartStrategy,
        generation: u64,
    ) {
        tracing::debug!(
            supervisor = %self.name,
            child = %id,
            delay_ms = %delay.as_millis(),
            "waiting before restart"
        );

        let control_tx = self.control_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let _send = control_tx.send(StatefulSupervisorCommand::RestartChild {
                id,
                strategy,
                generation,
            });
        });
    }

    /// Carries out a restart whose backoff has elapsed.
    async fn handle_restart_child(&mut self, id: &str, strategy: RestartStrategy, generation: u64) {
        let Some(position) = self
            .children
            .iter()
            .position(|c| c.id() == id && c.generation() == generation)
        else {
            // Terminated, dropped, or already restarted (by another child's
            // restart round) while it was backing off.
            tracing::debug!(
                supervisor = %self.name,
                child = %id,
                "child is no longer supervised, skipping deferred restart"
            );
            return;
        };

        self.apply_restart(position, strategy).await;
    }

    async fn apply_restart(&mut self, position: usize, strategy: RestartStrategy) {
        match strategy {
            RestartStrategy::OneForOne => self.restart_child(position).await,
            RestartStrategy::OneForAll => self.restart_all_children().await,
            RestartStrategy::RestForOne => self.restart_from(position).await,
        }
    }

    #[allow(clippy::indexing_slicing)]
    async fn restart_child(&mut self, position: usize) {
        self.children[position]
            .shutdown(self.shutdown_timeout)
            .await;

        let generation = self.next_generation();
        let id = Self::respawn(
            &mut self.children[position],
            &self.name,
            &self.control_tx,
            generation,
        );
        tracing::debug!(
            supervisor = %self.name,
            child = %id,
            "child restarted"
        );
    }

    async fn restart_all_children(&mut self) {
        tracing::debug!(
            supervisor = %self.name,
            "restarting all children (one_for_all)"
        );

        let children = std::mem::take(&mut self.children);
        self.children = self.stop_and_respawn(children).await;
    }

    #[allow(clippy::indexing_slicing)]
    async fn restart_from(&mut self, position: usize) {
        tracing::debug!(
            supervisor = %self.name,
            position = %position,
            "restarting from position (rest_for_one)"
        );

        let affected = self.children.split_off(position);
        let mut restarted = self.stop_and_respawn(affected).await;
        self.children.append(&mut restarted);
    }

    /// Restart round shared by `OneForAll` and `RestForOne`, following OTP:
    /// stop every affected child in reverse start order, then start them again
    /// in start order. `Temporary` children leave supervision instead.
    async fn stop_and_respawn(
        &mut self,
        mut children: Vec<StatefulChild<W>>,
    ) -> Vec<StatefulChild<W>> {
        for child in children.iter_mut().rev() {
            child.shutdown(self.shutdown_timeout).await;
        }

        let mut kept = Vec::with_capacity(children.len());
        for mut child in children {
            if Self::leaves_supervision(&child) {
                tracing::debug!(
                    supervisor = %self.name,
                    child = %child.id(),
                    "temporary child not restarted, dropping from supervision"
                );
                self.backoff_tracker.reset_child(child.id());
                continue;
            }

            // Nested supervisors were just shut down too, so skipping them would
            // strand a dead handle in the children list.
            let generation = self.next_generation();
            let id = Self::respawn(&mut child, &self.name, &self.control_tx, generation);
            tracing::debug!(
                supervisor = %self.name,
                child = %id,
                "child restarted"
            );
            kept.push(child);
        }
        kept
    }

    /// Whether a child stopped as part of a restart round is gone for good.
    ///
    /// A `Temporary` child is never restarted, not even when it was a sibling's
    /// failure that stopped it, so it leaves supervision here.
    fn leaves_supervision(child: &StatefulChild<W>) -> bool {
        child.restart_policy() == Some(RestartPolicy::Temporary)
    }

    /// Replaces a child that has been shut down with a freshly started one,
    /// returning its id.
    fn respawn(
        child: &mut StatefulChild<W>,
        supervisor_name: &str,
        control_tx: &mpsc::UnboundedSender<StatefulSupervisorCommand<W>>,
        generation: u64,
    ) -> ChildId {
        match child {
            StatefulChild::Worker(worker) => {
                let spec = worker.spec.clone();
                let id = spec.id.clone();
                *child = StatefulChild::Worker(StatefulWorkerProcess::spawn(
                    spec,
                    supervisor_name.to_owned(),
                    control_tx.clone(),
                    generation,
                ));
                id
            }
            StatefulChild::Supervisor { spec, .. } => {
                let shared_spec = Arc::clone(spec);
                let id = shared_spec.name.clone();
                let handle = StatefulSupervisorHandle::start_supervised(
                    (*shared_spec).clone(),
                    control_tx.clone(),
                    generation,
                );
                *child = StatefulChild::Supervisor {
                    handle,
                    spec: shared_spec,
                    generation,
                };
                id
            }
        }
    }

    /// Stops every child in reverse start order (as OTP does, so a child is
    /// never left running without the earlier siblings it may depend on), and
    /// abandons linked children that are still initializing.
    async fn shutdown_children(&mut self) {
        for (id, pending) in self.pending_linked.drain() {
            // It never finished `initialize`, so it has not started: abort it
            // rather than wait out the shutdown timeout (`initialize` does not
            // observe the shutdown signal).
            let mut worker = pending.worker;
            let _graceful = worker.stop(std::time::Duration::ZERO).await;
            let _send = pending
                .respond_to
                .send(Err(StatefulSupervisorError::ShuttingDown(
                    self.name.clone(),
                )));
            tracing::debug!(
                supervisor = %self.name,
                child = %id,
                "abandoned initializing child"
            );
        }

        while let Some(mut child) = self.children.pop() {
            let id = child.id().to_owned();
            child.shutdown(self.shutdown_timeout).await;
            tracing::debug!(
                supervisor = %self.name,
                child = %id,
                "shut down child"
            );
        }
    }
}

// ============================================================================
// Supervisor Handle (Stateful)
// ============================================================================

/// Handle used to interact with a running stateful supervisor tree.
pub struct StatefulSupervisorHandle<W: Worker> {
    pub(crate) name: Arc<String>,
    pub(crate) control_tx: mpsc::UnboundedSender<StatefulSupervisorCommand<W>>,
    /// Becomes `true` once the supervisor task has finished, i.e. after every
    /// child has been stopped and its `shutdown` hook has run (or timed out).
    pub(crate) stopped: tokio::sync::watch::Receiver<bool>,
}

// Hand-written so cloning a handle does not require the worker itself to be
// `Clone`; a handle is just a name and a channel.
impl<W: Worker> Clone for StatefulSupervisorHandle<W> {
    fn clone(&self) -> Self {
        Self {
            name: Arc::clone(&self.name),
            control_tx: self.control_tx.clone(),
            stopped: self.stopped.clone(),
        }
    }
}

impl<W: Worker> StatefulSupervisorHandle<W> {
    /// Spawns a stateful supervisor tree based on the provided specification.
    #[must_use]
    pub fn start(spec: StatefulSupervisorSpec<W>) -> Self {
        Self::start_inner(spec, None)
    }

    /// Spawns a supervisor as the child of another one, reporting its own
    /// termination to `parent_tx`.
    ///
    /// Without this a nested supervisor that escalates simply stops: the parent
    /// never learns, and keeps a dead handle in its children list forever.
    pub(crate) fn start_supervised(
        spec: StatefulSupervisorSpec<W>,
        parent_tx: mpsc::UnboundedSender<StatefulSupervisorCommand<W>>,
        generation: u64,
    ) -> Self {
        Self::start_inner(spec, Some((parent_tx, generation)))
    }

    fn start_inner(
        spec: StatefulSupervisorSpec<W>,
        parent: Option<(mpsc::UnboundedSender<StatefulSupervisorCommand<W>>, u64)>,
    ) -> Self {
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (stopped_tx, stopped) = tokio::sync::watch::channel(false);
        let name_arc = Arc::new(spec.name.clone());
        let runtime = StatefulSupervisorRuntime::new(spec, control_rx, control_tx.clone());

        let runtime_name = Arc::clone(&name_arc);
        tokio::spawn(async move {
            let reason = runtime.run().await;
            tracing::debug!(name = %*runtime_name, reason = ?reason, "supervisor stopped");

            // A `Shutdown` reason is the parent's own doing and is ignored there;
            // an abnormal exit is what makes the parent restart this subtree.
            if let Some((tx, generation)) = parent {
                let _send = tx.send(StatefulSupervisorCommand::ChildTerminated {
                    id: (*runtime_name).clone(),
                    reason,
                    generation,
                });
            }
            stopped_tx.send_replace(true);
        });

        Self {
            name: name_arc,
            control_tx,
            stopped,
        }
    }

    /// Dynamically starts a new child worker
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down or a child with this ID already exists.
    pub async fn start_child(
        &self,
        id: impl Into<String>,
        factory: impl Fn(Arc<WorkerContext>) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
        context: Arc<WorkerContext>,
    ) -> Result<ChildId, StatefulSupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();
        let spec = StatefulWorkerSpec::new(id, factory, restart_policy, context);

        self.control_tx
            .send(StatefulSupervisorCommand::StartChild {
                spec,
                respond_to: result_tx,
            })
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?
    }

    /// Dynamically starts a new child worker with linked initialization.
    ///
    /// This method waits for the worker's initialization to complete before returning.
    /// If initialization fails or times out, an error is returned and the worker is not added.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the child
    /// * `factory` - Factory function to create the worker
    /// * `restart_policy` - How to handle worker termination after it starts running
    /// * `context` - Shared context for stateful workers
    /// * `timeout` - Maximum time to wait for initialization
    ///
    /// # Errors
    ///
    /// * `StatefulSupervisorError::InitializationFailed` - Worker initialization returned an error
    /// * `StatefulSupervisorError::InitializationTimeout` - Worker didn't initialize within timeout
    /// * `StatefulSupervisorError::ChildAlreadyExists` - A child with this ID already exists
    /// * `StatefulSupervisorError::ShuttingDown` - Supervisor is shutting down
    ///
    /// # Note
    ///
    /// Initialization failures do NOT trigger restart policies. The worker must successfully
    /// initialize before restart policies take effect.
    pub async fn start_child_linked(
        &self,
        id: impl Into<String>,
        factory: impl Fn(Arc<WorkerContext>) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
        context: Arc<WorkerContext>,
        timeout: std::time::Duration,
    ) -> Result<ChildId, StatefulSupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();
        let spec = StatefulWorkerSpec::new(id, factory, restart_policy, context);

        self.control_tx
            .send(StatefulSupervisorCommand::StartChildLinked {
                spec,
                timeout,
                respond_to: result_tx,
            })
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?
    }

    /// Dynamically terminates a child
    ///
    /// # Errors
    ///
    /// Returns an error if the child is not found or the supervisor is shutting down.
    pub async fn terminate_child(&self, id: &str) -> Result<(), StatefulSupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(StatefulSupervisorCommand::TerminateChild {
                id: id.to_owned(),
                respond_to: result_tx,
            })
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?
    }

    /// Returns information about all children
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down.
    pub async fn which_children(&self) -> Result<Vec<ChildInfo>, StatefulSupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(StatefulSupervisorCommand::WhichChildren {
                respond_to: result_tx,
            })
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?
    }

    /// Shuts the supervisor tree down gracefully and waits until it has stopped.
    ///
    /// Children are stopped in reverse start order, as in OTP. Each worker is
    /// asked to stop and given the shutdown timeout to finish, including its
    /// `shutdown` hook, before it is aborted. When this returns, every child has
    /// stopped, so it is safe to exit the process.
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor had already stopped before the call.
    pub async fn shutdown(&self) -> Result<(), StatefulSupervisorError> {
        self.control_tx
            .send(StatefulSupervisorCommand::Shutdown)
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;
        self.wait_stopped().await;
        Ok(())
    }

    /// Waits until the supervisor task has finished, whether it was shut down
    /// or stopped because its restart intensity was exceeded.
    pub async fn wait_stopped(&self) {
        let mut stopped_rx = self.stopped.clone();
        // An error means the sender is gone, which only happens once the
        // supervisor task has ended: it is stopped either way.
        let _stopped = stopped_rx.wait_for(|has_stopped| *has_stopped).await;
    }

    /// Returns the supervisor's name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the supervisor's restart strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down.
    pub async fn restart_strategy(&self) -> Result<RestartStrategy, StatefulSupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(StatefulSupervisorCommand::GetRestartStrategy {
                respond_to: result_tx,
            })
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))
    }

    /// Returns the supervisor's uptime in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down.
    pub async fn uptime(&self) -> Result<u64, StatefulSupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(StatefulSupervisorCommand::GetUptime {
                respond_to: result_tx,
            })
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| StatefulSupervisorError::ShuttingDown(self.name().to_owned()))
    }
}

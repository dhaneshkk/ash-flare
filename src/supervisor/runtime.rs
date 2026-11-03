//! Supervisor runtime - internal state machine

use super::child::Child;
use super::error::SupervisorError;
use super::handle::SupervisorHandle;
use super::spec::{ChildSpec, SupervisorSpec};
use crate::restart::{BackoffTracker, RestartPolicy, RestartStrategy, RestartTracker};
use crate::supervisor_common::{LinkedInit, RestartDecision, decide_restart};
use crate::types::{ChildExitReason, ChildId, ChildInfo};
use crate::worker::{Worker, WorkerProcess, WorkerSpec, WorkerTermination};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Internal commands sent to supervisor runtime
pub(crate) enum SupervisorCommand<W: Worker> {
    StartChild {
        spec: WorkerSpec<W>,
        respond_to: oneshot::Sender<Result<ChildId, SupervisorError>>,
    },
    StartChildLinked {
        spec: WorkerSpec<W>,
        timeout: std::time::Duration,
        respond_to: oneshot::Sender<Result<ChildId, SupervisorError>>,
    },
    TerminateChild {
        id: ChildId,
        respond_to: oneshot::Sender<Result<(), SupervisorError>>,
    },
    WhichChildren {
        respond_to: oneshot::Sender<Result<Vec<ChildInfo>, SupervisorError>>,
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
    worker: WorkerProcess<W>,
    respond_to: oneshot::Sender<Result<ChildId, SupervisorError>>,
    /// A termination reported before the init outcome was processed (for
    /// example `run` returning immediately). Applied once the child is admitted.
    early_exit: Option<ChildExitReason>,
}

impl<W: Worker> From<WorkerTermination> for SupervisorCommand<W> {
    fn from(term: WorkerTermination) -> Self {
        SupervisorCommand::ChildTerminated {
            id: term.id,
            reason: term.reason,
            generation: term.generation,
        }
    }
}

/// Whether the runtime loop should keep serving commands or terminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeControl {
    /// Keep processing commands.
    Continue,
    /// Terminate the supervisor task.
    Stop,
}

/// Internal state machine that manages supervisor lifecycle and child processes
pub(crate) struct SupervisorRuntime<W: Worker> {
    name: String,
    children: Vec<Child<W>>,
    control_rx: mpsc::UnboundedReceiver<SupervisorCommand<W>>,
    control_tx: mpsc::UnboundedSender<SupervisorCommand<W>>,
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

impl<W: Worker> SupervisorRuntime<W> {
    pub(crate) fn new(
        spec: SupervisorSpec<W>,
        control_rx: mpsc::UnboundedReceiver<SupervisorCommand<W>>,
        control_tx: mpsc::UnboundedSender<SupervisorCommand<W>>,
    ) -> Self {
        let mut children = Vec::with_capacity(spec.children.len());
        let mut next_generation = 0_u64;

        for child_spec in spec.children {
            next_generation = next_generation.saturating_add(1);
            match child_spec {
                ChildSpec::Worker(worker_spec) => {
                    let worker = WorkerProcess::spawn(
                        worker_spec,
                        spec.name.clone(),
                        control_tx.clone(),
                        next_generation,
                    );
                    children.push(Child::Worker(worker));
                }
                ChildSpec::Supervisor(supervisor_spec) => {
                    let supervisor = SupervisorHandle::start_supervised(
                        (*supervisor_spec).clone(),
                        control_tx.clone(),
                        next_generation,
                    );
                    children.push(Child::Supervisor {
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
                SupervisorCommand::StartChild { spec, respond_to } => {
                    let result = self.handle_start_child(spec);
                    let _send = respond_to.send(result);
                }
                SupervisorCommand::StartChildLinked {
                    spec,
                    timeout,
                    respond_to,
                } => {
                    self.handle_start_child_linked(spec, timeout, respond_to);
                }
                SupervisorCommand::LinkedInitDone {
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
                SupervisorCommand::TerminateChild { id, respond_to } => {
                    let result = self.handle_terminate_child(&id).await;
                    let _send = respond_to.send(result);
                }
                SupervisorCommand::WhichChildren { respond_to } => {
                    let result = self.handle_which_children();
                    let _send = respond_to.send(result);
                }
                SupervisorCommand::GetRestartStrategy { respond_to } => {
                    let _send = respond_to.send(self.restart_strategy);
                }
                SupervisorCommand::GetUptime { respond_to } => {
                    let uptime = self.created_at.elapsed().as_secs();
                    let _send = respond_to.send(uptime);
                }
                SupervisorCommand::ChildTerminated {
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
                SupervisorCommand::RestartChild {
                    id,
                    strategy,
                    generation,
                } => {
                    self.handle_restart_child(&id, strategy, generation).await;
                }
                SupervisorCommand::Shutdown => {
                    self.shutdown_children().await;
                    return ChildExitReason::Shutdown;
                }
            }
        }

        self.shutdown_children().await;
        ChildExitReason::Shutdown
    }

    fn handle_start_child(&mut self, spec: WorkerSpec<W>) -> Result<ChildId, SupervisorError> {
        // Check if child with same ID already exists
        if self.id_in_use(&spec.id) {
            return Err(SupervisorError::ChildAlreadyExists(spec.id.clone()));
        }

        let id = spec.id.clone();
        let generation = self.next_generation();
        let worker =
            WorkerProcess::spawn(spec, self.name.clone(), self.control_tx.clone(), generation);

        self.children.push(Child::Worker(worker));
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
    /// [`SupervisorCommand::LinkedInitDone`]. Meanwhile the supervisor keeps
    /// serving commands and handling other children's failures.
    fn handle_start_child_linked(
        &mut self,
        spec: WorkerSpec<W>,
        timeout: std::time::Duration,
        respond_to: oneshot::Sender<Result<ChildId, SupervisorError>>,
    ) {
        if self.id_in_use(&spec.id) {
            let _send = respond_to.send(Err(SupervisorError::ChildAlreadyExists(spec.id.clone())));
            return;
        }

        let id = spec.id.clone();
        let generation = self.next_generation();
        let (init_tx, init_rx) = oneshot::channel();

        let worker = WorkerProcess::spawn_with_link(
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
            let _send = control_tx.send(SupervisorCommand::LinkedInitDone {
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
                self.children.push(Child::Worker(pending.worker));
                let _send = pending.respond_to.send(Ok(id.clone()));
                return match pending.early_exit {
                    Some(reason) => self.handle_child_terminated(id, reason, generation).await,
                    None => RuntimeControl::Continue,
                };
            }
            LinkedInit::Failed(reason) => SupervisorError::InitializationFailed {
                child_id: id.clone(),
                reason,
            },
            LinkedInit::Vanished => SupervisorError::InitializationFailed {
                child_id: id.clone(),
                reason: "worker exited before reporting initialization".to_owned(),
            },
            LinkedInit::TimedOut(timeout) => SupervisorError::InitializationTimeout {
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

    async fn handle_terminate_child(&mut self, id: &str) -> Result<(), SupervisorError> {
        let position = self
            .children
            .iter()
            .position(|c| c.id() == id)
            .ok_or_else(|| SupervisorError::ChildNotFound(id.to_owned()))?;

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
    fn handle_which_children(&self) -> Result<Vec<ChildInfo>, SupervisorError> {
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
            let _send = control_tx.send(SupervisorCommand::RestartChild {
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
    async fn stop_and_respawn(&mut self, mut children: Vec<Child<W>>) -> Vec<Child<W>> {
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
    fn leaves_supervision(child: &Child<W>) -> bool {
        child.restart_policy() == Some(RestartPolicy::Temporary)
    }

    /// Replaces a child that has been shut down with a freshly started one,
    /// returning its id.
    fn respawn(
        child: &mut Child<W>,
        supervisor_name: &str,
        control_tx: &mpsc::UnboundedSender<SupervisorCommand<W>>,
        generation: u64,
    ) -> ChildId {
        match child {
            Child::Worker(worker) => {
                let spec = worker.spec.clone();
                let id = spec.id.clone();
                *child = Child::Worker(WorkerProcess::spawn(
                    spec,
                    supervisor_name.to_owned(),
                    control_tx.clone(),
                    generation,
                ));
                id
            }
            Child::Supervisor { spec, .. } => {
                let shared_spec = Arc::clone(spec);
                let id = shared_spec.name.clone();
                let handle = SupervisorHandle::start_supervised(
                    (*shared_spec).clone(),
                    control_tx.clone(),
                    generation,
                );
                *child = Child::Supervisor {
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
                .send(Err(SupervisorError::ShuttingDown(self.name.clone())));
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

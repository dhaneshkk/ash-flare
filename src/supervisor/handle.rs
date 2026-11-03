//! Supervisor handle - public API for interacting with supervisors

use super::error::SupervisorError;
use super::runtime::{SupervisorCommand, SupervisorRuntime};
use super::spec::SupervisorSpec;
use crate::restart::RestartPolicy;
use crate::types::{ChildId, ChildInfo};
use crate::worker::{Worker, WorkerSpec};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};

/// Handle used to interact with a running supervisor tree.
pub struct SupervisorHandle<W: Worker> {
    pub(crate) name: Arc<String>,
    pub(crate) control_tx: mpsc::UnboundedSender<SupervisorCommand<W>>,
    /// Becomes `true` once the supervisor task has finished, i.e. after every
    /// child has been stopped and its `shutdown` hook has run (or timed out).
    pub(crate) stopped: watch::Receiver<bool>,
}

// Hand-written so cloning a handle does not require the worker itself to be
// `Clone`; a handle is just a name and a channel.
impl<W: Worker> Clone for SupervisorHandle<W> {
    fn clone(&self) -> Self {
        Self {
            name: Arc::clone(&self.name),
            control_tx: self.control_tx.clone(),
            stopped: self.stopped.clone(),
        }
    }
}

impl<W: Worker> SupervisorHandle<W> {
    /// Spawns a supervisor tree based on the provided specification.
    #[must_use]
    pub fn start(spec: SupervisorSpec<W>) -> Self {
        Self::start_inner(spec, None)
    }

    /// Spawns a supervisor as the child of another one, reporting its own
    /// termination to `parent_tx`.
    ///
    /// Without this a nested supervisor that escalates simply stops: the parent
    /// never learns, and keeps a dead handle in its children list forever.
    pub(crate) fn start_supervised(
        spec: SupervisorSpec<W>,
        parent_tx: mpsc::UnboundedSender<SupervisorCommand<W>>,
        generation: u64,
    ) -> Self {
        Self::start_inner(spec, Some((parent_tx, generation)))
    }

    fn start_inner(
        spec: SupervisorSpec<W>,
        parent: Option<(mpsc::UnboundedSender<SupervisorCommand<W>>, u64)>,
    ) -> Self {
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (stopped_tx, stopped) = watch::channel(false);
        let name_arc = Arc::new(spec.name.clone());
        let runtime = SupervisorRuntime::new(spec, control_rx, control_tx.clone());

        let runtime_name = Arc::clone(&name_arc);
        tokio::spawn(async move {
            let reason = runtime.run().await;
            tracing::debug!(name = %*runtime_name, reason = ?reason, "supervisor stopped");

            // A `Shutdown` reason is the parent's own doing and is ignored there;
            // an abnormal exit is what makes the parent restart this subtree.
            if let Some((tx, generation)) = parent {
                let _send = tx.send(SupervisorCommand::ChildTerminated {
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
        factory: impl Fn() -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
    ) -> Result<ChildId, SupervisorError> {
        self.start_child_with_signal(id, move |_signal| factory(), restart_policy)
            .await
    }

    /// Dynamically starts a child whose factory receives the worker's
    /// [`ShutdownSignal`](crate::ShutdownSignal), so it can observe cooperative
    /// termination.
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down or a child with this ID already exists.
    pub async fn start_child_with_signal(
        &self,
        id: impl Into<String>,
        factory: impl Fn(crate::ShutdownSignal) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
    ) -> Result<ChildId, SupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();
        let spec = WorkerSpec::with_signal(id, factory, restart_policy);

        self.control_tx
            .send(SupervisorCommand::StartChild {
                spec,
                respond_to: result_tx,
            })
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?
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
    /// * `timeout` - Maximum time to wait for initialization
    ///
    /// # Errors
    ///
    /// * `SupervisorError::InitializationFailed` - Worker initialization returned an error
    /// * `SupervisorError::InitializationTimeout` - Worker didn't initialize within timeout
    /// * `SupervisorError::ChildAlreadyExists` - A child with this ID already exists
    /// * `SupervisorError::ShuttingDown` - Supervisor is shutting down
    ///
    /// # Note
    ///
    /// Initialization failures do NOT trigger restart policies. The worker must successfully
    /// initialize before restart policies take effect.
    pub async fn start_child_linked(
        &self,
        id: impl Into<String>,
        factory: impl Fn() -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
        timeout: std::time::Duration,
    ) -> Result<ChildId, SupervisorError> {
        self.start_child_linked_with_signal(id, move |_signal| factory(), restart_policy, timeout)
            .await
    }

    /// Like [`SupervisorHandle::start_child_linked`], but the factory receives
    /// the worker's [`ShutdownSignal`](crate::ShutdownSignal) so the worker can
    /// observe cooperative termination.
    ///
    /// # Errors
    ///
    /// Same as [`SupervisorHandle::start_child_linked`].
    pub async fn start_child_linked_with_signal(
        &self,
        id: impl Into<String>,
        factory: impl Fn(crate::ShutdownSignal) -> W + Send + Sync + 'static,
        restart_policy: RestartPolicy,
        timeout: std::time::Duration,
    ) -> Result<ChildId, SupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();
        let spec = WorkerSpec::with_signal(id, factory, restart_policy);

        self.control_tx
            .send(SupervisorCommand::StartChildLinked {
                spec,
                timeout,
                respond_to: result_tx,
            })
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?
    }

    /// Dynamically terminates a child
    ///
    /// # Errors
    ///
    /// Returns an error if the child is not found or the supervisor is shutting down.
    pub async fn terminate_child(&self, id: &str) -> Result<(), SupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(SupervisorCommand::TerminateChild {
                id: id.to_owned(),
                respond_to: result_tx,
            })
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?
    }

    /// Returns information about all children
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down.
    pub async fn which_children(&self) -> Result<Vec<ChildInfo>, SupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(SupervisorCommand::WhichChildren {
                respond_to: result_tx,
            })
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?
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
    pub async fn shutdown(&self) -> Result<(), SupervisorError> {
        self.control_tx
            .send(SupervisorCommand::Shutdown)
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;
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
    pub async fn restart_strategy(
        &self,
    ) -> Result<crate::restart::RestartStrategy, SupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(SupervisorCommand::GetRestartStrategy {
                respond_to: result_tx,
            })
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))
    }

    /// Returns the supervisor's uptime in seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the supervisor is shutting down.
    pub async fn uptime(&self) -> Result<u64, SupervisorError> {
        let (result_tx, result_rx) = oneshot::channel();

        self.control_tx
            .send(SupervisorCommand::GetUptime {
                respond_to: result_tx,
            })
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))?;

        result_rx
            .await
            .map_err(|_| SupervisorError::ShuttingDown(self.name().to_owned()))
    }
}

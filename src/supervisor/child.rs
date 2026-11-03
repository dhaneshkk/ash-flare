//! Child management - handles both worker and supervisor children

use super::handle::SupervisorHandle;
use super::spec::SupervisorSpec;
use crate::restart::RestartPolicy;
use crate::types::ChildType;
use crate::worker::{Worker, WorkerProcess};
use std::sync::Arc;

/// Represents either a worker or a nested supervisor in the supervision tree
pub(crate) enum Child<W: Worker> {
    Worker(WorkerProcess<W>),
    Supervisor {
        handle: SupervisorHandle<W>,
        spec: Arc<SupervisorSpec<W>>,
        /// Incarnation number of this nested supervisor; see
        /// [`WorkerProcess::generation`](crate::worker::WorkerProcess).
        generation: u64,
    },
}

impl<W: Worker> Child<W> {
    #[inline]
    pub fn id(&self) -> &str {
        match self {
            Child::Worker(w) => &w.spec.id,
            Child::Supervisor { spec, .. } => &spec.name,
        }
    }

    /// Incarnation number of the currently running instance of this child.
    #[inline]
    pub fn generation(&self) -> u64 {
        match self {
            Child::Worker(w) => w.generation,
            Child::Supervisor { generation, .. } => *generation,
        }
    }

    #[inline]
    pub fn child_type(&self) -> ChildType {
        match self {
            Child::Worker(_) => ChildType::Worker,
            Child::Supervisor { .. } => ChildType::Supervisor,
        }
    }

    #[inline]
    #[allow(clippy::unnecessary_wraps)]
    pub fn restart_policy(&self) -> Option<RestartPolicy> {
        match self {
            Child::Worker(w) => Some(w.spec.restart_policy),
            Child::Supervisor { .. } => Some(RestartPolicy::Permanent),
        }
    }

    /// Policy as seen by the shared restart decision: `None` for nested
    /// supervisors, which are always treated as permanent.
    #[inline]
    pub fn restart_policy_for_decision(&self) -> Option<RestartPolicy> {
        match self {
            Child::Worker(w) => Some(w.spec.restart_policy),
            Child::Supervisor { .. } => None,
        }
    }

    /// Stops the child, allowing `timeout` for a worker to wind down
    /// cooperatively before its task is aborted.
    pub async fn shutdown(&mut self, timeout: std::time::Duration) {
        match self {
            Child::Worker(w) => {
                let _graceful = w.stop(timeout).await;
            }
            Child::Supervisor { handle, .. } => {
                // Waits until the nested tree has fully stopped, so a restart
                // never runs the old and the new subtree side by side.
                let _shutdown_result = handle.shutdown().await;
            }
        }
    }
}

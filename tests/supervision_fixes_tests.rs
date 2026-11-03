//! Regression tests for supervision-correctness fixes:
//!
//! 1. A panicking worker is reported and restarted like a failed one.
//! 2. `shutdown().await` returns only after every child (and its `shutdown`
//!    hook) has stopped, and children stop in reverse start order.
//! 3. Termination reports from already-replaced instances do not trigger a
//!    second restart round.
//! 4. `start_child_linked` does not block the supervisor while it waits.
//! 5. Restarting a nested supervisor never overlaps the old and new subtree.

use ash_flare::{
    RestartBackoff, RestartIntensity, RestartPolicy, RestartStrategy, ShutdownSignal,
    StatefulSupervisorHandle, StatefulSupervisorSpec, SupervisorError, SupervisorHandle,
    SupervisorSpec, Worker,
};
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What a probe worker does in `run`.
#[derive(Clone, Copy)]
enum Behaviour {
    /// Panics immediately.
    Panic,
    /// Panics during `initialize`.
    PanicInInit,
    /// Fails after `ms`, but only on its first run.
    FailFirstRunAfter(u64),
    /// Returns `Ok` immediately (a short-lived job).
    ReturnImmediately,
    /// Runs until asked to stop.
    Idle,
    /// Takes `ms` to initialize, then runs until asked to stop.
    SlowInit(u64),
}

/// Shared observations about every probe worker in a test.
#[derive(Default)]
struct Probe {
    /// `run` calls, per child id, in call order.
    runs: Mutex<Vec<String>>,
    /// `shutdown` hooks completed, in completion order.
    stopped: Mutex<Vec<String>>,
    /// Instances currently between `run` start and `shutdown` hook end.
    alive: AtomicUsize,
    /// Highest value `alive` reached.
    max_alive: AtomicUsize,
}

impl Probe {
    fn runs_of(&self, id: &str) -> usize {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|r| *r == id)
            .count()
    }

    fn stopped(&self) -> Vec<String> {
        self.stopped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

struct ProbeWorker {
    id: String,
    behaviour: Behaviour,
    probe: Arc<Probe>,
    signal: Option<ShutdownSignal>,
    /// Extra time the `shutdown` hook takes, e.g. to flush.
    hook_ms: u64,
}

#[async_trait]
impl Worker for ProbeWorker {
    type Error = std::io::Error;

    async fn initialize(&mut self) -> Result<(), Self::Error> {
        match self.behaviour {
            Behaviour::PanicInInit => panic!("init panic in {}", self.id),
            Behaviour::SlowInit(ms) => tokio::time::sleep(Duration::from_millis(ms)).await,
            _ => {}
        }
        Ok(())
    }

    async fn run(&mut self) -> Result<(), Self::Error> {
        let first_run = {
            let mut runs = self
                .probe
                .runs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            runs.push(self.id.clone());
            runs.iter().filter(|r| **r == self.id).count() == 1
        };
        let now_alive = self.probe.alive.fetch_add(1, Ordering::SeqCst) + 1;
        self.probe.max_alive.fetch_max(now_alive, Ordering::SeqCst);

        match self.behaviour {
            Behaviour::Panic => panic!("run panic in {}", self.id),
            Behaviour::FailFirstRunAfter(ms) if first_run => {
                tokio::time::sleep(Duration::from_millis(ms)).await;
                Err(std::io::Error::other("first run fails"))
            }
            Behaviour::ReturnImmediately => Ok(()),
            _ => {
                match &self.signal {
                    Some(signal) => signal.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
                Ok(())
            }
        }
    }

    async fn shutdown(&mut self) -> Result<(), Self::Error> {
        if self.hook_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.hook_ms)).await;
        }
        self.probe
            .stopped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(self.id.clone());
        let _previous = self
            .probe
            .alive
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_sub(1))
            });
        Ok(())
    }
}

fn factory(
    id: &str,
    behaviour: Behaviour,
    probe: &Arc<Probe>,
) -> impl Fn(ShutdownSignal) -> ProbeWorker + Send + Sync + 'static {
    factory_with_hook(id, behaviour, probe, 0)
}

fn factory_with_hook(
    id: &str,
    behaviour: Behaviour,
    probe: &Arc<Probe>,
    hook_ms: u64,
) -> impl Fn(ShutdownSignal) -> ProbeWorker + Send + Sync + 'static {
    let id = id.to_owned();
    let probe = Arc::clone(probe);
    move |signal| ProbeWorker {
        id: id.clone(),
        behaviour,
        probe: Arc::clone(&probe),
        signal: Some(signal),
        hook_ms,
    }
}

fn quiet_panics() {
    // Expected panics would otherwise clutter the test output.
    std::panic::set_hook(Box::new(|_| {}));
}

fn fast_spec(name: &str) -> SupervisorSpec<ProbeWorker> {
    SupervisorSpec::new(name)
        .with_restart_backoff(RestartBackoff::none())
        .with_restart_intensity(RestartIntensity::new(100, 10))
}

// ---------------------------------------------------------------------------
// 1. Panics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn panicking_worker_is_restarted() {
    quiet_panics();
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(fast_spec("panic").with_worker_signal(
        "p",
        factory("p", Behaviour::Panic, &probe),
        RestartPolicy::Permanent,
    ));

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        probe.runs_of("p") > 1,
        "a panicking Permanent worker must be restarted, ran {} time(s)",
        probe.runs_of("p")
    );
    let _shutdown = handle.shutdown().await;
}

#[tokio::test]
async fn persistent_panics_exhaust_restart_intensity() {
    quiet_panics();
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(
        SupervisorSpec::new("panic-escalate")
            .with_restart_backoff(RestartBackoff::none())
            .with_restart_intensity(RestartIntensity::new(3, 10))
            .with_worker_signal(
                "p",
                factory("p", Behaviour::Panic, &probe),
                RestartPolicy::Permanent,
            ),
    );

    tokio::time::timeout(Duration::from_secs(2), handle.wait_stopped())
        .await
        .expect("a worker that keeps panicking must make the supervisor escalate");
    assert_eq!(probe.runs_of("p"), 4, "initial run plus 3 restarts");
}

#[tokio::test]
async fn panicking_stateful_worker_is_restarted() {
    quiet_panics();
    let probe = Arc::new(Probe::default());
    let worker_probe = Arc::clone(&probe);
    let handle = StatefulSupervisorHandle::start(
        StatefulSupervisorSpec::new("stateful-panic")
            .with_restart_backoff(RestartBackoff::none())
            .with_restart_intensity(RestartIntensity::new(100, 10))
            .with_worker_signal(
                "p",
                move |_ctx, signal| ProbeWorker {
                    id: "p".to_owned(),
                    behaviour: Behaviour::Panic,
                    probe: Arc::clone(&worker_probe),
                    signal: Some(signal),
                    hook_ms: 0,
                },
                RestartPolicy::Permanent,
            ),
    );

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        probe.runs_of("p") > 1,
        "stateful supervisors must restart panicking workers too"
    );
    let _shutdown = handle.shutdown().await;
}

#[tokio::test]
async fn panic_in_linked_initialize_is_reported_as_init_failure() {
    quiet_panics();
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(fast_spec("init-panic"));

    let result = handle
        .start_child_linked_with_signal(
            "bad",
            factory("bad", Behaviour::PanicInInit, &probe),
            RestartPolicy::Permanent,
            Duration::from_secs(2),
        )
        .await;

    match result {
        Err(SupervisorError::InitializationFailed { reason, .. }) => {
            assert!(reason.contains("panicked"), "unexpected reason: {reason}");
            assert!(
                reason.contains("init panic in bad"),
                "panic message lost: {reason}"
            );
        }
        other => panic!("expected InitializationFailed, got {other:?}"),
    }
    let children = handle
        .which_children()
        .await
        .expect("supervisor is running");
    assert!(
        children.is_empty(),
        "a failed linked child must not be supervised"
    );
    let _shutdown = handle.shutdown().await;
}

// ---------------------------------------------------------------------------
// 2. Shutdown waits, in reverse order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shutdown_waits_for_hooks_and_stops_in_reverse_order() {
    let probe = Arc::new(Probe::default());
    let mut spec = fast_spec("ordered").with_shutdown_timeout(Duration::from_secs(5));
    for id in ["a", "b", "c"] {
        spec = spec.with_worker_signal(
            id,
            factory_with_hook(id, Behaviour::Idle, &probe, 30),
            RestartPolicy::Permanent,
        );
    }
    let handle = SupervisorHandle::start(spec);
    tokio::time::sleep(Duration::from_millis(50)).await;

    handle.shutdown().await.expect("supervisor was running");

    assert_eq!(
        probe.stopped(),
        vec!["c", "b", "a"],
        "every hook must have run by the time shutdown returns, newest child first"
    );
}

#[tokio::test]
async fn stateful_shutdown_waits_for_hooks() {
    let probe = Arc::new(Probe::default());
    let mut spec = StatefulSupervisorSpec::new("stateful-ordered")
        .with_shutdown_timeout(Duration::from_secs(5));
    for id in ["a", "b", "c"] {
        let worker_probe = Arc::clone(&probe);
        spec = spec.with_worker_signal(
            id,
            move |_ctx, signal| ProbeWorker {
                id: id.to_owned(),
                behaviour: Behaviour::Idle,
                probe: Arc::clone(&worker_probe),
                signal: Some(signal),
                hook_ms: 30,
            },
            RestartPolicy::Permanent,
        );
    }
    let handle = StatefulSupervisorHandle::start(spec);
    tokio::time::sleep(Duration::from_millis(50)).await;

    handle.shutdown().await.expect("supervisor was running");

    assert_eq!(probe.stopped(), vec!["c", "b", "a"]);
}

#[tokio::test]
async fn shutdown_of_stopped_supervisor_still_errors() {
    let handle = SupervisorHandle::start(fast_spec("twice"));
    handle.shutdown().await.expect("first shutdown succeeds");
    assert!(
        handle.shutdown().await.is_err(),
        "shutting down an already-stopped supervisor keeps reporting an error"
    );
}

// ---------------------------------------------------------------------------
// 3. Stale termination reports
// ---------------------------------------------------------------------------

fn one_for_all_pair(
    probe: &Arc<Probe>,
    backoff: RestartBackoff,
    b_fails_after: u64,
) -> SupervisorSpec<ProbeWorker> {
    SupervisorSpec::new("ofa")
        .with_restart_strategy(RestartStrategy::OneForAll)
        .with_restart_backoff(backoff)
        .with_restart_intensity(RestartIntensity::new(100, 10))
        .with_worker_signal(
            "a",
            factory("a", Behaviour::FailFirstRunAfter(50), probe),
            RestartPolicy::Permanent,
        )
        .with_worker_signal(
            "b",
            factory("b", Behaviour::FailFirstRunAfter(b_fails_after), probe),
            RestartPolicy::Permanent,
        )
        .with_worker_signal(
            "c",
            factory("c", Behaviour::Idle, probe),
            RestartPolicy::Permanent,
        )
}

#[tokio::test(flavor = "current_thread")]
async fn simultaneous_failures_cause_one_restart_round() {
    for _ in 0..10 {
        let probe = Arc::new(Probe::default());
        let handle = SupervisorHandle::start(one_for_all_pair(&probe, RestartBackoff::none(), 50));
        tokio::time::sleep(Duration::from_millis(300)).await;

        assert_eq!(
            probe.runs_of("c"),
            2,
            "two children failing together must restart the group once, not once per failure"
        );
        let _shutdown = handle.shutdown().await;
    }
}

#[tokio::test]
async fn failures_during_backoff_cause_one_restart_round() {
    let probe = Arc::new(Probe::default());
    let backoff = RestartBackoff::exponential(Duration::from_millis(100), Duration::from_secs(5));
    let handle = SupervisorHandle::start(one_for_all_pair(&probe, backoff, 55));
    tokio::time::sleep(Duration::from_millis(600)).await;

    assert_eq!(
        probe.runs_of("c"),
        2,
        "a second failure while the first is backing off must not schedule another round"
    );
    let _shutdown = handle.shutdown().await;
}

// ---------------------------------------------------------------------------
// 4. Linked start does not block the supervisor
// ---------------------------------------------------------------------------

#[tokio::test]
async fn linked_start_does_not_block_other_commands() {
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(fast_spec("linked").with_worker_signal(
        "x",
        factory("x", Behaviour::Idle, &probe),
        RestartPolicy::Permanent,
    ));

    let starter = handle.clone();
    let slow_probe = Arc::clone(&probe);
    let linked = tokio::spawn(async move {
        starter
            .start_child_linked_with_signal(
                "slow",
                factory("slow", Behaviour::SlowInit(1000), &slow_probe),
                RestartPolicy::Permanent,
                Duration::from_secs(5),
            )
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let started = Instant::now();
    let children = handle
        .which_children()
        .await
        .expect("supervisor is running");
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "which_children took {:?} while a linked child was initializing",
        started.elapsed()
    );
    assert_eq!(children.len(), 1, "an initializing child is not listed yet");

    // The same id cannot be claimed twice while it is still initializing.
    let duplicate = handle
        .start_child_with_signal(
            "slow",
            factory("slow", Behaviour::Idle, &probe),
            RestartPolicy::Permanent,
        )
        .await;
    assert!(matches!(
        duplicate,
        Err(SupervisorError::ChildAlreadyExists(_))
    ));

    let id = linked.await.expect("join").expect("slow child initializes");
    assert_eq!(id, "slow");
    let children = handle
        .which_children()
        .await
        .expect("supervisor is running");
    assert_eq!(children.len(), 2, "the linked child joins once initialized");
    let _shutdown = handle.shutdown().await;
}

#[tokio::test]
async fn linked_start_timeout_is_still_reported() {
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(fast_spec("linked-timeout"));

    let result = handle
        .start_child_linked_with_signal(
            "slow",
            factory("slow", Behaviour::SlowInit(5000), &probe),
            RestartPolicy::Permanent,
            Duration::from_millis(100),
        )
        .await;

    assert!(matches!(
        result,
        Err(SupervisorError::InitializationTimeout { .. })
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        probe.runs_of("slow"),
        0,
        "a timed-out child must not go on to run"
    );
    let _shutdown = handle.shutdown().await;
}

#[tokio::test]
async fn linked_child_that_exits_right_after_init_is_supervised() {
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(fast_spec("linked-early-exit"));

    handle
        .start_child_linked_with_signal(
            "job",
            factory("job", Behaviour::ReturnImmediately, &probe),
            RestartPolicy::Permanent,
            Duration::from_secs(2),
        )
        .await
        .expect("job initializes");
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(
        probe.runs_of("job") > 1,
        "a Permanent child exiting before its init result was processed must still be restarted"
    );
    let _shutdown = handle.shutdown().await;
}

#[tokio::test]
async fn shutdown_while_linked_child_initializes() {
    let probe = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(fast_spec("linked-shutdown"));

    let starter = handle.clone();
    let slow_probe = Arc::clone(&probe);
    let linked = tokio::spawn(async move {
        starter
            .start_child_linked_with_signal(
                "slow",
                factory("slow", Behaviour::SlowInit(5000), &slow_probe),
                RestartPolicy::Permanent,
                Duration::from_secs(10),
            )
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    tokio::time::timeout(Duration::from_secs(2), handle.shutdown())
        .await
        .expect("shutdown must not wait for a pending initialization")
        .expect("supervisor was running");
    let result = linked.await.expect("join");
    assert!(
        matches!(result, Err(SupervisorError::ShuttingDown(_))),
        "got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 5. Nested supervisors never overlap on restart
// ---------------------------------------------------------------------------

#[tokio::test]
async fn restarted_nested_supervisor_does_not_overlap_its_predecessor() {
    let probe = Arc::new(Probe::default());
    let nested = fast_spec("nested")
        .with_shutdown_timeout(Duration::from_secs(5))
        .with_worker_signal(
            "inner",
            factory_with_hook("inner", Behaviour::Idle, &probe, 100),
            RestartPolicy::Permanent,
        );
    let failing = Arc::new(Probe::default());
    let handle = SupervisorHandle::start(
        fast_spec("outer")
            .with_restart_strategy(RestartStrategy::OneForAll)
            .with_supervisor(nested)
            .with_worker_signal(
                "trigger",
                factory("trigger", Behaviour::FailFirstRunAfter(50), &failing),
                RestartPolicy::Permanent,
            ),
    );

    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        probe.runs_of("inner"),
        2,
        "the nested worker was restarted once"
    );
    assert_eq!(
        probe.max_alive.load(Ordering::SeqCst),
        1,
        "the old nested worker must finish stopping before its replacement starts"
    );
    let _shutdown = handle.shutdown().await;
}

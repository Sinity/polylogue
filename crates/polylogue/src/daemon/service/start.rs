//! Opening the archive and owning application workers.
use super::{Daemon, ModeState, acts};
use crate::daemon::lifecycle::{Phase, Progress};
use polylogue_engine::{Config, Engine, Lifetime, LoopStatus, RunSummary, SharedLoopStatus};
use polylogue_store::{Archive, ReadPool, WriterSender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long a stop waits for running requests (docs/operations.md).
const STOP_WAIT: Duration = Duration::from_secs(5);

/// Owns the application workers and writer lock. Transports must stop
/// before this runtime; external daemon references cannot keep workers alive.
pub struct Runtime {
    pub daemon: Arc<Daemon>,
    /// Ends the signal worker: a child of the archive's stop request.
    pub(super) signals_stop: Option<polylogue_store::cancellation::Cancellation>,
    pub(super) signals: Option<JoinHandle<std::io::Result<()>>>,
    live_events_stop: Option<Sender<()>>,
    live_events: Option<JoinHandle<std::io::Result<()>>>,
    /// Records hosted-connection outcomes and runs resumes.
    acts: Option<JoinHandle<()>>,
    scope: Option<crate::daemon::scope::ScopeWorker>,
    pub(super) engine: Option<JoinHandle<(std::io::Result<()>, Engine)>>,
    lock: Option<polylogue_store::ArchiveLock>,
    pub(super) worker_events: tokio::sync::watch::Sender<Option<&'static str>>,
    /// Set once by a loop run until idle, when it ends.
    idle: Arc<tokio::sync::watch::Sender<Option<RunSummary>>>,
    /// Where opening and stopping steps are reported.
    pub(crate) progress: Progress,
}

impl Runtime {
    pub(super) fn start_signals(&mut self) -> std::io::Result<()> {
        let stop = self.daemon.archive.stop().child();
        let stopped = stop.clone();
        let daemon = Arc::clone(&self.daemon);
        let worker_events = self.worker_events.clone();
        let worker = polylogue_model::spawn_named("polylogue-signals", move || {
            let _exit = crate::daemon::lifecycle::WorkerExit::new(
                worker_events,
                "signal worker terminated",
                None,
            );
            let result = crate::signals::run(daemon, stopped);
            if let Err(error) = &result {
                polylogue_store::log!(error, "signal worker: {error}");
            }
            result
        })?;
        self.signals_stop = Some(stop);
        self.signals = Some(worker);
        Ok(())
    }

    /// Whether this runtime runs the ingest loop (writes are not paused).
    #[cfg(test)]
    pub(crate) fn runs_loop(&self) -> bool {
        self.engine.is_some()
    }

    #[cfg(test)]
    pub(crate) fn signal_stop_for_test(&self) -> polylogue_store::cancellation::Cancellation {
        self.signals_stop
            .as_ref()
            .expect("running signal worker")
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn pause_signals(&mut self) -> std::io::Result<()> {
        if let Some(stop) = self.signals_stop.take() {
            stop.cancel();
        }
        if let Some(worker) = self.signals.take() {
            worker
                .join()
                .map_err(|_| std::io::Error::other("signal worker panicked"))??;
        }
        Ok(())
    }

    /// Refuses new requests and makes the archive's stop request, before
    /// transports and workers are joined. The stop ends every read and wait
    /// in progress, the loop's pass and parsing, the signal worker and the
    /// writer's reconstructible work; writes the writer was handed, and the
    /// captured bytes in hand, still commit.
    pub fn stop_accepting(&self) {
        self.daemon.requests.refuse();
        self.daemon.archive.stop().cancel();
        self.daemon.warm.clear();
        self.daemon.reads.close();
    }

    pub(crate) fn worker_events(&self) -> tokio::sync::watch::Receiver<Option<&'static str>> {
        self.worker_events.subscribe()
    }

    /// Becomes `Some` when a loop run until idle has ended, with what it did.
    pub(crate) fn idle(&self) -> tokio::sync::watch::Receiver<Option<RunSummary>> {
        self.idle.subscribe()
    }

    /// A finished continuous worker is a failure; slow work remains live.
    pub fn worker_failure(&self) -> Option<&'static str> {
        if self.daemon.is_stopping() {
            return None;
        }
        if self.engine.as_ref().is_some_and(JoinHandle::is_finished) && self.idle.borrow().is_none()
        {
            return Some("engine worker terminated");
        }
        if self.signals.as_ref().is_some_and(JoinHandle::is_finished) {
            return Some("signal worker terminated");
        }
        if let Some(failure) = self
            .scope
            .as_ref()
            .and_then(|worker| worker.worker_failure())
        {
            return Some(failure);
        }
        if self
            .live_events
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            return Some("live event worker terminated");
        }
        if self.acts.as_ref().is_some_and(JoinHandle::is_finished) {
            return Some("acts worker terminated");
        }
        None
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop_accepting();
        let mut steps = self.progress.steps(Phase::Stopping);
        // Transports have stopped; queued requests are refused and running
        // ones end (the closed reader pool ends any still waiting for a read).
        // Work that ignores its cancellation is left behind after the bound,
        // so a stop never hangs on it.
        // A second call (from drop, after the stop) has nothing left to report.
        if self.lock.is_some() {
            steps.next("waiting for running requests");
        }
        let running = self.daemon.requests.wait(STOP_WAIT);
        if running > 0 {
            polylogue_store::log!(
                warning,
                "{running} request(s) did not stop within {} s; stopping without them",
                STOP_WAIT.as_secs()
            );
        }
        let mut failure = None;
        if let Some(scope) = self.scope.as_mut() {
            steps.next("scope worker");
            if let Err(error) = scope.shutdown() {
                failure.get_or_insert(error);
            }
        }
        self.scope.take();
        if let Some(stop) = self.live_events_stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.live_events.take() {
            steps.next("live event worker");
            let result = worker
                .join()
                .unwrap_or_else(|_| Err(std::io::Error::other("live event worker panicked")));
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        }
        if let Some(worker) = self.acts.take() {
            steps.next("acts worker");
            self.daemon.acts.stop_reports();
            if worker.join().is_err() {
                failure.get_or_insert_with(|| std::io::Error::other("acts worker panicked"));
            }
        }
        if let Some(signals) = self.signals.take() {
            steps.next("signal worker");
            let result = signals
                .join()
                .unwrap_or_else(|_| Err(std::io::Error::other("signal worker panicked")));
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        }
        if let Some(engine) = self.engine.take() {
            steps.next("engine drain");
            match engine.join() {
                Ok((result, engine)) => {
                    if let Err(error) = result {
                        failure.get_or_insert(error);
                    }
                    // Engine owns the writer. Its final acknowledgments precede close.
                    steps.next("writer drain");
                    drop(engine);
                }
                Err(_) => {
                    failure.get_or_insert_with(|| std::io::Error::other("loop thread panicked"));
                }
            }
        }
        steps.finish();
        self.daemon.reads.close();
        self.lock.take();
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            polylogue_store::log!(error, "shutdown: {error}");
        }
    }
}

/// Standalone, or blocked: a Sinex configuration that cannot be read; never
/// a fallback to another mode. With the `sinex` feature, a configuration
/// asking for primary mode is kept so that start refuses it: primary mode is
/// not built in this version (`docs/primary-mode.md`).
#[derive(Clone)]
pub enum ArchiveMode {
    Standalone,
    #[cfg(feature = "sinex")]
    Primary,
    Blocked {
        /// The mode the configuration asked for, when it could be read.
        configured: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

/// Primary mode is not built in this version (`docs/primary-mode.md`).
#[cfg(feature = "sinex")]
fn primary_not_built() -> std::io::Error {
    std::io::Error::other(polylogue_store::StoreError::from(
        polylogue_model::Reason::ModeNotBuilt {
            mode: "primary".into(),
        },
    ))
}

/// Opens the archive and starts owned application workers. Startup errors
/// join any workers already started before releasing the archive lock.
/// A loop run until idle ends the runtime's work when it ends ([`Runtime::idle`]).
pub fn start(
    deployment: Arc<crate::config::Config>,
    config: Config,
    mode: ArchiveMode,
    lifetime: Lifetime,
    progress: &Progress,
) -> std::io::Result<Runtime> {
    let mut steps = progress.steps(Phase::Opening);
    steps.next("opening the archive");
    let root = deployment.archive.as_path();
    if let Some(stopped) = crate::daemon::storage_stop::this_boot(&deployment.paths, root) {
        return Err(std::io::Error::other(stopped));
    }
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let budget = config.memory_budget;
    let config = config.with_memory_budget(budget);
    let settings = config.clone();
    // An archive recorded as primary is refused before anything opens it,
    // so it is left as it was.
    #[cfg(feature = "sinex")]
    if Archive::recorded_mode(root).map_err(std::io::Error::other)?
        == Some(polylogue_store::archive::Mode::Primary)
    {
        return Err(primary_not_built());
    }
    let (archive, lock, mode, engine) = match mode {
        ArchiveMode::Standalone => {
            let (archive, lock) = Archive::open_for_serving(root).map_err(std::io::Error::other)?;
            let archive = archive.with_memory_budget(settings.memory_budget);
            steps.next("starting the writer");
            let engine = Engine::new(config, archive.clone()).map_err(|e| to_io(&e))?;
            (archive, lock, ModeState::Standalone, Some(engine))
        }
        ArchiveMode::Blocked {
            configured,
            path,
            error,
        } => {
            let (archive, lock) = Archive::open_blocked(root).map_err(std::io::Error::other)?;
            let archive = archive.with_memory_budget(settings.memory_budget);
            let state = ModeState::Blocked {
                configured,
                path,
                error,
            };
            (archive, lock, state, None)
        }
        #[cfg(feature = "sinex")]
        ArchiveMode::Primary => return Err(primary_not_built()),
    };
    let mut engine = engine;
    steps.next("opening the readers");
    let writes = engine.as_ref().map(Engine::writer);
    let per_reader = settings
        .reader_memory_per_connection()
        .map_err(|e| to_io(&e))?;
    // Readers bound how many requests read at once: one per worker and one
    // for idle-priority work; the pool keeps one more for interactive answers.
    let readers = settings.workers + 1;
    let reads =
        ReadPool::serving(&archive, readers, per_reader * readers as u64).map_err(|e| to_io(&e))?;
    let (tx, rx) = crossbeam_channel::unbounded();
    let (act_reports, act_reports_received) = channel();
    // The loop's first pass starts as soon as it runs.
    let status = Arc::new(SharedLoopStatus::new(LoopStatus {
        busy: engine.is_some(),
        ..LoopStatus::default()
    }));
    let mail = crate::mailbox::dir(&deployment.paths, &archive.root);
    let hosts = crate::hosting::dir(&deployment.paths, &archive.root);
    let warm = crate::daemon::warm::Warm::new(&archive, settings.clock.clone());
    let daemon = Arc::new(Daemon {
        clock: settings.clock.clone(),
        config: deployment,
        archive,
        reads,
        requests: Default::default(),
        callers: crate::hook::open_calls::OpenCalls::new(
            settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
        ),
        warm,
        scope: crate::daemon::scope::SharedScopeIndex::default(),
        status_snapshot: Default::default(),
        live_spool: Default::default(),
        advisories: crate::advisory::Cooldowns::default(),
        acts: crate::live_acts::Acts::new(
            mail,
            hosts,
            settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
            polylogue_sources::origin::by_name("codex")
                .map(|origin| {
                    settings
                        .root(origin.root)
                        .join("app-server-control/app-server-control.sock")
                })
                .unwrap_or_default(),
            "codex".into(),
            act_reports,
        ),
        writes,
        mode,
        loop_tx: Mutex::new(tx.clone()),
        status: Arc::clone(&status),
        captures_sources: settings.reads_sources && engine.is_some(),
        started: Instant::now(),
        signals: Mutex::new(serde_json::Value::Null),
        idle_gate: crate::daemon::schema::IdleGate::default(),
    });
    let acts_worker = {
        let daemon = Arc::clone(&daemon);
        polylogue_model::spawn_named("polylogue-acts", move || {
            acts::run_reports(daemon, act_reports_received)
        })?
    };
    // Hosted sessions outlive a daemon restart: their hosts, or Codex's
    // app-server, kept them running; connect to each again.
    steps.next("reattaching hosted sessions");
    acts::recover_host_submissions(&daemon);
    // Messages and notices left pending raise their running sessions' mailbox
    // flags; those to sessions no longer running are closed.
    acts::finish_ended(&daemon, daemon.config.requests.now().unix_ms(), true);
    let mut runtime = Runtime {
        daemon,
        signals_stop: None,
        signals: None,
        live_events_stop: None,
        live_events: None,
        acts: Some(acts_worker),
        scope: None,
        engine: None,
        lock: Some(lock),
        worker_events: tokio::sync::watch::channel(None).0,
        idle: Arc::new(tokio::sync::watch::channel(None).0),
        progress: progress.clone(),
    };
    steps.next("starting the scope worker");
    runtime.scope = Some(crate::daemon::scope::ScopeWorker::start(
        Arc::clone(&runtime.daemon),
        runtime
            .daemon
            .writes
            .as_ref()
            .map(WriterSender::completion_commits),
        runtime.worker_events.clone(),
    )?);
    let Some(mut engine) = engine.take() else {
        // Without a loop (writes paused) there is no work to wait for.
        if lifetime == Lifetime::UntilIdle {
            runtime.idle.send_replace(Some(RunSummary::default()));
        }
        return Ok(runtime);
    };
    // A failed register sync stops the writer; the daemon then stops too,
    // with its storage exit code.
    let storage_events = runtime.worker_events.clone();
    polylogue_model::spawn_named("polylogue-storage", move || {
        polylogue_store::storage_failure::wait();
        storage_events.send_replace(Some("the register failed a disk sync"));
    })?;
    steps.next("starting the loop and watching its sources");
    let (ready, wait) = sync_channel(1);
    let worker_events = runtime.worker_events.clone();
    let idle = Arc::clone(&runtime.idle);
    runtime.engine = Some(polylogue_model::spawn_named("polylogue-loop", move || {
        let expected = Arc::new(AtomicBool::new(false));
        let _exit = crate::daemon::lifecycle::WorkerExit::new(
            worker_events,
            "engine worker terminated",
            Some(Arc::clone(&expected)),
        );
        let started = ready.clone();
        let result = engine.run(tx, rx, &status, lifetime, || {
            let _ = started.send(Ok(()));
        });
        let result = match result {
            Ok(summary) => {
                // Stopping a service loop is the runtime's own shutdown;
                // an idle loop's end is the run's end.
                if lifetime == Lifetime::UntilIdle {
                    expected.store(true, Ordering::Release);
                    idle.send_replace(Some(summary));
                }
                Ok(())
            }
            Err(error) => {
                let _ = ready.try_send(Err(error.to_string()));
                Err(std::io::Error::other(error.to_string()))
            }
        };
        // Keep the writer alive until Runtime has joined the outcome producers.
        (result, engine)
    })?);
    wait.recv()
        .map_err(|_| std::io::Error::other("loop thread exited during startup"))?
        .map_err(std::io::Error::other)?;
    steps.next("starting the signal and live event workers");
    runtime.start_signals()?;
    let (stop, stopped) = channel();
    runtime.live_events_stop = Some(stop);
    let daemon = Arc::clone(&runtime.daemon);
    let worker_events = runtime.worker_events.clone();
    let paths = crate::live_events::Paths {
        spool: settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
        claude_sessions: polylogue_sources::origin::by_name("claude-code")
            .map(|origin| settings.root(origin.root).join("sessions"))
            .unwrap_or_default(),
        codex_socket: runtime.daemon.acts.codex().1,
        codex_program: runtime.daemon.acts.codex().0,
    };
    let clock = settings.clock.clone();
    runtime.live_events = Some(polylogue_model::spawn_named("polylogue-live", move || {
        let _exit = crate::daemon::lifecycle::WorkerExit::new(
            worker_events,
            "live event worker terminated",
            None,
        );
        crate::live_events::run(daemon, paths, clock, stopped)
    })?);
    // Load the vectors `similar:` ranks by before the first query needs them.
    let daemon = Arc::clone(&runtime.daemon);
    polylogue_model::spawn_named("polylogue-warm", move || {
        let Ok(reader) = daemon.reads.reader() else {
            return;
        };
        if let Ok(settings) = reader.settings_snapshot(daemon.config.requests.machine())
            && let Ok(Some(provider)) = crate::embeddings::Provider::from_settings(
                &settings,
                &daemon.config.embedding_credentials,
            )
        {
            let _ = daemon.warm.vectors(&reader, &provider);
        }
    })?;
    Ok(runtime)
}

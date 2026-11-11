//! A real daemon over a temporary archive and home for in-process tests:
//! its loop reads only the temporary home, and its socket is served on a
//! thread of its own.

use crate::daemon::{
    server,
    service::{self, Daemon},
};
use polylogue_engine::Config;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub(crate) fn engine_config(home: PathBuf) -> Config {
    Config::new(
        home,
        Default::default(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        2,
        polylogue_model::WallClock::new(jiff::Timestamp::now),
    )
    .expect("valid fixture configuration")
}

pub(crate) fn start_runtime(
    root: &std::path::Path,
    engine: Config,
    mode: service::ArchiveMode,
) -> std::io::Result<service::Runtime> {
    let deployment = crate::config::Config::for_test(&engine, root);
    service::start(
        deployment,
        engine,
        mode,
        polylogue_engine::Lifetime::UntilStopped,
        Some(polylogue_testkit::stopped_attempt()),
    )
}

pub(crate) struct ReadObservation {
    pub limit: polylogue_store::read::limit::ReadLimit,
    pub owner: std::thread::ThreadId,
    /// Checkout evidence survives release before the observing thread runs.
    pub acquired: Arc<std::sync::atomic::AtomicBool>,
}

impl ReadObservation {
    pub(crate) fn reader_acquired(&self) -> bool {
        self.acquired.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl std::ops::Deref for ReadObservation {
    type Target = polylogue_store::read::limit::ReadLimit;

    fn deref(&self) -> &Self::Target {
        &self.limit
    }
}

pub struct TestDaemon {
    pub home: PathBuf,
    /// The loop's settings the daemon was started with.
    pub config: Config,
    pub daemon: Arc<Daemon>,
    server: server::Server,
    runtime: service::Runtime,
    // Removed only after both owners have joined their workers.
    pub _dir: tempfile::TempDir,
}

/// A daemon whose loop has parsed one Claude Code session, `claude-code:s1`,
/// holding `text`, with its first derivation pass finished and socket served.
pub fn start(text: &str) -> TestDaemon {
    start_with_workers(text, 2)
}

pub(crate) fn start_with_workers(text: &str, workers: usize) -> TestDaemon {
    start_with_clock_and_workers(
        text,
        workers,
        polylogue_model::WallClock::new(jiff::Timestamp::now),
        None,
        NO_SEED,
    )
}

/// Starts with `workers` loop and request workers over a home where `seed`
/// has also written sources holding `messages` more messages, so the
/// daemon's first pass reads them all as one first build, as it reads an
/// existing home.
pub(crate) fn start_with_sources_and_workers(
    text: &str,
    seed: &dyn Fn(&Path),
    messages: u64,
    workers: usize,
) -> TestDaemon {
    start_with_clock_and_workers(
        text,
        workers,
        polylogue_model::WallClock::new(jiff::Timestamp::now),
        None,
        (seed, messages),
    )
}

/// Writes sources into the home before the daemon starts, and how many
/// messages they hold.
type Seed<'a> = (&'a dyn Fn(&Path), u64);
const NO_SEED: Seed<'static> = (&|_| {}, 0);

pub(crate) fn start_with_clock(text: &str, clock: polylogue_model::WallClock) -> TestDaemon {
    start_with_clock_and_workers(text, 2, clock, None, NO_SEED)
}

pub(crate) fn start_with_clock_and_day(
    text: &str,
    clock: polylogue_model::WallClock,
    day: query_language::time::DaySetting,
) -> TestDaemon {
    start_with_clock_and_workers(text, 2, clock, Some(day), NO_SEED)
}

fn start_with_clock_and_workers(
    text: &str,
    workers: usize,
    clock: polylogue_model::WallClock,
    day: Option<query_language::time::DaySetting>,
    seed: Seed<'_>,
) -> TestDaemon {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        polylogue_testkit::claude_user("u1", "s1", text),
    )
    .unwrap();
    (seed.0)(&home);
    let mut config = engine_config(home.clone());
    config.home = home.clone();
    config.clock = clock;
    config.tool_dirs.clear();
    config.workers = workers;
    let root = dir.path().join("archive");
    let mut deployment = crate::config::Config::for_test(&config, &root);
    if let Some(day) = day {
        deployment = deployment.standalone_for_archive(root, day).unwrap();
    }
    let runtime = service::start(
        deployment,
        config.clone(),
        service::ArchiveMode::Standalone,
        polylogue_engine::Lifetime::UntilStopped,
        Some(polylogue_testkit::stopped_attempt()),
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let server =
        server::Server::start(Arc::clone(&daemon), Arc::clone(&runtime.operations)).unwrap();
    let daemon = TestDaemon {
        _dir: dir,
        home,
        config,
        daemon,
        server,
        runtime,
    };
    daemon.wait_for(|data| {
        data["store"]["messages"].as_u64() == Some(1 + seed.1)
            && data["store"]["search_pending"].as_u64() == Some(0)
    });
    daemon.wait_for_signal_checkpoint();
    daemon
}

impl TestDaemon {
    pub(crate) fn worker_events(&self) -> tokio::sync::watch::Receiver<Option<&'static str>> {
        self.runtime.worker_events()
    }

    pub(crate) fn pause_signals(&mut self) {
        self.runtime.pause_signals().unwrap();
    }

    /// Waits until `status` satisfies `ok` and then until the loop is idle,
    /// so the pass that made `ok` hold has also derived from its rows. A
    /// state that never comes is ended by the test runner's timeout.
    pub fn wait_for(&self, ok: impl Fn(&serde_json::Value) -> bool) {
        let status = || {
            crate::protocol::into_value(call(&self.daemon, "status", serde_json::json!({})))
                .unwrap_or_else(|failure| panic!("status failed: {failure:?}"))
        };
        while !ok(&status()) {
            std::thread::sleep(Duration::from_millis(10));
        }
        // Checked after `ok`: an idle loop seen now has finished any pass
        // that wrote what `ok` saw.
        while status()["busy"] != false {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Waits until the signals worker's first evaluation has committed its
    /// checkpoint to archive.db, then until the loop is idle. That commit is
    /// asynchronous to the parse, so without this a test's first two calls
    /// could see different archive positions (plnext-0sqw).
    pub(crate) fn wait_for_signal_checkpoint(&self) {
        let machine = self.config.machine.clone();
        let committed = || {
            self.daemon
                .reads
                .reader()
                .unwrap()
                .signal_checkpoint(&machine)
                .unwrap()
                .is_some()
        };
        while !committed() {
            std::thread::sleep(Duration::from_millis(10));
        }
        self.wait_for(|_| true);
    }

    /// Restarts the daemon over the same synthetic archive and home.
    pub fn restart(&mut self) {
        self.restart_after_stop(|| {});
    }

    /// Stops the daemon the way SIGTERM does: no new requests, the socket
    /// closed, then every worker joined.
    pub(crate) fn stop(&mut self) -> std::io::Result<()> {
        self.runtime.stop_accepting();
        self.server.shutdown()?;
        self.runtime.shutdown()
    }

    /// The request [`Self::stop`] makes of the loop, to observe it.
    pub(crate) fn engine_stop(&self) -> polylogue_engine::StopRequest {
        self.runtime
            .engine_stop
            .clone()
            .expect("the daemon runs a loop")
    }

    /// Stops the daemon, runs `reset` over its stopped archive, and starts it again.
    pub(crate) fn restart_after_stop(&mut self, reset: impl FnOnce()) {
        self.stop().unwrap();
        reset();
        self.runtime = service::start(
            Arc::clone(&self.daemon.config),
            self.config.clone(),
            service::ArchiveMode::Standalone,
            polylogue_engine::Lifetime::UntilStopped,
            Some(polylogue_testkit::stopped_attempt()),
        )
        .unwrap();
        self.daemon = Arc::clone(&self.runtime.daemon);
        self.server = server::Server::start(
            Arc::clone(&self.daemon),
            Arc::clone(&self.runtime.operations),
        )
        .unwrap();
    }

    pub fn operations(&self) -> &Arc<command_surface::OperationTable> {
        &self.runtime.operations
    }

    /// How often the socket's listener thread has woken.
    pub(crate) fn server_wakeups(&self) -> u64 {
        self.server.wakeups()
    }

    pub(crate) fn stop_server(&mut self) -> std::io::Result<()> {
        self.server.shutdown()
    }

    pub(crate) fn watch_reads(&self) -> std::sync::mpsc::Receiver<ReadObservation> {
        let (sender, receiver) = std::sync::mpsc::channel();
        *self.daemon.read_limits.lock().unwrap() = Some(sender);
        receiver
    }

    pub(crate) fn wait_for_sql(limit: &polylogue_store::read::limit::ReadLimit) {
        let start = std::time::Instant::now();
        while limit.sqlite_steps() < 10_000 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "request never reached SQL"
            );
            std::thread::yield_now();
        }
    }

    /// Waits until the observed read has given its reader slot back, which
    /// it must within 2 s, and returns how long that took.
    pub(crate) fn assert_read_released(&self, read: &ReadObservation) -> Duration {
        let start = std::time::Instant::now();
        while self.daemon.reads.checked_out_by(read.owner) > 0 {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "the read kept its reader slot"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        start.elapsed()
    }

    /// Sets an archive setting through the daemon.
    pub fn set(&self, key: &str, value: &str) {
        let r = call(
            &self.daemon,
            "setting",
            serde_json::json!({ "key": key, "value": value }),
        );
        assert!(
            matches!(
                r,
                command_surface::Answer {
                    outcome: command_surface::Outcome::Ok(_),
                    ..
                }
            ),
            "{r:?}"
        );
    }
}

impl Drop for TestDaemon {
    fn drop(&mut self) {
        self.runtime.stop_accepting();
        if let Err(error) = self.server.shutdown() {
            eprintln!("test daemon socket: {error}");
        }
        if let Err(error) = self.runtime.shutdown() {
            eprintln!("test daemon runtime: {error}");
        }
    }
}

pub mod assets;
pub(crate) mod embeddings;

/// Answers one request on the calling thread as the daemon's own caller,
/// with the default read deadline and a new request ID.
pub(crate) fn call(
    daemon: &crate::daemon::service::Daemon,
    operation: &str,
    arguments: serde_json::Value,
) -> command_surface::Answer<serde_json::Value> {
    Call::new(operation, arguments).answer(daemon)
}

/// One request as a test sends it; the defaults are the daemon's own
/// caller on the command line, a new request ID and the default deadline.
#[derive(Clone)]
pub(crate) struct Call {
    pub operation: String,
    pub arguments: serde_json::Value,
    pub request_id: String,
    pub cursor: Option<String>,
    pub surface: crate::commands::Surface,
    pub cwd: Option<PathBuf>,
    pub authorship: polylogue_model::authorship::Authorship,
    pub limit: polylogue_store::read::limit::ReadLimit,
}

impl Call {
    pub(crate) fn new(operation: &str, arguments: serde_json::Value) -> Self {
        Self {
            operation: operation.into(),
            arguments,
            request_id: polylogue_store::user_writes::new_request_id(),
            cursor: None,
            surface: crate::commands::Surface::Cli,
            cwd: None,
            authorship: polylogue_model::authorship::compute_authorship(
                polylogue_model::authorship::Channel::Daemon,
                &Default::default(),
                None,
            ),
            limit: crate::daemon::execution::limit(None, Default::default()),
        }
    }
    /// The call a ready request names, with its identity and cursor.
    pub(crate) fn from_request(request: &command_surface::RunnableRequest) -> Self {
        Self::new(&request.operation, request.arguments.clone())
            .id(&request.request_id)
            .cursor(request.cursor.as_deref())
    }
    pub(crate) fn id(mut self, request_id: &str) -> Self {
        self.request_id = request_id.into();
        self
    }
    pub(crate) fn cursor(mut self, cursor: Option<&str>) -> Self {
        self.cursor = cursor.map(str::to_owned);
        self
    }
    pub(crate) fn surface(mut self, surface: crate::commands::Surface) -> Self {
        self.surface = surface;
        self
    }
    pub(crate) fn authorship(
        mut self,
        authorship: polylogue_model::authorship::Authorship,
    ) -> Self {
        self.authorship = authorship;
        self
    }
    pub(crate) fn limit(mut self, limit: polylogue_store::read::limit::ReadLimit) -> Self {
        self.limit = limit;
        self
    }
    pub(crate) fn answer(
        self,
        daemon: &crate::daemon::service::Daemon,
    ) -> command_surface::Answer<serde_json::Value> {
        let invocation = daemon.invocation(&self.authorship);
        self.answer_with(daemon, &invocation)
    }
    pub(crate) fn answer_with(
        self,
        daemon: &crate::daemon::service::Daemon,
        invocation: &crate::operation::Invocation,
    ) -> command_surface::Answer<serde_json::Value> {
        crate::daemon::handler::local(
            &self.operation,
            self.arguments,
            &crate::daemon::handler::Local {
                daemon,
                invocation,
                caller: &crate::commands::Caller {
                    surface: self.surface,
                    cwd: self.cwd,
                },
                limit: &self.limit,
                request_id: &self.request_id,
                cursor: self.cursor.as_deref(),
            },
        )
    }
}

/// A read the test resolves itself, sent as the read operation's arguments.
pub(crate) fn read(request: crate::daemon::service::read::ReadRequest) -> Call {
    let cursor = request.cursor.clone();
    let arguments = crate::protocol::read::Arguments::from_request(&request)
        .expect("a test read has public arguments");
    Call::new(
        "read",
        serde_json::to_value(arguments).expect("read arguments serialize"),
    )
    .cursor(cursor.as_deref())
}

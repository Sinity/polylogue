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

/// Loop and request workers of a test daemon unless a test asks for others.
const DEFAULT_WORKERS: usize = 2;

pub(crate) fn engine_config(home: PathBuf) -> Config {
    Config::new(
        home,
        Default::default(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        DEFAULT_WORKERS,
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
    start_with_deployment(deployment, engine, mode)
}

/// Starts a fixture runtime and waits for its initial index build and the
/// status snapshot read after it before status reads.
fn start_with_deployment(
    deployment: Arc<crate::config::Config>,
    engine: Config,
    mode: service::ArchiveMode,
) -> std::io::Result<service::Runtime> {
    let runtime = service::start(
        deployment,
        engine,
        mode,
        polylogue_engine::Lifetime::UntilStopped,
        Some(polylogue_testkit::stopped_attempt()),
        &Default::default(),
    )?;
    // Reads answer `index_building` until the loop's first pass installs the
    // index (docs/operations.md, Replacing an index from another schema).
    // Status then answers from the snapshot read after the install, which
    // counts as a status change.
    let until = std::time::Instant::now() + Duration::from_secs(60);
    let mut seen = runtime.daemon.status.changes();
    while runtime.runs_loop()
        && (runtime.daemon.archive.index_build().is_some()
            || !runtime.daemon.status_snapshot.is_read())
    {
        if std::time::Instant::now() >= until {
            return Err(std::io::Error::other(
                "the test daemon's first index was not installed within a minute",
            ));
        }
        seen = runtime.daemon.status.wait_past(seen, until);
    }
    Ok(runtime)
}

pub struct TestDaemon {
    pub home: PathBuf,
    /// The loop's settings the daemon was started with.
    pub config: Config,
    pub daemon: Arc<Daemon>,
    server: server::Server,
    runtime: service::Runtime,
    // Removed only after both holders have joined their workers.
    pub _dir: tempfile::TempDir,
}

/// A daemon whose loop has parsed one Claude Code session, `claude-code:s1`,
/// holding `text`, with its first derivation pass finished and socket served.
pub fn start(text: &str) -> TestDaemon {
    start_with_workers(text, DEFAULT_WORKERS)
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
    start_with_clock_and_workers(text, DEFAULT_WORKERS, clock, None, NO_SEED)
}

pub(crate) fn start_with_clock_and_day(
    text: &str,
    clock: polylogue_model::WallClock,
    day: query_language::time::DaySetting,
) -> TestDaemon {
    start_with_clock_and_workers(text, DEFAULT_WORKERS, clock, Some(day), NO_SEED)
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
    config.clock = clock;
    config.workers = workers;
    let root = dir.path().join("archive");
    let mut deployment = crate::config::Config::for_test(&config, &root);
    if let Some(day) = day {
        deployment = deployment.standalone_for_archive(root, day).unwrap();
    }
    let runtime =
        start_with_deployment(deployment, config.clone(), service::ArchiveMode::Standalone)
            .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let server = server::Server::start(Arc::clone(&daemon)).unwrap();
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
            &Default::default(),
        )
        .unwrap();
        self.daemon = Arc::clone(&self.runtime.daemon);
        self.server = server::Server::start(Arc::clone(&self.daemon)).unwrap();
    }

    pub(crate) fn stop_server(&mut self) -> std::io::Result<()> {
        self.server.shutdown()
    }

    /// Waits until a request holds a reader, which it must within 10 s.
    pub(crate) fn wait_for_a_read(&self) {
        let start = std::time::Instant::now();
        while self.daemon.reads.checked_out() == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "no request took a reader"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Waits until every reader is back in the pool, which must happen
    /// within 2 s, and returns how long that took.
    pub(crate) fn assert_reads_released(&self) -> Duration {
        let start = std::time::Instant::now();
        while self.daemon.reads.checked_out() > 0 {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "a request kept its reader"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        start.elapsed()
    }

    /// Sets an archive setting through the daemon.
    pub fn set(&self, key: &str, value: &str) {
        crate::test_answers::ok(call(
            &self.daemon,
            "set",
            serde_json::json!({ "key": key, "value": value }),
        ));
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
/// caller, a new request ID and the default deadline.
#[derive(Clone)]
pub(crate) struct Call {
    pub operation: String,
    pub arguments: serde_json::Value,
    pub request_id: String,
    pub cursor: Option<String>,
    pub deadline: command_surface::Deadline,
    pub caller: crate::operation::Caller,
}

impl Call {
    pub(crate) fn new(operation: &str, arguments: serde_json::Value) -> Self {
        Self {
            operation: operation.into(),
            arguments,
            request_id: polylogue_store::user_writes::new_request_id(),
            cursor: None,
            deadline: command_surface::Deadline::Default,
            caller: crate::operation::Caller::local(
                command_surface::Via::Daemon,
                Default::default(),
            ),
        }
    }
    /// The call a ready request names, with its identity, cursor and deadline.
    pub(crate) fn from_request(request: &command_surface::RunnableRequest) -> Self {
        Self {
            deadline: request.deadline,
            ..Self::new(&request.operation, request.arguments.clone())
                .id(&request.request_id)
                .cursor(request.cursor.as_deref())
        }
    }
    pub(crate) fn id(mut self, request_id: &str) -> Self {
        self.request_id = request_id.into();
        self
    }
    pub(crate) fn cursor(mut self, cursor: Option<&str>) -> Self {
        self.cursor = cursor.map(str::to_owned);
        self
    }
    /// The channel the request comes by, keeping its author.
    pub(crate) fn channel(mut self, via: command_surface::Via) -> Self {
        let channel: polylogue_model::authorship::Authorship =
            command_surface::author_of(&Default::default(), via, None).into();
        self.caller.authorship.via = channel.via;
        self
    }
    pub(crate) fn authorship(
        mut self,
        authorship: polylogue_model::authorship::Authorship,
    ) -> Self {
        self.caller.authorship = authorship;
        self
    }
    /// The moment the request is accepted, in place of the daemon's clock.
    pub(crate) fn at(mut self, at: polylogue_model::Timestamp) -> Self {
        self.caller.accepted_at = Some(at);
        self
    }
    pub(crate) fn calling_session(mut self, session: &str) -> Self {
        self.caller.calling_session = Some(session.into());
        self
    }
    pub(crate) fn cwd(mut self, cwd: PathBuf) -> Self {
        self.caller.cwd = Some(cwd);
        self
    }
    /// The request's read limit, which the test can observe: its bound,
    /// cancellation and stages.
    pub(crate) fn within(mut self, limit: &polylogue_store::read::limit::ReadLimit) -> Self {
        self.caller.stop = limit.cancellation.clone();
        self.caller.outer_deadline = limit.deadline();
        self.caller.read_limit = Some(limit.clone());
        self
    }
    pub(crate) fn request(&self) -> command_surface::RunnableRequest {
        command_surface::Request {
            cursor: self.cursor.clone(),
            deadline: self.deadline,
            ..crate::client::request(
                &self.operation,
                self.arguments.clone(),
                self.request_id.clone(),
            )
        }
    }
    /// The answer, through the daemon's one `answer`.
    pub(crate) fn answer(
        self,
        daemon: &crate::daemon::service::Daemon,
    ) -> command_surface::Answer<serde_json::Value> {
        self.answer_with_events(daemon).1
    }
    /// The events and the answer.
    pub(crate) fn answer_with_events(
        self,
        daemon: &crate::daemon::service::Daemon,
    ) -> (Vec<u8>, command_surface::Answer<serde_json::Value>) {
        crate::daemon::handler::answer_collected(daemon, &self.request(), &self.caller)
    }
}

/// A read through the declared line, retaining its typed cursor and options.
pub(crate) fn read_call(request: polylogue_reading::ReadRequest) -> Call {
    let step = crate::daemon::verbs::read::step_from(&request, request.cursor.clone());
    let rows = crate::protocol::line::Rows {
        selection: request.references.is_none().then_some(request.target),
        references: request
            .references
            .unwrap_or_default()
            .iter()
            .map(ToString::to_string)
            .collect(),
        steps: vec![crate::protocol::line::Step::Read(Box::new(step))],
        ..Default::default()
    };
    Call::new(
        "rows",
        serde_json::to_value(rows).expect("read line serializes"),
    )
}

/// What an operation's call borrows, for a test that runs a verb or an
/// operation directly: the daemon's own caller with `authorship`, no
/// deadline, and events kept in memory.
pub(crate) struct CallParts {
    caller: crate::operation::Caller,
    invocation: polylogue_reading::operation::Invocation,
    limit: polylogue_store::read::limit::ReadLimit,
    query: std::cell::RefCell<Option<String>>,
    events: KeptEvents,
    request_id: String,
    budget: command_surface::TokenBudget,
}

/// Events written by a call run directly, kept in memory.
#[derive(Default)]
pub(crate) struct KeptEvents(pub std::cell::RefCell<Vec<u8>>);

impl crate::daemon::handler::Events for KeptEvents {
    fn write(&self, frames: &[u8]) -> std::io::Result<()> {
        self.0.borrow_mut().extend_from_slice(frames);
        Ok(())
    }
}

impl CallParts {
    pub(crate) fn new(
        daemon: &crate::daemon::service::Daemon,
        authorship: polylogue_model::authorship::Authorship,
        request_id: &str,
    ) -> Self {
        let mut caller = crate::operation::Caller::local(
            command_surface::Via::Daemon,
            daemon.archive.stop().child(),
        );
        caller.authorship = authorship;
        let request = crate::client::request("rows", serde_json::json!({}), request_id.into());
        Self {
            invocation: caller.invocation(daemon, &request),
            limit: polylogue_store::read::limit::ReadLimit::without_deadline(caller.stop.clone()),
            caller,
            query: Default::default(),
            events: KeptEvents::default(),
            request_id: request_id.into(),
            budget: Default::default(),
        }
    }
    pub(crate) fn cwd(mut self, cwd: PathBuf) -> Self {
        self.caller.cwd = Some(cwd);
        self
    }
    pub(crate) fn budget(mut self, budget: command_surface::TokenBudget) -> Self {
        self.budget = budget;
        self
    }
    pub(crate) fn call<'a>(
        &'a self,
        daemon: &'a crate::daemon::service::Daemon,
    ) -> crate::daemon::handler::Call<'a> {
        crate::daemon::handler::Call {
            daemon,
            caller: &self.caller,
            invocation: &self.invocation,
            request_id: &self.request_id,
            cursor: None,
            budget: self.budget,
            limit: &self.limit,
            lane: crate::daemon::handler::Lane::Ordinary,
            output: &self.events,
            query: &self.query,
        }
    }
}

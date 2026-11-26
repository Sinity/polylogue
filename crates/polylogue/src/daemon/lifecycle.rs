//! OS signals and service supervision around the daemon runtime.
use super::{server::Server, service::Runtime};
use crate::config::{EnvSnapshot, Error};
#[cfg(target_os = "linux")]
use std::os::linux::net::SocketAddrExt;
use std::{
    ffi::OsStr,
    future::pending,
    io,
    os::unix::{
        ffi::OsStrExt,
        net::{SocketAddr, UnixDatagram},
    },
    path::Path,
    sync::{Arc, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::{
    signal::unix::{SignalKind, signal},
    sync::{oneshot, watch},
};

/// Reports termination even while unwinding; joins keep the result.
/// A finite completion snapshot explicitly marks its expected return.
pub(crate) struct WorkerExit {
    events: tokio::sync::watch::Sender<Option<&'static str>>,
    name: &'static str,
    expected: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}
impl WorkerExit {
    pub(crate) fn new(
        events: tokio::sync::watch::Sender<Option<&'static str>>,
        name: &'static str,
        expected: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Self {
        Self {
            events,
            name,
            expected,
        }
    }
}
impl Drop for WorkerExit {
    fn drop(&mut self) {
        if self
            .expected
            .as_ref()
            .is_some_and(|expected| expected.load(std::sync::atomic::Ordering::Acquire))
        {
            return;
        }
        self.events.send_if_modified(|event| {
            if event.is_some() {
                return false;
            }
            *event = Some(self.name);
            true
        });
    }
}

/// Service-manager inputs captured with the binary's other environment inputs.
#[derive(Debug, Clone)]
pub(crate) struct Inputs {
    notify: Option<SocketAddr>,
    watchdog: Option<Duration>,
}
impl Inputs {
    pub(crate) fn from_environment(env: &EnvSnapshot, pid: u32) -> Result<Self, Error> {
        let notify = env
            .get_os("NOTIFY_SOCKET")
            .filter(|value| !value.is_empty())
            .map(notification_address)
            .transpose()
            .map_err(|error| Error::Invalid(format!("NOTIFY_SOCKET: {error}")))?;
        let watchdog = env
            .text("WATCHDOG_USEC")?
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|_| Error::Invalid("WATCHDOG_USEC must be an unsigned integer".into()))
            })
            .transpose()?;
        let watchdog_owner = env
            .text("WATCHDOG_PID")?
            .map(|value| {
                value
                    .parse::<u32>()
                    .map_err(|_| Error::Invalid("WATCHDOG_PID must be a process ID".into()))
            })
            .transpose()?;
        let watchdog = watchdog
            .filter(|value| *value > 0)
            .filter(|_| watchdog_owner.is_none_or(|owner| owner == pid));
        if watchdog.is_some() && notify.is_none() {
            return Err(Error::Invalid(
                "WATCHDOG_USEC requires NOTIFY_SOCKET".into(),
            ));
        }
        Ok(Self {
            notify,
            watchdog: watchdog.map(Duration::from_micros),
        })
    }
}

fn notification_address(value: &OsStr) -> io::Result<SocketAddr> {
    let bytes = value.as_bytes();
    if let Some(name) = bytes.strip_prefix(b"@") {
        if name.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "abstract socket name is empty",
            ));
        }
        #[cfg(target_os = "linux")]
        {
            SocketAddr::from_abstract_name(name)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "abstract notification sockets require Linux",
            ))
        }
    } else if Path::new(value).is_absolute() {
        SocketAddr::from_pathname(Path::new(value))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "socket path must be absolute or abstract",
        ))
    }
}

struct Notifications {
    socket: UnixDatagram,
    address: SocketAddr,
}
impl Notifications {
    fn new(address: SocketAddr) -> io::Result<Self> {
        let socket = UnixDatagram::unbound()?;
        socket.set_nonblocking(true)?;
        Ok(Self { socket, address })
    }
    fn send(&self, message: &[u8]) -> io::Result<()> {
        let sent = self.socket.send_to_addr(message, &self.address)?;
        if sent != message.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "incomplete service notification",
            ));
        }
        Ok(())
    }
    /// Sends a status report. Readiness and the watchdog are the sends whose
    /// failure stops the daemon; a lost report is replaced by the next.
    fn report(&self, message: &str) {
        let _ = self.send(message.as_bytes());
    }
}

/// How often the supervisor reports a step in progress, and how long a step
/// runs before each report names it in the journal.
const REPORT_EVERY: Duration = Duration::from_secs(10);
/// How long the service manager waits for the next report while the daemon
/// opens: a long open (a large index to open, many sources to watch) keeps
/// extending the start timeout, and a process that stops reporting fails it.
const OPEN_EXTENSION: Duration = Duration::from_secs(60);

/// Opening or stopping, the two phases whose steps are reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Opening,
    Stopping,
}
impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Stopping => "stopping",
        }
    }
    fn title(self) -> &'static str {
        match self {
            Self::Opening => "Opening",
            Self::Stopping => "Stopping",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Step {
    phase: Phase,
    what: &'static str,
    since: Instant,
}

/// The step the daemon is in while it opens or stops. The thread doing the
/// work sets it ([`Steps`]); the supervisor reports it to the service
/// manager and, while a step runs long, to the journal.
#[derive(Clone)]
pub(crate) struct Progress(Arc<watch::Sender<Option<Step>>>);
impl Default for Progress {
    fn default() -> Self {
        Self(Arc::new(watch::channel(None).0))
    }
}
impl Progress {
    pub(crate) fn steps(&self, phase: Phase) -> Steps {
        Steps {
            progress: self.clone(),
            phase,
            current: None,
        }
    }
    fn subscribe(&self) -> watch::Receiver<Option<Step>> {
        self.0.subscribe()
    }
}

/// One phase's steps on the thread running them. Each step's end is logged
/// with its duration; the step is also what `status` shows the thread doing.
pub(crate) struct Steps {
    progress: Progress,
    phase: Phase,
    current: Option<(Step, polylogue_store::activity::Doing)>,
}
impl Steps {
    /// Ends the step in progress and starts `what`.
    pub(crate) fn next(&mut self, what: &'static str) {
        self.finish();
        let step = Step {
            phase: self.phase,
            what,
            since: Instant::now(),
        };
        self.progress.0.send_replace(Some(step));
        self.current = Some((step, polylogue_store::activity::doing(what)));
    }
    /// Ends the step in progress.
    pub(crate) fn finish(&mut self) {
        if let Some((step, _doing)) = self.current.take() {
            polylogue_store::log!(
                info,
                "{}: {} took {}",
                step.phase.name(),
                step.what,
                seconds(step.since.elapsed().as_secs_f64())
            );
        }
    }
}
impl Drop for Steps {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Every duration the lifecycle logs, in one format.
fn seconds(value: f64) -> String {
    format!("{value:.3} s")
}

/// The journal line for a step that has run at least [`REPORT_EVERY`]:
/// the step, how long it has run, and what every busy thread is doing,
/// which names what the step waits on.
fn slow_step(
    step: &Step,
    now: Instant,
    threads: &[polylogue_store::activity::Thread],
) -> Option<String> {
    let elapsed = now.saturating_duration_since(step.since);
    if elapsed < REPORT_EVERY {
        return None;
    }
    let mut line = format!(
        "{}: still {} after {}",
        step.phase.name(),
        step.what,
        seconds(elapsed.as_secs_f64())
    );
    let busy: Vec<String> = threads.iter().map(thread_activity).collect();
    if busy.is_empty() {
        line.push_str("; no thread reports what it is doing");
    } else {
        line.push_str("; waiting on ");
        line.push_str(&busy.join("; "));
    }
    Some(line)
}

fn thread_activity(thread: &polylogue_store::activity::Thread) -> String {
    let mut text = format!("{}:", thread.name);
    if !thread.doing.is_empty() {
        let doing: Vec<String> = thread
            .doing
            .iter()
            .map(|work| match &work.subject {
                Some(subject) => format!("{} ({subject})", work.what),
                None => work.what.clone(),
            })
            .collect();
        text.push_str(&format!(
            " {} for {}",
            doing.join(" > "),
            seconds(thread.doing[0].seconds)
        ));
        if let Some(idle) = thread.seconds_since_progress {
            text.push_str(&format!(", last progress {} ago", seconds(idle)));
        }
    }
    for statement in &thread.statements {
        text.push_str(&format!(
            ", running `{}` on the {} for {}",
            statement.sql,
            statement.connection,
            seconds(statement.seconds)
        ));
    }
    text
}

/// What the service manager shows as the unit's status: the phase and step.
fn step_status(step: &Step) -> String {
    format!("STATUS={}: {}", step.phase.title(), step.what)
}

/// What the service manager shows while the daemon serves.
fn serving_status(archive: &polylogue_store::Archive) -> String {
    match archive.index_build() {
        Some(build) => format!("STATUS=Serving; {build}"),
        None => "STATUS=Serving".into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Signal {
    Terminate,
    Interrupt,
}
impl Signal {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Terminate => "SIGTERM",
            Self::Interrupt => "SIGINT",
        }
    }
    fn number(self) -> i32 {
        match self {
            Self::Terminate => libc::SIGTERM,
            Self::Interrupt => libc::SIGINT,
        }
    }
}
struct Signals {
    terminate: tokio::signal::unix::Signal,
    interrupt: tokio::signal::unix::Signal,
}
impl Signals {
    fn new() -> io::Result<Self> {
        Ok(Self {
            terminate: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
        })
    }
    async fn next(&mut self) -> io::Result<Signal> {
        let (received, kind) = tokio::select! {
            received = self.terminate.recv() => (received, Signal::Terminate),
            received = self.interrupt.recv() => (received, Signal::Interrupt),
        };
        received
            .map(|()| kind)
            .ok_or_else(|| io::Error::other("OS signal stream closed"))
    }
}

// Field order preserves transport-before-runtime cleanup on a startup error or panic.
struct Started {
    socket: Server,
    web: Option<crate::web::Server>,
    runtime: Runtime,
}

/// Installs signal handling before startup can block. The independent supervisor
/// remains responsive while startup and graceful draining run on other threads.
pub(crate) struct Supervisor {
    started: Option<oneshot::Sender<Started>>,
    thread: Option<JoinHandle<io::Result<()>>>,
    progress: Progress,
}
impl Supervisor {
    pub(crate) fn start(inputs: &Inputs) -> io::Result<Self> {
        let inputs = inputs.clone();
        let progress = Progress::default();
        let reported = progress.clone();
        let (started, receive) = oneshot::channel();
        let (ready, ready_receiver) = mpsc::sync_channel(1);
        let thread = polylogue_model::spawn_named("polylogue-lifecycle", move || {
            let prepared = (|| {
                let executor = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let signals = {
                    let _entered = executor.enter();
                    Signals::new()?
                };
                let notifications = inputs.notify.map(Notifications::new).transpose()?;
                Ok::<_, io::Error>((executor, signals, notifications))
            })();
            match prepared {
                Ok((executor, signals, notifications)) => {
                    if ready.send(Ok(())).is_err() {
                        return Ok(());
                    }
                    executor.block_on(supervise(
                        receive,
                        signals,
                        notifications,
                        inputs.watchdog,
                        reported,
                    ))
                }
                Err(error) => {
                    let _ = ready.send(Err(error));
                    Ok(())
                }
            }
        })?;
        match ready_receiver.recv() {
            Ok(Ok(())) => Ok(Self {
                started: Some(started),
                thread: Some(thread),
                progress,
            }),
            result => {
                drop(started);
                let _ = thread.join();
                match result {
                    Ok(Err(error)) => Err(error),
                    _ => Err(io::Error::other("signal supervisor did not start")),
                }
            }
        }
    }
    /// Where the opening and stopping steps are set for this supervisor to report.
    pub(crate) fn progress(&self) -> &Progress {
        &self.progress
    }
    pub(crate) fn run(
        mut self,
        runtime: Runtime,
        socket: Server,
        web: Option<crate::web::Server>,
    ) -> io::Result<()> {
        let started = Started {
            socket,
            web,
            runtime,
        };
        if self
            .started
            .take()
            .is_none_or(|sender| sender.send(started).is_err())
        {
            return Err(io::Error::other(
                "signal supervisor stopped during daemon startup",
            ));
        }
        self.join()
    }
    fn join(&mut self) -> io::Result<()> {
        self.thread.take().map_or(Ok(()), |thread| {
            thread
                .join()
                .map_err(|_| io::Error::other("signal supervisor panicked"))?
        })
    }
}
impl Drop for Supervisor {
    fn drop(&mut self) {
        self.started.take();
        if let Err(error) = self.join() {
            polylogue_store::log!(error, "lifecycle: {error}");
        }
    }
}

enum Stop {
    Signal(Signal),
    Listener,
    /// A loop run until idle has no work left.
    Idle,
    Failed(io::Error),
}

async fn supervise(
    mut receive: oneshot::Receiver<Started>,
    mut signals: Signals,
    notifications: Option<Notifications>,
    watchdog: Option<Duration>,
    progress: Progress,
) -> io::Result<()> {
    let mut steps = progress.subscribe();
    let mut reports = tokio::time::interval(REPORT_EVERY);
    reports.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut first_signal = None;
    let mut app = loop {
        tokio::select! { biased;
            signal = signals.next() => {
                let signal = signal?;
                if first_signal.is_some() { end_at_once(signal, &steps); }
                polylogue_store::log!(info, "stopping on {}", signal.name());
                first_signal = Some(signal);
            }
            started = &mut receive => match started { Ok(started) => break started, Err(_) => return Ok(()) },
            // The first tick is immediate, so the extension precedes any step.
            _ = reports.tick() => {
                if let Some(notifications) = &notifications {
                    notifications.report(&format!("EXTEND_TIMEOUT_USEC={}", OPEN_EXTENSION.as_micros()));
                }
                log_slow_step(Phase::Opening, &steps);
            }
            _ = steps.changed() => report_step(notifications.as_ref(), Phase::Opening, &mut steps),
        }
    };
    let stop = match app.socket.take_listener_stopped() {
        Err(error) => Stop::Failed(error),
        Ok(mut listener_ended) => {
            if let Some(signal) = first_signal {
                Stop::Signal(signal)
            } else {
                monitor(
                    &app,
                    &mut signals,
                    &mut listener_ended,
                    notifications.as_ref(),
                    watchdog,
                )
                .await
            }
        }
    };
    if let Stop::Idle = &stop {
        polylogue_store::log!(info, "the loop is idle; stopping");
    }
    if let Stop::Signal(signal) = &stop {
        if first_signal.is_none() {
            polylogue_store::log!(info, "stopping on {}", signal.name());
        }
        first_signal = Some(*signal);
    }
    if let Some(notifications) = &notifications {
        if let Err(error) = notifications.send(b"STOPPING=1") {
            polylogue_store::log!(error, "service stopping notification: {error}");
        }
        notifications.report("STATUS=Stopping");
    }
    // An opening step reported after startup ended is not a stopping step.
    steps.mark_unchanged();
    reports.reset();
    let mut drained = tokio::task::spawn_blocking(move || drain(app));
    let result = loop {
        tokio::select! { biased;
            signal = signals.next() => {
                let signal = signal?;
                if first_signal.is_some() { end_at_once(signal, &steps); }
                first_signal = Some(signal);
            }
            result = &mut drained => break result.map_err(|_| io::Error::other("daemon shutdown panicked"))?,
            _ = steps.changed() => report_step(notifications.as_ref(), Phase::Stopping, &mut steps),
            _ = reports.tick() => log_slow_step(Phase::Stopping, &steps),
        }
    };
    result?;
    match stop {
        Stop::Failed(error) => Err(error),
        Stop::Signal(_) | Stop::Listener | Stop::Idle => Ok(()),
    }
}

/// A second signal ends the process at once; the line says what it cut short.
fn end_at_once(signal: Signal, steps: &watch::Receiver<Option<Step>>) -> ! {
    match *steps.borrow() {
        Some(step) => polylogue_store::log!(
            warning,
            "{} again while {} ({}); ending at once",
            signal.name(),
            step.phase.name(),
            step.what
        ),
        None => polylogue_store::log!(warning, "{} again; ending at once", signal.name()),
    }
    std::process::exit(128 + signal.number());
}

fn report_step(
    notifications: Option<&Notifications>,
    phase: Phase,
    steps: &mut watch::Receiver<Option<Step>>,
) {
    let step = *steps.borrow_and_update();
    if let Some(notifications) = notifications
        && let Some(step) = step.filter(|step| step.phase == phase)
    {
        notifications.report(&step_status(&step));
    }
}

fn log_slow_step(phase: Phase, steps: &watch::Receiver<Option<Step>>) {
    let step = *steps.borrow();
    if let Some(step) = step.filter(|step| step.phase == phase)
        && let Some(line) = slow_step(
            &step,
            Instant::now(),
            &polylogue_store::activity::threads(polylogue_store::activity::SLOW_STATEMENT),
        )
    {
        polylogue_store::log!(warning, "{line}");
    }
}

async fn monitor(
    app: &Started,
    signals: &mut Signals,
    listener_ended: &mut oneshot::Receiver<()>,
    notifications: Option<&Notifications>,
    watchdog: Option<Duration>,
) -> Stop {
    let mut workers = app.runtime.worker_events();
    let mut idle = app.runtime.idle();
    if let Some(failure) = *workers.borrow_and_update() {
        return Stop::Failed(io::Error::other(failure));
    }
    if idle.borrow_and_update().is_some() {
        return Stop::Idle;
    }
    if let Some(failure) = app.runtime.worker_failure() {
        return Stop::Failed(io::Error::other(failure));
    }
    if let Some(notifications) = notifications {
        notifications.report("STATUS=Opening: waiting for the socket to answer");
        let responsive = tokio::select! { biased;
            signal = signals.next() => return signal.map_or_else(Stop::Failed, Stop::Signal),
            _ = &mut *listener_ended => return Stop::Listener,
            changed = workers.changed() => return worker_stop(changed, &mut workers),
            _ = idle.changed() => return Stop::Idle,
            responsive = probe(&app.runtime.daemon.archive, READINESS_BUDGET) => responsive,
        };
        if !responsive {
            return Stop::Failed(io::Error::new(
                io::ErrorKind::TimedOut,
                "daemon socket did not answer the readiness probe",
            ));
        }
        if let Some(failure) = app.runtime.worker_failure() {
            return Stop::Failed(io::Error::other(failure));
        }
        if let Err(error) = notifications.send(b"READY=1") {
            return Stop::Failed(error);
        }
    }
    // The status line follows the index build while the daemon serves.
    let mut status = notifications.map(|notifications| {
        let mut ticker = tokio::time::interval(REPORT_EVERY);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        (notifications, ticker, String::new())
    });
    let mut ticker = watchdog.map(|period| {
        let mut ticker = tokio::time::interval(period / 2);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        (ticker, period / 4)
    });
    loop {
        let pulse = async {
            let Some((ticker, budget)) = &mut ticker else {
                return pending::<bool>().await;
            };
            ticker.tick().await;
            probe(&app.runtime.daemon.archive, *budget).await
        };
        tokio::select! { biased;
            signal = signals.next() => return signal.map_or_else(Stop::Failed, Stop::Signal),
            _ = &mut *listener_ended => return Stop::Listener,
            changed = workers.changed() => return worker_stop(changed, &mut workers),
            _ = idle.changed() => return Stop::Idle,
            responsive = pulse => {
                if let Some(failure) = app.runtime.worker_failure() { return Stop::Failed(io::Error::other(failure)); }
                if responsive && let Some(notifications) = notifications && let Err(error) = notifications.send(b"WATCHDOG=1") { return Stop::Failed(error); }
            }
            () = report_serving(&mut status, &app.runtime.daemon.archive) => {}
        }
    }
}

/// Sends the serving status on each report tick when it changed; pending
/// forever without a service manager.
async fn report_serving(
    status: &mut Option<(&Notifications, tokio::time::Interval, String)>,
    archive: &polylogue_store::Archive,
) {
    let Some((notifications, ticker, sent)) = status else {
        return pending().await;
    };
    ticker.tick().await;
    let line = serving_status(archive);
    if *sent != line {
        notifications.report(&line);
        *sent = line;
    }
}

fn worker_stop(
    changed: Result<(), tokio::sync::watch::error::RecvError>,
    workers: &mut tokio::sync::watch::Receiver<Option<&'static str>>,
) -> Stop {
    let failure = if changed.is_err() {
        "worker lifecycle channel closed"
    } else {
        (*workers.borrow_and_update()).unwrap_or("worker lifecycle event was empty")
    };
    Stop::Failed(io::Error::other(failure))
}

/// How long readiness waits for the socket to answer `ping`.
const READINESS_BUDGET: Duration = Duration::from_secs(2);

async fn probe(archive: &polylogue_store::Archive, budget: Duration) -> bool {
    let archive = archive.clone();
    tokio::task::spawn_blocking(move || {
        let request = command_surface::Request {
            operation: super::server::PING.into(),
            arguments: serde_json::json!({}),
            request_id: "daemon-liveness".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        crate::client::Exchange {
            caller: Default::default(),
            budget: Some(budget),
            ..crate::client::Exchange::new(&request)
        }
        .call_ready(&archive)
        .is_ok_and(|answer| !answer.is_error())
    })
    .await
    .unwrap_or(false)
}

fn drain(mut app: Started) -> io::Result<()> {
    let started = Instant::now();
    let mut steps = app.runtime.progress.steps(Phase::Stopping);
    steps.next("refusing new requests");
    app.runtime.stop_accepting();
    steps.next("socket drain");
    let mut failure = app.socket.shutdown().err();
    if let Some(web) = &mut app.web {
        steps.next("HTTP drain");
        if let Err(error) = web.shutdown() {
            failure.get_or_insert(error);
        }
    }
    steps.finish();
    if let Err(error) = app.runtime.shutdown() {
        failure.get_or_insert(error);
    }
    polylogue_store::log!(
        info,
        "stopped in {}",
        seconds(started.elapsed().as_secs_f64())
    );
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;

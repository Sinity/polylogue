//! OS signals and service supervision around the ordinary daemon runtime.
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
    sync::mpsc,
    thread::JoinHandle,
    time::Duration,
};
use tokio::{
    signal::unix::{SignalKind, signal},
    sync::oneshot,
};
use tokio_util::sync::CancellationToken;

/// Publishes termination even while unwinding; joins retain the actual result.
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
        let owner = env
            .text("WATCHDOG_PID")?
            .map(|value| {
                value
                    .parse::<u32>()
                    .map_err(|_| Error::Invalid("WATCHDOG_PID must be a process ID".into()))
            })
            .transpose()?;
        let watchdog = watchdog
            .filter(|value| *value > 0)
            .filter(|_| owner.is_none_or(|owner| owner == pid));
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
}
impl Supervisor {
    pub(crate) fn start(inputs: &Inputs) -> io::Result<Self> {
        let inputs = inputs.clone();
        let (started, receive) = oneshot::channel();
        let (ready, ready_receiver) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("polylogue-lifecycle".into())
            .spawn(move || {
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
            eprintln!("polylogue: lifecycle: {error}");
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
) -> io::Result<()> {
    let mut first_signal = None;
    let mut app = loop {
        tokio::select! { biased;
            signal = signals.next() => {
                let signal = signal?;
                if first_signal.is_some() { std::process::exit(128 + signal.number()); }
                eprintln!("polylogue: stopping on {}", signal.name());
                first_signal = Some(signal);
            }
            started = &mut receive => match started { Ok(started) => break started, Err(_) => return Ok(()) },
        }
    };
    let stop = match app.socket.take_quiescing() {
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
        eprintln!("polylogue: the loop is idle; stopping");
    }
    if let Stop::Signal(signal) = &stop {
        if first_signal.is_none() {
            eprintln!("polylogue: stopping on {}", signal.name());
        }
        first_signal = Some(*signal);
    }
    if let Some(notifications) = &notifications
        && let Err(error) = notifications.send(b"STOPPING=1")
    {
        eprintln!("polylogue: service stopping notification: {error}");
    }
    let mut drained = tokio::task::spawn_blocking(move || drain(app));
    let result = loop {
        tokio::select! { biased;
            signal = signals.next() => {
                let signal = signal?;
                if first_signal.is_some() { std::process::exit(128 + signal.number()); }
                first_signal = Some(signal);
            }
            result = &mut drained => break result.map_err(|_| io::Error::other("daemon shutdown panicked"))?,
        }
    };
    result?;
    match stop {
        Stop::Failed(error) => Err(error),
        Stop::Signal(_) | Stop::Listener | Stop::Idle => Ok(()),
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
        }
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
    matches!(
        tokio::time::timeout(
            budget,
            crate::client::call_ready_as(
                archive,
                &request,
                &CancellationToken::new(),
                Default::default(),
            )
        )
        .await,
        Ok(Ok(_))
    )
}

fn drain(mut app: Started) -> io::Result<()> {
    app.runtime.stop_accepting();
    let mut failure = app.socket.shutdown().err();
    if let Some(web) = &mut app.web
        && let Err(error) = web.shutdown()
    {
        failure.get_or_insert(error);
    }
    if let Err(error) = app.runtime.shutdown() {
        failure.get_or_insert(error);
    }
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;

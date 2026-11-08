use super::*;
use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
};

fn inputs(values: &[(&str, &str)], pid: u32) -> Result<Inputs, Error> {
    Inputs::from_environment(
        &EnvSnapshot::from_values(
            values
                .iter()
                .map(|(key, value)| ((*key).into(), (*value).into())),
        ),
        pid,
    )
}

#[cfg(target_os = "linux")]
#[test]
fn service_inputs_observe_the_watchdog_owner_and_keep_native_socket_names() {
    assert!(inputs(&[], 7).unwrap().notify.is_none());
    let configured = inputs(
        &[
            ("NOTIFY_SOCKET", "@synthetic-notify"),
            ("WATCHDOG_USEC", "1000000"),
            ("WATCHDOG_PID", "7"),
        ],
        7,
    )
    .unwrap();
    assert_eq!(configured.watchdog, Some(Duration::from_secs(1)));
    assert_eq!(
        configured.notify.unwrap().as_abstract_name(),
        Some(b"synthetic-notify".as_slice())
    );
    assert!(
        inputs(
            &[
                ("NOTIFY_SOCKET", "@synthetic-notify"),
                ("WATCHDOG_USEC", "1000000"),
                ("WATCHDOG_PID", "8")
            ],
            7
        )
        .unwrap()
        .watchdog
        .is_none()
    );
    for values in [
        vec![("NOTIFY_SOCKET", "relative")],
        vec![("NOTIFY_SOCKET", "@")],
        vec![("WATCHDOG_USEC", "many")],
        vec![("WATCHDOG_USEC", "1000000")],
        vec![("WATCHDOG_PID", "other")],
    ] {
        assert!(inputs(&values, 7).is_err(), "{values:?}");
    }
}

#[test]
fn notifications_are_sent_to_the_captured_native_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("notify");
    let receiver = UnixDatagram::bind(&path).unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let configured = Inputs::from_environment(
        &EnvSnapshot::from_values([("NOTIFY_SOCKET".into(), path.into_os_string())]),
        7,
    )
    .unwrap();
    let sender = Notifications::new(configured.notify.unwrap()).unwrap();
    sender.send(b"READY=1").unwrap();
    let mut bytes = [0; 128];
    let n = receiver.recv(&mut bytes).unwrap();
    assert_eq!(&bytes[..n], b"READY=1");
}

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn spawn(mode: &str, root: &Path, notify: Option<&Path>, watchdog: bool) -> Process {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "daemon::lifecycle::tests::isolated_signal_child",
            "--ignored",
            "--nocapture",
        ])
        .env("POLYLOGUE_LIFECYCLE_TEST", mode)
        .env("POLYLOGUE_LIFECYCLE_ROOT", root)
        .env_remove("WATCHDOG_PID")
        .env_remove("WATCHDOG_USEC")
        .env_remove("NOTIFY_SOCKET")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(notify) = notify {
        command.env("NOTIFY_SOCKET", notify);
    }
    if watchdog {
        command.env("WATCHDOG_USEC", "2000000");
    }
    Process(polylogue_testkit::spawn_daemon(&mut command).unwrap())
}
fn send(process: &Process, signal: rustix::process::Signal) {
    let pid = rustix::process::Pid::from_raw(process.0.id().try_into().unwrap()).unwrap();
    rustix::process::kill_process(pid, signal).unwrap();
}
fn notification(receiver: &UnixDatagram) -> io::Result<String> {
    let mut bytes = [0; 128];
    receiver
        .recv(&mut bytes)
        .map(|n| String::from_utf8(bytes[..n].to_vec()).unwrap())
}
fn until(receiver: &UnixDatagram, expected: &str) {
    loop {
        if notification(receiver).unwrap() == expected {
            return;
        }
    }
}

#[test]
fn sigterm_and_sigint_drain_real_daemons_and_preserve_admitted_writes() {
    for signal in [rustix::process::Signal::TERM, rustix::process::Signal::INT] {
        let dir = tempfile::tempdir().unwrap();
        let notify = dir.path().join("notify");
        let receiver = UnixDatagram::bind(&notify).unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let root = dir.path().join("archive");
        let mut process = spawn("daemon", &root, Some(&notify), false);
        until(&receiver, "READY=1");
        let archive = polylogue_store::Archive::at(&root);
        let call = command_surface::Request {
            operation: "setting".into(),
            arguments: serde_json::json!({"key":"browser.actions", "value":"on"}),
            request_id: "accepted-before-signal".into(),
            deadline: command_surface::Deadline::Milliseconds(5000),
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let answer = crate::client::call_ready(&archive, &call).unwrap();
        assert!(
            matches!(
                answer,
                command_surface::Answer {
                    outcome: command_surface::Outcome::Ok(_),
                    ..
                }
            ),
            "{answer:?}"
        );
        let stopping = std::time::Instant::now();
        send(&process, signal);
        until(&receiver, "STOPPING=1");
        assert!(process.0.wait().unwrap().success());
        eprintln!(
            "{signal:?} shutdown_seconds={:.6}",
            stopping.elapsed().as_secs_f64()
        );
        let (archive, lock) = polylogue_store::Archive::open_or_create(&root).unwrap();
        let reads = polylogue_store::ReadPool::open(&archive).unwrap();
        let reader = reads.reader().unwrap();
        assert_eq!(
            reader
                .setting_value("browser.actions", "synthetic-machine")
                .unwrap(),
            Some(polylogue_model::settings::Value::Boolean(true))
        );
        assert!(
            reader
                .user_request("accepted-before-signal")
                .unwrap()
                .is_some()
        );
        drop(reader);
        drop(reads);
        drop(lock);
    }
}

#[test]
fn a_second_signal_ends_the_process_while_startup_is_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let mut process = spawn("blocked-startup", dir.path(), None, false);
    let mut output = BufReader::new(process.0.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        assert_ne!(output.read_line(&mut line).unwrap(), 0);
        if line.contains("signal-supervisor-ready") {
            break;
        }
        line.clear();
    }
    send(&process, rustix::process::Signal::TERM);
    let mut errors = BufReader::new(process.0.stderr.take().unwrap());
    line.clear();
    loop {
        assert_ne!(errors.read_line(&mut line).unwrap(), 0);
        if line.contains("stopping on SIGTERM") {
            break;
        }
        line.clear();
    }
    // `Child::wait` closes the child's stdin, and end of input ends the
    // blocked startup normally. Holding stdin leaves the second signal as
    // the only way the child can exit.
    let _blocked_startup = process.0.stdin.take().unwrap();
    send(&process, rustix::process::Signal::INT);
    assert_eq!(process.0.wait().unwrap().code(), Some(128 + libc::SIGINT));
}

#[test]
fn watchdog_requires_a_response_from_the_actual_request_path() {
    let dir = tempfile::tempdir().unwrap();
    let notify = dir.path().join("notify");
    let receiver = UnixDatagram::bind(&notify).unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let root = dir.path().join("archive");
    let mut process = spawn("daemon", &root, Some(&notify), true);
    until(&receiver, "READY=1");
    until(&receiver, "WATCHDOG=1");
    until(&receiver, "WATCHDOG=1");
    let archive = polylogue_store::Archive::at(&root);
    let socket = archive.socket();
    std::fs::rename(&socket, dir.path().join("responding-socket")).unwrap();
    let blackhole = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    receiver.set_nonblocking(true).unwrap();
    while notification(&receiver).is_ok() {}
    receiver.set_nonblocking(false).unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    // At most one probe already in flight may have answered before the path changed.
    let quiet = (0..2).any(|_| match notification(&receiver) {
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            true
        }
        Ok(message) if message == "WATCHDOG=1" => false,
        other => panic!("unexpected watchdog observation: {other:?}"),
    });
    assert!(quiet, "watchdog was fed without request responses");
    drop(blackhole);
    send(&process, rustix::process::Signal::TERM);
    assert!(process.0.wait().unwrap().success());
}

#[test]
fn continuous_worker_exit_wakes_supervision_without_a_watchdog() {
    for notify_enabled in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let notify = dir.path().join("notify");
        let _receiver = notify_enabled.then(|| UnixDatagram::bind(&notify).unwrap());
        let mut process = spawn(
            "worker-exit",
            &dir.path().join("archive"),
            notify_enabled.then_some(notify.as_path()),
            false,
        );
        assert!(process.0.wait().unwrap().success());
        let mut output = String::new();
        std::io::Read::read_to_string(process.0.stdout.as_mut().unwrap(), &mut output).unwrap();
        assert!(
            output.contains("observed:signal worker terminated"),
            "{output}"
        );
    }
}

#[test]
fn exit_events_are_retained_report_panics_and_allow_finite_completion() {
    let (events, mut received) = tokio::sync::watch::channel(None);
    let worker_events = events.clone();
    assert!(
        std::thread::spawn(move || {
            let _exit = WorkerExit::new(worker_events, "synthetic worker terminated", None);
            panic!("synthetic worker panic");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        *received.borrow_and_update(),
        Some("synthetic worker terminated")
    );
    let (events, received) = tokio::sync::watch::channel(None);
    let expected = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    drop(WorkerExit::new(
        events,
        "finite completion terminated",
        Some(expected),
    ));
    assert_eq!(*received.borrow(), None);
}

#[test]
fn busy_workers_remain_live_until_their_actual_return() {
    let (events, received) = tokio::sync::watch::channel(None);
    let (entered, active) = std::sync::mpsc::sync_channel(0);
    let (release, blocked) = std::sync::mpsc::sync_channel(0);
    let thread = std::thread::spawn(move || {
        let _exit = WorkerExit::new(events, "synthetic worker terminated", None);
        entered.send(()).unwrap();
        blocked.recv().unwrap();
    });
    active.recv().unwrap();
    assert_eq!(*received.borrow(), None);
    assert!(!received.has_changed().unwrap());
    release.send(()).unwrap();
    thread.join().unwrap();
    assert_eq!(*received.borrow(), Some("synthetic worker terminated"));
}

#[test]
#[ignore = "isolated subprocess helper"]
fn isolated_signal_child() {
    let mode = std::env::var("POLYLOGUE_LIFECYCLE_TEST").unwrap();
    let inputs = Inputs::from_environment(&EnvSnapshot::capture(), std::process::id()).unwrap();
    if mode == "blocked-startup" {
        let _supervisor = Supervisor::start(&inputs).unwrap();
        println!("signal-supervisor-ready");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        return;
    }
    let root = std::path::PathBuf::from(std::env::var_os("POLYLOGUE_LIFECYCLE_ROOT").unwrap());
    let home = root.parent().unwrap().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut engine = crate::test_daemon::engine_config(home);
    engine.tool_dirs.clear();
    engine.workers = 1;
    let mut startup = crate::config::Startup::for_test(engine, &root);
    startup.lifecycle = inputs;
    if mode == "worker-exit" {
        let runtime = super::super::service::start(
            std::sync::Arc::clone(&startup.config),
            startup.engine.clone(),
            startup.mode.clone(),
            polylogue_engine::Lifetime::UntilStopped,
            startup.stopped_attempt,
        )
        .unwrap();
        let stop_worker = runtime.signal_stop_for_test();
        let socket = Server::start(
            std::sync::Arc::clone(&runtime.daemon),
            std::sync::Arc::clone(&runtime.operations),
        )
        .unwrap();
        let mut app = Started {
            socket,
            web: None,
            runtime,
        };
        let mut listener_ended = app.socket.take_quiescing().unwrap();
        let executor = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let stop = executor.block_on(async {
            let mut signals = Signals::new().unwrap();
            let notifications = startup
                .lifecycle
                .notify
                .clone()
                .map(Notifications::new)
                .transpose()
                .unwrap();
            let monitored = monitor(
                &app,
                &mut signals,
                &mut listener_ended,
                notifications.as_ref(),
                None,
            );
            tokio::pin!(monitored);
            // Poll until supervision waits on events, then terminate an actual
            // continuous worker. No watchdog tick can discover this exit.
            std::future::poll_fn(
                |cx| match std::future::Future::poll(monitored.as_mut(), cx) {
                    std::task::Poll::Pending => std::task::Poll::Ready(()),
                    std::task::Poll::Ready(_) => panic!("supervision stopped before the worker"),
                },
            )
            .await;
            stop_worker.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(5), monitored)
                .await
                .expect("worker exit did not wake supervision")
        });
        let Stop::Failed(error) = stop else {
            panic!("worker exit was not a failure")
        };
        assert_eq!(error.to_string(), "signal worker terminated");
        drain(app).unwrap();
        println!("observed:{error}");
        return;
    }
    super::super::server::serve(
        &startup,
        super::super::server::Run {
            reads_sources: true,
            lifetime: polylogue_engine::Lifetime::UntilStopped,
        },
    )
    .unwrap();
}

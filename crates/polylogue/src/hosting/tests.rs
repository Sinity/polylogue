use super::*;
use std::os::unix::fs::PermissionsExt;

/// A fixture ACP agent: answers `initialize` and `session/new`, and for
/// each prompt counts it in `prompts`, asks a permission, and finishes the
/// prompt once the permission is answered. `polylogue_test/quit` exits.
const ACP_AGENT: &str = r#"#!/bin/sh
count="$1"
n=0
prompt=
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\("[^"]*"\).*/\1/p')
  case "$line" in
    *polylogue_test/quit*) exit 0 ;;
    *'"method":"initialize"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":1}}\n' "$id" ;;
    *'"method":"session/new"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"sessionId":"s1"}}\n' "$id" ;;
    *'"method":"session/prompt"'*)
      n=$((n+1)); echo "$n" > "$count"; prompt=$id
      printf '{"jsonrpc":"2.0","id":90%s,"method":"session/request_permission","params":{"sessionId":"s1","toolCall":{"toolCallId":"t%s"},"options":[{"optionId":"yes","name":"Allow","kind":"allow_once"},{"optionId":"no","name":"Reject","kind":"reject_once"}]}}\n' "$n" "$n" ;;
    *'"optionId":"yes"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn"}}\n' "$prompt" ;;
    *'"method":"session/cancel"'*) echo cancel >> "$count.cancel" ;;
    *'"outcome":"cancelled"'*) echo cancelled >> "$count.cancel"; printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"cancelled"}}\n' "$prompt" ;;
  esac
done
"#;

/// A fixture Claude Code in stream-json mode: each user message asks to
/// run a tool; an allowed tool finishes the turn; an interrupt is noted.
const CLAUDE_AGENT: &str = r#"#!/bin/sh
log="$1"
n=0
while IFS= read -r line; do
  case "$line" in
    *polylogue_test/quit*) exit 0 ;;
    *'"type":"user"'*)
      n=$((n+1))
      printf '{"type":"control_request","request_id":"r%s","request":{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"ls"}}}\n' "$n" ;;
    *'"behavior":"allow"'*) echo "$line" >> "$log"; printf '{"type":"result","subtype":"success"}\n' ;;
    *'"behavior":"deny"'*) echo "$line" >> "$log"; printf '{"type":"result","subtype":"success"}\n' ;;
    *'"subtype":"interrupt"'*) echo interrupted >> "$log" ;;
  esac
done
"#;

fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

#[test]
fn hosted_statuses_survive_a_failed_spool_and_are_written_once_after_recovery() {
    let mut hosting = Hosting::start(
        "claude-code",
        HostProtocol::ClaudeStreamJson,
        CLAUDE_AGENT,
        Some("s1"),
    );
    std::fs::write(&hosting.config.spool, b"blocked").unwrap();
    let client = HostedStdio::connect(
        &hosting.config.socket,
        HostProtocol::ClaudeStreamJson,
        String::new(),
    )
    .unwrap();
    client.send("synthetic turn", None).unwrap();
    wait_until("the permission after failed status writes", || {
        client.request().is_some()
    });
    assert!(status_lines(&hosting.config.spool).is_empty());
    std::fs::remove_file(&hosting.config.spool).unwrap();
    client.answer(true, None).unwrap();
    wait_until("the recovered status writes", || {
        status_lines(&hosting.config.spool)
            .iter()
            .any(|(_, word)| word.as_deref() == Some("idle"))
    });
    hosting.quit(&client);
    let lines = status_lines(&hosting.config.spool);
    let words: Vec<_> = lines
        .iter()
        .filter_map(|(_, word)| word.as_deref())
        .collect();
    assert_eq!(words, ["busy", "blocked", "busy", "idle"]);
    assert_eq!(
        lines
            .iter()
            .filter(|(event, _)| event == "HostedEnded")
            .count(),
        1
    );
}

#[test]
fn a_host_reports_a_failed_final_spool_flush_after_stopping_its_producers() {
    let mut hosting = Hosting::start(
        "claude-code",
        HostProtocol::ClaudeStreamJson,
        CLAUDE_AGENT,
        Some("s1"),
    );
    std::fs::write(&hosting.config.spool, b"blocked").unwrap();
    let client = HostedStdio::connect(
        &hosting.config.socket,
        HostProtocol::ClaudeStreamJson,
        String::new(),
    )
    .unwrap();
    client.send("synthetic turn", None).unwrap();
    wait_until("the permission after failed status writes", || {
        client.request().is_some()
    });
    client
        .state
        .0
        .locked()
        .write(&json!({"method":"polylogue_test/quit"}))
        .unwrap();
    let error = hosting.server.take().unwrap().join().unwrap().unwrap_err();
    assert!(
        error.to_string().contains("host live event spool drain"),
        "{error}"
    );
    assert!(status_lines(&hosting.config.spool).is_empty());
    assert!(!hosting.config.socket.exists());
}

#[test]
fn an_agent_descendant_holding_stdin_does_not_delay_the_hosts_final_status() {
    struct Holder(i32);
    impl Drop for Holder {
        fn drop(&mut self) {
            // SAFETY: this exact PID was returned by the synthetic agent below.
            #[allow(unsafe_code)]
            unsafe {
                libc::kill(self.0, libc::SIGKILL);
            }
        }
    }
    let mut hosting = Hosting::start(
        "claude-code",
        HostProtocol::ClaudeStreamJson,
        r#"#!/bin/sh
sleep 60 <&0 >/dev/null 2>&1 &
echo "$!" > "$1.holder"
dd bs=1 count=1 of="$1.started" 2>/dev/null
while [ ! -f "$1.exit" ]; do sleep 0.01; done
"#,
        Some("s1"),
    );
    let log = hosting.dir.path().join("log");
    let holder_file = log.with_extension("holder");
    wait_until("the agent's stdin holder", || {
        std::fs::read_to_string(&holder_file).is_ok_and(|text| text.trim().parse::<i32>().is_ok())
    });
    let holder = Holder(
        std::fs::read_to_string(holder_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    );
    let client = HostedStdio::connect(
        &hosting.config.socket,
        HostProtocol::ClaudeStreamJson,
        String::new(),
    )
    .unwrap();
    client.send(&"x".repeat(256 * 1024), None).unwrap();
    // One byte read by the agent proves the producer has reached the pipe.
    // Its remaining payload exceeds the pipe capacity; the holder reads none.
    wait_until("the blocked pipe write", || {
        std::fs::metadata(log.with_extension("started")).is_ok_and(|metadata| metadata.len() == 1)
    });
    std::fs::write(log.with_extension("exit"), b"exit").unwrap();
    let server = hosting.server.take().unwrap();
    let (finished, outcome) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let _ = finished.send(server.join());
    });
    let result = outcome.recv_timeout(std::time::Duration::from_secs(5));
    // Release the pipe before asserting, including when the old joining path hung.
    drop(holder);
    waiter.join().unwrap();
    result
        .expect("host final status waited for a descendant's unrelated stdin pipe")
        .unwrap()
        .unwrap();
    let lines = status_lines(&hosting.config.spool);
    assert_eq!(
        lines
            .iter()
            .filter(|(event, _)| event == "HostedEnded")
            .count(),
        1
    );
    assert_eq!(lines.last().unwrap().0, "HostedEnded");
    assert!(!hosting.config.socket.exists());
    drop(client);
}

#[test]
fn a_stopped_host_appends_neither_late_status_nor_a_late_retry() {
    let directory = tempfile::tempdir().unwrap();
    let config = HostConfig {
        harness: "claude-code".into(),
        protocol: HostProtocol::ClaudeStreamJson,
        argv: vec![],
        cwd: directory.path().into(),
        socket: directory.path().join("host.sock"),
        spool: directory.path().join("spool"),
        session_id: Some("s1".into()),
    };
    let session = SessionId::parse("claude-code:s1").unwrap();
    let relay = Mutex::new(Relay {
        client: None,
        backlog: VecDeque::new(),
        tracker: Tracker::new(config.protocol),
        session: Some(session.clone()),
        spool: crate::live_events::PendingLines::new("registry", "host", Default::default()),
        stopping: Arc::new(AtomicBool::new(false)),
    });
    std::fs::write(&config.spool, b"blocked").unwrap();
    status_line(
        &mut relay.locked(),
        &config,
        &session,
        None,
        "HostedStatus",
        Some("busy"),
    );
    {
        let relay = relay.locked();
        relay.stopping.store(true, Ordering::Release);
    }
    std::fs::remove_file(&config.spool).unwrap();
    retry_status(&mut relay.locked(), &config);
    assert!(
        !config.spool.exists(),
        "a stopped producer retried after handing over final flush"
    );
    status_line(
        &mut relay.locked(),
        &config,
        &session,
        None,
        "HostedEnded",
        None,
    );
    let expected = status_lines(&config.spool);
    assert_eq!(
        expected,
        [
            ("HostedStatus".into(), Some("busy".into())),
            ("HostedEnded".into(), None)
        ]
    );
    status_line(
        &mut relay.locked(),
        &config,
        &session,
        None,
        "HostedStatus",
        Some("idle"),
    );
    retry_status(&mut relay.locked(), &config);
    assert_eq!(status_lines(&config.spool), expected);
}

fn wait_until(what: &str, ok: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn host_startup_reports_a_child_exit_before_socket_publication() {
    let dir = tempfile::tempdir().unwrap();
    let mut command = std::process::Command::new("sh");
    command.args(["-c", "exit 7"]);
    let error = start_host(
        &mut command,
        &dir.path().join("host.sock"),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "polylogue internal host exited at start (exit status: 7)"
    );
}

#[test]
fn host_startup_returns_after_publication_and_keeps_the_child_running() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("host.sock");
    let entered = dir.path().join("child.pid");
    let starting = {
        let socket = socket.clone();
        let entered = entered.clone();
        std::thread::spawn(move || {
            let mut command = std::process::Command::new("sh");
            command
                .args([
                    "-c",
                    "printf '%s' $$ > \"$1.pending\"; mv \"$1.pending\" \"$1\"; exec sleep 60",
                    "synthetic-host",
                ])
                .arg(entered);
            start_host(
                &mut command,
                &socket,
                Instant::now() + Duration::from_secs(10),
            )
        })
    };
    wait_until("the child starting", || entered.exists());
    let _listener = crate::socket::bind(&socket).unwrap();
    let result = starting.join().unwrap();
    let pid: u32 = std::fs::read_to_string(entered).unwrap().parse().unwrap();
    let running = crate::live_events::identity(pid).is_some();
    stop_host(pid);
    assert_eq!(result.unwrap().0, pid);
    assert!(
        running,
        "socket readiness must keep the hosted process alive"
    );
}

#[test]
fn host_startup_observes_reaper_pipe_exit_without_a_pidfd() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("host.sock");
    let watch = crate::socket::Watch::new(&socket).unwrap();
    let (exited, notify_exit) = UnixStream::pair().unwrap();
    let mut child = std::process::Command::new("sh")
        .args(["-c", "exit 9"])
        .spawn()
        .unwrap();
    let reaper = std::thread::spawn(move || {
        let status = child.wait().unwrap();
        drop(notify_exit);
        status
    });
    let status = reaper.join().unwrap();
    let _listener = crate::socket::bind(&socket).unwrap();
    let outcome = wait_for_host_socket(
        &watch,
        &socket,
        &exited.into(),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert!(matches!(outcome, HostSocket::Exited));
    assert_eq!(status.code(), Some(9));
}

#[test]
fn host_socket_wait_honours_its_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("host.sock");
    let watch = crate::socket::Watch::new(&socket).unwrap();
    let (exited, running) = UnixStream::pair().unwrap();
    let (answer, answered) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        answer
            .send(wait_for_host_socket(
                &watch,
                &socket,
                &exited.into(),
                Instant::now() + Duration::from_millis(50),
            ))
            .unwrap();
    });
    let result = answered.recv_timeout(Duration::from_secs(2));
    // Release a broken deadline wait before asserting its observed result.
    drop(running);
    worker.join().unwrap();
    let error = result.unwrap().unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
}

/// The registry lines the host wrote: (event, status word).
fn status_lines(spool: &Path) -> Vec<(String, Option<String>)> {
    let Ok(entries) = std::fs::read_dir(spool.join("registry")) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        for line in std::fs::read_to_string(entry.path()).unwrap().lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            out.push((
                v["event"].as_str().unwrap().to_owned(),
                v["payload"]["status"].as_str().map(str::to_owned),
            ));
        }
    }
    out
}

struct Hosting {
    dir: tempfile::TempDir,
    config: HostConfig,
    server: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}

impl Hosting {
    fn start(harness: &str, protocol: HostProtocol, agent: &str, session: Option<&str>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let program = script(dir.path(), "agent", agent);
        let config = HostConfig {
            harness: harness.into(),
            protocol,
            argv: vec![
                program.to_string_lossy().into_owned(),
                dir.path().join("log").to_string_lossy().into_owned(),
            ],
            cwd: dir.path().to_owned(),
            socket: dir.path().join("hosts").join("key.sock"),
            spool: dir.path().join("spool"),
            session_id: session.map(str::to_owned),
        };
        let server = {
            let config = config.clone();
            std::thread::spawn(move || serve(&config))
        };
        wait_until("the host's socket", || {
            UnixStream::connect(&config.socket).is_ok()
        });
        Self {
            dir,
            config,
            server: Some(server),
        }
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.dir.path().join("log")).unwrap_or_default()
    }

    /// Ends the agent and waits for its host to exit.
    fn quit(&mut self, through: &HostedStdio) {
        through
            .state
            .0
            .locked()
            .write(&json!({"method": "polylogue_test/quit"}))
            .unwrap();
        self.server.take().unwrap().join().unwrap().unwrap();
    }
}

fn record(hosting: &Hosting) -> Record {
    Record {
        session: "gemini-cli:s1".into(),
        protocol: HostProtocol::Acp,
        socket: Some(hosting.config.socket.clone()),
        protocol_session: Some("s1".into()),
    }
}

/// A daemon connection that goes away leaves the hosted agent running: a
/// new connection reattaches within 2 s, gets the permission request the
/// first connection never answered, answers it, and the turn finishes;
/// the next prompt runs too. The agent counts every prompt: none is lost
/// and none is sent twice.
#[test]
fn a_hosted_session_survives_its_daemon_connection() {
    let mut hosting = Hosting::start("gemini-cli", HostProtocol::Acp, ACP_AGENT, None);
    let first =
        HostedStdio::connect(&hosting.config.socket, HostProtocol::Acp, String::new()).unwrap();
    first
        .call("initialize", json!({"protocolVersion": 1}))
        .unwrap();
    let created = first
        .call("session/new", json!({"cwd": "/", "mcpServers": []}))
        .unwrap();
    assert_eq!(created["sessionId"], "s1");
    first.state.0.locked().protocol_session = "s1".into();
    first.send("first turn", None).unwrap();
    wait_until("the permission request", || first.request().is_some());
    // The daemon stops before answering.
    drop(first);
    let second = HostedStdio::reattach(&record(&hosting)).unwrap();
    wait_until("the replayed request", || second.request().is_some());
    assert!(second.state.0.locked().turn, "the hello says a turn runs");
    let (sent, outcomes) = std::sync::mpsc::channel();
    second.on_sent(Arc::new(move |delivery, result| {
        // A failed test may already have dropped its observer during unwind.
        let _ = sent.send((delivery.to_owned(), result));
    }));
    // A message sent while the turn runs waits for it.
    assert!(!second.send("second turn", Some("queued-success")).unwrap());
    assert_eq!(second.state.0.locked().queued.len(), 1);
    second.answer(true, None).unwrap();
    // The first prompt finishes and the queued one is sent.
    wait_until("the second prompt", || {
        std::fs::read_to_string(hosting.dir.path().join("log")).is_ok_and(|n| n.trim() == "2")
    });
    wait_until("its permission request", || second.request().is_some());
    assert_eq!(
        outcomes.try_recv().unwrap(),
        ("queued-success".into(), Ok(()))
    );
    second.answer(true, None).unwrap();
    wait_until("the second turn's end", || !second.state.0.locked().turn);
    assert_eq!(hosting.log().trim(), "2", "every prompt ran exactly once");
    hosting.quit(&second);
    wait_until("the closed host connection", || !second.open());
    assert!(
        outcomes.try_recv().is_err(),
        "a sent prompt has no drop outcome"
    );
    // The host wrote the session's statuses, then its end, and left.
    let lines = status_lines(&hosting.config.spool);
    let words: Vec<&str> = lines.iter().filter_map(|(_, w)| w.as_deref()).collect();
    assert_eq!(
        words,
        [
            "busy", "blocked", "busy", "idle", "busy", "blocked", "busy", "idle"
        ]
    );
    assert_eq!(lines.last().unwrap().0, "HostedEnded");
    assert!(!hosting.config.socket.exists());
}

/// A host that exits during a prompt never receives the local waiting
/// prompts. Each waiting delivery gets one drop outcome; a prompt already
/// written keeps its outcome and is never resent.
#[test]
fn acp_eof_reports_unsent_queued_deliveries_once() {
    let mut hosting = Hosting::start("gemini-cli", HostProtocol::Acp, ACP_AGENT, None);
    let client =
        HostedStdio::connect(&hosting.config.socket, HostProtocol::Acp, "s1".into()).unwrap();
    let (sent, outcomes) = std::sync::mpsc::channel();
    client.on_sent(Arc::new(move |delivery, result| {
        // A failed test may already have dropped its observer during unwind.
        let _ = sent.send((delivery.to_owned(), result));
    }));
    assert!(
        client
            .send("running turn", Some("already-written"))
            .unwrap()
    );
    wait_until("the running prompt's permission", || {
        client.request().is_some()
    });
    assert!(!client.send("waiting one", Some("queued-one")).unwrap());
    assert!(!client.send("waiting two", Some("queued-two")).unwrap());
    assert!(!client.send("waiting without a delivery", None).unwrap());
    hosting.quit(&client);
    wait_until("EOF from the host", || !client.open());
    assert_eq!(
        hosting.log().trim(),
        "1",
        "only the running prompt reached the agent"
    );
    let reported: Vec<_> = outcomes.try_iter().collect();
    assert_eq!(reported.len(), 2, "only queued deliveries are reported");
    for ((delivery, result), expected) in reported.iter().zip(["queued-one", "queued-two"]) {
        assert_eq!(delivery, expected);
        assert_eq!(
            result.as_ref().unwrap_err(),
            "the host connection closed before the queued prompt was sent"
        );
    }
    assert!(client.state.0.locked().queued.is_empty());
    assert!(client.send("after EOF", Some("rejected")).is_err());
    drop(client);
    assert!(
        outcomes.try_recv().is_err(),
        "closing twice reports nothing again"
    );
}

/// Messages waiting behind a running prompt are counted as they join the
/// queue, and the count drops to zero when the connection closes and drops
/// them.
#[test]
fn the_waiting_messages_are_reported_each_time_their_number_changes() {
    let mut hosting = Hosting::start("gemini-cli", HostProtocol::Acp, ACP_AGENT, None);
    let client =
        HostedStdio::connect(&hosting.config.socket, HostProtocol::Acp, "s1".into()).unwrap();
    let (changes, counts) = std::sync::mpsc::channel();
    client.on_queue(Arc::new(move |waiting| {
        let _ = changes.send(waiting);
    }));
    assert!(client.send("running turn", None).unwrap());
    assert!(!client.send("waiting one", None).unwrap());
    assert!(!client.send("waiting two", None).unwrap());
    hosting.quit(&client);
    wait_until("EOF from the host", || !client.open());
    assert_eq!(counts.try_iter().collect::<Vec<_>>(), [1, 2, 0]);
}

/// ACP round trips: a message prompts an idle agent, an interrupt cancels the
/// running prompt, and a denial selects the reject option.
#[test]
fn acp_prompts_cancels_and_answers_through_the_protocol() {
    let mut hosting = Hosting::start("gemini-cli", HostProtocol::Acp, ACP_AGENT, None);
    let client =
        HostedStdio::connect(&hosting.config.socket, HostProtocol::Acp, "s1".into()).unwrap();
    assert!(client.interrupt().is_err(), "no prompt runs yet");
    client.send("work", None).unwrap();
    wait_until("the permission request", || client.request().is_some());
    client.interrupt().unwrap();
    wait_until("the cancelled prompt's end", || {
        !client.state.0.locked().turn
    });
    // The cancel came first, then the pending permission was answered
    // `cancelled`, and only then did the agent end the prompt.
    assert_eq!(
        std::fs::read_to_string(hosting.dir.path().join("log.cancel")).unwrap(),
        "cancel\ncancelled\n"
    );
    assert!(client.request().is_none());
    let request = json!({"id": 7, "params": {"options": [
        {"optionId": "always", "kind": "allow_always"},
        {"optionId": "never", "kind": "reject_once"}]}});
    assert_eq!(
        answer_message(HostProtocol::Acp, &request, false, None),
        json!({"jsonrpc": "2.0", "id": 7, "result": {"outcome": {"outcome": "selected", "optionId": "never"}}})
    );
    assert_eq!(
        answer_message(
            HostProtocol::Acp,
            &json!({"id": 8, "params": {"options": []}}),
            true,
            None
        ),
        json!({"jsonrpc": "2.0", "id": 8, "result": {"outcome": {"outcome": "cancelled"}}})
    );
    hosting.quit(&client);
}

/// Claude Code stream-json round trips: a message is a user message, the
/// tool permission the agent asks is answered with its input, a denial
/// carries its reason, and an interrupt is a control request; the host
/// writes busy, blocked and idle for the session it was started with.
#[test]
fn claude_stream_json_steers_answers_and_interrupts() {
    let id = "0199a1b2-3c4d-7e8f-9012-345678901234";
    let mut hosting = Hosting::start(
        "claude-code",
        HostProtocol::ClaudeStreamJson,
        CLAUDE_AGENT,
        Some(id),
    );
    let client = HostedStdio::connect(
        &hosting.config.socket,
        HostProtocol::ClaudeStreamJson,
        String::new(),
    )
    .unwrap();
    client.send("look around", None).unwrap();
    wait_until("the tool permission", || client.request().is_some());
    assert_eq!(client.request().unwrap(), "r1");
    client.answer(true, None).unwrap();
    wait_until("the approval", || hosting.log().contains("allow"));
    let approval: Value = serde_json::from_str(hosting.log().lines().next().unwrap()).unwrap();
    assert_eq!(approval["response"]["request_id"], "r1");
    assert_eq!(
        approval["response"]["response"]["updatedInput"],
        json!({"command": "ls"})
    );
    // The second message starts a second turn only after the first ended.
    wait_until("the first turn's end", || {
        status_lines(&hosting.config.spool)
            .iter()
            .any(|(_, word)| word.as_deref() == Some("idle"))
    });
    client.send("and again", None).unwrap();
    wait_until("the second permission", || client.request().is_some());
    client.answer(false, Some("not now")).unwrap();
    wait_until("the denial", || hosting.log().contains("not now"));
    client.interrupt().unwrap();
    wait_until("the interrupt", || hosting.log().contains("interrupted"));
    hosting.quit(&client);
    let lines = status_lines(&hosting.config.spool);
    let words: Vec<&str> = lines.iter().filter_map(|(_, w)| w.as_deref()).collect();
    assert_eq!(
        words,
        [
            "busy", "blocked", "busy", "idle", "busy", "blocked", "busy", "idle"
        ]
    );
    let rows: Vec<Value> = std::fs::read_dir(hosting.config.spool.join("registry"))
        .unwrap()
        .flatten()
        .flat_map(|e| {
            std::fs::read_to_string(e.path())
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect::<Vec<Value>>()
        })
        .collect();
    for row in &rows {
        assert_eq!(row["session_id"], id);
        assert_eq!(row["provider"], "claude-code");
    }
}

/// The host's status follows the protocol: a turn is busy from its prompt
/// to its end, blocked while the agent's request is unanswered, and the
/// unanswered request is what a new connection gets again.
#[test]
fn the_tracker_follows_turns_and_unanswered_requests() {
    let mut acp = Tracker::new(HostProtocol::Acp);
    assert_eq!(
        acp.client_sent(&json!({"id": "a", "method": "session/prompt"})),
        Some("busy")
    );
    let ask = json!({"id": 5, "method": "session/request_permission"});
    assert_eq!(acp.agent_sent(&ask, b"ask"), Some("blocked"));
    assert_eq!(acp.unanswered().collect::<Vec<_>>(), [b"ask".as_slice()]);
    assert_eq!(
        acp.agent_sent(&json!({"method": "session/update"}), b"u"),
        None
    );
    assert_eq!(
        acp.client_sent(&json!({"id": 5, "result": {}})),
        Some("busy")
    );
    assert_eq!(acp.unanswered().count(), 0);
    assert_eq!(
        acp.agent_sent(
            &json!({"id": "a", "result": {"stopReason": "end_turn"}}),
            b"r"
        ),
        Some("idle")
    );
    let mut claude = Tracker::new(HostProtocol::ClaudeStreamJson);
    assert_eq!(claude.client_sent(&json!({"type": "user"})), Some("busy"));
    assert_eq!(claude.client_sent(&json!({"type": "user"})), None);
    assert_eq!(
        claude.agent_sent(&json!({"type": "result"}), b"r"),
        None,
        "the second message's turn runs next"
    );
    assert_eq!(
        claude.agent_sent(&json!({"type": "result"}), b"r"),
        Some("idle")
    );
}

/// Records round-trip, and removing one takes its socket with it.
#[test]
fn records_list_every_hosted_session() {
    let dir = tempfile::tempdir().unwrap();
    let codex = Record {
        session: "codex:t1".into(),
        protocol: HostProtocol::CodexAppServer,
        socket: None,
        protocol_session: Some("t1".into()),
    };
    write_record(dir.path(), "codex-t1", &codex).unwrap();
    std::fs::write(dir.path().join("hermes-x.sock"), b"").unwrap();
    let hermes = Record {
        session: "hermes:x".into(),
        protocol: HostProtocol::Acp,
        socket: Some(dir.path().join("hermes-x.sock")),
        protocol_session: Some("x".into()),
    };
    write_record(dir.path(), "hermes-x", &hermes).unwrap();
    assert_eq!(
        records(dir.path())
            .into_iter()
            .collect::<std::io::Result<Vec<_>>>()
            .unwrap(),
        [
            (String::from("codex-t1"), codex),
            (String::from("hermes-x"), hermes)
        ]
    );
    remove(dir.path(), "hermes-x");
    assert!(!dir.path().join("hermes-x.sock").exists());
    assert_eq!(records(dir.path()).len(), 1);
    assert_eq!(
        agent_command("claude-code", Some("u")).unwrap(),
        [
            "claude",
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-hook-events",
            "--permission-prompts",
            "host",
            "--session-id",
            "u"
        ]
    );
    assert_eq!(agent_command("claude-code", None), None);
    assert_eq!(agent_command("hermes", None).unwrap(), ["hermes", "acp"]);
}

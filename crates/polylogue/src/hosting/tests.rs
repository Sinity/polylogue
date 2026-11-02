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

fn wait_until(what: &str, ok: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
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
        lock(&through.state.0)
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

/// law: pl.live.hosted-survives
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
    lock(&first.state.0).protocol_session = "s1".into();
    first.send("first turn", None).unwrap();
    wait_until("the permission request", || first.request().is_some());
    // The daemon stops before answering.
    drop(first);
    let restarted = Instant::now();
    let second = HostedStdio::reattach(&record(&hosting)).unwrap();
    wait_until("the replayed request", || second.request().is_some());
    assert!(
        restarted.elapsed() < Duration::from_secs(2),
        "reattached within 2 s"
    );
    assert!(lock(&second.state.0).turn, "the hello says a turn runs");
    // A message sent while the turn runs waits for it.
    second.send("second turn", None).unwrap();
    assert_eq!(lock(&second.state.0).queued.len(), 1);
    second.answer(true, None).unwrap();
    // The first prompt finishes and the queued one is sent.
    wait_until("the second prompt", || {
        std::fs::read_to_string(hosting.dir.path().join("log")).is_ok_and(|n| n.trim() == "2")
    });
    wait_until("its permission request", || second.request().is_some());
    second.answer(true, None).unwrap();
    wait_until("the second turn's end", || !lock(&second.state.0).turn);
    assert_eq!(hosting.log().trim(), "2", "every prompt ran exactly once");
    hosting.quit(&second);
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

/// ACP round trips: a ping prompts an idle agent, an interrupt cancels the
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
    wait_until("the cancelled prompt's end", || !lock(&client.state.0).turn);
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

/// Claude Code stream-json round trips: a ping is a user message, the
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
    let raw = std::fs::read_dir(hosting.config.spool.join("registry"))
        .unwrap()
        .flatten()
        .map(|e| std::fs::read_to_string(e.path()).unwrap())
        .collect::<String>();
    assert!(raw.contains(&format!("\"session_id\":\"{id}\"")));
    assert!(raw.contains("\"provider\":\"claude-code\""));
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
        records(dir.path()),
        [
            (String::from("codex-t1"), codex),
            (String::from("hermes-x"), hermes)
        ]
    );
    remove(dir.path(), "hermes-x");
    assert!(!dir.path().join("hermes-x.sock").exists());
    assert_eq!(records(dir.path()).len(), 1);
    assert_eq!(
        agent_command("claude-code", Some("u"))
            .unwrap()
            .last()
            .map(String::as_str),
        Some("u")
    );
    assert_eq!(agent_command("claude-code", None), None);
    assert_eq!(agent_command("hermes", None).unwrap(), ["hermes", "acp"]);
}

use super::*;

/// The headless resume runs the harness's own resume with the ping as its
/// prompt; a harness without one has none.
#[test]
fn headless_resume_uses_each_harness_own_command() {
    let claude = SessionId::parse("claude-code:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    assert_eq!(
        headless_resume(&claude, "ping text").unwrap(),
        [
            "claude",
            "--resume",
            "0199a1b2-3c4d-7e8f-9012-345678901234",
            "-p",
            "ping text"
        ]
    );
    let codex = SessionId::parse("codex:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    assert_eq!(
        headless_resume(&codex, "t").unwrap(),
        [
            "codex",
            "exec",
            "resume",
            "0199a1b2-3c4d-7e8f-9012-345678901234",
            "t"
        ]
    );
    let gemini = SessionId::parse("gemini-cli:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    assert_eq!(headless_resume(&gemini, "t"), None);
}

/// One resumed run per session at a time: a second waits behind the first
/// and starts when it finishes.
#[test]
fn resumed_runs_of_one_session_queue_in_order() {
    let (reports, received) = std::sync::mpsc::channel();
    let acts = Acts::new(
        PathBuf::from("/nonexistent"),
        PathBuf::from("/nonexistent"),
        PathBuf::new(),
        PathBuf::new(),
        PathBuf::new(),
        reports,
    );
    drop(received);
    let session = SessionId::parse("codex:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    let run = |n: u8| ResumeRun {
        session: session.clone(),
        delivery: format!("d{n}"),
        text: String::new(),
    };
    // With no acts worker nothing runs, and the session is free again.
    assert!(acts.resume(run(1)));
    assert_eq!(acts.resume_finished(&session), None);
    acts.resuming
        .lock()
        .unwrap()
        .insert(session.to_string(), VecDeque::new());
    assert!(!acts.resume(run(2)));
    assert!(!acts.resume(run(3)));
    assert_eq!(
        acts.resume_finished(&session).map(|r| r.delivery),
        Some("d2".into())
    );
    assert_eq!(
        acts.resume_finished(&session).map(|r| r.delivery),
        Some("d3".into())
    );
    assert_eq!(acts.resume_finished(&session), None);
    assert!(acts.resume(run(4)));
}

/// Answers to a hosted Codex thread take the shape each request asks for.
#[test]
fn hosted_answers_take_each_request_shape() {
    assert_eq!(
        answer_result(
            "item/commandExecution/requestApproval",
            &Value::Null,
            true,
            None
        )
        .unwrap(),
        json!({"decision": "accept"})
    );
    assert_eq!(
        answer_result("item/fileChange/requestApproval", &Value::Null, false, None).unwrap(),
        json!({"decision": "decline"})
    );
    let questions = json!({"questions": [{"id": "q1"}, {"id": "q2"}]});
    assert_eq!(
        answer_result(
            "item/tool/requestUserInput",
            &questions,
            true,
            Some("use main")
        )
        .unwrap(),
        json!({"answers": {"q1": {"answers": ["use main"]}, "q2": {"answers": ["use main"]}}})
    );
    assert!(answer_result("item/permissions/requestApproval", &Value::Null, true, None).is_err());
}

/// A fixture Codex app-server behind `codex app-server proxy`: each
/// `thread/start` makes thread `t<N>`; a `turn/start` whose text says
/// `block` is held unanswered until the file `release` exists, and marks
/// `blocked` meanwhile.
fn blocking_codex(dir: &std::path::Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let program = dir.join("codex");
    std::fs::write(
        &program,
        r#"#!/bin/sh
dir=$(dirname "$0")
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  [ -z "$id" ] && continue
  case "$line" in
    *'"thread/start"'*)
      n=$(cat "$dir/threads" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$dir/threads"
      printf '{"id":%s,"result":{"thread":{"id":"t%s"}}}\n' "$id" "$n" ;;
    *'"turn/start"'*'block'*)
      touch "$dir/blocked"
      while [ ! -e "$dir/release" ]; do sleep 0.01; done
      printf '{"id":%s,"result":{}}\n' "$id" ;;
    *) printf '{"id":%s,"result":{}}\n' "$id" ;;
  esac
done
"#,
    )
    .unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    program
}

/// An act waiting on one hosted session's adapter holds only that
/// session: another session's act completes meanwhile.
#[test]
fn one_session_adapter_block_does_not_delay_another() {
    let fixture = tempfile::tempdir().unwrap();
    let program = blocking_codex(fixture.path());
    let socket = fixture.path().join("app-server-control.sock");
    let hosts = fixture.path().join("hosts");
    std::fs::create_dir_all(&hosts).unwrap();
    let (reports, _received) = std::sync::mpsc::channel();
    let acts = Arc::new(Acts::new(
        fixture.path().join("mail"),
        hosts,
        fixture.path().join("spool"),
        socket.clone(),
        program.clone(),
        reports,
    ));
    let start = |text: &str| {
        let hosted = HostedCodex::start(&program, &socket, "/home/op/work", text).unwrap();
        acts.host_codex(hosted).unwrap()
    };
    let (blocked, free) = (start("first turn"), start("first turn"));
    // The first turns finished: further sends start new turns.
    let waiting = {
        let acts = Arc::clone(&acts);
        let blocked = blocked.clone();
        std::thread::spawn(move || acts.host_send(&blocked, "block here", None))
    };
    let marked = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !fixture.path().join("blocked").exists() {
        assert!(
            std::time::Instant::now() < marked,
            "the adapter never blocked"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let (sent, outcome) = std::sync::mpsc::channel();
    {
        let acts = Arc::clone(&acts);
        std::thread::spawn(move || sent.send(acts.host_send(&free, "hello", None)));
    }
    let free_outcome = outcome
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the other session's act waited for the blocked adapter");
    assert_eq!(free_outcome, Ok(true));
    // With one lock for every session the free act would have waited for
    // the blocked one to give up (its adapter answer limit).
    assert!(!waiting.is_finished(), "the blocked act was still waiting");
    std::fs::write(fixture.path().join("release"), "").unwrap();
    let _ = waiting.join().unwrap();
}

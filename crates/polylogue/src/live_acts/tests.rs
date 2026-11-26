use super::*;

/// The headless resume runs the harness's own resume with the message as its
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
    let session = SessionId::parse("codex:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    let run = |n: u8| ResumeRun {
        session: session.clone(),
        delivery: format!("d{n}"),
        argv: Vec::new(),
        cwd: PathBuf::new(),
    };
    let reported = || match received.try_recv() {
        Ok(Report::Resume(run, _)) => Some(run.delivery),
        Ok(_) => panic!("only resumes are reported"),
        Err(_) => None,
    };
    assert!(acts.resume(run(1)).unwrap().is_some());
    assert_eq!(reported(), Some("d1".into()));
    assert!(acts.resume(run(2)).unwrap().is_none());
    assert!(acts.resume(run(3)).unwrap().is_none());
    assert_eq!(reported(), None, "queued runs wait for the running one");
    assert_eq!(
        acts.resume_finished(&session).map(|r| r.delivery),
        Some("d2".into())
    );
    assert_eq!(
        acts.resume_finished(&session).map(|r| r.delivery),
        Some("d3".into())
    );
    assert_eq!(acts.resume_finished(&session), None);
    assert!(acts.resume(run(4)).unwrap().is_some());
    assert_eq!(reported(), Some("d4".into()));
}

/// With no acts worker nothing runs, and the session is free again.
#[test]
fn a_resume_nobody_can_run_leaves_the_session_free() {
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
        argv: Vec::new(),
        cwd: PathBuf::new(),
    };
    assert!(acts.resume(run(1)).is_err());
    assert_eq!(acts.resume_finished(&session), None);
    assert!(acts.resume(run(2)).is_err());
    assert_eq!(acts.resume_finished(&session), None);
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
    assert_eq!(
        answer_result(
            "item/tool/requestUserInput",
            &json!({"questions": [{"prompt": "no id"}]}),
            true,
            None
        )
        .unwrap_err()
        .to_string(),
        "the user-input request names no question to answer"
    );
    assert_eq!(
        answer_result("item/permissions/requestApproval", &Value::Null, true, None)
            .unwrap_err()
            .to_string(),
        "unsupported: Polylogue does not answer item/permissions/requestApproval requests"
    );
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
    assert_eq!(waiting.join().unwrap(), Ok(true));
}

/// A delivery is released before the flag is raised, so a failed raise still
/// leaves the delivery for the next claim.
#[test]
fn a_delivery_is_released_even_when_the_flag_cannot_be_raised() {
    let (reports, _received) = std::sync::mpsc::channel();
    let acts = Acts::new(
        PathBuf::from("/nonexistent"),
        PathBuf::new(),
        PathBuf::new(),
        PathBuf::new(),
        PathBuf::new(),
        reports,
    );
    let session = SessionId::parse("claude-code:s1").unwrap();
    acts.take(vec![("ping".to_owned(), ())]);
    assert!(acts.handed_out("ping"));
    assert!(acts.hand_to_hook("ping", &session).is_err());
    assert!(!acts.handed_out("ping"));
}

/// A Codex answer that the app-server could not be sent leaves the request
/// pending, so the same answer can be given again.
#[test]
fn a_failed_codex_answer_keeps_the_request_pending() {
    let dir = tempfile::tempdir().unwrap();
    let program = blocking_codex(dir.path());
    let mut hosted =
        HostedCodex::start(&program, &dir.path().join("socket"), "/home/op", "hello").unwrap();
    hosted.requests.push_back((
        json!(7),
        "item/permissions/requestApproval".into(),
        Value::Null,
    ));
    hosted.requests.push_back((
        json!(8),
        "item/commandExecution/requestApproval".into(),
        Value::Null,
    ));
    // The first request is a kind Polylogue does not answer: nothing is
    // sent and both requests remain, oldest first.
    assert!(hosted.answer(true, None).is_err());
    assert_eq!(hosted.requests.len(), 2);
    assert_eq!(hosted.requests.front().unwrap().0, json!(7));
    // A closed connection fails the send and keeps the request as well.
    hosted.requests.pop_front();
    let _ = hosted.connection.child.kill();
    let _ = hosted.connection.child.wait();
    assert!(hosted.answer(true, None).is_err());
    assert_eq!(hosted.requests.len(), 1);
    assert!(!hosted.connection.is_running());
}

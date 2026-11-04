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
    let acts = Acts::new(
        PathBuf::from("/nonexistent"),
        PathBuf::from("/nonexistent"),
        PathBuf::new(),
        PathBuf::new(),
        PathBuf::new(),
    );
    let session = SessionId::parse("codex:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    let run = |n: u8| ResumeRun {
        session: session.clone(),
        delivery: format!("d{n}"),
        text: String::new(),
        cwd: None,
    };
    // With no daemon bound nothing runs, and the session is free again.
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

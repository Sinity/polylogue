use super::*;

#[test]
fn native_resume_uses_each_harness_command() {
    let s = SessionId::parse("claude-code:7f3e").unwrap();
    let c = harness_resume(&s, Some("/workspace/project x/repo")).unwrap();
    assert_eq!(
        (c.program.as_str(), c.args.as_slice(), c.cwd.as_deref()),
        (
            "claude",
            &["--resume".to_owned(), "7f3e".to_owned()][..],
            Some("/workspace/project x/repo")
        )
    );
    let c = harness_resume(&SessionId::parse("codex:0199").unwrap(), None).unwrap();
    assert_eq!(
        (c.program.as_str(), c.args.as_slice(), c.cwd.as_deref()),
        ("codex", &["resume".to_owned(), "0199".to_owned()][..], None)
    );
    assert!(harness_resume(&SessionId::parse("claude-code:agent-a1").unwrap(), None).is_none());
    assert!(harness_resume(&SessionId::parse("chatgpt:abc").unwrap(), None).is_none());
}

#[test]
fn resume_uses_launch_directory_not_last_used() {
    let home = polylogue_testkit::ClaudeHome::new();
    let record = |uuid: &str, cwd: &str| {
        format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": uuid, "sessionId": "s1", "cwd": cwd,
                "timestamp": "2026-03-01T10:00:00Z",
                "message": {"role": "user", "content": uuid}
            })
        )
    };
    home.append(
        "s1.jsonl",
        &(record("u1", "/home/op/launch") + &record("u2", "/home/op/launch/crates/sub")),
    );
    let session = SessionId::parse("claude-code:s1").unwrap();
    let directory = |home: &polylogue_testkit::ClaudeHome| {
        let (mut engine, _lock) = home.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        let reads = home.reads();
        let reader = reads.reader().unwrap();
        assert_eq!(
            reader
                .session(&session)
                .unwrap()
                .unwrap()
                .cwd
                .last()
                .map(String::as_str),
            Some("/home/op/launch/crates/sub"),
            "the last directory used is still recorded"
        );
        let plan = plan(&reader, &session).unwrap();
        plan.harness_command.unwrap().cwd
    };
    assert_eq!(directory(&home).as_deref(), Some("/home/op/launch"));
    // A relocation by the tool moves where the session resumes.
    home.append(
        "s1.jsonl",
        &format!(
            "{}\n{}",
            serde_json::json!({"type": "relocated", "sessionId": "s1",
                               "relocatedCwd": "/home/op/moved",
                               "timestamp": "2026-03-01T10:01:00Z"}),
            record("u3", "/home/op/moved/crates/sub")
        ),
    );
    assert_eq!(directory(&home).as_deref(), Some("/home/op/moved"));
}

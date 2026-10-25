use super::*;

#[test]
fn native_resume_uses_each_harness_command() {
    let s = SessionId::parse("claude-code:7f3e").unwrap();
    let c = harness_resume(&s, Some("/realm/project/x repo")).unwrap();
    assert_eq!(
        (c.program.as_str(), c.args.as_slice(), c.cwd.as_deref()),
        (
            "claude",
            &["--resume".to_owned(), "7f3e".to_owned()][..],
            Some("/realm/project/x repo")
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

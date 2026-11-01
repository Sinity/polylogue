use super::*;

#[test]
fn side_files_link_only_explicit_session_ids_or_complete_path_ids() {
    for (path, bytes, expected) in [
        (
            "/h/.claude/teams/example/config.json",
            br#"{"leadSessionId":"s1"}"#.as_slice(),
            Some("claude-code:s1"),
        ),
        (
            "/h/.claude/usage-data/facets/summary.json",
            br#"{"session_id":"s2"}"#.as_slice(),
            Some("claude-code:s2"),
        ),
        (
            "/h/.claude/history-summaries/short-name.md",
            b"# summary".as_slice(),
            None,
        ),
        (
            "/h/.claude/paste-cache/abcd.txt",
            b"pasted text".as_slice(),
            None,
        ),
    ] {
        let parser = open_claude(Path::new(path)).unwrap();
        let (steps, _) = parser.steps(None, bytes, 0).unwrap();
        assert_eq!(
            steps[0].output.events[0]
                .session
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            expected
        );
        assert_eq!(
            steps[0].output.events[0].range,
            Some((0, bytes.len() as u64))
        );
    }
    let parser = open_codex(Path::new(
        "/h/.codex/attachments/thread-one/pasted-text-1.txt",
    ))
    .unwrap();
    let (steps, _) = parser.steps(None, b"prompt context", 0).unwrap();
    assert_eq!(
        steps[0].output.events[0]
            .session
            .as_ref()
            .unwrap()
            .to_string(),
        "codex:thread-one"
    );
}

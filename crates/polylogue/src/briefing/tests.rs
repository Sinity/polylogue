use super::*;

#[test]
fn a_checkout_root_joins_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("proj/repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("crates/x")).unwrap();
    let dirs = scope_dirs(&repo.join("crates/x"));
    assert_eq!(dirs.len(), 2);
    assert_eq!(dirs[1], repo.to_string_lossy());
}

#[test]
fn start_sources_parse_from_harness_names() {
    assert_eq!(StartSource::parse("compact"), StartSource::Compact);
    assert_eq!(StartSource::parse("anything"), StartSource::Startup);
    assert_eq!(StartSource::parse("resume"), StartSource::Resume);
}

#[test]
fn recall_finds_prose_then_tool_output_as_a_query_sees_them() {
    use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:s1",
            title: Some("quokka work"),
            ..SessionFacts::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "m1",
            role: "user",
            written_by: "human",
            text: Some("the quokka migration plan"),
            ..MessageFacts::default()
        },
    )
    .unwrap();
    let m2 = f
        .message(
            s,
            MessageFacts {
                id: "m2",
                role: "assistant",
                written_by: "model",
                ..MessageFacts::default()
            },
        )
        .unwrap();
    f.tool_call(
        s,
        m2,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("ls"),
            outcome: "ok",
            output: Some("quokka.txt"),
            ..CallFacts::default()
        },
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let ask = Ask::Recall {
        text: "Quokka!".into(),
        cwd: None,
        for_session: None,
    };
    let p = prepare(&reader, &ask, "d1", Timestamp::from_unix_ms(0).unwrap()).unwrap();
    assert_eq!(
        p.refs,
        ["session:test:s1", "message:m1", "message:m2"],
        "{}",
        p.text
    );
    assert!(
        p.text.contains("(prose) the quokka migration plan"),
        "{}",
        p.text
    );
    assert!(p.text.contains("(output) quokka.txt"), "{}", p.text);
    let search_summary = p.search_summary.unwrap();
    assert_eq!(
        (
            search_summary.hits,
            search_summary.sessions,
            search_summary.more
        ),
        (2, 1, false)
    );
}

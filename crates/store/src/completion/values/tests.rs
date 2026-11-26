use super::*;
use crate::fixture::{Fixture, SessionFacts};

fn origins() -> Vec<String> {
    vec!["codex".into(), "claude-code".into()]
}

#[test]
fn sessions_match_full_and_native_id_prefixes_with_their_shortest_unique_prefix() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    for (id, last) in [
        ("codex:abc123", Some(300)),
        ("codex:abd456", Some(200)),
        ("claude-code:abx", None),
        ("codex:zzz", Some(100)),
    ] {
        fixture
            .session(SessionFacts {
                id,
                last,
                ..Default::default()
            })
            .unwrap();
    }
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let found = |prefix: &str| {
        reader
            .completion_sessions(prefix, &origins())
            .unwrap()
            .into_iter()
            .map(|session| (session.id, session.prefix))
            .collect::<Vec<_>>()
    };
    // Most recent first; sessions without activity last.
    assert_eq!(
        found("ab"),
        [
            ("codex:abc123".into(), "codex:abc".into()),
            ("codex:abd456".into(), "codex:abd".into()),
            ("claude-code:abx".into(), "claude-code:a".into()),
        ]
    );
    assert_eq!(found("codex:z"), [("codex:zzz".into(), "codex:z".into())]);
    assert!(found("codex:q").is_empty());
}

#[test]
fn a_prefix_matching_no_id_matches_titles_in_order() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "codex:one",
            title: Some("Back off on 429"),
            ..Default::default()
        })
        .unwrap();
    fixture
        .session(SessionFacts {
            id: "codex:two",
            title: Some("Unrelated"),
            ..Default::default()
        })
        .unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let titles: Vec<_> = reader
        .completion_sessions("bkoff", &origins())
        .unwrap()
        .into_iter()
        .map(|session| session.title)
        .collect();
    assert_eq!(titles, ["Back off on 429"]);
}

#[test]
fn values_count_sessions_and_current_tags() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    for (id, repo, cwd) in [
        ("codex:a", "sinex", "/home/op/sinex"),
        ("codex:b", "sinex", "/home/op/sinex/crates"),
        ("codex:c", "polylogue", "/home/op/polylogue"),
    ] {
        fixture
            .session(SessionFacts {
                id,
                repo: Some(repo),
                cwd: Some(cwd),
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .note("tag", Some("rate-limits"), None, "session:codex:a")
        .unwrap();
    fixture
        .note("tag", Some("rate-limits"), None, "session:codex:b")
        .unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let values = |field: &str, prefix: &str| {
        reader
            .completion_values(field, prefix)
            .unwrap()
            .into_iter()
            .map(|value| (value.value, value.count))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        values("repo", ""),
        [("sinex".into(), 2), ("polylogue".into(), 1)]
    );
    assert_eq!(
        values("cwd", "/home/op/s"),
        [
            ("/home/op/sinex".into(), 2),
            ("/home/op/sinex/crates".into(), 1)
        ]
    );
    assert_eq!(values("tag", "rate"), [("rate-limits".into(), 2)]);
    assert!(values("model", "").is_empty());
}

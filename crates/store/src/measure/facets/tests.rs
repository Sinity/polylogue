use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

#[test]
fn distinct_sessions_global_idf_and_optional_families_share_one_reader() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let a = f
        .session(SessionFacts {
            id: "codex:a",
            repo: Some("https://example.test/team/one.git"),
            ..Default::default()
        })
        .unwrap();
    let b = f
        .session(SessionFacts {
            id: "codex:b",
            repo: Some("alternate repository label"),
            ..Default::default()
        })
        .unwrap();
    let c = f
        .session(SessionFacts {
            id: "claude-code:c",
            ..Default::default()
        })
        .unwrap();
    for pk in [a, b, c] {
        f.note(
            "tag",
            None,
            Some("common"),
            &format!(
                "session:{}",
                if pk == a {
                    "codex:a"
                } else if pk == b {
                    "codex:b"
                } else {
                    "claude-code:c"
                }
            ),
        )
        .unwrap();
    }
    f.note("tag", None, Some("Rare"), "session:codex:a")
        .unwrap();
    f.note("tag", None, Some("rare"), "session:codex:a")
        .unwrap();
    for i in 0..4 {
        let m = f
            .message(
                a,
                MessageFacts {
                    id: &format!("m{i}"),
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        f.tool_call(
            a,
            m,
            CallFacts {
                tool: "Read",
                action: "file_read",
                outcome: "unknown",
                ..Default::default()
            },
        )
        .unwrap();
    }
    f.derive().unwrap();
    // This fixture supplies the already-derived repository identity.
    f.live(a, "hook_stop", 0, None, Some("root-one")).unwrap();
    f.live(b, "hook_stop", 0, None, Some("root-one")).unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selected = Selection {
        sessions: Some(vec![a]),
        ..Default::default()
    };
    let result = facets(&reader, &selected, "deep").unwrap();
    assert_eq!(result.selected.total_sessions, 1);
    assert_eq!(result.global.total_sessions, 3);
    assert_eq!(result.selected.families["tags"]["rare"], 1);
    assert_eq!(result.selected.families["roles"]["assistant"], 1);
    assert_eq!(result.selected.families["actions"]["file_read"], 1);
    assert_eq!(result.global.families["repositories"]["root-one"], 2);
    assert_eq!(result.idf["tags"]["common"], 0.0);
    assert!((result.idf["tags"]["rare"] - 3.0_f64.ln()).abs() < 1e-12);
    assert!(result.idf["tags"]["rare"] > result.idf["origins"]["codex"]);
    let basic = facets(&reader, &Selection::default(), "no-idf").unwrap();
    assert!(basic.idf.is_empty());
    assert!(basic.selected.omitted.contains(&"actions"));
    assert!(!basic.selected.families.contains_key("omitted"));
    // The reader keeps the selected/global population even after another commit.
    f.session(SessionFacts {
        id: "codex:later",
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        facets(&reader, &selected, "basic")
            .unwrap()
            .global
            .total_sessions,
        3
    );
}

#[test]
fn empty_selection_and_empty_archive_are_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        let x = facets(&r, &Selection::default(), "").unwrap();
        assert_eq!(x.global.total_sessions, 0);
        assert!(x.idf.is_empty());
    }
    f.session(SessionFacts {
        id: "codex:a",
        ..Default::default()
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = facets(
        &r,
        &Selection {
            sessions: Some(vec![]),
            ..Default::default()
        },
        "",
    )
    .unwrap();
    assert_eq!(x.selected.total_sessions, 0);
    assert_eq!(x.global.total_sessions, 1);
}

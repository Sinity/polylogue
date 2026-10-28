use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

fn all(reader: &Reader<'_>, selection: &str, options: &str) -> Vec<Value> {
    reader
        .query(
            &query(
                selection,
                "SELECT pk AS _pk FROM session WHERE kind='main' AND stub=0",
                options,
                "UTC",
                0,
            )
            .unwrap(),
            &[],
        )
        .unwrap()
}
fn find<'a>(rows: &'a [Value], family: &str, value: Option<&str>) -> &'a Value {
    rows.iter()
        .find(|r| r["family"] == family && r["value"].as_str() == value)
        .unwrap()
}

#[test]
fn facets_group_distinct_sessions_and_compute_idf_and_lift_at_read() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, subagent) in [
        ("codex:a", false),
        ("codex:b", false),
        ("claude-code:c", false),
        ("codex:child", true),
    ] {
        let pk = f
            .session(SessionFacts {
                id,
                kind: if subagent { "subagent" } else { "main" },
                ..Default::default()
            })
            .unwrap();
        for n in 0..3 {
            let m = f
                .message(
                    pk,
                    MessageFacts {
                        id: &format!("{id}-{n}"),
                        role: "assistant",
                        written_by: "model",
                        model: Some("synthetic-model"),
                        at: Some(0),
                        ..Default::default()
                    },
                )
                .unwrap();
            f.tool_call(
                pk,
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
        f.note("tag", None, Some("common"), &format!("session:{id}"))
            .unwrap();
    }
    f.note("tag", None, Some("Rare"), "session:codex:a")
        .unwrap();
    f.note("tag", None, Some("rare"), "session:codex:a")
        .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut rows = all(
        &reader,
        "SELECT pk AS _pk FROM session WHERE id='codex:a'",
        "deep",
    );
    let summary = Summary {
        outcome: "ok",
        selected_sessions: 1,
        global_sessions: 3,
        pending: 0,
    };
    weights(&mut rows, &summary);
    assert_eq!(find(&rows, "tags", Some("rare"))["selected_count"], 1);
    assert_eq!(find(&rows, "tags", Some("rare"))["global_count"], 1);
    assert_eq!(find(&rows, "roles", Some("assistant"))["selected_count"], 1);
    assert_eq!(find(&rows, "actions", Some("file_read"))["global_count"], 3);
    assert_eq!(find(&rows, "tags", Some("common"))["idf"], 0.0);
    assert!(
        (find(&rows, "tags", Some("rare"))["idf"].as_f64().unwrap() - 3.0_f64.ln()).abs() < 1e-12
    );
    assert_eq!(find(&rows, "tags", Some("rare"))["lift"], 3.0);
    assert_eq!(
        find(&rows, "models", Some("synthetic-model"))["global_count"],
        3
    );
}

#[test]
fn empty_selection_keeps_global_counts_and_unknowns() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:empty",
            ..Default::default()
        })
        .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let scope = Selection {
        sessions: Some(vec![]),
        population: Some(vec![s]),
        ..Default::default()
    };
    let summary = summary(&reader, &scope).unwrap();
    assert_eq!(summary.outcome, "empty");
    let mut rows = all(&reader, "SELECT pk AS _pk FROM session WHERE 0", "basic");
    weights(&mut rows, &summary);
    assert_eq!(find(&rows, "origins", Some("codex"))["selected_count"], 0);
    assert_eq!(find(&rows, "repositories", None)["global_count"], 1);
    assert!(rows.iter().all(|r| r["lift"].is_null()));
    assert!(
        query(
            "SELECT 1 AS _pk",
            "SELECT 1 AS _pk",
            "unsupported",
            "UTC",
            0
        )
        .is_err()
    );
}

#[test]
fn withdrawn_and_revised_tags_change_counts_without_derivation() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.session(SessionFacts {
        id: "codex:tag",
        ..Default::default()
    })
    .unwrap();
    let note = f
        .note("tag", None, Some("old"), "session:codex:tag")
        .unwrap();
    let conn = crate::archive::open_write(&f.archive.archive_db(), 64).unwrap();
    conn.execute("INSERT INTO note(id,first_note,kind,author,machine,value,created_at,commit_seq) SELECT 'revision',id,kind,author,machine,'new',created_at+1,commit_seq+1 FROM note WHERE id=?1",[&note]).unwrap();
    conn.execute(
        "INSERT INTO note_target(note_id,ref) VALUES('revision','session:codex:tag')",
        [],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let rows = all(&reader, "SELECT pk AS _pk FROM session", "tags");
    assert!(rows.iter().all(|r| r["value"] != "old"));
    assert_eq!(find(&rows, "tags", Some("new"))["selected_count"], 1);
    drop(reader);
    drop(pool);
    conn.execute(
        "INSERT INTO note_relation(note_id,kind,other) VALUES('revision','withdraws',?1)",
        [format!("note:{note}")],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert!(all(&reader, "SELECT pk AS _pk FROM session", "tags").is_empty());
}

use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

fn all(reader: &Reader<'_>, selection: &str, options: &str) -> Vec<Value> {
    let clock = query_language::time::Clock::new(
        "2026-10-07T10:00:00Z".parse().unwrap(),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::midnight(),
        )
        .unwrap(),
    );
    reader.with_clock(&clock, || {
        reader
            .query(
                &query(
                    selection,
                    "SELECT pk AS _pk FROM session WHERE kind='main' AND stub=0",
                    options,
                )
                .unwrap(),
                &[],
            )
            .unwrap()
    })
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
        f.note("tag", Some("common"), None, &format!("session:{id}"))
            .unwrap();
    }
    f.note("tag", Some("Rare"), None, "session:codex:a")
        .unwrap();
    f.note("tag", Some("rare"), None, "session:codex:a")
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
    assert!(query("SELECT 1 AS _pk", "SELECT 1 AS _pk", "unsupported").is_err());
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
        .note("tag", Some("old"), None, "session:codex:tag")
        .unwrap();
    use polylogue_model::{
        Reference, RevisionReference, Timestamp,
        assertion::{AssertionKind, AssertionLink, LinkRole},
    };
    let mut conn = crate::archive::open_write(&f.archive.archive_db(), 64).unwrap();
    let mut revision = crate::assertions::get(&conn, &note).unwrap().unwrap().fact;
    revision.id = "revision".into();
    revision.key = Some("new".into());
    revision.created_at = Timestamp::from_unix_ms(2);
    revision.links.push(AssertionLink {
        role: LinkRole::Revises,
        reference: Reference::Assertion(RevisionReference {
            root: note.clone(),
            revision: Some(note.clone()),
        }),
        seen_hash: None,
        quote: None,
    });
    let tx = conn.transaction().unwrap();
    crate::assertions::apply(&tx, std::slice::from_ref(&revision), 2).unwrap();
    tx.commit().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let rows = all(&reader, "SELECT pk AS _pk FROM session", "tags");
    assert!(rows.iter().all(|r| r["value"] != "old"));
    assert_eq!(find(&rows, "tags", Some("new"))["selected_count"], 1);
    drop(reader);
    drop(pool);
    let mut withdrawal = revision;
    withdrawal.id = "withdrawal".into();
    withdrawal.root = "withdrawal".into();
    withdrawal.kind = AssertionKind::Withdrawal;
    withdrawal.key = None;
    withdrawal.value = None;
    withdrawal.created_at = Timestamp::from_unix_ms(3);
    withdrawal.links = vec![AssertionLink {
        role: LinkRole::Withdraws,
        reference: Reference::Assertion(RevisionReference {
            root: note,
            revision: None,
        }),
        seen_hash: None,
        quote: None,
    }];
    let tx = conn.transaction().unwrap();
    crate::assertions::apply(&tx, &[withdrawal], 3).unwrap();
    tx.commit().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert!(all(&reader, "SELECT pk AS _pk FROM session", "tags").is_empty());
}

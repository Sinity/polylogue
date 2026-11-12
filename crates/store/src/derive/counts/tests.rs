use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

#[test]
fn stored_counts_match_the_shown_path_and_action_population() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let pk = f
        .session(SessionFacts {
            id: "codex:counts",
            ..Default::default()
        })
        .unwrap();
    for (id, by, kind, text, path) in [
        ("human", "human", Some("prose"), "one two", None),
        ("command", "human", Some("command"), "/help", None),
        ("model", "model", None, "three four five", None),
        ("harness", "harness", None, "not authored prose", None),
        ("off-path", "model", None, "off path", Some(false)),
        ("duplicate", "model", None, "another copy", None),
    ] {
        let m = f
            .message(
                pk,
                MessageFacts {
                    id,
                    role: if by == "model" { "assistant" } else { "user" },
                    written_by: by,
                    prompt_kind: kind,
                    text: Some(text),
                    on_active_path: path,
                    thinking: (id == "model").then_some("thinking"),
                    ..Default::default()
                },
            )
            .unwrap();
        if id == "model" {
            for (tool, action) in [("exec", "code"), ("shell", "shell"), ("unknown", "")] {
                f.tool_call(
                    pk,
                    m,
                    CallFacts {
                        tool,
                        action,
                        outcome: "error",
                        ..Default::default()
                    },
                )
                .unwrap();
            }
        }
    }
    let conn = crate::archive::open_write(&f.archive.index_db(), 64).unwrap();
    conn.execute(
        "UPDATE message SET duplicate_of='model' WHERE id='duplicate'",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE tool_call SET run_by_script=1 WHERE tool='shell'",
        [],
    )
    .unwrap();
    let c = compute(&conn, pk).unwrap();
    assert_eq!(
        (
            c.messages,
            c.human_prompts,
            c.human_commands,
            c.words,
            c.human_words,
            c.model_words,
            c.thinking,
            c.tool_calls,
            c.errors
        ),
        (4, 1, 1, 6, 3, 3, 1, 2, 2)
    );
    let tx = conn.unchecked_transaction().unwrap();
    apply(&tx, pk, &compute(&tx, pk).unwrap()).unwrap();
    tx.commit().unwrap();
    let stored = read(&conn, pk, true).unwrap();
    assert_eq!(
        serde_json::to_value(c).unwrap(),
        serde_json::to_value(stored).unwrap()
    );
    crate::sql_functions::register(&conn).unwrap();
    for (_, column) in crate::session_sql::COUNT_FIELDS {
        let computed: Option<i64> = conn
            .query_row(
                &format!(
                    "SELECT {} FROM session s WHERE pk=?1",
                    crate::session_sql::compute_count("s", column)
                ),
                [pk],
                |r| r.get(0),
            )
            .unwrap();
        let stored: Option<i64> = conn
            .query_row(
                &format!("SELECT {column} FROM session WHERE pk=?1"),
                [pk],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, computed, "{column}");
    }
}

/// The stored counts of `session` and the full recompute over its rows.
fn stored_and_recomputed(w: &crate::fixture::writer::WriterArchive, session: &str) -> (Json, Json) {
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let conn = reader.conn().unwrap();
    let pk: i64 = conn
        .query_row(
            "SELECT pk FROM session WHERE id = ?1",
            [format!("claude-code:{session}")],
            |r| r.get(0),
        )
        .unwrap();
    let stored = serde_json::to_value(read(conn, pk, true).unwrap()).unwrap();
    let recomputed = serde_json::to_value(compute(conn, pk).unwrap()).unwrap();
    (stored, recomputed)
}

use polylogue_model::Json;

#[test]
fn fold_after_append_equals_full_recompute() {
    use crate::fixture::writer::{WriterArchive, bash, message, prose, sid};
    use polylogue_model::{Block, Event, Output, Role, Session, SessionKind, TimeQuality};
    let mut w = WriterArchive::new();
    let mut seen = Vec::new();
    let mut check = |w: &WriterArchive, stage: &str| {
        let (stored, recomputed) = stored_and_recomputed(w, "s");
        assert_eq!(stored, recomputed, "{stage}");
        seen.push(stored);
    };
    let answer = |n: u32, at: i64, text: &str| {
        message(
            &format!("a{n}"),
            "s",
            Role::Assistant,
            at,
            vec![
                Block::Thinking {
                    text: Some("considering".into()),
                    signature: None,
                    redacted: false,
                },
                Block::Text { text: text.into() },
            ],
        )
    };
    // Appends: a prompt, an answer with thinking, a call without its result.
    w.write("s", prose("s", 1, 1_000, "one two three"));
    check(&w, "first prompt");
    w.write("s", vec![answer(1, 2_000, "four five")]);
    check(&w, "append");
    let mut call = bash("s", 1, 3_000, "cargo test", false, "exit 101");
    let result = call.pop().unwrap();
    w.write("s", call);
    check(&w, "call without result");
    // A late result: the call's outcome changes in a later transaction.
    w.write("s", vec![result]);
    check(&w, "late result");
    // A compaction and a new active-path marker: every message but the
    // marked leaf leaves the path in the derive step.
    let event = |kind: &str, after: Option<&str>| Event {
        session: Some(sid("s")),
        kind: kind.into(),
        at: None,
        at_quality: TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: after.map(|id| polylogue_model::MessageId::parse(id).unwrap()),
        range: None,
        data: if kind == "compaction" {
            serde_json::to_value(polylogue_model::CompactionBoundary::default()).unwrap()
        } else {
            Json::Null
        },
    };
    let offset = w.offset();
    w.write_output(
        "input-s",
        "claude-code",
        "/p/s",
        Output {
            sessions: vec![Session::new(sid("s"), SessionKind::Main)],
            events: vec![
                event("compaction", None),
                event("active_leaf", Some("claude-code:s-a1")),
            ],
            ..Output::default()
        },
        2,
    );
    assert!(w.offset() > offset);
    w.writer.derive().unwrap();
    check(&w, "compaction and new marker");
    // A rewrite: a snapshot replaces the answer's text.
    w.write_snapshot("s", vec![answer(1, 2_000, "a longer replacement answer")]);
    check(&w, "rewrite");
    // A new input states the same session's prompt again.
    w.write_from(
        "input-copy",
        Session::new(sid("s"), SessionKind::Main),
        prose("s", 1, 1_000, "one two three"),
    );
    check(&w, "new input");
    w.write_from(
        "input-copy",
        Session::new(sid("s"), SessionKind::Main),
        prose("s", 9, 9_000, "a prompt only the copy has"),
    );
    check(&w, "append to the new input");
    // The stages changed the counts, so agreement is not of unchanged values.
    seen.dedup();
    assert!(seen.len() >= 6, "{seen:?}");
}

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
    run(&tx, pk).unwrap();
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

use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn session_kind_reads(supplied: bool) -> u64 {
    let dir = tempfile::tempdir().unwrap();
    let mut conn = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    let tx = conn.transaction().unwrap();
    let seed = batch("input1", 0, 0, vec![], &["main", "child"]);
    write_rows(&tx, &seed, &seed.parse[0], 1, 1, true).unwrap();
    tx.execute(
        "UPDATE session SET kind='subagent' WHERE id=?1",
        [sid("child").to_string()],
    )
    .unwrap();
    tx.flush_prepared_statement_cache();
    let observed = Arc::new(AtomicU64::new(0));
    let count = observed.clone();
    tx.authorizer(Some(move |context: AuthContext<'_>| {
        if matches!(
            context.action,
            AuthAction::Read {
                table_name: "session",
                column_name: "kind"
            }
        ) {
            count.fetch_add(1, Ordering::Relaxed);
        }
        Authorization::Allow
    }))
    .unwrap();
    let rows = (0..32)
        .map(|n| {
            row(
                n,
                text_message(
                    &format!("message-{n}"),
                    if n % 2 == 0 { "main" } else { "child" },
                    Role::User,
                    "Synthetic body",
                    None,
                ),
            )
        })
        .collect();
    let mut b = batch(
        "input1",
        0,
        100,
        rows,
        if supplied { &["main", "child"] } else { &[] },
    );
    if supplied {
        b.parse[0].output.sessions[1] = Session::new(sid("child"), SessionKind::Subagent);
    }
    write_rows(&tx, &b, &b.parse[0], 1, 2, true).unwrap();
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(
        tx.query_row(
            "SELECT count(*) FROM message_source WHERE main=1",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        16
    );
    assert_eq!(
        tx.query_row(
            "SELECT count(*) FROM message_source WHERE main=0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        16
    );
    observed.load(Ordering::Relaxed)
}

#[test]
fn supplied_session_kinds_skip_the_database_fallback() {
    let supplied = session_kind_reads(true);
    let fallback = session_kind_reads(false);
    println!("session-kind SQL preparations: supplied={supplied}, fallback={fallback}");
    assert!(
        supplied < fallback,
        "supplied facts still execute the database fallback"
    );
}

#[test]
fn prepared_and_fallback_bodies_keep_exact_messages_and_events() {
    for mode in ["unprepared", "prepared", "incomplete"] {
        let w = world();
        let mut message = text_message(
            "answer",
            "s",
            Role::Assistant,
            "Synthetic Ω body",
            Some("tool_use"),
        );
        message.blocks.push(Block::ToolCall {
            call_id: Some("call-1".into()),
            tool: Some("Bash".into()),
            action: polylogue_model::ToolAction::Shell,
            input: serde_json::json!({"command":"printf x", "nested":[null, false, "Ω"]}),
            command: Some("printf x".into()),
            path: None,
        });
        let event = Event {
            session: sid("s"),
            kind: "notice".into(),
            at: None,
            after: Some(message.id.clone()),
            range: Some((10, 20)),
            data: serde_json::json!({"nested":[null, false, "Ω"]}),
        };
        let mut b = batch("input1", 0, 0, vec![row(1, message.clone())], &["s"]);
        b.parse[0].output.events.push(event.clone());
        match mode {
            "prepared" => b.parse[0].prepare(),
            "incomplete" => {
                b.parse[0].bodies = vec![vec![]];
                b.parse[0].events = vec![("wrong event".into(), vec![]); 2];
            }
            _ => {}
        }
        w.writer.send(b).unwrap();
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{mode}: {:?}", report.errors);
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader.messages(&sid("s"), false).unwrap(),
            [message.clone()],
            "{mode}"
        );
        assert_eq!(reader.events(&sid("s")).unwrap(), [event], "{mode}");
        let versions = reader.message_versions(&message.id).unwrap();
        assert_eq!(versions.len(), 1, "{mode}");
        assert_eq!(versions[0].message, message, "{mode}");
    }
}

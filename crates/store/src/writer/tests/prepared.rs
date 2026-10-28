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
    let first_build = crate::first_build::FirstBuild::default();
    let tx = conn.transaction().unwrap();
    let mut seed = batch("input1", 0, 0, vec![], &["main", "child"]);
    seed.parse[0].output.sessions[1] = Session::new(sid("child"), SessionKind::Subagent);
    write_rows(&tx, &seed, &seed.parse[0], 1, 1, Some(&first_build)).unwrap();
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
    write_rows(&tx, &b, &b.parse[0], 1, 2, Some(&first_build)).unwrap();
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
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
fn new_messages_do_not_read_a_previous_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut conn = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first_build = crate::first_build::FirstBuild::default();
    // With caching disabled, authorization sees every statement execution.
    conn.set_prepared_statement_cache_capacity(0);
    let tx = conn.transaction().unwrap();
    let observed = Arc::new(AtomicU64::new(0));
    let count = observed.clone();
    tx.authorizer(Some(move |context: AuthContext<'_>| {
        if matches!(
            context.action,
            AuthAction::Read {
                table_name: "message_source",
                column_name: "version_hash"
            }
        ) {
            count.fetch_add(1, Ordering::Relaxed);
        }
        Authorization::Allow
    }))
    .unwrap();
    let messages = 32;
    let rows = (0..messages)
        .map(|n| {
            row(
                n,
                text_message(&format!("new-{n}"), "main", Role::User, "Body", None),
            )
        })
        .collect();
    let mut b = batch("input1", 0, 100, rows, &["main"]);
    b.parse[0].at_end = false;
    assert_eq!(
        write_rows(&tx, &b, &b.parse[0], 1, 1, Some(&first_build)).unwrap(),
        messages
    );
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        messages as i64
    );
    let reads = observed.load(Ordering::Relaxed);
    println!("source-version reads for {messages} new messages: {reads}");
    // A first build finds no row under a new message's key, so it reads no
    // shown version.
    assert_eq!(reads, 0, "new sources still query a previous value");
}

#[test]
fn prepared_and_fallback_bodies_keep_exact_messages_and_events() {
    for mode in ["unprepared", "prepared", "repeated"] {
        let w = test_archive();
        let mut message = text_message(
            "answer",
            "s",
            Role::Assistant,
            "Synthetic Ω body",
            Some("tool_use"),
        );
        message.blocks.push(Block::ToolCall {
            run_by_script: false,
            call_id: Some("call-1".into()),
            tool: Some("Bash".into()),
            action: polylogue_model::ToolAction::Shell,
            input: serde_json::json!({"command":"printf x", "nested":[null, false, "Ω"]}),
            command: Some("printf x".into()),
            path: None,
        });
        message.blocks.push(Block::ToolResult {
            call_id: Some("call-1".into()),
            outcome: ToolOutcome::Ok,
            output: Some("Synthetic Ω output".into()),
            data: Some(serde_json::json!({"result":[false, "Ω"]})),
            attachments: vec![],
        });
        message.blocks.push(Block::Other {
            type_name: "synthetic".into(),
            data: serde_json::json!({"opaque":[null, "Ω"]}),
        });
        let event = Event {
            session: Some(sid("s")),
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
            "repeated" => {
                b.parse[0].prepare();
                b.parse[0].prepare();
            }
            _ => {}
        }
        if mode != "unprepared" {
            let prepared = &b.parse[0].prepared.messages[0];
            assert_eq!(
                prepared
                    .blocks
                    .iter()
                    .map(|b| crate::encoding::unpack_block(b))
                    .collect::<Result<Vec<_>>>()
                    .unwrap(),
                message
                    .blocks
                    .iter()
                    .map(pack_block)
                    .map(|b| crate::encoding::unpack_block(&b.unwrap()))
                    .collect::<Result<Vec<_>>>()
                    .unwrap()
            );
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
    }
}

#[test]
fn filtering_prepared_messages_keeps_original_hashes_and_structured_payloads() {
    for removed in [vec![0], vec![1], vec![0, 1, 2]] {
        let w = test_archive();
        let messages = (0..3)
            .map(|n| {
                let mut m = text_message(
                    &format!("m{n}"),
                    "s",
                    Role::Assistant,
                    "Synthetic text",
                    Some("end_turn"),
                );
                m.blocks.push(Block::Other {
                    type_name: "synthetic".into(),
                    data: serde_json::json!({"message":n,"nested":[false,"Ω"]}),
                });
                m
            })
            .collect::<Vec<_>>();
        let mut b = batch(
            "input1",
            0,
            0,
            messages
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, m)| row(i as u64, m))
                .collect(),
            &["s"],
        );
        b.parse[0].prepare();
        let mut refused = crate::excise::RefusedIds::default();
        for n in &removed {
            refused.messages.insert(messages[*n].id.to_string());
        }
        let mut filtered = without_refused(&b, &refused);
        filtered.parse[0].prepare();
        w.writer.send(filtered).unwrap();
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        let expected = messages
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed.contains(i))
            .map(|(_, m)| m.clone())
            .collect::<Vec<_>>();
        assert_eq!(reader.messages(&sid("s"), false).unwrap(), expected);
    }
}

#[test]
fn misaligned_prepared_rows_refuse_before_writing() {
    for change_id in [false, true] {
        let w = test_archive();
        let mut b = batch(
            "input1",
            0,
            0,
            vec![row(
                1,
                text_message("m", "s", Role::User, "Synthetic", None),
            )],
            &["s"],
        );
        b.parse[0].prepare();
        if change_id {
            b.parse[0].output.messages[0].message.id =
                polylogue_model::MessageId::parse("claude-code:other").unwrap();
        } else {
            b.parse[0].prepared.messages[0].blocks.clear();
        }
        b.parse[0].prepare();
        w.writer.send(b).unwrap();
        assert!(!w.writer.flush().unwrap().errors.is_empty());
        let pool = ReadPool::open(&w.archive).unwrap();
        assert_eq!(
            pool.reader()
                .unwrap()
                .query("SELECT count(*) AS n FROM message", &[])
                .unwrap()[0]["n"],
            0
        );
    }
}

#[test]
fn prepared_hook_events_keep_waiting_reasons_and_current_tool_facts() {
    for prepared in [false, true] {
        for (kind, data, waiting, todo) in [
            (
                "hook_notification",
                serde_json::json!({"notification_type":"permission_prompt"}),
                Some("permission"),
                None,
            ),
            (
                "hook_pre_tool_use",
                serde_json::json!({"tool_use_id":"call-1","tool_name":"TodoWrite","tool_input":{"todos":[{"status":"in_progress","activeForm":"Checking the parser"}]}}),
                None,
                Some("Checking the parser"),
            ),
        ] {
            let w = test_archive();
            let mut message = text_message("m", "s", Role::Assistant, "Synthetic reply", None);
            message.blocks.push(Block::ToolCall {
                run_by_script: false,
                call_id: Some("call-1".into()),
                tool: Some("TodoWrite".into()),
                action: polylogue_model::ToolAction::Other,
                input: serde_json::json!({}),
                command: None,
                path: None,
            });
            let mut b = batch("input1", 0, 0, vec![row(1, message)], &["s"]);
            b.parse[0].output.events.push(Event {
                session: Some(sid("s")),
                kind: kind.into(),
                at: Timestamp::from_unix_ms(2_000),
                after: None,
                range: Some((10, 20)),
                data: data.clone(),
            });
            if prepared {
                b.parse[0].prepare();
            }
            w.writer.send(b).unwrap();
            assert!(w.writer.flush().unwrap().errors.is_empty());
            let pool = ReadPool::open(&w.archive).unwrap();
            let reader = pool.reader().unwrap();
            assert_eq!(reader.events(&sid("s")).unwrap()[0].data, data);
            let rows=reader.query("SELECT s.live_waiting_on AS waiting,s.live_todo AS todo,t.call_id FROM session s LEFT JOIN tool_call t ON t.pk=s.live_open_tool_pk WHERE s.id='claude-code:s'",&[]).unwrap();
            assert_eq!(
                rows[0]["waiting"].as_str(),
                waiting,
                "prepared={prepared} {kind}"
            );
            assert_eq!(rows[0]["todo"].as_str(), todo, "prepared={prepared} {kind}");
            if todo.is_some() {
                assert_eq!(rows[0]["call_id"], "call-1");
            }
        }
    }
}

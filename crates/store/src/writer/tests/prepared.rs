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
    let mut seed = batch("input1", 0, 0, vec![], &["main", "child"]);
    seed.parse[0].output.sessions[1] = Session::new(sid("child"), SessionKind::Subagent);
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
fn new_messages_do_not_read_a_previous_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut conn = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
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
        write_rows(&tx, &b, &b.parse[0], 1, 1, true).unwrap(),
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
    // The current-message lookup reads the preferred version once per message.
    assert_eq!(reads, messages, "new sources still query a previous value");
}

#[test]
fn prepared_and_fallback_bodies_keep_exact_messages_and_events() {
    for mode in ["unprepared", "prepared", "repeated"] {
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
            let mut attrs = message.clone();
            attrs.blocks.clear();
            assert_eq!(
                crate::encoding::unpack_json::<Message>(&prepared.attrs).unwrap(),
                attrs
            );
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
        let versions = reader.message_versions(&message.id).unwrap();
        assert_eq!(versions.len(), 1, "{mode}");
        assert_eq!(versions[0].message, message, "{mode}");
    }
}

#[test]
fn filtering_prepared_messages_keeps_original_hashes_and_structured_payloads() {
    for removed in [vec![0], vec![1], vec![0, 1, 2]] {
        let w = world();
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
        for m in expected {
            let versions = reader.message_versions(&m.id).unwrap();
            assert_eq!(versions.len(), 1);
            assert_eq!(versions[0].message, m);
        }
    }
}

#[test]
fn misaligned_prepared_rows_refuse_before_writing() {
    for change_id in [false, true] {
        let w = world();
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
fn retiring_a_prepared_snapshot_promotes_the_complete_losing_payload() {
    let w = world();
    let mut preferred = text_message("m", "s", Role::Assistant, "Preferred", Some("end_turn"));
    preferred.blocks.push(Block::Other {
        type_name: "synthetic".into(),
        data: serde_json::json!({"winner":true}),
    });
    let mut alternative = preferred.clone();
    alternative.blocks[1] = Block::Other {
        type_name: "synthetic".into(),
        data: serde_json::json!({"alternative":[null,"Ω"]}),
    };
    for (input, rank, message) in [
        ("preferred", 0, preferred.clone()),
        ("alternative", 1, alternative.clone()),
    ] {
        let mut b = batch(input, rank, 0, vec![row(1, message)], &["s"]);
        if rank == 0 {
            b.parse[0].snapshot = Some(polylogue_model::Snapshot {
                start: 0,
                end: 100,
                covered: vec![sid("s")],
                complete: true,
            });
        }
        b.parse[0].prepare();
        w.writer.send(b).unwrap();
    }
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut empty = batch("preferred", 0, 100, vec![], &["s"]);
    empty.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: 100,
        end: 200,
        covered: vec![sid("s")],
        complete: true,
    });
    empty.parse[0].prepare();
    w.writer.send(empty).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.messages(&sid("s"), false).unwrap(),
        [alternative.clone()]
    );
    let versions = reader.message_versions(&alternative.id).unwrap();
    assert_eq!(versions.len(), 2);
    assert!(versions.iter().any(|v| v.message == preferred));
    assert!(versions.iter().any(|v| v.message == alternative));
}

#[test]
fn partial_message_frames_do_not_repeat_session_or_input_snapshot_scans() {
    let dir = tempfile::tempdir().unwrap();
    let mut conn = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.set_prepared_statement_cache_capacity(0);
    let tx = conn.transaction().unwrap();
    let other = batch("other", 0, 0, vec![], &["child"]);
    write_rows(&tx, &other, &other.parse[0], 2, 1, true).unwrap();
    let mut seed = batch("input", 0, 0, vec![], &["child"]);
    seed.parse[0].output.sessions[0] = Session::new(sid("child"), SessionKind::Subagent);
    seed.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: 900,
        end: 1000,
        covered: vec![sid("child")],
        complete: false,
    });
    write_rows(&tx, &seed, &seed.parse[0], 1, 1, true).unwrap();
    let scans = Arc::new(AtomicU64::new(0));
    let updates = scans.clone();
    tx.authorizer(Some(move |ctx: AuthContext<'_>| {
        if matches!(
            ctx.action,
            AuthAction::Read {
                table_name: "snapshot" | "snapshot_session",
                column_name: "complete"
            }
        ) {
            updates.fetch_add(1, Ordering::Relaxed);
        }
        Authorization::Allow
    }))
    .unwrap();
    for n in 0..128 {
        let mut b = batch(
            "input",
            0,
            n,
            vec![row(
                n,
                text_message(
                    &format!("m{n}"),
                    "child",
                    Role::User,
                    "synthetic body",
                    None,
                ),
            )],
            &[],
        );
        b.parse[0].snapshot = Some(polylogue_model::Snapshot {
            start: 900,
            end: 1000,
            covered: vec![],
            complete: false,
        });
        b.parse[0].at_end = false;
        write_rows(&tx, &b, &b.parse[0], 1, 2 + n as i64, true).unwrap();
    }
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    let count = scans.load(Ordering::Relaxed);
    println!("128 partial message frames: whole-snapshot coverage queries={count}");
    assert_eq!(count, 0);
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        128
    );
    assert_eq!(
        tx.query_row(
            "SELECT json_extract(facts, '$.kind') FROM session_record WHERE input_pk=(SELECT pk FROM input_ref WHERE id='input')",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "subagent"
    );
    assert_eq!(
        tx.query_row(
            "SELECT count(*) FROM message_source WHERE main=0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        128
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM snapshot_session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    // The other input still describes a main session; it must not classify
    // this input's child messages or make an older view current.
    assert_eq!(
        tx.query_row("SELECT kind FROM session", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "main"
    );
    let mut late = batch(
        "input",
        0,
        200,
        vec![row(
            200,
            text_message("late", "child", Role::User, "old view", None),
        )],
        &[],
    );
    late.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: 800,
        end: 900,
        covered: vec![],
        complete: false,
    });
    late.parse[0].at_end = false;
    write_rows(&tx, &late, &late.parse[0], 1, 200, true).unwrap();
    assert_eq!(tx.query_row("SELECT main,current FROM message_source s JOIN message m ON m.pk=s.message_pk WHERE m.id='claude-code:late'",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?))).unwrap(),(0,0));
}

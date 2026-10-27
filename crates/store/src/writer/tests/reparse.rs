use super::*;
use polylogue_model::{Document, DocumentId, DocumentPurpose, DocumentRow};
use serde_json::json;

fn fixture(changed: bool) -> Vec<Batch> {
    let message = |id, text| text_message(id, "main", Role::Assistant, text, Some("end_turn"));
    let mut main = batch(
        "a-main-file",
        0,
        0,
        vec![
            row(10, message("shared", "the main session's reading")),
            row(
                if changed { 80 } else { 20 },
                message("moved", "same text, later record"),
            ),
            row(
                30,
                message(
                    if changed { "renamed" } else { "before-name" },
                    "renamed message",
                ),
            ),
        ],
        &["main"],
    );
    if !changed {
        main.parse[0]
            .output
            .messages
            .push(row(40, message("dropped", "removed by the new parser")));
    }
    let mut child = batch(
        "child-file",
        0,
        0,
        vec![row(
            1,
            text_message(
                "shared",
                "child",
                Role::Assistant,
                "copied into the subagent",
                Some("end_turn"),
            ),
        )],
        &["child"],
    );
    child.parse[0].output.sessions[0].kind = SessionKind::Subagent;
    let browser_message = || {
        let mut m = message("browser-shared", "page reading");
        m.id = MessageId::parse("chatgpt:shared").unwrap();
        m.session = SessionId::parse("chatgpt:conversation").unwrap();
        m
    };
    let mut browser = batch("browser", 1, 0, vec![row(9, browser_message())], &[]);
    browser.input.format = "browser-capture".into();
    browser.parse[0].output.sessions.push(Session::new(
        SessionId::parse("chatgpt:conversation").unwrap(),
        SessionKind::Main,
    ));
    let mut export = browser.clone();
    export.input.id = "export".into();
    export.input.arrival_rank = 2;
    export.input.format = "chatgpt".into();
    export.observation.as_mut().unwrap().path = "/p/export".into();
    export.parse[0].output.messages[0].offset = 50;
    export.parse[0].output.messages[0].message.blocks = vec![Block::Text {
        text: "export reading".into(),
    }];
    for (b, limit) in [
        (&mut main, if changed { 90 } else { 100 }),
        (&mut child, 100),
    ] {
        b.parse[0].output.events.push(Event {
            session: Some(sid("main")),
            kind: "quota_report".into(),
            at: Timestamp::from_unix_ms(1000),
            after: None,
            range: Some((1, 2)),
            data: json!({"report_id":"quota-shared","limit":limit}),
        });
        b.parse[0].output.links.push(Link {
            parent: sid("main"),
            child: sid("child"),
            kind: LinkKind::Spawn,
            at: None,
            call_id: None,
            evidence: b.input.id.clone(),
        });
    }
    if !changed {
        main.parse[0].output.events.push(Event {
            session: Some(sid("main")),
            kind: "hook_notification".into(),
            at: Timestamp::from_unix_ms(2000),
            after: None,
            range: Some((50, 60)),
            data: json!({"notification_type":"permission_prompt"}),
        });
    }
    main.parse[0].output.links[0].at = Some(MessageId::parse("claude-code:shared").unwrap());
    child.parse[0].output.links[0].call_id = Some("spawn-call".into());
    let mut doc = Document::new(
        DocumentId::parse("claude-ai:instructions").unwrap(),
        DocumentPurpose::Instruction,
    );
    doc.blocks = vec![Block::Text {
        text: if changed {
            "new document"
        } else {
            "old document"
        }
        .into(),
    }];
    doc.source_time = Timestamp::from_unix_ms(if changed { 2000 } else { 1000 });
    main.parse[0].output.documents.push(DocumentRow {
        offset: 60,
        end: 70,
        document: doc.clone(),
    });
    doc.blocks = vec![Block::Text {
        text: "other input's document".into(),
    }];
    doc.source_time = Timestamp::from_unix_ms(1500);
    child.parse[0].output.documents.push(DocumentRow {
        offset: 61,
        end: 71,
        document: doc,
    });
    for b in [&mut main, &mut child, &mut browser, &mut export] {
        b.parse[0].reading_version = if changed {
            "synthetic=2"
        } else {
            "synthetic=1"
        }
        .into();
        b.parse[0].counts = counts(b.parse[0].output.messages.len() as u64);
    }
    vec![main, child, browser, export]
}

fn write(w: &TestArchive, batches: impl IntoIterator<Item = Batch>) {
    for mut b in batches {
        for step in &mut b.parse {
            step.prepare();
        }
        w.writer.send(b).unwrap();
    }
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    w.writer.derive().unwrap();
}

fn natural_rows(w: &TestArchive) -> Json {
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let mut result = serde_json::Map::new();
    result.insert("live".into(),json!(r.query("SELECT id,live_last_event,live_last_event_at,live_waiting_on,live_waiting_since,live_todo,live_task FROM session ORDER BY id",&[]).unwrap()));
    for id in [
        "claude-code:main",
        "claude-code:child",
        "chatgpt:conversation",
    ] {
        let id = SessionId::parse(id).unwrap();
        result.insert(
            id.to_string(),
            json!({"session":r.session(&id).unwrap(),"messages":r.messages(&id,false).unwrap()}),
        );
    }
    for (name, sql) in [
        (
            "sources",
            "SELECT m.id,i.id AS input,s.chain,s.source_start,s.source_end,s.shown,hex(s.version_hash) AS version FROM message_source s JOIN message m ON m.pk=s.message_pk JOIN input_ref i ON i.pk=s.input_pk ORDER BY m.id,i.id,s.chain",
        ),
        (
            "order",
            "SELECT m.id,i.id AS input,m.seq,m.record_position FROM message m JOIN input_ref i ON i.pk=m.order_input_pk ORDER BY m.id",
        ),
        (
            "events",
            "SELECT e.id,s.id AS session,e.kind,e.at,i.id AS input,e.source_chain,e.source_start,e.source_end,hex(e.data) AS data FROM event e LEFT JOIN session s ON s.pk=e.session_pk JOIN input_ref i ON i.pk=e.input_pk ORDER BY e.id",
        ),
        (
            "links",
            "SELECT parent_id,child_id,kind,how,branch_point,spawning_call,evidence,i.id AS input,source_chain FROM link l JOIN input_ref i ON i.pk=l.input_pk ORDER BY parent_id,child_id,kind,how",
        ),
        (
            "event_sources",
            "SELECT e.id,i.id AS input,s.chain,hex(v.hash) AS hash FROM event_source s JOIN event e ON e.pk=s.event_pk JOIN input_ref i ON i.pk=s.input_pk JOIN event_value v ON v.pk=s.value_pk ORDER BY e.id,i.id,s.chain",
        ),
        (
            "link_sources",
            "SELECT l.parent_id,l.child_id,i.id AS input,s.chain,hex(v.hash) AS hash FROM link_source s JOIN link l ON l.pk=s.link_pk JOIN input_ref i ON i.pk=s.input_pk JOIN link_value v ON v.pk=s.value_pk ORDER BY l.parent_id,l.child_id,i.id,s.chain",
        ),
        (
            "documents",
            "SELECT d.id,i.id AS input,source_chain,source_start,source_end,hex(d.version_hash) AS version FROM document d JOIN input_ref i ON i.pk=d.input_pk ORDER BY d.id",
        ),
        (
            "parse",
            "SELECT i.id,p.chain,p.reading_version,p.records,p.parsed_upto,p.status FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk ORDER BY i.id,p.chain",
        ),
    ] {
        result.insert(name.into(), json!(r.query(sql, &[]).unwrap()));
    }
    Json::Object(result)
}

#[test]
fn replacing_each_chain_equals_a_fresh_build_and_plan_order_does_not_matter() {
    let reparsed = test_archive();
    write(&reparsed, fixture(false));
    for mut b in fixture(true) {
        let mut reading = ChainReading::new().unwrap();
        reading.push(&mut b.parse[0]).unwrap();
        reparsed
            .writer
            .sender()
            .unwrap()
            .replace_chain(b.input, 1, b.length, reading)
            .unwrap();
    }
    reparsed.writer.derive().unwrap();
    let fresh = test_archive();
    write(&fresh, fixture(true));
    let reverse = test_archive();
    write(&reverse, fixture(true).into_iter().rev());
    assert_eq!(natural_rows(&reparsed), natural_rows(&fresh));
    assert_eq!(natural_rows(&fresh), natural_rows(&reverse));
    let pool = ReadPool::open(&reparsed.archive).unwrap();
    let r = pool.reader().unwrap();
    assert!(r.query("PRAGMA foreign_key_check", &[]).unwrap().is_empty());
    assert_eq!(r.query("SELECT count(*) AS n FROM message WHERE id IN ('claude-code:dropped','claude-code:before-name')",&[]).unwrap()[0]["n"],0);
    assert_eq!(
        r.query(
            "SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'removed'",
            &[]
        )
        .unwrap()[0]["n"],
        0
    );
}

#[test]
fn retracting_the_winner_restores_complete_structured_payloads_without_parsing_survivors() {
    let w = test_archive();
    let original = super::hash_order::structured_message(false);
    let mut winner = original.clone();
    winner.blocks = vec![Block::Text {
        text: "shown first".into(),
    }];
    write(
        &w,
        [
            batch("a", 0, 0, vec![row(1, winner)], &["hash-order"]),
            batch("b", 2, 0, vec![row(3, original.clone())], &["hash-order"]),
        ],
    );
    let mut reading = ChainReading::new().unwrap();
    reading
        .push(&mut ParseStep::new(
            Output::default(),
            RecordCounts::default(),
            0..100,
            true,
        ))
        .unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(
            InputKey {
                id: "a".into(),
                format: "claude-code".into(),
                arrival_rank: 0,
            },
            1,
            100,
            reading,
        )
        .unwrap();
    w.writer.derive().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(r.messages(&sid("hash-order"), false).unwrap(), [original]);
}

#[test]
fn newest_chain_is_selected_before_comparing_inputs() {
    let mut expected = None;
    for order in [
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
    ] {
        let w = test_archive();
        let batches = [
            batch(
                "a",
                0,
                0,
                vec![row(
                    1,
                    text_message(
                        "same",
                        "s",
                        Role::Assistant,
                        "old complete",
                        Some("end_turn"),
                    ),
                )],
                &["s"],
            ),
            batch(
                "b",
                1,
                0,
                vec![row(
                    1,
                    text_message(
                        "same",
                        "s",
                        Role::Assistant,
                        "other complete",
                        Some("end_turn"),
                    ),
                )],
                &["s"],
            ),
            {
                let mut b = batch(
                    "a",
                    0,
                    0,
                    vec![row(
                        1,
                        text_message("same", "s", Role::Assistant, "new open", None),
                    )],
                    &["s"],
                );
                b.change = Change::Rewritten;
                b
            },
        ];
        // Chain numbers are archive facts, independent of parse-plan order.
        for b in &batches {
            let mut bytes = b.clone();
            bytes.parse.clear();
            w.writer.send(bytes).unwrap();
        }
        assert!(w.writer.flush().unwrap().errors.is_empty());
        for i in order {
            let mut b = batches[i].clone();
            b.change = Change::Reparse {
                chain: if i == 2 { 2 } else { 1 },
            };
            w.writer.send(b).unwrap();
        }
        assert!(w.writer.flush().unwrap().errors.is_empty());
        let got = texts(&w, "s");
        assert_eq!(got, ["other complete"]);
        if let Some(e) = &expected {
            assert_eq!(&got, e);
        } else {
            expected = Some(got);
        }
    }
}

#[test]
fn a_later_reading_failure_rolls_back_retraction_and_prior_steps() {
    let w = test_archive();
    write(&w, fixture(false));
    let before = natural_rows(&w);
    let mut b = fixture(true).remove(0);
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut b.parse[0]).unwrap();
    b.parse[0].output.events.push(Event {
        session: None,
        kind: "unbound".into(),
        at: None,
        after: None,
        range: None,
        data: Json::Null,
    });

    reading.push(&mut b.parse[0]).unwrap();
    assert!(
        w.writer
            .sender()
            .unwrap()
            .replace_chain(b.input, 1, b.length, reading)
            .is_err()
    );
    assert_eq!(natural_rows(&w), before);
}

#[test]
fn a_chain_reading_preserves_deep_structured_values() {
    let w = test_archive();
    write(
        &w,
        [batch(
            "a",
            0,
            0,
            vec![row(
                1,
                text_message("deep", "s", Role::User, "before", None),
            )],
            &["s"],
        )],
    );
    let nested = (0..200).fold(json!("leaf"), |value, _| json!({"nested":value}));
    let mut message = text_message("deep", "s", Role::User, "", None);
    message.blocks = vec![Block::Other {
        type_name: "nested".into(),
        data: nested,
    }];
    let mut b = batch("a", 0, 0, vec![row(1, message.clone())], &["s"]);
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut b.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(b.input, 1, b.length, reading)
        .unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        reads.reader().unwrap().messages(&sid("s"), false).unwrap(),
        [message]
    );
}

#[test]
fn a_later_complete_conversation_updates_its_shown_message_body() {
    let w = test_archive();
    write(
        &w,
        [batch(
            "a",
            0,
            0,
            vec![row(
                1,
                text_message(
                    "same",
                    "s",
                    Role::Assistant,
                    "first reading",
                    Some("end_turn"),
                ),
            )],
            &["s"],
        )],
    );
    let mut appended = batch(
        "a",
        0,
        100,
        vec![row(
            120,
            text_message(
                "same",
                "s",
                Role::Assistant,
                "later reading",
                Some("end_turn"),
            ),
        )],
        &["s"],
    );
    appended.parse[0].repeats_whole_conversations = true;
    write(&w, [appended]);
    assert_eq!(texts(&w, "s"), ["later reading"]);
}

#[test]
fn hook_event_times_produce_the_same_live_state_in_both_input_orders() {
    let make = |input, kind: &str, at, data| {
        let mut b = batch(input, 0, 0, Vec::new(), &["s"]);
        b.parse[0].output.events.push(Event {
            session: Some(sid("s")),
            kind: kind.into(),
            at: Timestamp::from_unix_ms(at),
            after: None,
            range: Some((1, 2)),
            data,
        });
        b
    };
    for end_at in [1000, 2000] {
        let batches = [
            make("a", "hook_permission_request", 1000, json!({})),
            make("b", "hook_stop", end_at, json!({})),
            make(
                "c",
                "hook_notification",
                3000,
                json!({"notification_type":"other"}),
            ),
        ];
        let first = test_archive();
        write(&first, batches.clone());
        let reverse = test_archive();
        write(&reverse, batches.into_iter().rev());
        assert_eq!(natural_rows(&first), natural_rows(&reverse));
    }
}

#[test]
fn a_session_stated_later_in_a_reading_is_not_removed_mid_transaction() {
    let w = test_archive();
    let original = batch(
        "a",
        0,
        0,
        vec![row(
            1,
            text_message("same", "s", Role::User, "retained", None),
        )],
        &["s"],
    );
    write(
        &w,
        [original.clone(), batch("b", 0, 0, Vec::new(), &["other"])],
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let before = pool
        .reader()
        .unwrap()
        .query("SELECT pk FROM session WHERE id='claude-code:s'", &[])
        .unwrap();
    let mut reading = ChainReading::new().unwrap();
    reading
        .push(&mut ParseStep::new(
            Output::default(),
            RecordCounts::default(),
            0..0,
            true,
        ))
        .unwrap();
    let mut replacement = original;
    reading.push(&mut replacement.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(replacement.input, 1, replacement.length, reading)
        .unwrap();
    w.writer.derive().unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .query("SELECT pk FROM session WHERE id='claude-code:s'", &[])
            .unwrap(),
        before
    );
    assert_eq!(texts(&w, "s"), ["retained"]);
}

#[test]
fn changing_a_shared_event_marks_the_old_and_restored_sessions() {
    let w = test_archive();
    let make = |input, session, rank, limit| {
        let mut b = batch(input, rank, 0, Vec::new(), &[session]);
        b.parse[0].output.events.push(Event {
            session: Some(sid(session)),
            kind: "quota_report".into(),
            at: Timestamp::from_unix_ms(1000),
            after: None,
            range: Some((1, 2)),
            data: json!({"report_id":"shared","limit":limit}),
        });
        b
    };
    let pool = ReadPool::open(&w.archive).unwrap();
    let changed = || {
        pool.reader()
            .unwrap()
            .query("SELECT changed FROM session WHERE id='claude-code:b'", &[])
            .unwrap()[0]["changed"]
            .as_i64()
            .unwrap()
    };
    write(&w, [make("b", "b", 1, 100)]);
    let before = changed();
    let winner = make("a", "a", 0, 90);
    write(&w, [winner.clone()]);
    let moved = changed();
    assert!(moved > before);
    let mut reading = ChainReading::new().unwrap();
    reading
        .push(&mut ParseStep::new(
            Output::default(),
            RecordCounts::default(),
            0..100,
            true,
        ))
        .unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(winner.input, 1, winner.length, reading)
        .unwrap();
    w.writer.derive().unwrap();
    assert!(changed() > moved);
    assert_eq!(
        pool.reader()
            .unwrap()
            .query(
                "SELECT s.id FROM event e JOIN session s ON s.pk=e.session_pk",
                &[]
            )
            .unwrap()[0]["id"],
        "claude-code:b"
    );
}

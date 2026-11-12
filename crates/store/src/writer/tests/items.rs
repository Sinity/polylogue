use super::*;

fn stored(w: &TestArchive, session: &str) -> crate::items::StoredItems {
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    reader.item_index(&sid(session)).unwrap().expect("indexed")
}

/// The stored index equals entries computed from the rows alone.
fn assert_matches_rows(w: &TestArchive, session: &str) {
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let index = reader.item_index(&sid(session)).unwrap().expect("indexed");
    let computed =
        crate::items::computed_entries(reader.conn().unwrap(), index.session_pk).unwrap();
    assert_eq!(index.entries, computed, "session {session}");
    // What is stored beside the entries equals a recount of every entry.
    let classes = polylogue_model::read::entry_classes(&computed);
    let (columns, entries, total) =
        crate::items::stored_columns(reader.conn().unwrap(), index.session_pk).unwrap();
    assert_eq!(entries, computed.len() as u64, "session {session}");
    let mut running = polylogue_model::read::ItemCounts::default();
    let units: Vec<bool> = {
        let conn = reader.conn().unwrap();
        let mut statement = conn
            .prepare("SELECT unit IS NOT NULL FROM session_item WHERE session_pk = ?1 ORDER BY seq")
            .unwrap();
        statement
            .query_map([index.session_pk], |row| row.get::<_, bool>(0))
            .unwrap()
            .map(|unit| unit.unwrap())
            .collect()
    };
    for (at, ((entry, class), (stored_class, call_id, counts))) in
        computed.iter().zip(&classes).zip(&columns).enumerate()
    {
        assert_eq!(stored_class, class, "session {session} entry {at}");
        assert_eq!(call_id, &entry.call_id, "session {session} entry {at}");
        assert_eq!(
            counts.as_ref(),
            units[at].then_some(&running),
            "session {session} entry {at}"
        );
        running.add(entry, *class);
    }
    assert_eq!(total, running, "session {session}");
}

fn message(id: &str, role: Role, text: &str) -> Message {
    text_message(id, "s", role, text, Some("end_turn"))
}

#[test]
fn identical_lower_priority_event_claims_leave_selected_events_and_items_untouched() {
    let w = test_archive();
    let mut main = batch(
        "a-events",
        Route::Live,
        0,
        vec![row(0, message("anchor", Role::User, "anchor"))],
        &["s"],
    );
    for n in 0..196 {
        main.parse[0].output.events.push(Event {
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            session: Some(sid("s")),
            kind: "synthetic.observation".into(),
            at: None,
            after: Some(MessageId::parse("claude-code:anchor").unwrap()),
            range: None,
            data: serde_json::json!({"number": n}),
        });
    }
    w.store_reading(main.clone(), Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let before = stored(&w, "s");
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    conn.execute_batch(
        "CREATE TABLE test_event_updates(n INTEGER NOT NULL);
        INSERT INTO test_event_updates VALUES(0);
        CREATE TRIGGER test_event_updates AFTER UPDATE ON event
        BEGIN UPDATE test_event_updates SET n=n+1; END;",
    )
    .unwrap();
    let mut extra = batch("z-extra-events", Route::Capture, 0, vec![], &[]);
    extra.parse[0].output.events = main.parse[0].output.events.clone();
    extra.parse[0].output.event_ids = main.parse[0].output.event_ids.clone();
    w.store_reading(extra, Some(&SOURCE_RECORD)).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM event_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        392
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let id: String = conn
        .query_row("SELECT id FROM event ORDER BY pk LIMIT 1", [], |r| r.get(0))
        .unwrap();
    let record = reader.event_record(&id).unwrap().unwrap();
    assert_eq!(
        record.identity.sources.len(),
        2,
        "both retained source claims are exported"
    );
    assert_eq!(stored(&w, "s"), before);
    assert_matches_rows(&w, "s");
    let updates: i64 = conn
        .query_row("SELECT n FROM test_event_updates", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        (updates, report.item_units_read),
        (0, 0),
        "unchanged selected event updates and session item reads"
    );
    // Exercise the production validity lookup, with unrelated cached entries
    // before the selected header so a missing reverse index cannot pass early.
    let (pk, session, kind, at, data_bytes): (i64, i64, String, Option<i64>, u64) = conn
        .query_row(
            "SELECT pk,session_pk,kind,at,length(data) FROM event WHERE id=?1",
            [&id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    let work = || {
        use std::sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        };
        let steps = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&steps);
        conn.flush_prepared_statement_cache();
        conn.progress_handler(
            1,
            Some(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
        assert!(
            crate::items::event_is_current(&conn, pk, session, &id, &kind, at, data_bytes).unwrap()
        );
        conn.progress_handler(0, None::<fn() -> bool>).unwrap();
        steps.load(Ordering::Relaxed)
    };
    let small = work();
    conn.execute_batch(
        "SAVEPOINT unrelated_headers;
        WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<8192)
        INSERT INTO session_item
        SELECT s.session_pk,-100000+n.x,s.unit,NULL,100000+n.x,s.time_bound,
               s.entry,s.class,s.call_id,s.counts
        FROM n,session_item s WHERE s.event_pk=(SELECT min(pk) FROM event);",
    )
    .unwrap();
    let large = work();
    assert!(
        large <= small + 4,
        "keyed header work grew: {small} -> {large}"
    );
    conn.execute_batch("DROP INDEX session_item_event").unwrap();
    let unindexed = work();
    assert!(
        unindexed > 8192,
        "positive control did not scan unrelated headers: {unindexed}"
    );
    conn.execute_batch("ROLLBACK TO unrelated_headers; RELEASE unrelated_headers")
        .unwrap();
}

#[test]
fn an_identical_event_claim_repairs_retained_old_item_metadata() {
    let w = test_archive();
    let mut input = batch("a-events", Route::Live, 0, vec![], &["s"]);
    input.parse[0].output.events.push(Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(sid("s")),
        kind: "synthetic.observation".into(),
        at: None,
        after: None,
        range: None,
        data: serde_json::json!({"value":1}),
    });
    w.store_reading(input.clone(), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let (pk, entry): (i64, Vec<u8>) = conn
        .query_row(
            "SELECT event_pk,entry FROM session_item WHERE event_pk IS NOT NULL",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    let mut stale: polylogue_model::read::IndexEntry = serde_json::from_slice(&entry).unwrap();
    stale.at = Some(123);
    let mut old = blake3::Hasher::new();
    old.update(b"event\0");
    old.update(&pk.to_le_bytes());
    old.update(b"synthetic.observation");
    conn.execute(
        "UPDATE session_item SET unit=?1,entry=?2 WHERE event_pk=?3",
        params![
            old.finalize().as_bytes().as_slice(),
            serde_json::to_vec(&stale).unwrap(),
            pk
        ],
    )
    .unwrap();
    let before = stored(&w, "s");
    let mut extra = batch("z-extra-events", Route::Capture, 0, vec![], &[]);
    extra.parse[0].output.events = input.parse[0].output.events.clone();
    w.store_reading(extra, Some(&SOURCE_RECORD)).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(
        report.item_units_read > 0,
        "retained stale items must be refreshed"
    );
    assert!(
        stored(&w, "s").rewrite_version > before.rewrite_version,
        "the existing item cursor version invalidates the stale reading"
    );
    assert_matches_rows(&w, "s");
    assert_eq!(
        conn.query_row("SELECT count(*) FROM event_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn selected_event_changes_keep_their_payload_fields_and_placement() {
    for change in 0..6 {
        let w = test_archive();
        let mut input = batch(
            "events",
            Route::Live,
            0,
            vec![
                row(0, message("anchor", Role::User, "first")),
                row(1, message("next", Role::User, "second")),
            ],
            &["s"],
        );
        let event = Event {
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            session: Some(sid("s")),
            kind: "synthetic.observation".into(),
            at: None,
            after: Some(MessageId::parse("claude-code:anchor").unwrap()),
            range: Some((0, 1)),
            data: serde_json::json!({"value": 1}),
        };
        input.parse[0].output.events.push(event);
        input.parse[0]
            .output
            .event_ids
            .insert(0, "synthetic-event".into());
        w.store_reading(input.clone(), Some(&SOURCE_RECORD))
            .unwrap();
        assert!(w.writer.flush().unwrap().errors.is_empty());
        let event = &mut input.parse[0].output.events[0];
        match change {
            0 => {
                event.data =
                    serde_json::json!({"value": "a payload with a different stored length"})
            }
            1 => event.kind = "synthetic.revised".into(),
            2 => event.at = Timestamp::from_unix_ms(1000),
            3 => event.after = Some(MessageId::parse("claude-code:next").unwrap()),
            4 => event.range = Some((1, 2)),
            5 => input.parse[0].order_from = Some(9),
            _ => unreachable!(),
        }
        input.parse[0].reading_version = "synthetic=2".into();
        let expected = input.parse[0].output.events[0].clone();
        let mut reading = ChainReading::new().unwrap();
        reading.push(&mut input.parse[0]).unwrap();
        w.writer
            .sender()
            .unwrap()
            .replace_chain(input.input, 1, 100, reading)
            .unwrap();
        w.writer.derive().unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .event_record("synthetic-event")
                .unwrap()
                .unwrap()
                .event,
            expected,
            "changed event field {change}"
        );
        if change == 5 {
            assert_eq!(
                reader.query("SELECT seq FROM event", &[]).unwrap()[0]["seq"],
                9
            );
        }
        assert_matches_rows(&w, "s");
    }
}

#[test]
fn identical_lower_priority_claims_leave_shown_messages_and_items_untouched() {
    let w = test_archive();
    let messages: Vec<_> = (0..256)
        .map(|n| {
            let mut m = message(&format!("m{n}"), Role::Assistant, "same answer");
            m.at = Timestamp::from_unix_ms(1000 + n * 10);
            row(n as u64 / 3, m)
        })
        .collect();
    w.store_reading(
        batch("a-main", Route::Live, 0, messages.clone(), &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s");
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    conn.execute_batch(
        "CREATE TABLE test_message_updates(n INTEGER NOT NULL);
         INSERT INTO test_message_updates VALUES(0);
         CREATE TRIGGER test_message_updates AFTER UPDATE OF
         input_pk,source_chain,source_start,source_end,session_order,in_progress ON message
         BEGIN UPDATE test_message_updates SET n=n+1; END;",
    )
    .unwrap();
    w.store_reading(
        batch("z-extra", Route::Capture, 0, messages, &[]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let claims: i64 = conn
        .query_row("SELECT count(*) FROM message_source", [], |r| r.get(0))
        .unwrap();
    assert_eq!(claims, 512);
    assert_eq!(stored(&w, "s").entries, before.entries);
    assert_matches_rows(&w, "s");
    let updates: i64 = conn
        .query_row("SELECT n FROM test_message_updates", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        (updates, report.item_units_read),
        (0, 0),
        "unchanged displayed message updates and session item reads"
    );
}

#[test]
fn a_new_citation_chain_does_not_reread_unchanged_message_items() {
    let w = test_archive();
    let messages: Vec<_> = (0..256)
        .map(|n| {
            let mut m = message(&format!("m{n}"), Role::Assistant, "same answer");
            m.at = Timestamp::from_unix_ms(1000 + n * 10);
            row(n as u64 / 3, m)
        })
        .collect();
    w.store_reading(
        batch("main", Route::Live, 0, messages.clone(), &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s").entries;
    let mut rewrite = batch("main", Route::Live, 0, messages, &[]);
    rewrite.change = Change::Rewritten;
    w.store_reading(rewrite, Some(&[b'x'; 100])).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM message WHERE source_chain=2",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        256
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM message_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        512
    );
    assert_eq!(stored(&w, "s").entries, before);
    assert_matches_rows(&w, "s");
    assert_eq!(
        report.item_units_read, 0,
        "citation-only changes preserve the indexed message units"
    );
}

#[test]
fn unchanged_shown_message_still_moves_events_to_its_new_source_sequence() {
    let w = test_archive();
    let rows = vec![row(10, message("anchor", Role::User, "anchor"))];
    w.store_reading(
        batch("a-order", Route::Capture, 0, rows.clone(), &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    let mut main = batch("z-main", Route::Live, 0, rows, &["s"]);
    main.parse[0].order_from = Some(9);
    w.store_reading(main.clone(), Some(&SOURCE_RECORD)).unwrap();
    let mut events = batch("events", Route::Live, 0, vec![], &[]);
    events.parse[0].output.events.push(Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(sid("s")),
        kind: "message_position".into(),
        at: None,
        after: Some(MessageId::parse("claude-code:anchor").unwrap()),
        range: None,
        data: serde_json::json!({}),
    });
    w.store_reading(events, Some(&SOURCE_RECORD)).unwrap();
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let shown = || {
        pool.reader().unwrap().query("SELECT input_pk,source_chain,source_start,source_end,session_order,in_progress FROM message", &[]).unwrap()
    };
    let before = shown();
    assert_eq!(
        pool.reader()
            .unwrap()
            .query("SELECT seq FROM event", &[])
            .unwrap()[0]["seq"],
        9
    );
    main.parse[0].order_from = Some(1);
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut main.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(main.input, 1, 100, reading)
        .unwrap();
    w.writer.derive().unwrap();
    assert_eq!(shown(), before);
    assert_eq!(
        pool.reader()
            .unwrap()
            .query("SELECT seq FROM event", &[])
            .unwrap()[0]["seq"],
        1
    );
    assert_matches_rows(&w, "s");
}

#[test]
fn removing_a_redundant_content_claim_still_invalidates_session_derivations() {
    let w = test_archive();
    let reading = MessageRow {
        id_kind: polylogue_model::IdKind::Content,
        ..row(10, message("shared", Role::User, "same prompt"))
    };
    for (input, route, sessions) in [
        ("a-main", Route::Live, &["s"][..]),
        ("z-copy", Route::Capture, &[][..]),
    ] {
        w.store_reading(
            batch(input, route, 0, vec![reading.clone()], sessions),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
        assert!(w.writer.flush().unwrap().errors.is_empty());
    }
    w.writer.derive().unwrap();
    let before = stored(&w, "s").entries;
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM derive_due", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    conn.execute_batch(
        "CREATE TABLE test_message_updates(n INTEGER NOT NULL);
         INSERT INTO test_message_updates VALUES(0);
         CREATE TRIGGER test_message_updates AFTER UPDATE OF
         input_pk,source_chain,source_start,source_end,session_order,in_progress ON message
         BEGIN UPDATE test_message_updates SET n=n+1; END;",
    )
    .unwrap();
    let mut empty = batch("z-copy", Route::Capture, 0, vec![], &[]);
    empty.change = Change::Rewritten;
    w.store_reading(empty, Some(&[b'x'; 100])).unwrap();
    let report = w.writer.flush_rows().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM message_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT n FROM test_message_updates", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(report.item_units_read, 0);
    assert_eq!(stored(&w, "s").entries, before);
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM derive_due WHERE rows_changed=1",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1,
        "source membership remains an input to session derivations"
    );
}

#[test]
fn every_committed_batch_leaves_the_index_equal_to_the_rows() {
    let w = test_archive();
    w.store_reading(
        batch(
            "i1",
            Route::Live,
            0,
            vec![
                row(10, message("a", Role::User, "first question")),
                row(20, message("b", Role::Assistant, "first answer")),
            ],
            &["s"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    assert_matches_rows(&w, "s");
    assert_eq!(stored(&w, "s").entries.len(), 4);
    w.store_reading(
        batch(
            "i1",
            Route::Live,
            100,
            vec![row(110, message("c", Role::User, "second question"))],
            &["s"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    assert_matches_rows(&w, "s");
    assert_eq!(stored(&w, "s").entries.len(), 6);
}

#[test]
fn an_append_moves_only_the_append_counter() {
    let w = test_archive();
    w.store_reading(
        batch(
            "i1",
            Route::Live,
            0,
            vec![row(10, message("a", Role::User, "question"))],
            &["s"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s");
    for n in 1..=20u64 {
        w.store_reading(
            batch(
                "i1",
                Route::Live,
                n * 100,
                vec![row(
                    n * 100 + 10,
                    message(&format!("m{n}"), Role::Assistant, "more"),
                )],
                &["s"],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
        w.writer.flush().unwrap();
    }
    let after = stored(&w, "s");
    assert_eq!(after.rewrite_version, before.rewrite_version);
    assert!(after.append_seq > before.append_seq);
    assert_eq!(after.entries[..before.entries.len()], before.entries[..]);
    assert_matches_rows(&w, "s");
}

#[test]
fn a_changed_earlier_message_moves_the_rewrite_version() {
    let w = test_archive();
    w.store_reading(
        batch(
            "i1",
            Route::Live,
            0,
            vec![
                row(0, text_message("r", "s", Role::Assistant, "part", None)),
                row(10, message("q", Role::User, "later question")),
            ],
            &["s"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s");
    w.store_reading(
        batch(
            "i1",
            Route::Live,
            100,
            vec![row(
                0,
                text_message(
                    "r",
                    "s",
                    Role::Assistant,
                    "part, said otherwise",
                    Some("end_turn"),
                ),
            )],
            &["s"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let after = stored(&w, "s");
    assert!(after.rewrite_version > before.rewrite_version);
    assert_matches_rows(&w, "s");
}

#[test]
fn a_newly_stored_parent_link_moves_the_child_rewrite_version() {
    let w = test_archive();
    w.store_reading(
        batch(
            "child-file",
            Route::Live,
            0,
            vec![row(
                0,
                text_message("c1", "child", Role::User, "go on", None),
            )],
            &["child"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "child");
    let mut linked = batch("child-file", Route::Live, 100, vec![], &["child"]);
    linked.parse[0].output.links.push(polylogue_model::Link {
        parent: sid("parent"),
        child: sid("child"),
        kind: polylogue_model::LinkKind::Fork,
        copies_prefix: false,
        how: None,
        at: Some(MessageId::parse("claude-code:p1").unwrap()),
        call_id: None,
        evidence: "fixture".into(),
    });
    linked.parse[0].counts = counts(0);
    w.store_reading(linked, Some(&SOURCE_RECORD)).unwrap();
    w.writer.flush().unwrap();
    let after = stored(&w, "child");
    assert!(after.rewrite_version > before.rewrite_version);
    assert_eq!(after.entries, before.entries);
}

#[test]
fn a_growing_last_message_is_an_append() {
    let w = test_archive();
    let mut first = message("g", Role::Assistant, "start");
    first.stop_reason = None;
    w.store_reading(
        batch("i1", Route::Live, 0, vec![row(0, first.clone())], &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s");
    let mut grown = first;
    grown.blocks.push(Block::Text {
        text: "and a second block".into(),
    });
    grown.stop_reason = Some("end_turn".into());
    w.store_reading(
        batch("i1", Route::Live, 100, vec![row(0, grown)], &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let after = stored(&w, "s");
    assert_eq!(after.rewrite_version, before.rewrite_version);
    assert_eq!(after.entries.len(), before.entries.len() + 1);
    assert_matches_rows(&w, "s");
}

#[test]
fn an_event_naming_no_message_keeps_later_messages_appends() {
    let w = test_archive();
    let mut first = message("a", Role::User, "question");
    first.at = Timestamp::from_unix_ms(1_000);
    let mut opening = batch("i1", Route::Live, 0, vec![row(10, first)], &["s"]);
    opening.parse[0].output.events.push(polylogue_model::Event {
        session: Some(sid("s")),
        kind: "hook_run".into(),
        at: Timestamp::from_unix_ms(2_000),
        at_quality: polylogue_model::TimeQuality::Captured,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((20, 30)),
        data: serde_json::json!({"hook": "synthetic"}),
    });
    w.store_reading(opening, Some(&SOURCE_RECORD)).unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s");
    let mut later = message("b", Role::Assistant, "answer");
    later.at = Timestamp::from_unix_ms(3_000);
    w.store_reading(
        batch("i1", Route::Live, 100, vec![row(110, later)], &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let after = stored(&w, "s");
    assert_eq!(after.rewrite_version, before.rewrite_version);
    assert_eq!(after.entries[..before.entries.len()], before.entries[..]);
    assert_eq!(
        after.entries[before.entries.len()].key,
        polylogue_model::read::EntryKey::header(&MessageId::parse("claude-code:b").unwrap())
    );
    assert_matches_rows(&w, "s");
}

fn hook_event(at: i64, range: (u64, u64)) -> polylogue_model::Event {
    polylogue_model::Event {
        session: Some(sid("s")),
        kind: "hook_run".into(),
        at: Timestamp::from_unix_ms(at),
        at_quality: polylogue_model::TimeQuality::Captured,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some(range),
        data: serde_json::json!({"hook": "synthetic"}),
    }
}

#[test]
fn long_session_append_visits_only_suffix() {
    let w = test_archive();
    // 300 timed messages, with hook lines between them, in one reading.
    let rows = (0..300u64)
        .map(|n| {
            let mut m = message(&format!("m{n}"), Role::Assistant, "earlier work");
            m.at = Timestamp::from_unix_ms(1_000 + n as i64 * 10);
            // One 100-byte chunk holds them all, several to a record.
            row(n / 3, m)
        })
        .collect();
    let mut opening = batch("i1", Route::Live, 0, rows, &["s"]);
    for n in 0..300u64 {
        opening.parse[0]
            .output
            .events
            .push(hook_event(1_005 + n as i64 * 10, (n / 3, n / 3 + 1)));
    }
    w.store_reading(opening, Some(&SOURCE_RECORD)).unwrap();
    w.writer.flush().unwrap();
    let before = stored(&w, "s");
    // A live append: one message and the hook line after it.
    let mut appended = message("next", Role::User, "one more question");
    appended.at = Timestamp::from_unix_ms(5_000);
    let mut append = batch("i1", Route::Live, 100, vec![row(110, appended)], &["s"]);
    append.parse[0]
        .output
        .events
        .push(hook_event(5_005, (115, 116)));
    w.store_reading(append, Some(&SOURCE_RECORD)).unwrap();
    let report = w.writer.flush().unwrap();
    // The last earlier message and its hook line, then the two new units.
    assert_eq!(report.item_units_read, 4, "{report:?}");
    let after = stored(&w, "s");
    assert_eq!(after.rewrite_version, before.rewrite_version);
    assert_eq!(after.append_seq, before.append_seq + 1);
    assert_eq!(after.entries[..before.entries.len()], before.entries[..]);
    assert_matches_rows(&w, "s");
}

#[test]
fn a_late_hook_line_is_placed_by_its_time_from_the_suffix() {
    let w = test_archive();
    let rows = (0..20u64)
        .map(|n| {
            let mut m = message(&format!("m{n}"), Role::Assistant, "work");
            m.at = Timestamp::from_unix_ms(1_000 + n as i64 * 10);
            row(n * 5, m)
        })
        .collect();
    w.store_reading(
        batch("i1", Route::Live, 0, rows, &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    // A hook line from another input, timed between the 15th and 16th messages.
    let mut late = batch("i2", Route::Live, 0, Vec::new(), &["s"]);
    late.parse[0].output.events.push(hook_event(1_155, (1, 2)));
    w.store_reading(late, Some(&SOURCE_RECORD)).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.item_units_read < 10, "{report:?}");
    assert_matches_rows(&w, "s");
}

#[test]
fn lineage_walks_links_transitively_and_skips_references() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    for id in ["test:a", "test:b", "test:c", "test:d"] {
        fixture
            .session(crate::fixture::SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
    }
    fixture.link("test:a", "test:b", "spawn", None).unwrap();
    fixture.link("test:b", "test:c", "resume", None).unwrap();
    fixture
        .link("test:a", "test:d", "reference", Some("mention"))
        .unwrap();
    fixture.link("test:c", "test:a", "fork", None).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let a = SessionId::parse("test:a").unwrap();
    let c = SessionId::parse("test:c").unwrap();
    assert_eq!(
        reader
            .lineage(&a, crate::lineage::Lineage::Descendants)
            .unwrap(),
        ["test:b", "test:c"]
    );
    assert_eq!(
        reader
            .lineage(&c, crate::lineage::Lineage::Ancestors)
            .unwrap(),
        ["test:b", "test:a"]
    );
}

/// A failed item refresh must reject the rows, even when the old index exists.
#[test]
fn indexing_errors_abort_the_changed_rows_transaction() {
    for invalid_json in [false, true] {
        let w = test_archive();
        w.store_reading(
            batch(
                "i1",
                Route::Live,
                0,
                vec![row(10, message("a", Role::User, "original"))],
                &["s"],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
        w.writer.flush().unwrap();
        let before = stored(&w, "s");
        let mut conn =
            crate::archive::open_write(&w.archive.index_db(), crate::archive::cache::SETUP)
                .unwrap();
        {
            let tx = conn.transaction().unwrap();
            let counter = bump_counter(&tx).unwrap();
            tx.execute(
                "UPDATE session SET changed = ?1 WHERE pk = ?2",
                params![counter, before.session_pk],
            )
            .unwrap();
            let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
            let messages: Vec<i64> = tx
                .prepare("SELECT pk FROM message WHERE session_pk = ?1")
                .unwrap()
                .query_map([before.session_pk], |row| row.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            for pk in messages {
                changes.message(&tx, pk).unwrap();
            }
            tx.execute("UPDATE message SET on_active_path = 0", [])
                .unwrap();
            if invalid_json {
                let body =
                    crate::encoding::pack_json(&serde_json::json!({"type":"not-a-block"})).unwrap();
                tx.execute("UPDATE block SET body = ?1", [body]).unwrap();
            } else {
                tx.execute("UPDATE block SET idx = -1, ordinal = NULL", [])
                    .unwrap();
            }
            let error = crate::items::refresh(&tx, &changes)
                .expect_err("a transaction cannot commit rows whose item index did not refresh");
            if invalid_json {
                assert!(
                    matches!(
                        error,
                        crate::StoreError::Invalid(_) | crate::StoreError::Json(_)
                    ),
                    "{error}"
                );
            } else {
                assert!(
                    matches!(error, crate::StoreError::StoredValue { column: 2, .. }),
                    "{error}"
                );
            }
            // Dropping the same transaction production callers own rolls all
            // changed rows back together with any attempted index updates.
        }
        assert_eq!(stored(&w, "s"), before);
        assert_matches_rows(&w, "s");
    }
}

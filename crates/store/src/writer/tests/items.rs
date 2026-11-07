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
}

fn message(id: &str, role: Role, text: &str) -> Message {
    text_message(id, "s", role, text, Some("end_turn"))
}

/// law: pl.read.idx-same-tx
#[test]
fn every_committed_batch_leaves_the_index_equal_to_the_rows() {
    let w = test_archive();
    w.store_reading(
        batch(
            "i1",
            0,
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
            0,
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

/// law: pl.read.page-append-stable
#[test]
fn an_append_moves_only_the_append_counter() {
    let w = test_archive();
    w.store_reading(
        batch(
            "i1",
            0,
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
                0,
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

/// law: pl.read.page-rewrite-stale
#[test]
fn a_changed_earlier_message_moves_the_rewrite_version() {
    let w = test_archive();
    w.store_reading(
        batch(
            "i1",
            0,
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
            0,
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

/// law: pl.read.page-rewrite-stale
#[test]
fn a_newly_stored_parent_link_moves_the_child_rewrite_version() {
    let w = test_archive();
    w.store_reading(
        batch(
            "child-file",
            0,
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
    let mut linked = batch("child-file", 0, 100, vec![], &["child"]);
    linked.parse[0].output.links.push(polylogue_model::Link {
        parent: sid("parent"),
        child: sid("child"),
        kind: polylogue_model::LinkKind::Fork,
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
        batch("i1", 0, 0, vec![row(0, first.clone())], &["s"]),
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
        batch("i1", 0, 100, vec![row(0, grown)], &["s"]),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    w.writer.flush().unwrap();
    let after = stored(&w, "s");
    assert_eq!(after.rewrite_version, before.rewrite_version);
    assert_eq!(after.entries.len(), before.entries.len() + 1);
    assert_matches_rows(&w, "s");
}

/// law: pl.read.page-append-stable
#[test]
fn an_event_naming_no_message_keeps_later_messages_appends() {
    let w = test_archive();
    let mut first = message("a", Role::User, "question");
    first.at = Timestamp::from_unix_ms(1_000);
    let mut opening = batch("i1", 0, 0, vec![row(10, first)], &["s"]);
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
        batch("i1", 0, 100, vec![row(110, later)], &["s"]),
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
    fixture
        .link("test:a", "test:b", "subagent", "fresh")
        .unwrap();
    fixture
        .link("test:b", "test:c", "continuation", "resume")
        .unwrap();
    fixture
        .link("test:a", "test:d", "reference", "mention")
        .unwrap();
    fixture
        .link("test:c", "test:a", "continuation", "fork")
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let a = SessionId::parse("test:a").unwrap();
    let c = SessionId::parse("test:c").unwrap();
    assert_eq!(
        reader
            .lineage(&a, crate::items::Lineage::Descendants)
            .unwrap(),
        ["test:b", "test:c"]
    );
    assert_eq!(
        reader
            .lineage(&c, crate::items::Lineage::Ancestors)
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
                0,
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
            let error = crate::items::refresh_changed(&tx, counter)
                .expect_err("a transaction cannot commit rows whose item index did not refresh");
            assert!(
                matches!(
                    error,
                    crate::StoreError::Invalid(_) | crate::StoreError::Json(_)
                ),
                "{error}"
            );
            // Dropping the same transaction production callers own rolls all
            // changed rows back together with any attempted index updates.
        }
        assert_eq!(stored(&w, "s"), before);
        assert_matches_rows(&w, "s");
    }
}

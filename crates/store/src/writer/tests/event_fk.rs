use super::*;

fn removal_steps(unrelated: usize, indexed: bool) -> u64 {
    let w = test_archive();
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    assert!(crate::schema::building(&conn).unwrap());
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name='event_after'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0,
        "the supported first-build writer currently defers this index"
    );
    let tx = conn.transaction().unwrap();
    let first = crate::first_build::FirstBuild::load(&tx).unwrap();
    let target = batch(
        "target",
        Route::Live,
        0,
        vec![row(
            0,
            text_message("gone", "target", Role::User, "gone", None),
        )],
        &["target"],
    );
    write_test_rows(&tx, &target, &target.parse[0], 1, 1, Some(&first)).unwrap();
    let mut other = batch(
        "other",
        Route::Live,
        0,
        vec![row(
            0,
            text_message("anchor", "other", Role::User, "anchor", None),
        )],
        &["other"],
    );
    let event = |session: &str, after: &str, n: usize| Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(sid(session)),
        kind: "synthetic.observation".into(),
        at: None,
        after: Some(MessageId::parse(&format!("claude-code:{after}")).unwrap()),
        range: None,
        data: serde_json::json!({"number":n}),
    };
    for n in 0..unrelated {
        other.parse[0]
            .output
            .events
            .push(event("other", "anchor", n));
    }
    write_test_rows(&tx, &other, &other.parse[0], 1, 2, Some(&first)).unwrap();
    let mut attached = batch("attached", Route::Live, 0, vec![], &[]);
    attached.parse[0]
        .output
        .events
        .push(event("target", "gone", unrelated));
    write_test_rows(&tx, &attached, &attached.parse[0], 1, 3, Some(&first)).unwrap();
    if indexed {
        tx.execute_batch("CREATE INDEX event_after ON event(after_message_pk) WHERE after_message_pk IS NOT NULL").unwrap();
    }
    assert_eq!(
        tx.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    tx.execute_batch("SAVEPOINT replacement").unwrap();
    let mut removed = batch("target", Route::Live, 0, vec![], &["target"]);
    removed.parse[0].reading_version = "synthetic=2".into();
    // Production prepares keys/choices before entering the writer transaction.
    // The fixture resolver reloads its first-build map; exclude that test-only work.
    resolve::prepare_index(&tx, &mut removed, 1, true).unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let receipt = fixture_receipt();
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    let steps = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&steps);
    tx.progress_handler(
        1,
        Some(move || {
            count.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    write_rows(
        &tx,
        &removed,
        &removed.parse[0],
        removed.parse[0].resolution.as_ref().unwrap(),
        4,
        Some(&first),
        Some(&receipt),
        &changes,
    )
    .unwrap();
    tx.progress_handler(0, None::<fn() -> bool>).unwrap();
    let measured = steps.load(Ordering::Relaxed);
    assert_eq!(
        tx.query_row(
            "SELECT count(*) FROM message WHERE id='claude-code:gone'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM event", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        (unrelated + 1) as i64
    );
    assert_eq!(tx.query_row("SELECT count(*) FROM event WHERE after_message_id='claude-code:gone' AND after_message_pk IS NULL",[],|r|r.get::<_,i64>(0)).unwrap(),1);
    assert_eq!(
        tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
    tx.execute_batch("ROLLBACK TO replacement; RELEASE replacement")
        .unwrap();
    assert_eq!(
        tx.query_row(
            "SELECT count(*) FROM message WHERE id='claude-code:gone'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(tx.query_row("SELECT count(*) FROM event WHERE after_message_id='claude-code:gone' AND after_message_pk IS NOT NULL",[],|r|r.get::<_,i64>(0)).unwrap(),1);
    measured
}

#[test]
fn first_build_message_removal_uses_keyed_event_foreign_keys() {
    let small = removal_steps(0, false);
    let large = removal_steps(8192, false);
    let indexed_small = removal_steps(0, true);
    let indexed_large = removal_steps(8192, true);
    eprintln!(
        "first-build removal VM: {small} -> {large}; indexed: {indexed_small} -> {indexed_large}"
    );
    assert!(
        large > small + 8192,
        "missing-index positive control: {small} -> {large}"
    );
    assert!(
        indexed_large <= indexed_small + 64,
        "existing event_after must bound the real writer: {indexed_small} -> {indexed_large}"
    );
}

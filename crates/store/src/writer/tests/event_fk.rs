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
        1,
        "the supported first-build writer keeps this foreign-key lookup"
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
    if !indexed {
        tx.execute_batch("DROP INDEX event_after").unwrap();
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
        &crate::text_rows::TextIds::new(&tx, false).unwrap(),
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

fn session_removal_steps(unrelated: usize, indexed: bool) -> u64 {
    let w = test_archive();
    let mut conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    assert!(crate::schema::building(&conn).unwrap());
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name='link_child'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1,
        "the first-build writer keeps its session foreign-key lookup"
    );
    let tx = conn.transaction().unwrap();
    let first = crate::first_build::FirstBuild::load(&tx).unwrap();
    let mut input = batch("sessions", Route::Live, 0, vec![], &["target", "other"]);
    for n in 0..unrelated {
        input.parse[0].output.links.push(polylogue_model::Link {
            kind: LinkKind::Fork,
            copies_prefix: false,
            how: None,
            parent: sid(&format!("p{n}")),
            child: sid("other"),
            at: None,
            call_id: None,
            evidence: "synthetic".into(),
        });
    }
    write_test_rows(&tx, &input, &input.parse[0], 1, 1, Some(&first)).unwrap();
    let target: i64 = tx
        .query_row(
            "SELECT pk FROM session WHERE id='claude-code:target'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let snapshot =
        || {
            tx.prepare(
        "SELECT json_array(pk,parent_id,child_id,child_pk,kind,how,copies_prefix,branch_point,
                           spawning_call,evidence,source_chain,input_pk) FROM link ORDER BY pk"
    ).unwrap().query_map([],|r|r.get::<_,String>(0)).unwrap()
        .collect::<rusqlite::Result<Vec<_>>>().unwrap()
        };
    let links = snapshot();
    if !indexed {
        tx.execute_batch("DROP INDEX link_child").unwrap();
    }
    tx.execute_batch("SAVEPOINT removal").unwrap();
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
    crate::session_row::remove(&tx, target, "claude-code:target", 2).unwrap();
    tx.progress_handler(0, None::<fn() -> bool>).unwrap();
    let measured = steps.load(Ordering::Relaxed);
    assert_eq!(
        tx.query_row("SELECT count(*) FROM session WHERE pk=?1", [target], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
    assert_eq!(
        snapshot(),
        links,
        "unrelated child links must remain byte-equivalent"
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
    tx.execute_batch("ROLLBACK TO removal; RELEASE removal")
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM session WHERE pk=?1", [target], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        1
    );
    assert_eq!(snapshot(), links);
    measured
}

#[test]
fn first_build_session_removal_keeps_its_link_foreign_key_lookup() {
    let small = session_removal_steps(0, false);
    let large = session_removal_steps(8192, false);
    let indexed_small = session_removal_steps(0, true);
    let indexed_large = session_removal_steps(8192, true);
    eprintln!(
        "first-build session removal VM: {small} -> {large}; indexed: {indexed_small} -> {indexed_large}"
    );
    // The existing cleanup also examines links' natural IDs. Isolate the
    // additional FK scan without claiming the entire helper is constant work.
    assert!(
        large - small > indexed_large - indexed_small + 8192,
        "the numeric child FK must not add another history scan"
    );
}

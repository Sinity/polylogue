use super::*;
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
struct Work {
    checks: u64,
    message_reads: u64,
    updates: u64,
    vm_steps: u64,
}

thread_local! {
    static WORK: Cell<Work> = Cell::new(Work::default());
}

fn observe(event: TraceEvent<'_>) {
    if let TraceEvent::Profile(statement, _) = event {
        let sql = statement.sql();
        WORK.with(|work| {
            let mut value = work.get();
            let relevant =
                if sql.starts_with("SELECT EXISTS(SELECT 1 FROM event WHERE after_message_id") {
                    value.checks += 1;
                    true
                } else if sql.starts_with("SELECT m.id,coalesce(s.seq,0)") {
                    value.message_reads += 1;
                    true
                } else if sql.starts_with("UPDATE event SET after_message_pk=") {
                    value.updates += 1;
                    true
                } else {
                    false
                };
            if relevant {
                // These fixtures disable caching, so VM steps belong to this execution.
                value.vm_steps +=
                    u64::try_from(statement.get_status(rusqlite::StatementStatus::VmStep)).unwrap();
            }
            work.set(value);
        });
    }
}

fn connection() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.set_prepared_statement_cache_capacity(0);
    conn
}

fn apply(tx: &Transaction<'_>, mut input: Batch, counter: i64) -> Work {
    resolve::prepare_index(tx, &mut input, 1, false).unwrap();
    let changes = crate::change_summary::ChangeSummary::start(tx).unwrap();
    let texts = crate::text_rows::TextIds::new(tx, false).unwrap();
    WORK.with(|work| work.set(Work::default()));
    tx.trace_v2(TraceEventCodes::SQLITE_TRACE_PROFILE, Some(observe));
    write_rows(
        tx,
        &input,
        &input.parse[0],
        rows::ReadingWrite {
            texts: &texts,
            resolution: input.parse[0].resolution.as_ref().unwrap(),
            counter,
            first_build: None,
            source_commit: Some(&fixture_commit()),
            changes: &changes,
            retraction: false,
            replacement: false,
        },
    )
    .unwrap();
    tx.trace_v2(TraceEventCodes::empty(), None);
    crate::items::refresh(tx, &changes).unwrap();
    WORK.with(Cell::get)
}

fn placed_after(session: &str, after: &str, explicit: bool) -> Event {
    Event {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        session: Some(sid(session)),
        kind: if explicit {
            "own_position"
        } else {
            "message_position"
        }
        .into(),
        at: None,
        after: Some(MessageId::parse(&format!("claude-code:{after}")).unwrap()),
        range: explicit.then_some((7, 8)),
        data: serde_json::json!({}),
    }
}

fn state(tx: &Transaction<'_>) -> Vec<(String, String, Option<String>, i64)> {
    tx.prepare(
        "SELECT e.id,e.kind,m.id,e.seq FROM event e LEFT JOIN message m
         ON m.pk=e.after_message_pk ORDER BY e.id",
    )
    .unwrap()
    .query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
    })
    .unwrap()
    .collect::<rusqlite::Result<_>>()
    .unwrap()
}

#[test]
fn fresh_messages_without_events_have_constant_placement_work() {
    let measure = |count: usize| {
        let mut conn = connection();
        let tx = conn.transaction().unwrap();
        let input = batch(
            "input",
            Route::Live,
            0,
            (0..count)
                .map(|n| {
                    row(
                        0,
                        text_message(&format!("m{n}"), "s", Role::User, "synthetic", None),
                    )
                })
                .collect(),
            &["s"],
        );
        let work = apply(&tx, input, 1);
        assert_eq!(
            tx.query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            count as i64
        );
        assert_eq!(
            tx.query_row("SELECT count(*) FROM event", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            work.message_reads, 0,
            "no message/source readback for absent anchors: {work:?}"
        );
        assert_eq!(
            work.updates, 0,
            "no event update per fresh message: {work:?}"
        );
        assert_eq!(
            work.checks, 1,
            "one application owns anchor readiness: {work:?}"
        );
        work
    };
    let small = measure(1);
    let large = measure(1024);
    eprintln!("empty-event writer anchor work: {small:?} -> {large:?}");
    assert_eq!(large.vm_steps, small.vm_steps);
    assert!(
        large.vm_steps > 0,
        "the work observer must see the real query"
    );
}

#[test]
fn late_placements_use_final_sources_across_rollback() {
    let mut conn = connection();
    let tx = conn.transaction().unwrap();
    let mut events = batch("events", Route::Live, 0, vec![], &["events"]);
    events.parse[0].output.events = vec![
        placed_after("events", "target", false),
        placed_after("events", "target", true),
    ];
    apply(&tx, events, 1);
    let before = state(&tx);
    assert!(before.iter().all(|row| row.2.is_none()));
    tx.execute_batch("SAVEPOINT messages").unwrap();
    let mut input = batch(
        "messages",
        Route::Live,
        0,
        vec![
            row(10, text_message("target", "s", Role::User, "target", None)),
            row(20, text_message("other", "s", Role::User, "other", None)),
        ],
        &["s"],
    );
    input.parse[0].order_from = Some(29);
    let work = apply(&tx, input.clone(), 2);
    let after = state(&tx);
    assert!(
        after
            .iter()
            .all(|row| row.2.as_deref() == Some("claude-code:target"))
    );
    assert_eq!(
        after
            .iter()
            .find(|row| row.1 == "message_position")
            .unwrap()
            .3,
        29
    );
    assert_eq!(
        after.iter().find(|row| row.1 == "own_position").unwrap().3,
        7
    );
    assert_eq!(
        work.updates, 1,
        "all present anchors share one final repair: {work:?}"
    );
    assert_eq!(work.message_reads, 0);
    assert_eq!(
        tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
    let event_session: i64 = tx
        .query_row(
            "SELECT pk FROM session WHERE id='claude-code:events'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let stored = crate::items::stored_entries(&tx, event_session, 0).unwrap();
    assert_eq!(
        crate::items::computed_entries(&tx, event_session).unwrap(),
        stored
    );
    tx.execute_batch("ROLLBACK TO messages; RELEASE messages")
        .unwrap();
    assert_eq!(state(&tx), before);
    apply(&tx, input, 2);
    assert_eq!(
        state(&tx),
        after,
        "retry preserves event identities and positions"
    );
}

#[test]
fn event_placement_repair_skips_unrelated_history() {
    let measure = |unrelated: usize| {
        let mut conn = connection();
        let tx = conn.transaction().unwrap();
        apply(
            &tx,
            batch(
                "old",
                Route::Live,
                0,
                vec![row(0, text_message("old", "old", Role::User, "old", None))],
                &["old"],
            ),
            1,
        );
        let mut events = batch("events", Route::Live, 0, vec![], &["events"]);
        events.parse[0].output.events = (0..unrelated)
            .map(|n| {
                let mut event = placed_after("events", "old", false);
                event.data = serde_json::json!({"number":n});
                event
            })
            .chain(std::iter::once(placed_after("events", "new", false)))
            .collect();
        apply(&tx, events, 2);
        let before = state(&tx);
        let work = apply(
            &tx,
            batch(
                "new",
                Route::Live,
                0,
                vec![row(0, text_message("new", "new", Role::User, "new", None))],
                &["new"],
            ),
            3,
        );
        let after = state(&tx);
        assert_eq!(after.len(), before.len());
        for (old, new) in before.iter().zip(&after) {
            assert_eq!((&old.0, &old.1), (&new.0, &new.1));
            if old.2.is_some() {
                assert_eq!(old, new);
            }
        }
        assert_eq!(work.updates, 1);
        work.vm_steps
    };
    let small = measure(0);
    let large = measure(4096);
    eprintln!("keyed writer anchor VM steps: {small} -> {large}");
    assert!(
        large <= small + 128,
        "unrelated event history must not add a scan: {small} -> {large}"
    );
}

use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn fixture(messages: i64, successor: bool) -> (Connection, SessionChange) {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE message(pk INTEGER PRIMARY KEY, session_pk INTEGER NOT NULL,
        id TEXT NOT NULL, session_order TEXT NOT NULL, duplicate_of INTEGER, at INTEGER);
        CREATE INDEX message_session ON message(session_pk,session_order,id);
        CREATE TABLE session_item(session_pk INTEGER NOT NULL,seq INTEGER NOT NULL,
            message_pk INTEGER,event_pk INTEGER,time_bound INTEGER,PRIMARY KEY(session_pk,seq)) WITHOUT ROWID;
        CREATE INDEX session_item_message ON session_item(message_pk) WHERE message_pk IS NOT NULL;
        CREATE INDEX session_item_event ON session_item(event_pk) WHERE event_pk IS NOT NULL;
        CREATE TABLE event(pk INTEGER PRIMARY KEY,session_pk INTEGER,after_message_pk INTEGER,at INTEGER);
        INSERT INTO message VALUES(1,1,'prefix','00000000',NULL,1);
        INSERT INTO session_item VALUES(1,0,1,NULL,1);")
        .unwrap();
    let mut change = SessionChange::default();
    for order in 1..=messages {
        // PK order differs from message order; the minimum must use the latter.
        let pk = messages + 2 - order;
        conn.execute(
            "INSERT INTO message VALUES(?1,1,?2,?3,NULL,1)",
            params![pk, format!("message-{pk}"), format!("{order:08x}")],
        )
        .unwrap();
        change.messages.insert(pk);
    }
    if successor {
        let pk = messages + 2;
        conn.execute(
            "INSERT INTO message VALUES(?1,1,'successor',?2,NULL,1)",
            params![pk, format!("{:08x}", messages + 1)],
        )
        .unwrap();
        conn.execute("INSERT INTO session_item VALUES(1,1,?1,NULL,1)", [pk])
            .unwrap();
    }
    (conn, change)
}

fn counted<T>(conn: &Connection, run: impl FnOnce() -> T) -> (T, u64) {
    conn.flush_prepared_statement_cache();
    let steps = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&steps);
    conn.progress_handler(
        1,
        Some(move || {
            count.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let result = run();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    (result, steps.load(Ordering::Relaxed))
}

fn boundary_steps(messages: i64, successor: bool) -> u64 {
    let (conn, change) = fixture(messages, successor);
    let (boundary, steps) = counted(&conn, || start(&conn, 1, &change).unwrap());
    assert!(matches!(
        boundary,
        Start::From(Anchor {
            seq: 0,
            pk: 1,
            time_bound: Some(1)
        })
    ));
    steps
}

#[test]
fn insertion_boundary_work_follows_changed_batch_size() {
    for successor in [false, true] {
        let small = boundary_steps(64, successor);
        let large = boundary_steps(1024, successor);
        assert!(
            large <= small * 24,
            "16 times the changed rows took {small} -> {large} VM steps"
        );
    }
    let (conn, change) = fixture(1024, true);
    let (rows, old_steps) = counted(&conn, || {
        let mut advanced = 0;
        for &pk in &change.messages {
            let order: (String, String) = conn
                .query_row(
                    "SELECT session_order,id FROM message WHERE pk=?1",
                    [pk],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            let mut statement = conn
                .prepare_cached(
                    "SELECT pk FROM message WHERE session_pk=?1
                AND duplicate_of IS NULL AND (session_order,id)>(?2,?3)
                ORDER BY session_order,id",
                )
                .unwrap();
            let mut rows = statement.query(params![1, order.0, order.1]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                advanced += 1;
                if !change.messages.contains(&row.get::<_, i64>(0).unwrap()) {
                    break;
                }
            }
        }
        advanced
    });
    assert_eq!(
        rows,
        1024 * 1025 / 2,
        "the original loop revisits the changed tail"
    );
    let current = boundary_steps(1024, true);
    assert!(
        old_steps > current * 8,
        "positive control {old_steps}, current {current}"
    );
}

#[test]
fn moved_and_removed_messages_keep_the_earliest_boundary() {
    let (conn, mut change) = fixture(1, true);
    conn.execute("INSERT INTO session_item VALUES(1,2,2,NULL,1)", [])
        .unwrap();
    // An existing message moved before the retained prefix.
    conn.execute("UPDATE message SET session_order='-1' WHERE pk=2", [])
        .unwrap();
    assert!(matches!(start(&conn, 1, &change).unwrap(), Start::Whole));
    // Its removed row still invalidates its previous indexed position.
    conn.execute("DELETE FROM message WHERE pk=2", []).unwrap();
    assert!(matches!(
        start(&conn, 1, &change).unwrap(),
        Start::From(Anchor { seq: 1, .. })
    ));
    change.messages.clear();
    assert!(matches!(start(&conn, 1, &change).unwrap(), Start::Links));
}

#[test]
fn reply_parent_boundary_work_follows_the_changed_batch() {
    let work = |messages| {
        let (conn, mut change) = fixture(messages, false);
        change.parents.insert("prefix".into());
        change.parents.insert("absent-parent".into());
        // In this reversed-PK reply chain, every parent except the retained
        // prefix is itself a visible message in the changed batch.
        for pk in 3..=messages + 1 {
            change.parents.insert(format!("message-{pk}"));
        }
        let (boundary, steps) = counted(&conn, || start(&conn, 1, &change).unwrap());
        assert!(matches!(boundary, Start::Whole));
        steps
    };
    let small = work(64);
    let large = work(1024);
    eprintln!("reply-parent VM steps: 64={small}, 1024={large}");
    assert!(
        large <= small * 24,
        "16 times the changed reply rows took {small} -> {large} VM steps"
    );
}

use super::*;

fn index() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.execute_batch(
        "INSERT INTO input_ref(pk, id) VALUES (1, 'input');
         INSERT INTO session(pk, id, origin, native_id, kind, changed) VALUES (1, 'claude-code:s', 'claude-code', 's', 'main', 0);",
    )
    .unwrap();
    conn
}

fn insert_event(conn: &Connection, id: &str) -> i64 {
    conn.execute(
        "INSERT INTO event(id, seq, kind, input_pk, data) VALUES (?1, 0, 'note', 1, X'00')",
        [id],
    )
    .unwrap();
    conn.last_insert_rowid()
}

#[test]
fn a_key_finds_the_row_written_under_it() {
    let conn = index();
    let keys = FirstBuild;
    assert_eq!(keys.event(&conn, "first").unwrap(), None);
    let pk = insert_event(&conn, "first");
    keys.add_event(&conn, "first", pk).unwrap();
    assert_eq!(keys.event(&conn, "first").unwrap(), Some(pk));
    assert_eq!(keys.event(&conn, "second").unwrap(), None);
}

#[test]
fn a_row_rolled_back_or_replaced_under_its_key_is_not_found() {
    let mut conn = index();
    let keys = FirstBuild;
    let tx = conn.transaction().unwrap();
    let pk = insert_event(&tx, "gone");
    keys.add_event(&tx, "gone", pk).unwrap();
    tx.rollback().unwrap();
    assert_eq!(keys.event(&conn, "gone").unwrap(), None);
    // The same row number, now holding another key.
    let reused = insert_event(&conn, "other");
    assert_eq!(reused, pk);
    keys.add_event(&conn, "other", reused).unwrap();
    assert_eq!(keys.event(&conn, "gone").unwrap(), None);
    assert_eq!(keys.event(&conn, "other").unwrap(), Some(reused));
    let again = insert_event(&conn, "gone");
    keys.add_event(&conn, "gone", again).unwrap();
    assert_eq!(keys.event(&conn, "gone").unwrap(), Some(again));
}

#[test]
fn keys_sharing_a_hash_each_find_their_own_row() {
    let conn = index();
    let first = insert_event(&conn, "one");
    let second = insert_event(&conn, "two");
    let keys = FirstBuild;
    keys.add_event(&conn, "one", first).unwrap();
    keys.add_event(&conn, "two", second).unwrap();
    let hash = blake3::hash(b"one");
    conn.execute("UPDATE build_key SET hash=?1 WHERE row=?2",
        rusqlite::params![&hash.as_bytes()[..8], second]).unwrap();
    assert_eq!(keys.event(&conn, "one").unwrap(), Some(first));
    assert_eq!(keys.event(&conn, "three").unwrap(), None);
}

#[test]
fn texts_and_messages_are_found_by_their_keys() {
    let conn = index();
    let keys = FirstBuild;
    let hash = *blake3::hash(b"walrus").as_bytes();
    conn.execute(
        "INSERT INTO text(hash, body) VALUES (?1, X'00')",
        [&hash[..]],
    )
    .unwrap();
    keys.add_text(&conn, &hash, conn.last_insert_rowid()).unwrap();
    assert_eq!(
        keys.text(&conn, &hash).unwrap(),
        Some(conn.last_insert_rowid())
    );
    assert_eq!(
        keys.text(&conn, blake3::hash(b"narwhal").as_bytes())
            .unwrap(),
        None
    );
    conn.execute(
        "INSERT INTO message(id, session_pk, order_input_pk, order_input_id, seq, role, written_by, input_pk)
         VALUES ('claude-code:m', 1, 1, 'input', 0, 'user', 'human', 1)",
        [],
    )
    .unwrap();
    keys.add_message(&conn, "claude-code:m", conn.last_insert_rowid()).unwrap();
    assert_eq!(
        keys.message(&conn, "claude-code:m").unwrap(),
        Some(conn.last_insert_rowid())
    );
    assert_eq!(keys.message(&conn, "claude-code:n").unwrap(), None);
}

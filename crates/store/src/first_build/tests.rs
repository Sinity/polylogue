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
    let keys = FirstBuild::default();
    assert_eq!(keys.event(&conn, "first").unwrap(), None);
    let pk = insert_event(&conn, "first");
    keys.add_event("first", pk);
    assert_eq!(keys.event(&conn, "first").unwrap(), Some(pk));
    assert_eq!(keys.event(&conn, "second").unwrap(), None);
}

#[test]
fn a_row_rolled_back_or_replaced_under_its_key_is_not_found() {
    let mut conn = index();
    let keys = FirstBuild::default();
    let tx = conn.transaction().unwrap();
    let pk = insert_event(&tx, "gone");
    keys.add_event("gone", pk);
    tx.rollback().unwrap();
    assert_eq!(keys.event(&conn, "gone").unwrap(), None);
    // The same row number, now holding another key.
    let reused = insert_event(&conn, "other");
    assert_eq!(reused, pk);
    keys.add_event("other", reused);
    assert_eq!(keys.event(&conn, "gone").unwrap(), None);
    assert_eq!(keys.event(&conn, "other").unwrap(), Some(reused));
    let again = insert_event(&conn, "gone");
    keys.add_event("gone", again);
    assert_eq!(keys.event(&conn, "gone").unwrap(), Some(again));
}

#[test]
fn keys_sharing_a_hash_each_find_their_own_row() {
    let conn = index();
    let first = insert_event(&conn, "one");
    let second = insert_event(&conn, "two");
    let mut rows = Rows::default();
    rows.add(7, first);
    rows.add(7, second);
    let check = "SELECT id FROM event WHERE pk = ?1";
    assert_eq!(rows.find(&conn, check, 7, b"one").unwrap(), Some(first));
    assert_eq!(rows.find(&conn, check, 7, b"two").unwrap(), Some(second));
    assert_eq!(rows.find(&conn, check, 7, b"three").unwrap(), None);
}

#[test]
fn texts_and_messages_are_found_by_their_keys() {
    let conn = index();
    let keys = FirstBuild::default();
    let hash = *blake3::hash(b"walrus").as_bytes();
    conn.execute(
        "INSERT INTO text(hash, body) VALUES (?1, X'00')",
        [&hash[..]],
    )
    .unwrap();
    keys.add_text(&hash, conn.last_insert_rowid());
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
    keys.add_message("claude-code:m", conn.last_insert_rowid());
    assert_eq!(
        keys.message(&conn, "claude-code:m").unwrap(),
        Some(conn.last_insert_rowid())
    );
    assert_eq!(keys.message(&conn, "claude-code:n").unwrap(), None);
}

#[test]
fn loading_committed_rows_restores_every_key_map() {
    let conn = index();
    insert_event(&conn, "event");
    let hash = *blake3::hash(b"kept text").as_bytes();
    conn.execute("INSERT INTO text(hash,body) VALUES (?1,X'00')", [&hash[..]])
        .unwrap();
    conn.execute("INSERT INTO message(id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk) VALUES ('claude-code:message',1,1,'input',0,'user','human',1)", []).unwrap();
    let keys = FirstBuild::load(&conn).unwrap();
    assert!(
        keys.message(&conn, "claude-code:message")
            .unwrap()
            .is_some()
    );
    assert!(keys.event(&conn, "event").unwrap().is_some());
    assert!(keys.text(&conn, &hash).unwrap().is_some());
}

#[test]
fn startup_key_snapshot_never_reads_payload_columns() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    use rusqlite::params;
    let conn = index();
    let pk = insert_event(&conn, "kept");
    conn.execute(
        "UPDATE event SET data=?1 WHERE pk=?2",
        params![vec![7u8; 1024 * 1024], pk],
    )
    .unwrap();
    let hash = *blake3::hash(b"text key").as_bytes();
    conn.execute(
        "INSERT INTO text(hash,body) VALUES (?1,?2)",
        params![hash.as_slice(), vec![8u8; 1024 * 1024]],
    )
    .unwrap();
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            column_name: "body" | "data",
            ..
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }));
    let keys = FirstBuild::load(&conn).unwrap();
    assert_eq!(keys.event(&conn, "kept").unwrap(), Some(pk));
    assert!(keys.text(&conn, &hash).unwrap().is_some());
}

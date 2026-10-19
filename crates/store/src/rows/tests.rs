use super::*;

fn archive() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    c.execute_batch(
        "CREATE TABLE note (id TEXT PRIMARY KEY, kind TEXT, body BLOB, commit_seq INTEGER);
         CREATE TABLE note_target (note_id TEXT, ref TEXT, PRIMARY KEY (note_id, ref));
         CREATE TABLE note_relation (note_id TEXT, kind TEXT, other TEXT);",
    )
    .unwrap();
    c
}

#[test]
fn a_unit_round_trips_and_restoring_it_twice_changes_nothing() {
    let a = archive();
    a.execute_batch(
        "INSERT INTO note VALUES ('n1', 'lesson', x'00ff', 7);
         INSERT INTO note_target VALUES ('n1', 'session:claude-code:s1');",
    )
    .unwrap();
    let unit = Unit {
        commit_seq: 7,
        kind: "note",
        key: "n1".into(),
    };
    let batch = read(&a, &unit).unwrap();
    assert_eq!(batch.rows, 2);
    let b = archive();
    assert_eq!(apply(&b, &batch.lines).unwrap(), 2);
    assert_eq!(apply(&b, &batch.lines).unwrap(), 0);
    let body: Vec<u8> = b
        .query_row("SELECT body FROM note WHERE id = 'n1'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(body, [0x00, 0xff]);
}

#[test]
fn a_row_for_an_unknown_table_is_refused() {
    let b = archive();
    let line = br#"{"table":"sqlite_master","row":{"name":"x"}}"#;
    assert!(apply(&b, line).is_err());
}

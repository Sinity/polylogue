//! What a live append costs in the index: rows written grow with what was
//! appended, never with how long the session already is.
use crate::fixture::writer::{WriterArchive, prose};

/// Counts every row inserted, updated or deleted in index.db's ordinary
/// tables from here on, through triggers on each table.
fn count_index_writes(w: &WriterArchive) {
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    conn.execute_batch("CREATE TABLE test_index_writes (n INTEGER NOT NULL); INSERT INTO test_index_writes VALUES (0);")
        .unwrap();
    let tables: Vec<String> = conn
        .prepare(
            "SELECT name FROM sqlite_schema WHERE type='table' AND sql NOT LIKE 'CREATE VIRTUAL%'
             AND name NOT LIKE 'sqlite_%' AND name <> 'test_index_writes'
             AND name NOT IN (SELECT name FROM sqlite_schema s WHERE EXISTS
               (SELECT 1 FROM sqlite_schema v WHERE v.type='table' AND v.sql LIKE 'CREATE VIRTUAL%'
                AND s.name LIKE v.name || '_%'))",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    for table in tables {
        for event in ["INSERT", "UPDATE", "DELETE"] {
            conn.execute_batch(&format!(
                "CREATE TRIGGER \"test_count_{table}_{event}\" AFTER {event} ON \"{table}\"
                 BEGIN UPDATE test_index_writes SET n = n + 1; END;"
            ))
            .unwrap();
        }
    }
}

fn index_writes(w: &WriterArchive) -> i64 {
    rusqlite::Connection::open(w.archive.index_db())
        .unwrap()
        .query_row("SELECT n FROM test_index_writes", [], |r| r.get(0))
        .unwrap()
}

/// Index rows one appended prompt writes to a session already `length`
/// prompts long.
fn rows_for_one_append(length: u32) -> i64 {
    let mut w = WriterArchive::new();
    let mut messages = Vec::new();
    for n in 0..length {
        messages.extend(prose("s", n, 1_000 + i64::from(n) * 20, "a question"));
    }
    w.write("s", messages);
    count_index_writes(&w);
    let before = index_writes(&w);
    w.write("s", prose("s", length, 900_000, "one more question"));
    index_writes(&w) - before
}

#[test]
fn index_rows_written_per_live_append() {
    let short = rows_for_one_append(5);
    let long = rows_for_one_append(400);
    assert!(short > 0);
    assert_eq!(
        long, short,
        "rows one append writes to a 5- and a 400-prompt session"
    );
}

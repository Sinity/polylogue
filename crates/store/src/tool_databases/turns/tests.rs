use super::*;
use rusqlite::Connection;
use serde_json::Value;
use std::fs;

fn setup() -> (tempfile::TempDir, PathBuf, PathBuf, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch(
            "PRAGMA journal_mode=WAL;
        CREATE TABLE a(id INTEGER PRIMARY KEY,n INTEGER);
        CREATE TABLE z(id INTEGER PRIMARY KEY,n INTEGER);
        WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<50)
        INSERT INTO a SELECT n,10 FROM ids;
        INSERT INTO z VALUES(1,10);",
        )
        .unwrap();
    (dir, db, out, source)
}

fn finish(captures: &mut Captures, mut turn: CaptureTurn) {
    let mut turns = 0;
    loop {
        assert!(turn.errors.is_empty(), "{:?}", turn.errors);
        if !turn.pending {
            break;
        }
        turns += 1;
        assert!(turns < 5000, "capture did not finish");
        turn = captures.turn(Vec::new(), 7).unwrap();
    }
}

#[test]
fn turns_keep_one_source_view_and_continue_without_rediscovery() {
    let (_dir, db, out, source) = setup();
    let mut captures = Captures::default();
    let first = captures
        .turn(vec![(db.clone(), out.clone(), None)], 1)
        .unwrap();
    assert!(first.pending);
    assert!(captures.paths().next().is_none());
    assert!(fs::read(&out).unwrap().is_empty());
    source
        .execute_batch("BEGIN; UPDATE a SET n=20; UPDATE z SET n=20; COMMIT;")
        .unwrap();
    finish(&mut captures, first);
    let first = fs::read_to_string(&out).unwrap();
    let values: Vec<Value> = first
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(values.iter().filter(|row| row["r"]["n"] == 10).count(), 51);
    assert!(!values.iter().any(|row| row["r"]["n"] == 20));
    assert_eq!(captures.paths().next(), Some(out.as_path()));
    let quiet = captures
        .turn(vec![(db.clone(), out.clone(), None)], 1000)
        .unwrap();
    finish(&mut captures, quiet);
    let changed = captures.turn(vec![(db, out.clone(), None)], 3).unwrap();
    assert!(changed.pending);
    assert_eq!(fs::read_to_string(&out).unwrap(), first);
    finish(&mut captures, changed);
    let after = fs::read_to_string(&out).unwrap();
    assert!(after.starts_with(&first));
    assert_eq!(
        after
            .lines()
            .filter(|line| serde_json::from_str::<Value>(line).unwrap()["r"]["n"] == 20)
            .count(),
        51
    );
}

#[test]
fn dropping_paused_capture_releases_source_view_and_publishes_nothing() {
    let (_dir, db, out, source) = setup();
    let mut captures = Captures::default();
    assert!(
        captures
            .turn(vec![(db.clone(), out.clone(), None)], 1)
            .unwrap()
            .pending
    );
    source.execute_batch("UPDATE a SET n=20").unwrap();
    drop(captures);
    assert_eq!(fs::read(&out).unwrap(), b"");
    source.busy_timeout(std::time::Duration::ZERO).unwrap();
    let busy: i64 = source
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))
        .unwrap();
    assert_eq!(busy, 0);
    let mut fresh = Captures::default();
    let turn = fresh.turn(vec![(db, out.clone(), None)], 1000).unwrap();
    finish(&mut fresh, turn);
    assert_eq!(
        fs::read_to_string(&out)
            .unwrap()
            .lines()
            .filter(|line| serde_json::from_str::<Value>(line).unwrap()["r"]["n"] == 20)
            .count(),
        50
    );
}

#[test]
fn an_existing_batch_cannot_be_displaced_by_new_discovery() {
    let (dir, db, out, _source) = setup();
    let second = dir.path().join("second.sqlite");
    let second_out = dir.path().join("second.jsonl");
    Connection::open(&second)
        .unwrap()
        .execute_batch("CREATE TABLE t(n); INSERT INTO t VALUES(42)")
        .unwrap();
    let mut captures = Captures::default();
    let mut turn = captures
        .turn(
            vec![
                (db.clone(), out.clone(), None),
                (second, second_out.clone(), None),
            ],
            1,
        )
        .unwrap();
    while turn.pending {
        assert!(turn.errors.is_empty());
        // A newly supplied batch does not replace the remaining second DB.
        turn = captures
            .turn(vec![(db.clone(), out.clone(), None)], 7)
            .unwrap();
    }
    assert!(fs::read_to_string(second_out).unwrap().contains("42"));
}

#[test]
fn revalidation_withholds_an_earlier_output_until_its_bytes_are_verified() {
    let (_dir, db, out, source) = setup();
    source
        .execute_batch(
            "ALTER TABLE a ADD COLUMN text TEXT;
        UPDATE a SET text=printf('%04000d',id)",
        )
        .unwrap();
    let mut captures = Captures::default();
    let turn = captures
        .turn(vec![(db.clone(), out.clone(), None)], 1000)
        .unwrap();
    finish(&mut captures, turn);
    assert_eq!(captures.paths().next(), Some(out.as_path()));
    let mut changed = fs::read(&out).unwrap();
    changed[0] = b'!';
    changed.extend_from_slice(b"unpublished tail");
    fs::write(&out, &changed).unwrap();
    let mut turn = captures.turn(vec![(db, out.clone(), None)], 1).unwrap();
    assert!(turn.pending);
    assert!(captures.paths().next().is_none());
    let mut errors = Vec::new();
    loop {
        errors.extend(turn.errors);
        if !turn.pending {
            break;
        }
        turn = captures.turn(Vec::new(), 1).unwrap();
    }
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("completed-prefix hash differs"));
    assert!(captures.paths().next().is_none());
    assert_eq!(fs::read(&out).unwrap(), changed);
}

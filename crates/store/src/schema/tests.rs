use super::*;

#[test]
fn every_schema_applies_to_an_empty_database() {
    for (name, create) in [
        ("archive", ensure_archive as fn(&Connection) -> Result<()>),
        ("sinex", ensure_sinex),
        ("index", ensure_index),
        ("embeddings", ensure_embeddings),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
        create(&conn).unwrap_or_else(|e| panic!("{name} again: {e}"));
    }
}

/// `archive.sql` as version 1 declared it. The archive's migrations start
/// from these tables.
const ARCHIVE_1: &str = include_str!("tests/archive_1.sql");

/// An archive at `version`: version 1's tables and the migrations after it.
/// A test for a migration fills an archive at the version before, migrates
/// it and checks that every row is still there.
pub(crate) fn archive_at(version: u32) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(ARCHIVE_1).unwrap();
    for step in &ARCHIVE_MIGRATIONS[..version as usize - 1] {
        conn.execute_batch(step).unwrap();
    }
    set_version(&conn, version).unwrap();
    conn
}

#[test]
fn version_1_and_the_migrations_give_the_declared_archive() {
    for from in 1..=ARCHIVE_VERSION {
        let conn = archive_at(from);
        ensure_archive(&conn).unwrap_or_else(|e| panic!("from version {from}: {e}"));
        assert_eq!(version(&conn).unwrap(), ARCHIVE_VERSION);
        assert!(
            same_schema(&conn, &[ARCHIVE]).unwrap(),
            "from version {from}"
        );
    }
}

#[test]
fn migration_2_adds_backfill_jobs_and_keeps_every_row() {
    let conn = archive_at(1);
    conn.execute_batch(
        "INSERT INTO setting(id, machine, key, value, author, set_at, commit_seq)
             VALUES ('s1', '*', 'browser.actions', 'on', 'operator', 1, 1);
         INSERT INTO chat_action(id, request_key, request_hash, request_json, stage_only,
             needs_approval, created_at, commit_seq)
             VALUES ('a1', 'k1', x'00', '{}', 1, 0, 2, 2);
         INSERT INTO chat_action_event(action, n, at, kind, phase, commit_seq)
             VALUES ('a1', 1, 2, 'queued', 'queued', 2);",
    )
    .unwrap();
    let rows = |conn: &Connection| -> Vec<String> {
        ["setting", "chat_action", "chat_action_event"]
            .iter()
            .flat_map(|table| {
                conn.prepare(&format!("SELECT * FROM {table}"))
                    .unwrap()
                    .query_map([], |r| {
                        Ok((0..r.as_ref().column_count())
                            .map(|i| format!("{:?}", r.get_ref(i).unwrap()))
                            .collect::<Vec<_>>()
                            .join("|"))
                    })
                    .unwrap()
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .unwrap()
            })
            .collect()
    };
    let before = rows(&conn);
    ensure_archive(&conn).unwrap();
    assert!(version(&conn).unwrap() >= 2);
    assert_eq!(rows(&conn), before);
    conn.execute_batch(
        "INSERT INTO backfill_job(id, provider, since, created_at, commit_seq)
             VALUES ('j1', 'chatgpt', NULL, 3, 3);",
    )
    .unwrap();
}

#[test]
fn migration_3_keeps_every_request_and_leaves_old_ones_without_their_content() {
    let conn = archive_at(2);
    conn.execute_batch(
        "INSERT INTO client_request(id, content_hash, created_at) VALUES ('r1', x'01', 5);",
    )
    .unwrap();
    ensure_archive(&conn).unwrap();
    assert!(version(&conn).unwrap() >= 3);
    let row: (String, Vec<u8>, i64, Option<Vec<u8>>, i64) = conn
        .query_row(
            "SELECT id, content_hash, created_at, request, commit_seq FROM client_request",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(row, ("r1".into(), vec![1], 5, None, 0));
}

#[test]
fn migration_4_counts_the_rows_it_finds_and_keeps_every_row() {
    let conn = archive_at(3);
    conn.execute_batch(
        "INSERT INTO input(id, format) VALUES ('i1', 'f'), ('i2', 'f');
         INSERT INTO revision(input, number, produced, chain, length, commit_seq)
             VALUES ('i1', 1, 'first', 1, 10, 1), ('i1', 2, 'appended', 1, 20, 2),
                    ('i2', 1, 'imported', 1, 5, 3);
         INSERT INTO blob(hash, size, stored_size, place, encoding, state)
             VALUES (x'01', 100, 50, 'file', 'plain', 'present'),
                    (x'02', 7, 7, 'file', 'plain', 'writing');",
    )
    .unwrap();
    let rows = |conn: &Connection| -> Vec<String> {
        ["input", "revision", "blob"]
            .iter()
            .flat_map(|table| {
                conn.prepare(&format!("SELECT * FROM {table}"))
                    .unwrap()
                    .query_map([], |r| {
                        Ok((0..r.as_ref().column_count())
                            .map(|i| format!("{:?}", r.get_ref(i).unwrap()))
                            .collect::<Vec<_>>()
                            .join("|"))
                    })
                    .unwrap()
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .unwrap()
            })
            .collect()
    };
    let before = rows(&conn);
    ensure_archive(&conn).unwrap();
    assert!(version(&conn).unwrap() >= 4);
    assert_eq!(rows(&conn), before);
    let counts = |conn: &Connection| -> Vec<(String, i64)> {
        conn.prepare("SELECT name, n FROM row_count ORDER BY name")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    let expect = |input: i64, revision: i64, stored: i64| {
        vec![
            ("input".to_owned(), input),
            ("revision".to_owned(), revision),
            ("stored_bytes".to_owned(), stored),
        ]
    };
    assert_eq!(counts(&conn), expect(2, 3, 100));
    // The triggers keep the counts from here on.
    conn.execute_batch(
        "UPDATE blob SET state = 'present' WHERE hash = x'02';
         INSERT INTO input(id, format) VALUES ('i3', 'f');
         DELETE FROM revision WHERE input = 'i2';
         UPDATE blob SET state = 'deleting' WHERE hash = x'01';",
    )
    .unwrap();
    assert_eq!(counts(&conn), expect(3, 2, 7));
    conn.execute_batch("DELETE FROM blob WHERE hash = x'02';")
        .unwrap();
    assert_eq!(counts(&conn), expect(3, 2, 0));
}

const TWO_TABLES: &str = "CREATE TABLE kept(value TEXT NOT NULL) STRICT;
    CREATE TABLE added(value TEXT NOT NULL) STRICT;";

fn version_1_with_rows() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE kept(value TEXT NOT NULL) STRICT;
         INSERT INTO kept VALUES ('one'), ('two');
         PRAGMA user_version = 1;",
    )
    .unwrap();
    conn
}

fn rows(conn: &Connection) -> Vec<String> {
    conn.prepare("SELECT value FROM kept ORDER BY value")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn migrations_bring_an_older_database_forward_and_keep_its_rows() {
    let conn = version_1_with_rows();
    let steps = ["CREATE TABLE added(value TEXT NOT NULL) STRICT;"];
    ensure(&conn, "archive", &[TWO_TABLES], &steps).unwrap();
    assert_eq!(version(&conn).unwrap(), 2);
    assert_eq!(rows(&conn), ["one", "two"]);
    ensure(&conn, "archive", &[TWO_TABLES], &steps).unwrap();
    // A table rebuilt under a new name compares as declared: SQLite quotes
    // the renamed table's name in its declaration.
    let conn = version_1_with_rows();
    let steps = ["CREATE TABLE added(value TEXT NOT NULL) STRICT;
        CREATE TABLE kept_new(value TEXT NOT NULL) STRICT;
        INSERT INTO kept_new SELECT value FROM kept;
        DROP TABLE kept;
        ALTER TABLE kept_new RENAME TO kept;"];
    ensure(&conn, "archive", &[TWO_TABLES], &steps).unwrap();
    assert_eq!(rows(&conn), ["one", "two"]);
}

#[test]
fn a_failed_or_wrong_migration_changes_nothing() {
    for (step, wrong_tables) in [
        (
            "CREATE TABLE added(value TEXT NOT NULL) STRICT; INSERT INTO missing VALUES (1);",
            false,
        ),
        (
            "CREATE TABLE added(other TEXT) STRICT; DELETE FROM kept;",
            true,
        ),
    ] {
        let conn = version_1_with_rows();
        let result = ensure(&conn, "archive", &[TWO_TABLES], &[step]);
        if wrong_tables {
            assert!(matches!(
                result,
                Err(StoreError::NoMigration {
                    database: "archive",
                    found: 1
                })
            ));
        } else {
            assert!(result.is_err());
        }
        assert_eq!(version(&conn).unwrap(), 1);
        assert_eq!(rows(&conn), ["one", "two"]);
        assert!(!same_schema(&conn, &[TWO_TABLES]).unwrap());
    }
}

#[test]
fn a_newer_database_or_one_with_unknown_tables_is_refused() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    set_version(&conn, ARCHIVE_VERSION + 1).unwrap();
    assert!(matches!(
        ensure_archive(&conn),
        Err(StoreError::NewerSchema { database: "archive", found, current })
            if found == ARCHIVE_VERSION + 1 && current == ARCHIVE_VERSION
    ));
    let conn = Connection::open_in_memory().unwrap();
    ensure_embeddings(&conn).unwrap();
    set_version(&conn, 2).unwrap();
    assert!(matches!(
        ensure_embeddings(&conn),
        Err(StoreError::NewerSchema {
            database: "embeddings",
            ..
        })
    ));
    // The current version with a table missing, as an archive written by a
    // build that declared the same version with other tables.
    for change in [
        "DROP TABLE excised",
        "ALTER TABLE note ADD COLUMN synthetic TEXT",
    ] {
        let conn = Connection::open_in_memory().unwrap();
        ensure_archive(&conn).unwrap();
        conn.execute_batch(change).unwrap();
        assert!(
            matches!(
                ensure_archive(&conn),
                Err(StoreError::NoMigration {
                    database: "archive",
                    ..
                })
            ),
            "{change}"
        );
    }
}

#[test]
fn comments_and_spacing_in_the_declarations_do_not_change_the_schema() {
    let respaced: String = ARCHIVE
        .lines()
        .map(|line| {
            format!(
                "{}   -- synthetic remark, 'quoted'\n\n",
                line.split_whitespace().collect::<Vec<_>>().join("  ")
            )
        })
        .collect();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&respaced).unwrap();
    set_version(&conn, ARCHIVE_VERSION).unwrap();
    ensure_archive(&conn).unwrap();
}

#[test]
fn index_format_changes_require_a_version_and_digest_update() {
    assert_eq!(
        (
            INDEX_VERSION,
            blake3::hash(INDEX.as_bytes()).to_hex().to_string(),
            blake3::hash(include_bytes!("../encoding/rows.dict"))
                .to_hex()
                .to_string()
        ),
        (
            49,
            "45132588d9fe71c1fc2c45895574621208673c6ea4ce117512a93803b5f94013".to_owned(),
            "e27529a9578815c3ee02058decc971ecf7c7125774a6d0e9b885eddd8a928490".to_owned()
        ),
        "record current source digests; bump INDEX_VERSION for schema or row-format changes, not comments"
    );
}

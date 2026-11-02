use super::*;

#[test]
fn every_schema_applies_to_an_empty_database() {
    for (name, create) in [
        ("archive", ensure_archive as fn(&Connection) -> Result<()>),
        ("primary", ensure_primary_index),
        ("index", ensure_index),
        ("embeddings", ensure_embeddings),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
        create(&conn).unwrap_or_else(|e| panic!("{name} again: {e}"));
    }
}

/// law: one-baseline
#[test]
fn every_database_starts_at_its_one_baseline() {
    assert_eq!(ARCHIVE_VERSION, 1);
    assert_eq!(INDEX_VERSION, 1);
    assert!(ARCHIVE_MIGRATIONS.is_empty());
    for create in [
        ensure_archive as fn(&Connection) -> Result<()>,
        ensure_index,
        ensure_primary_index,
        ensure_embeddings,
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), 1);
    }
}

/// law: one-baseline
#[test]
fn an_index_at_another_version_is_refused_without_removing_rows() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_index(&conn).unwrap();
    conn.execute("INSERT INTO input_ref(id) VALUES ('synthetic')", [])
        .unwrap();
    // An index written by an older build of another lineage.
    set_version(&conn, 17).unwrap();
    assert!(matches!(
        ensure_index(&conn),
        Err(StoreError::IndexVersion {
            found: 17,
            wanted: 1
        })
    ));
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE input_ref(pk INTEGER PRIMARY KEY, id TEXT) STRICT;
         INSERT INTO input_ref(id) VALUES ('synthetic');
         PRAGMA user_version = 1;",
    )
    .unwrap();
    // No baseline format identity: an index written by an earlier Polylogue.
    assert!(matches!(
        ensure_index(&conn),
        Err(StoreError::OlderFormat { database: "index" })
    ));
    assert_eq!(
        ensure_index(&conn).unwrap_err().to_string(),
        "older format: import with `polylogue import --from-archive PATH`"
    );
    assert_eq!(
        conn.query_row("SELECT id FROM input_ref", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "synthetic"
    );
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
        "DROP TABLE revision",
        "ALTER TABLE assertion ADD COLUMN synthetic TEXT",
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
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    ensure_archive(&conn).unwrap();
}

#[test]
fn read_indexes_are_added_in_place_and_survive_reopen() {
    let conn = Connection::open_in_memory().unwrap();
    base_index(&conn).unwrap();
    set_version(&conn, INDEX_VERSION).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    conn.execute("INSERT INTO input_ref(id) VALUES ('synthetic')", [])
        .unwrap();
    assert!(same_index_schema_in(&conn, false).unwrap());
    ensure_index(&conn).unwrap();
    let base = Connection::open_in_memory().unwrap();
    base_index(&base).unwrap();
    let base_shape = shape(&base).unwrap();
    base.execute_batch(READ_INDEXES).unwrap();
    let expected = shape(&base).unwrap();
    let read_indexes: Vec<_> = shape(&conn)
        .unwrap()
        .into_iter()
        .filter(|item| item.0 == "index" && !base_shape.contains(item))
        .map(|item| item.1)
        .collect();
    assert!(!read_indexes.is_empty());
    for name in read_indexes {
        conn.execute_batch(&format!("DROP INDEX {name}")).unwrap();
        assert!(
            same_index_schema_in(&conn, false).unwrap(),
            "missing {name}"
        );
        ensure_index(&conn).unwrap();
        assert_eq!(shape(&conn).unwrap(), expected);
    }
    assert_eq!(shape(&conn).unwrap(), expected);
    assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
    assert_eq!(
        conn.query_row("SELECT id FROM input_ref", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "synthetic"
    );
    conn.execute_batch("DROP INDEX event_at; CREATE INDEX event_at ON event(kind)")
        .unwrap();
    assert!(!same_index_schema_in(&conn, false).unwrap());
}

/// law: one-baseline
#[test]
fn an_archive_written_before_the_baseline_is_an_older_format_and_stays_as_it_is() {
    // An archive with tables and a version but no format identity, as every
    // earlier Polylogue wrote it (master's archives are at version 4).
    for found in [1, 4] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "CREATE TABLE input(id TEXT PRIMARY KEY) STRICT;
             INSERT INTO input VALUES ('synthetic');
             PRAGMA user_version = {found};"
        ))
        .unwrap();
        let error = ensure_archive(&conn).unwrap_err();
        assert!(
            matches!(
                error,
                StoreError::OlderFormat {
                    database: "archive"
                }
            ),
            "{error}"
        );
        assert_eq!(
            error.to_string(),
            "older format: import with `polylogue import --from-archive PATH`"
        );
        assert_eq!(version(&conn).unwrap(), found);
        assert_eq!(
            conn.query_row("SELECT id FROM input", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "synthetic"
        );
    }
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    assert_eq!(format_id(&conn).unwrap(), FORMAT_ID);
}

/// The conversion layer for the rewrite's own retired formats is gone: no
/// file of it may survive in the tree, compiled or not.
#[test]
fn no_retired_format_converter_remains_in_the_tree() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for path in [
        "crates/store/src/assertions/conversion.rs",
        "crates/store/src/assertions/conversion",
        "crates/store/src/primary/authored/conserve.rs",
        "crates/sinex/src/client/confirmed/source3.rs",
        "crates/sinex/src/client/confirmed/source3",
        "crates/sinex/src/client/retained.rs",
        "crates/sinex/src/client/retained",
        "crates/sinex/src/producer/requests/wire.rs",
        "crates/model/src/historical.rs",
    ] {
        assert!(!workspace.join(path).exists(), "{path} remains");
    }
}

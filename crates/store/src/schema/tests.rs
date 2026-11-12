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

#[test]
fn every_database_starts_at_its_current_version() {
    assert_eq!(ARCHIVE_VERSION, 2);
    assert_eq!(INDEX_VERSION, 2);
    for (create, expected) in [
        (
            ensure_archive as fn(&Connection) -> Result<()>,
            ARCHIVE_VERSION,
        ),
        (ensure_index, INDEX_VERSION),
        (ensure_primary_index, INDEX_VERSION),
        (ensure_embeddings, 1),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), expected);
    }
}

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
            wanted: INDEX_VERSION
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
fn open_indexes_are_added_in_place_and_survive_reopen() {
    let conn = Connection::open_in_memory().unwrap();
    base_index(&conn).unwrap();
    set_version(&conn, INDEX_VERSION).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    conn.execute("INSERT INTO input_ref(id) VALUES ('synthetic')", [])
        .unwrap();
    // Views are part of the schema; open indexes are recreated before writing.
    conn.execute_batch(OPEN_INDEXES).unwrap();
    assert!(same_index_schema_in(&conn, false).unwrap());
    ensure_index(&conn).unwrap();
    let base = Connection::open_in_memory().unwrap();
    base_index(&base).unwrap();
    let base_shape = shape(&base).unwrap();
    base.execute_batch(OPEN_INDEXES).unwrap();
    let expected = shape(&base).unwrap();
    let open_indexes: Vec<_> = shape(&conn)
        .unwrap()
        .into_iter()
        .filter(|item| item.0 == "index" && !base_shape.contains(item))
        .map(|item| item.1)
        .collect();
    assert!(!open_indexes.is_empty());
    for name in open_indexes {
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

#[test]
fn recent_read_indexes_wait_for_first_build_completion_on_reopen() {
    let conn = Connection::open_in_memory().unwrap();
    create_index(&conn).unwrap();
    conn.execute_batch("DROP INDEX session_recent; DROP INDEX message_recent;")
        .unwrap();
    let recent = || {
        conn.query_row(
            "SELECT count(*) FROM sqlite_master
             WHERE name IN ('session_recent','message_recent')",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
    };
    ensure_index(&conn).unwrap();
    assert_eq!(recent(), 0);
    conn.execute("DELETE FROM meta WHERE key='building'", [])
        .unwrap();
    ensure_index(&conn).unwrap();
    assert_eq!(recent(), 2);
    ensure_index(&conn).unwrap();
    assert_eq!(recent(), 2);
}

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

/// An unversioned file that already holds tables is not ours to create into:
/// it is refused before it takes the format identity or any table.
#[test]
fn an_unversioned_file_with_tables_is_refused_unchanged() {
    for (name, create) in [
        ("archive", ensure_archive as fn(&Connection) -> Result<()>),
        ("index", ensure_index),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE unrelated(x INTEGER) STRICT; INSERT INTO unrelated VALUES (1);",
        )
        .unwrap();
        let error = create(&conn).unwrap_err();
        assert!(
            matches!(error, StoreError::NoMigration { found: 0, .. }),
            "{name}: {error}"
        );
        assert_eq!(format_id(&conn).unwrap(), 0, "{name}");
        assert_eq!(version(&conn).unwrap(), 0, "{name}");
        assert_eq!(
            shape(&conn).unwrap(),
            vec![(
                "table".to_owned(),
                "unrelated".to_owned(),
                statement("CREATE TABLE unrelated(x INTEGER) STRICT")
            )],
            "{name}"
        );
    }
}

#[test]
fn parse_standings_require_their_length_and_reason() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_index(&conn).unwrap();
    conn.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(1,'waiting','synthetic'),(2,'done','synthetic'),(3,'collected','synthetic');
        INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,at_length) VALUES
          (1,1,'v1','waiting',40,100),(2,1,'v1','done',90,NULL),(3,1,'v1','no_parser_yet',30,NULL);").unwrap();
    assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
    let rows: Vec<(String, i64, Option<i64>)> = conn
        .prepare("SELECT status,parsed_upto,at_length FROM parse_state ORDER BY input_pk")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        rows,
        [
            ("waiting".to_owned(), 40, Some(100)),
            ("done".to_owned(), 90, None),
            ("no_parser_yet".to_owned(), 30, None),
        ]
    );
    let detail: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('parse_state') WHERE name='detail')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!detail);
    // A waiting status always names its length.
    assert!(
        conn.execute("UPDATE parse_state SET at_length=NULL WHERE input_pk=1", [])
            .is_err()
    );
    // Every designed standing is storable, and a refusal or failure keeps its reason.
    for (status, reason) in [
        ("refused", "synthetic reason"),
        ("failed", "synthetic error"),
    ] {
        conn.execute(
            "UPDATE parse_state SET status=?1,reason=?2,at_length=90 WHERE input_pk=2",
            [status, reason],
        )
        .unwrap();
        assert!(
            conn.execute("UPDATE parse_state SET reason=NULL WHERE input_pk=2", [])
                .is_err()
        );
    }
    assert!(
        conn.execute(
            "UPDATE parse_state SET status='done',reason='stray',at_length=NULL WHERE input_pk=2",
            []
        )
        .is_err()
    );
}

mod context_cleanup;
mod embeddings;

fn archive_before_unknown_time(conn: &Connection) {
    let previous = ARCHIVE.replacen(
        crate::assertions::SCHEMA,
        include_str!("assertions_before_unknown_time.sql"),
        1,
    );
    conn.execute_batch(&previous).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    set_version(conn, 1).unwrap();
    conn.execute_batch("INSERT INTO meta VALUES ('write_counter','19');
        INSERT INTO machine VALUES ('synthetic-machine','host',12);
        INSERT INTO input VALUES ('synthetic-input','kept-text');
        INSERT INTO assertion(rowid,id,root,kind,value,author,via,machine,created_at,commit_seq)
        VALUES(17,'epoch','epoch','note','\"Known epoch\"','unattributed','import','host',0,11),
              (29,'later','later','note','\"Known later\"','operator','cli','host',1234,19);
        INSERT INTO link(record,role,ref) VALUES ('epoch','about','repo:synthetic'),('later','about','repo:synthetic');
        INSERT INTO revision_state VALUES ('epoch','epoch',1,0),('later','later',1,0);
        INSERT INTO assertion_slot VALUES (zeroblob(32),'epoch');").unwrap();
}

fn durable_rows(
    conn: &Connection,
) -> std::collections::BTreeMap<String, Vec<Vec<rusqlite::types::Value>>> {
    let names:Vec<String> = conn.prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name<>'assertion_import' ORDER BY name").unwrap()
        .query_map([],|r|r.get(0)).unwrap().collect::<rusqlite::Result<_>>().unwrap();
    names
        .into_iter()
        .map(|name| {
            let mut statement = conn
                .prepare(&format!("SELECT * FROM \"{name}\" ORDER BY 1"))
                .unwrap();
            let count = statement.column_count();
            let rows = statement
                .query_map([], |r| (0..count).map(|i| r.get(i)).collect())
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            (name, rows)
        })
        .collect()
}

#[test]
fn register_transition_preserves_all_durable_rows_and_known_times_and_repeats() {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    archive_before_unknown_time(&conn);
    let before = durable_rows(&conn);
    let changes = conn.total_changes();
    ensure_archive(&conn).unwrap();
    assert_eq!(version(&conn).unwrap(), 2);
    assert_eq!(conn.total_changes(), changes);
    assert_eq!(durable_rows(&conn), before);
    assert_eq!(
        conn.query_row(
            "SELECT group_concat(rowid) FROM assertion ORDER BY rowid",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "17,29"
    );
    assert!(
        conn.pragma_query_value(None, "foreign_keys", |r| r.get::<_, bool>(0))
            .unwrap()
    );
    assert!(
        !conn
            .prepare("PRAGMA foreign_key_check")
            .unwrap()
            .exists([])
            .unwrap()
    );
    ensure_archive(&conn).unwrap();
    assert_eq!(durable_rows(&conn), before);
    assert_eq!(conn.total_changes(), changes);
    conn.execute("INSERT INTO assertion(id,root,kind,value,author,via,machine,created_at,commit_seq) VALUES ('undated','undated','rubric','{}','unattributed','import','host',NULL,20)",[]).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT created_at FROM assertion WHERE id='undated'",
            [],
            |r| r.get::<_, Option<i64>>(0)
        )
        .unwrap(),
        None
    );
}

#[cfg(feature = "faults")]
#[test]
fn register_transition_failure_rolls_back_rows_shape_version_and_foreign_keys() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("authored-transition-failure.db");
    let conn = Connection::open(&file).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    archive_before_unknown_time(&conn);
    let before = durable_rows(&conn);
    let old_shape = shape(&conn).unwrap();
    crate::fault::arm(
        "archive.authored_time_transition",
        "authored-transition-failure.db",
        crate::fault::Action::Fail,
        0,
        1,
    );
    assert!(ensure_archive(&conn).is_err());
    assert_eq!(version(&conn).unwrap(), 1);
    assert_eq!(shape(&conn).unwrap(), old_shape);
    assert_eq!(durable_rows(&conn), before);
    assert!(
        conn.pragma_query_value(None, "foreign_keys", |r| r.get::<_, bool>(0))
            .unwrap()
    );
    assert!(
        !conn
            .prepare("PRAGMA foreign_key_check")
            .unwrap()
            .exists([])
            .unwrap()
    );
    ensure_archive(&conn).unwrap();
    assert_eq!(durable_rows(&conn), before);
}

#[test]
fn ordinary_register_transition_keeps_partial_index_cursor_checkpoint_and_inode() {
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    std::fs::create_dir(&root).unwrap();
    let mode = b"{\"mode\":\"standalone\"}\n";
    std::fs::write(root.join("meta.json"), mode).unwrap();
    let source = Connection::open(root.join("archive.db")).unwrap();
    archive_before_unknown_time(&source);
    let originals = durable_rows(&source);
    drop(source);
    let path = root.join("index.db");
    let index = Connection::open(&path).unwrap();
    ensure_index(&index).unwrap();
    index.execute_batch("INSERT INTO input_ref(pk,id) VALUES(1,'synthetic-input');
      INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,records,used,commit_seq,checkpoint)
      VALUES(1,7,'synthetic-reading','parsing',42,3,3,19,x'0001ff');").unwrap();
    let cursor = |c: &Connection| {
        c.query_row("SELECT chain,reading_version,status,parsed_upto,records,used,commit_seq,checkpoint FROM parse_state WHERE input_pk=1",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?,r.get::<_,i64>(5)?,r.get::<_,i64>(6)?,r.get::<_,Vec<u8>>(7)?))).unwrap()
    };
    let before = cursor(&index);
    let inode = std::fs::metadata(&path).unwrap().ino();
    drop(index);
    let (archive, lock) = crate::Archive::open_or_create(&root).unwrap();
    let source = Connection::open(archive.archive_db()).unwrap();
    assert_eq!(version(&source).unwrap(), ARCHIVE_VERSION);
    assert_eq!(durable_rows(&source), originals);
    let index = Connection::open(archive.index_db()).unwrap();
    assert_eq!(cursor(&index), before);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    assert_eq!(std::fs::read(root.join("meta.json")).unwrap(), mode);
    drop(index);
    drop(source);
    drop(lock);
    drop(archive);
    let (archive, _lock) = crate::Archive::open_or_create(&root).unwrap();
    assert_eq!(
        cursor(&Connection::open(archive.index_db()).unwrap()),
        before
    );
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
}

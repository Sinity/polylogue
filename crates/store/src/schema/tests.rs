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
fn every_database_starts_at_its_current_version() {
    assert_eq!(ARCHIVE_VERSION, 4);
    assert_eq!(INDEX_VERSION, 5);
    assert_eq!(ARCHIVE_MIGRATIONS.len(), 3);
    for (create, expected) in [
        (
            ensure_archive as fn(&Connection) -> Result<()>,
            ARCHIVE_VERSION,
        ),
        (ensure_index, INDEX_VERSION),
        (ensure_primary_index, INDEX_VERSION),
        (ensure_embeddings, 2),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), expected);
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
    let steps = [Migration::Sql(
        "CREATE TABLE added(value TEXT NOT NULL) STRICT;",
    )];
    ensure(&conn, "archive", &[TWO_TABLES], &steps).unwrap();
    assert_eq!(version(&conn).unwrap(), 2);
    assert_eq!(rows(&conn), ["one", "two"]);
    ensure(&conn, "archive", &[TWO_TABLES], &steps).unwrap();
    // A table rebuilt under a new name compares as declared: SQLite quotes
    // the renamed table's name in its declaration.
    let conn = version_1_with_rows();
    let steps = [Migration::Sql(
        "CREATE TABLE added(value TEXT NOT NULL) STRICT;
        CREATE TABLE kept_new(value TEXT NOT NULL) STRICT;
        INSERT INTO kept_new SELECT value FROM kept;
        DROP TABLE kept;
        ALTER TABLE kept_new RENAME TO kept;",
    )];
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
        let result = ensure(&conn, "archive", &[TWO_TABLES], &[Migration::Sql(step)]);
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
    set_version(&conn, 3).unwrap();
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
    // Read views are part of the schema; only read indexes may be absent.
    conn.execute_batch(READ_INDEXES).unwrap();
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

#[test]
fn old_kept_output_upgrade_marks_missing_policy_without_changing_stored_values() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    let raw = r#"{ "unit":"sessions", "columns":[], "rows":[{"text":"\u03bb","value":1.00}] }"#;
    let evolved = polylogue_model::kept::evolve_body(raw).unwrap();
    assert!(evolved.contains(r#""rows":[{"text":"\u03bb","value":1.00}]"#));
    assert_eq!(
        polylogue_model::kept::evolve_body(&evolved).unwrap(),
        evolved
    );
    let original = serde_json::json!({"unit":"sessions","shape":"rows","columns":[{"name":"cwd","kind":"text"}],"rows":[{"cwd":"/work/module"}],"identities":[["test:one"]]});
    conn.execute("INSERT INTO kept_result(id,query,ran_at,read_at,rows_json) VALUES ('old','sessions',0,'{}',?1)", [original.to_string()]).unwrap();
    conn.execute_batch("DROP INDEX query_commit_seq; ALTER TABLE query DROP COLUMN commit_seq;")
        .unwrap();
    set_version(&conn, 2).unwrap();
    ensure_archive(&conn).unwrap();
    let stored: String = conn
        .query_row(
            "SELECT rows_json FROM kept_result WHERE id='old'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let mut upgraded: serde_json::Value = serde_json::from_str(&stored).unwrap();
    let metadata = upgraded
        .as_object_mut()
        .unwrap()
        .remove("output_metadata")
        .unwrap();
    assert_eq!(metadata["version"], 2);
    assert_eq!(metadata["rows"]["kind"], "unproven");
    assert_eq!(upgraded, original);
    ensure_archive(&conn).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT rows_json FROM kept_result WHERE id='old'",
            [],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        stored
    );
}

#[test]
fn primary_kept_output_upgrade_preserves_stored_values() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_primary_index(&conn).unwrap();
    let original = serde_json::json!({"unit":"sessions","columns":[{"name":"cwd","kind":"text"}],"rows":[{"cwd":"/work/module"}]});
    conn.execute("INSERT INTO source_kept_result(id,query,ran_at,read_at,rows_json) VALUES ('old','sessions',0,'{}',?1)", [original.to_string()]).unwrap();
    conn.execute_batch(
        "DROP INDEX source_query_commit_seq; ALTER TABLE source_query DROP COLUMN commit_seq;",
    )
    .unwrap();
    remove_context_schema(&conn);
    set_version(&conn, 2).unwrap();
    ensure_primary_index(&conn).unwrap();
    assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
    let body: String = conn
        .query_row(
            "SELECT rows_json FROM source_kept_result WHERE id='old'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let mut body: serde_json::Value = serde_json::from_str(&body).unwrap();
    let metadata = body
        .as_object_mut()
        .unwrap()
        .remove("output_metadata")
        .unwrap();
    assert_eq!(metadata["rows"]["kind"], "unproven");
    assert_eq!(body, original);
}

#[test]
fn query_position_upgrade_preserves_unknown_original_objects() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        if primary {
            ensure_primary_index(&conn).unwrap();
        } else {
            ensure_archive(&conn).unwrap();
        }
        let table = if primary { "source_query" } else { "query" };
        conn.execute_batch(&format!(
            "DROP INDEX {table}_commit_seq; ALTER TABLE {table} DROP COLUMN commit_seq;"
        ))
        .unwrap();
        // The upgrade owns metadata only and need not decode/rewrite the body.
        conn.execute(
            &format!("INSERT INTO {table}(hash,normal_form,object) VALUES (?1,'{{}}',?2)"),
            rusqlite::params!["a".repeat(64), b"original sealed bytes".as_slice()],
        )
        .unwrap();
        if primary {
            remove_context_schema(&conn);
        }
        set_version(&conn, 3).unwrap();
        if primary {
            ensure_primary_index(&conn).unwrap();
        } else {
            ensure_archive(&conn).unwrap();
        }
        let (body, position): (Vec<u8>, Option<i64>) = conn
            .query_row(
                &format!("SELECT object,commit_seq FROM {table}"),
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(body, b"original sealed bytes");
        assert_eq!(
            position, None,
            "a missing historical admission must remain unknown"
        );
        if primary {
            ensure_primary_index(&conn).unwrap();
        } else {
            ensure_archive(&conn).unwrap();
        }
        assert_eq!(
            version(&conn).unwrap(),
            if primary {
                INDEX_VERSION
            } else {
                ARCHIVE_VERSION
            }
        );
    }
}

fn remove_context_schema(conn: &Connection) {
    conn.execute_batch("DROP TABLE parser_context; DROP TABLE parser_context_state; ALTER TABLE parse_state DROP COLUMN stopped_attempt;").unwrap();
}

#[test]
fn index_v5_keeps_rows_and_leaves_old_attempts_and_context_unrecorded() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        if primary {
            ensure_primary_index(&conn).unwrap();
        } else {
            ensure_index(&conn).unwrap();
        }
        conn.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(991,'synthetic','claude-code'); INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,at_length,reason) VALUES(991,1,'old-reading','failed',7,10,'original reason');").unwrap();
        remove_context_schema(&conn);
        set_version(&conn, 4).unwrap();
        if primary {
            ensure_primary_index(&conn).unwrap();
        } else {
            ensure_index(&conn).unwrap();
        }
        let row: (i64, String, Option<Vec<u8>>) = conn
            .query_row(
                "SELECT parsed_upto,reason,stopped_attempt FROM parse_state WHERE input_pk=991",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (7, "original reason".into(), None));
        assert_eq!(
            conn.query_row("SELECT count(*) FROM parser_context_state", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
    }
}

#[test]
fn embeddings_revision_migration_preserves_paid_rows_and_rejections() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(include_str!("fixtures/embeddings_1.sql"))
        .unwrap();
    set_version(&conn, 1).unwrap();
    conn.execute("INSERT INTO embedded(id,text_hash,model,chunk,dims,vector,embedded_at) VALUES (37,x'01','synthetic/1',2,1,x'0000803f',123)", []).unwrap();
    conn.execute(
        "INSERT INTO embed_failure VALUES (x'02','synthetic/1','original rejection',456)",
        [],
    )
    .unwrap();
    ensure_embeddings(&conn).unwrap();
    assert_eq!(version(&conn).unwrap(), 2);
    assert_eq!(conn.query_row("SELECT id || ':' || hex(text_hash) || ':' || model || ':' || chunk || ':' || dims || ':' || hex(vector) || ':' || embedded_at FROM embedded", [], |r|r.get::<_,String>(0)).unwrap(), "37:01:synthetic/1:2:1:0000803F:123");
    assert_eq!(conn.query_row("SELECT hex(text_hash) || ':' || model || ':' || error || ':' || failed_at FROM embed_failure", [], |r|r.get::<_,String>(0)).unwrap(), "02:synthetic/1:original rejection:456");
    ensure_embeddings(&conn).unwrap();
    let revision = |model: &str| {
        conn.query_row(
            "SELECT revision FROM embedding_revision WHERE model=?1",
            [model],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
    };
    assert_eq!(revision("synthetic/1"), 1);
    conn.execute("UPDATE embedded SET vector=x'0000003f' WHERE id=37", [])
        .unwrap();
    assert_eq!(revision("synthetic/1"), 2);
    conn.execute("UPDATE embedded SET model='synthetic/2' WHERE id=37", [])
        .unwrap();
    assert_eq!(revision("synthetic/1"), 3);
    assert_eq!(revision("synthetic/2"), 1);
    conn.execute("UPDATE embed_failure SET error='new rejection'", [])
        .unwrap();
    assert_eq!(revision("synthetic/1"), 4);

}

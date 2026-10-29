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
fn archive_and_embedding_baselines_stay_at_one_while_the_index_migrates() {
    assert_eq!(ARCHIVE_VERSION, 1);
    assert_eq!(INDEX_VERSION, 4);
    for (create, expected) in [
        (ensure_archive as fn(&Connection) -> Result<()>, 1),
        (ensure_index, 4),
        (ensure_embeddings, 1),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), expected);
    }
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
fn read_indexes_are_added_in_place_and_survive_reopen() {
    let conn = Connection::open_in_memory().unwrap();
    base_index(&conn).unwrap();
    set_version(&conn, INDEX_VERSION).unwrap();
    conn.execute("INSERT INTO input_ref(id) VALUES ('synthetic')", [])
        .unwrap();
    assert!(same_index_schema(&conn).unwrap());
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
        assert!(same_index_schema(&conn).unwrap(), "missing {name}");
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
    assert!(!same_index_schema(&conn).unwrap());
}

#[test]
fn migration_keeps_rows_source_bytes_and_paid_embedding_hashes_when_mask_state_retires() {
    let index = Connection::open_in_memory().unwrap();
    index.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
    index
        .execute_batch(include_str!("fixtures/index_3.sql"))
        .unwrap();
    index.execute_batch("PRAGMA user_version=3;
      INSERT INTO input_ref(pk,id,format) VALUES(1,'synthetic-source','fixture');
      INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'claude-code:migration','claude-code','migration','main',1);
      INSERT INTO message(pk,id,session_pk,order_input_pk,seq,arrival_rank,role,written_by,prompt_kind,input_pk,cost_nusd,provider_cost_nusd,pem,order_input_id)
      VALUES(1,'claude-code:migration/m',1,1,1,0,'user','human','prose',1,12345,67890,2,'synthetic-source');").unwrap();
    let source = "--secret=synthetic-preserved-value\n\0";
    let text_body = crate::encoding::pack_text(source).unwrap();
    let hash = blake3::hash(source.as_bytes());
    let block_body =
        crate::encoding::pack_json(&serde_json::json!({"type":"text","text":null})).unwrap();
    index
        .execute(
            "INSERT INTO text(id,hash,body) VALUES(1,?1,?2)",
            rusqlite::params![hash.as_bytes().as_slice(), text_body],
        )
        .unwrap();
    index.execute("INSERT INTO block(message_pk,session_pk,idx,kind,text_id,body) VALUES(1,1,0,'text',1,?1)",[&block_body]).unwrap();
    let reading_body = crate::encoding::pack_json(&serde_json::json!({"source":source})).unwrap();
    index
        .execute(
            "INSERT INTO message_version(version_hash,body) VALUES(?1,?2)",
            rusqlite::params![hash.as_bytes().as_slice(), reading_body],
        )
        .unwrap();
    let embeddings = Connection::open_in_memory().unwrap();
    ensure_embeddings(&embeddings).unwrap();
    let vector = [0_f32, 1_f32, 0_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    embeddings.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES(?1,'synthetic/model/3/document/unit',0,3,?2,1)",
        rusqlite::params![hash.as_bytes().as_slice(),vector]).unwrap();
    let values = |conn: &Connection| {
        conn.query_row(
            "SELECT id,cost_nusd,provider_cost_nusd FROM message",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .unwrap()
    };
    let before = values(&index);
    ensure_index(&index).unwrap();
    ensure_index(&index).unwrap();
    assert_eq!(version(&index).unwrap(), INDEX_VERSION);
    assert_eq!(values(&index), before);
    assert_eq!(
        index
            .query_row("SELECT body FROM block", [], |row| row.get::<_, Vec<u8>>(0))
            .unwrap(),
        block_body
    );
    let (stored_hash, stored_body): (Vec<u8>, Vec<u8>) = index
        .query_row("SELECT hash,body FROM text", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(stored_hash, hash.as_bytes());
    assert_eq!(crate::encoding::unpack_text(&stored_body).unwrap(), source);
    assert_eq!(
        index
            .query_row(
                "SELECT body FROM message_version WHERE version_hash=?1",
                [&stored_hash],
                |row| row.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
        reading_body
    );
    let (cached_hash, cached_vector): (Vec<u8>, Vec<u8>) = embeddings
        .query_row("SELECT text_hash,vector FROM embedded", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(cached_hash, stored_hash);
    assert_eq!(cached_vector, vector);
    assert!(
        !index
            .prepare("SELECT name FROM pragma_table_info('message') WHERE name='pem'")
            .unwrap()
            .exists([])
            .unwrap()
    );
}

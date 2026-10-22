use super::*;

#[test]
fn every_schema_applies_to_an_empty_database() {
    for (name, create) in [
        ("archive", create_archive as fn(&Connection) -> Result<()>),
        ("index", create_index),
        ("embeddings", create_embeddings),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn a_newer_archive_is_refused() {
    let mut conn = Connection::open_in_memory().unwrap();
    create_archive(&conn).unwrap();
    set_version(&conn, 99).unwrap();
    assert!(matches!(
        migrate_archive(&mut conn),
        Err(StoreError::TooNew { found: 99, .. })
    ));
}

#[test]
fn index_schema_changes_require_a_version_and_digest_update() {
    assert_eq!(
        (
            INDEX_VERSION,
            blake3::hash(INDEX.as_bytes()).to_hex().to_string()
        ),
        (
            25,
            "3b2fe4bcb37c0a9e51e069051cdcb5c3eebd98d2fb9bed29b1ef8544302e2aa7".to_owned()
        ),
        "bump INDEX_VERSION and record its schema digest when index.sql changes"
    );
}

#[test]
fn excision_revision_migration_preserves_rows_and_runs_once() {
    let mut conn = Connection::open_in_memory().unwrap();
    create_archive(&conn).unwrap();
    conn.execute("INSERT INTO excision(id,scope,reason,created_at,commit_seq) VALUES ('old','session:claude-code:gone','retained reason',1,7)",[]).unwrap();
    migrate_archive(&mut conn).unwrap();
    assert_eq!(version(&conn).unwrap(), 5);
    let before = crate::excise::recovery::current(&conn).unwrap();
    assert!(before.version > 0);
    migrate_archive(&mut conn).unwrap();
    assert_eq!(crate::excise::recovery::current(&conn).unwrap(), before);
    assert_eq!(
        conn.query_row("SELECT reason FROM excision WHERE id='old'", [], |r| r
            .get::<_, String>(
            0
        ))
        .unwrap(),
        "retained reason"
    );
    conn.execute("UPDATE excision SET reason='changed' WHERE id='old'", [])
        .unwrap();
    let changed = crate::excise::recovery::current(&conn).unwrap();
    assert_ne!(changed, before);
    conn.execute("DELETE FROM excision WHERE id='old'", [])
        .unwrap();
    assert_ne!(crate::excise::recovery::current(&conn).unwrap(), changed);
    let mut empty = Connection::open_in_memory().unwrap();
    create_archive(&empty).unwrap();
    migrate_archive(&mut empty).unwrap();
    assert_eq!(crate::excise::recovery::current(&empty).unwrap().version, 0);
}

#[test]
fn vector_cleanup_migration_preserves_old_policy_and_repeats_without_losing_work() {
    let mut conn = Connection::open_in_memory().unwrap();
    create_archive(&conn).unwrap();
    conn.execute_batch(ARCHIVE_MIGRATIONS[0]).unwrap();
    set_version(&conn, 2).unwrap();
    conn.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('old','session:claude-code:gone',1,7)", []).unwrap();
    let before = crate::excise::recovery::current(&conn).unwrap();
    migrate_archive(&mut conn).unwrap();
    assert_eq!(crate::excise::recovery::current(&conn).unwrap(), before);
    assert!(!crate::excise::vector_cleanup::pending(&conn).unwrap());
    conn.execute(
        "INSERT INTO pending_vector_excision VALUES (?1,'old')",
        [vec![1u8; 32]],
    )
    .unwrap();
    migrate_archive(&mut conn).unwrap();
    assert!(crate::excise::vector_cleanup::pending(&conn).unwrap());
    assert_eq!(crate::excise::recovery::current(&conn).unwrap(), before);
}

#[test]
fn append_fact_migration_assigns_only_positions_after_all_existing_units() {
    let mut conn = Connection::open_in_memory().unwrap();
    create_archive(&conn).unwrap();
    conn.execute_batch("INSERT INTO meta VALUES ('write_counter','50');
        INSERT INTO input VALUES ('input','synthetic');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',1,'first',1,0,70);
        INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at) VALUES ('later','input','m','synthetic','deleted','test',1);
        INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at) VALUES ('earlier','input','m','synthetic','deleted','test',2);
        INSERT INTO delivery(id,recipient_kind,sender,event,body,content_hash,refs_json,left_out_json,at,bytes,commit_seq) VALUES ('opaque:[id]','operator','cli','briefing',x'626f6479',zeroblob(32),'[]','[]',1,4,80);
        INSERT INTO delivery_event VALUES ('opaque:[id]','delivered',2,'detail');").unwrap();
    migrate_archive(&mut conn).unwrap();
    let rows: Vec<(String, i64)> = conn
        .prepare("SELECT id,commit_seq FROM observation ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(rows, vec![("earlier".into(), 83), ("later".into(), 82)]);
    assert_eq!(
        conn.query_row("SELECT commit_seq FROM delivery_event", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        81
    );
    assert_eq!(
        conn.query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "83"
    );
    assert_eq!(
        conn.query_row("SELECT commit_seq FROM revision", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        70
    );
    migrate_archive(&mut conn).unwrap();
    assert_eq!(
        conn.query_row("SELECT commit_seq FROM delivery_event", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        81
    );
}

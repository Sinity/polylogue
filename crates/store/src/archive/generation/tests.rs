use super::*;
use crate::archive::cache;

fn index() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn
}

fn value(conn: &Connection, key: &str) -> String {
    conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))
        .unwrap()
}

#[test]
fn same_boot_keeps_positions_but_changed_boot_ignores_a_stale_clean_stop() {
    let conn = index();
    begin(&conn, "boot-a", None).unwrap();
    let original = value(&conn, "index_instance");
    conn.execute_batch("INSERT INTO meta VALUES ('write_counter','50'),('clean_stop','1'); INSERT INTO removed_session VALUES ('claude-code:old',50)").unwrap();
    begin(&conn, "boot-a", None).unwrap();
    assert_eq!(value(&conn, "index_instance"), original);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM removed_session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    begin(&conn, "boot-b", None).unwrap();
    assert_ne!(value(&conn, "index_instance"), original);
    assert_eq!(value(&conn, "boot_id"), "boot-b");
    assert_eq!(value(&conn, "write_counter"), "50");
    assert_eq!(
        conn.query_row("SELECT count(*) FROM removed_session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let next = value(&conn, "index_instance");
    begin(&conn, "boot-b", None).unwrap();
    assert_eq!(value(&conn, "index_instance"), next);
}

#[test]
fn a_missing_generation_is_created_and_a_failed_update_changes_nothing() {
    let conn = index();
    begin(&conn, "boot-a", None).unwrap();
    // Same boot, so only the missing generation can trigger the creation.
    conn.execute("DELETE FROM meta WHERE key='index_instance'", [])
        .unwrap();
    begin(&conn, "boot-a", None).unwrap();
    assert_eq!(value(&conn, "boot_id"), "boot-a");
    let original = value(&conn, "index_instance");
    assert!(!original.is_empty());
    conn.execute_batch("INSERT INTO removed_session VALUES ('claude-code:old',5); CREATE TRIGGER refuse_boot BEFORE UPDATE ON meta WHEN NEW.key='boot_id' BEGIN SELECT RAISE(ABORT,'boot update failed'); END;").unwrap();
    let sync: i64 = conn
        .pragma_query_value(None, "synchronous", |r| r.get(0))
        .unwrap();
    assert!(begin(&conn, "boot-b", None).is_err());
    assert_eq!(value(&conn, "index_instance"), original);
    assert_eq!(value(&conn, "boot_id"), "boot-a");
    assert_eq!(
        conn.query_row("SELECT count(*) FROM removed_session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.pragma_query_value(None, "synchronous", |r| r.get::<_, i64>(0))
            .unwrap(),
        sync
    );
}

#[test]
fn a_deliberately_replaced_index_establishes_its_boot_before_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    // An installed index, which is then removed by hand.
    crate::fixture::install_empty_rebuild(
        &archive,
        &crate::Writer::start(&archive).unwrap().sender(),
    )
    .unwrap();
    let before_replacement = archive.scope_token().unwrap();
    let conn = super::super::open_write(&archive.index_db(), cache::SETUP).unwrap();
    let before = value(&conn, "index_instance");
    drop(conn);
    for suffix in ["", "-wal", "-shm"] {
        let path = archive.root.join(format!("index.db{suffix}"));
        if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
    }
    archive.prepare_index().unwrap();
    assert!(matches!(
        before_replacement.check(),
        Err(crate::StoreError::ScopeChanged)
    ));
    // The replacement is built and installed; its generation is the one
    // its build established.
    crate::fixture::install_empty_rebuild(
        &archive,
        &crate::Writer::start(&archive).unwrap().sender(),
    )
    .unwrap();
    let conn = super::super::open_read(&archive.index_db()).unwrap();
    let rebuilt = value(&conn, "index_instance");
    assert_ne!(rebuilt, before);
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap();
    assert_eq!(value(&conn, "boot_id"), boot.trim());
    drop(conn);
    drop(lock);
    let (archive, _lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    let conn = super::super::open_read(&archive.index_db()).unwrap();
    assert_eq!(value(&conn, "index_instance"), rebuilt);
}

#[test]
fn a_boot_generation_change_ends_earlier_reads() {
    let dir = tempfile::tempdir().unwrap();
    {
        let root = dir.path().join("standalone");
        std::fs::create_dir(&root).unwrap();
        let archive = crate::Archive::at(&root);
        let conn = index();
        let original = archive.scope_token().unwrap();
        begin(&conn, "boot-a", Some(&archive.read_scope)).unwrap();
        assert!(matches!(
            original.check(),
            Err(crate::StoreError::ScopeChanged)
        ));
        let current = archive.scope_token().unwrap();
        begin(&conn, "boot-a", Some(&archive.read_scope)).unwrap();
        assert!(current.check().is_ok());
        begin(&conn, "boot-b", Some(&archive.read_scope)).unwrap();
        assert!(matches!(
            current.check(),
            Err(crate::StoreError::ScopeChanged)
        ));
    }
}

#[test]
fn transport_fixture_keeps_source_rows() {
    let root = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(root.path()).unwrap();
    let source = super::super::open_write(&archive.archive_db(), cache::SETUP).unwrap();
    crate::input_rows::InputRows {
        capture_read: None,
        input: "kept-source".into(),
        origin: "synthetic".into(),
        chain: None,
        revision: None,
        extents: Vec::new(),
        observation: None,
    }
    .insert(&source, 1)
    .unwrap();
    let token = archive.scope_token().unwrap();
    let index = super::super::open_read(&archive.index_db()).unwrap();
    let old = value(&index, "index_instance");
    archive.advance_generation_for_test().unwrap();
    assert!(matches!(
        token.check(),
        Err(crate::StoreError::ScopeChanged)
    ));
    assert_ne!(value(&index, "index_instance"), old);
    assert_eq!(
        source
            .query_row("SELECT id FROM input", [], |row| row.get::<_, String>(0))
            .unwrap(),
        "kept-source"
    );
    assert!(archive.scope_token().unwrap().check().is_ok());
}

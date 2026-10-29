use super::*;

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
    let original = value(&conn, "generation");
    conn.execute_batch("INSERT INTO meta VALUES ('write_counter','50'),('clean_stop','1'); INSERT INTO removed_session VALUES ('claude-code:old',50)").unwrap();
    begin(&conn, "boot-a", None).unwrap();
    assert_eq!(value(&conn, "generation"), original);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM removed_session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    begin(&conn, "boot-b", None).unwrap();
    assert_ne!(value(&conn, "generation"), original);
    assert_eq!(value(&conn, "boot_id"), "boot-b");
    assert_eq!(value(&conn, "write_counter"), "50");
    assert_eq!(
        conn.query_row("SELECT count(*) FROM removed_session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let next = value(&conn, "generation");
    begin(&conn, "boot-b", None).unwrap();
    assert_eq!(value(&conn, "generation"), next);
}

#[test]
fn a_missing_generation_is_created_and_a_failed_update_changes_nothing() {
    let conn = index();
    conn.execute("DELETE FROM meta WHERE key='generation'", [])
        .unwrap();
    begin(&conn, "boot-a", None).unwrap();
    let original = value(&conn, "generation");
    conn.execute_batch("INSERT INTO removed_session VALUES ('claude-code:old',5); CREATE TRIGGER refuse_boot BEFORE UPDATE ON meta WHEN NEW.key='boot_id' BEGIN SELECT RAISE(ABORT,'boot update failed'); END;").unwrap();
    let sync: i64 = conn
        .pragma_query_value(None, "synchronous", |r| r.get(0))
        .unwrap();
    assert!(begin(&conn, "boot-b", None).is_err());
    assert_eq!(value(&conn, "generation"), original);
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
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let dir = tempfile::tempdir().unwrap();
    let pointer = Arc::new(AtomicBool::new(true));
    let invalidate = Arc::clone(&pointer);
    let _registration = crate::Archive::register_scope_invalidator(dir.path(), move || {
        invalidate.store(false, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    let (archive, lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    let publisher = archive.scope_token().unwrap();
    publisher
        .with_current(|| {
            pointer.store(true, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
    let conn = super::super::open_write(&archive.index_db(), cache::SETUP).unwrap();
    let before = value(&conn, "generation");
    drop(conn);
    for suffix in ["", "-wal", "-shm"] {
        let path = archive.root.join(format!("index.db{suffix}"));
        if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
    }
    archive.prepare_index().unwrap();
    assert!(!pointer.load(Ordering::SeqCst));
    assert!(matches!(
        publisher.with_current(|| Ok(())),
        Err(crate::StoreError::ScopeChanged)
    ));
    let conn = super::super::open_read(&archive.index_db()).unwrap();
    let rebuilt = value(&conn, "generation");
    assert_ne!(rebuilt, before);
    assert!(!value(&conn, "boot_id").is_empty());
    drop(conn);
    drop(lock);
    let (archive, _lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    let conn = super::super::open_read(&archive.index_db()).unwrap();
    assert_eq!(value(&conn, "generation"), rebuilt);
}

#[test]
fn a_boot_generation_change_fences_publishers_in_both_archive_modes() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let dir = tempfile::tempdir().unwrap();
    for primary in [false, true] {
        let root = dir
            .path()
            .join(if primary { "primary" } else { "standalone" });
        std::fs::create_dir(&root).unwrap();
        let archive = if primary {
            crate::Archive::on_sinex(&root, &root.join("bytes"))
        } else {
            crate::Archive::at(&root)
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let callback = Arc::clone(&calls);
        let _registration = crate::Archive::register_scope_invalidator(&root, move || {
            callback.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        let conn = index();
        let original = archive.scope_token().unwrap();
        begin(&conn, "boot-a", Some(&archive.read_scope)).unwrap();
        assert!(matches!(
            original.check(),
            Err(crate::StoreError::ScopeChanged)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let current = archive.scope_token().unwrap();
        begin(&conn, "boot-a", Some(&archive.read_scope)).unwrap();
        assert!(current.check().is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        begin(&conn, "boot-b", Some(&archive.read_scope)).unwrap();
        assert!(matches!(
            current.check(),
            Err(crate::StoreError::ScopeChanged)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}

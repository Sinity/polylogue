use super::*;
use crate::archive::{cache, open_write};
use std::fs::{File, TryLockError};
use std::sync::mpsc;

mod vfs;

#[test]
fn a_first_read_cannot_see_states_separated_by_two_commits() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let registration = vfs::Registration::install();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let (reached, gap) = mpsc::channel();
    let (proceed, resume) = mpsc::channel();
    registration.pause(reader.conn.as_ref().unwrap(), reached, resume);
    let location = fixture.archive.clone();
    let writer = std::thread::spawn(move || {
        gap.recv().unwrap();
        let probe = File::open(location.archive_db()).unwrap();
        let guarded = match probe.try_lock() {
            Ok(()) => {
                probe.unlock().unwrap();
                false
            }
            Err(TryLockError::WouldBlock) => true,
            Err(error) => panic!("write lock probe: {error}"),
        };
        // A first read under the shared lock must finish before the writer
        // can go on. Without the lock, finish all four commits before
        // resuming it.
        if guarded {
            proceed.send(()).unwrap();
        }
        let mut archive_conn = open_write(&location.archive_db(), cache::SETUP).unwrap();
        let mut index = open_write(&location.index_db(), cache::SETUP).unwrap();
        let mut states = vec![(0, 0)];
        for (archive_write, counter) in [(true, 1), (false, 1), (true, 2), (false, 2)] {
            let conn = if archive_write {
                &mut archive_conn
            } else {
                &mut index
            };
            let _lock = crate::write_lock::hold(conn).unwrap();
            let tx = conn.transaction().unwrap();
            tx.execute("INSERT INTO meta(key,value) VALUES ('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[counter.to_string()]).unwrap();
            tx.commit().unwrap();
            let counter = |conn: &Connection| {
                conn.query_row(
                    "SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
            };
            states.push((counter(&index), counter(&archive_conn)));
        }
        if !guarded {
            proceed.send(()).unwrap();
        }
        states
    });
    let position = reader.change_position().unwrap();
    let states = writer.join().unwrap();
    assert_eq!(states, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]);
    assert!(
        states.contains(&(position.index, position.archive)),
        "mixed pair {:?} was never committed: {states:?}",
        (position.index, position.archive)
    );
    assert_eq!(reader.change_position().unwrap(), position);
    let current = pool.reader().unwrap().change_position().unwrap();
    assert_eq!((current.index, current.archive), (2, 2));
    drop(reader);
    drop(pool);
    drop(registration);
}

#[test]
fn started_reads_do_not_hold_the_write_lock() {
    use crate::fixture::writer::{WriterArchive, prose};
    let mut fixture = WriterArchive::new();
    fixture.write("synthetic", prose("synthetic", 1, 1, "before"));
    fixture.writer.flush().unwrap();
    let pool = ReadPool::open(&fixture.archive).unwrap();
    let reader = pool.reader().unwrap();
    let before = reader.change_position().unwrap();
    fixture.write("synthetic", prose("synthetic", 2, 2, "after"));
    fixture.writer.flush().unwrap();
    let after = pool.reader().unwrap().change_position().unwrap();
    assert!(after.archive > before.archive);
    assert!(after.index > before.index);
    assert_eq!(reader.change_position().unwrap(), before);
}

#[test]
fn an_unused_slot_refuses_after_close_with_or_without_waiting_for_the_lock() {
    for contended in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        let lock = File::open(fixture.archive.archive_db()).unwrap();
        if contended {
            lock.lock().unwrap();
        }
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        pool.close();
        assert!(
            matches!(reader.change_position(), Err(StoreError::ReadPoolClosed)),
            "contended={contended}"
        );
        assert!(
            matches!(
                reader.query("SELECT 42 AS n", &[]),
                Err(StoreError::ReadPoolClosed)
            ),
            "repeat contended={contended}"
        );
    }
}

#[test]
fn an_interrupted_first_read_does_not_wait_for_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let lock = File::open(fixture.archive.archive_db()).unwrap();
    lock.lock().unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn
        .as_ref()
        .unwrap()
        .get_interrupt_handle()
        .interrupt();
    assert!(
        matches!(reader.change_position(),Err(StoreError::Sqlite(rusqlite::Error::SqliteFailure(error,_))) if error.code==rusqlite::ErrorCode::OperationInterrupted)
    );
}

fn mark_index_unsynced(archive: &Archive) {
    let conn = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let tx =
        crate::write_lock::transaction(&conn, rusqlite::TransactionBehavior::Immediate).unwrap();
    tx.execute("INSERT INTO meta(key,value) VALUES ('unsynced','1') ON CONFLICT(key) DO UPDATE SET value=excluded.value",[]).unwrap();
    tx.commit().unwrap();
    drop(conn);
}
fn replace_index(archive: &Archive) {
    mark_index_unsynced(archive);
    archive.prepare_index().unwrap();
}

#[test]
fn an_unused_slot_on_a_replaced_index_refuses_while_idle_connections_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let unused = pool.reader().unwrap();
    replace_index(&fixture.archive);
    assert!(matches!(
        unused.change_position(),
        Err(StoreError::ReadIndexReplaced)
    ));
    assert!(matches!(
        unused.change_position(),
        Err(StoreError::ReadIndexReplaced)
    ));
    drop(unused);
    let current = pool.reader().unwrap().change_position().unwrap();
    replace_index(&fixture.archive);
    let refreshed = pool.reader().unwrap().change_position().unwrap();
    assert_ne!(current.generation, refreshed.generation);
}

#[test]
fn a_started_reader_keeps_its_read_while_the_index_is_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let old = reader.change_position().unwrap();
    replace_index(&fixture.archive);
    assert_eq!(reader.change_position().unwrap(), old);
    drop(reader);
    let current = pool.reader().unwrap().change_position().unwrap();
    assert_ne!(current.generation, old.generation);
}

#[test]
fn a_first_read_creates_no_file_in_a_read_only_archive_directory() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let names = || {
        let mut names: Vec<_> = std::fs::read_dir(&fixture.archive.root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        names
    };
    let before = names();
    let original = std::fs::metadata(&fixture.archive.root)
        .unwrap()
        .permissions();
    std::fs::set_permissions(
        &fixture.archive.root,
        std::fs::Permissions::from_mode(0o555),
    )
    .unwrap();
    let result = (|| -> Result<_> {
        let pool = fixture.reads()?;
        pool.reader()?.change_position()
    })();
    std::fs::set_permissions(&fixture.archive.root, original).unwrap();
    result.unwrap();
    assert_eq!(names(), before);
}

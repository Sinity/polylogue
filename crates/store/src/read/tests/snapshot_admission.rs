use super::*;
use crate::archive::{cache, open_write};
use std::fs::{File, TryLockError};
use std::sync::mpsc;

mod vfs;

#[test]
fn first_pin_cannot_combine_states_separated_by_two_publications() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let registration = vfs::Registration::install();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let (reached, gap) = mpsc::channel();
    let (proceed, resume) = mpsc::channel();
    registration.pause(reader.conn.as_ref().unwrap(), reached, resume);
    let location = fixture.archive.clone();
    let publisher = std::thread::spawn(move || {
        gap.recv().unwrap();
        let probe = File::open(location.archive_db()).unwrap();
        let guarded = match probe.try_lock() {
            Ok(()) => {
                probe.unlock().unwrap();
                false
            }
            Err(TryLockError::WouldBlock) => true,
            Err(error) => panic!("publication probe: {error}"),
        };
        // A guarded first pin must finish before publication can proceed.
        // With the old path, finish all four commits before resuming it.
        if guarded {
            proceed.send(()).unwrap();
        }
        let mut canonical = open_write(&location.archive_db(), cache::SETUP).unwrap();
        let mut index = open_write(&location.index_db(), cache::SETUP).unwrap();
        let mut states = vec![(0, 0)];
        for (canonical_write, counter) in [(true, 1), (false, 1), (true, 2), (false, 2)] {
            let conn = if canonical_write {
                &mut canonical
            } else {
                &mut index
            };
            let _publication = crate::publication::write(conn).unwrap();
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
            states.push((counter(&index), counter(&canonical)));
        }
        if !guarded {
            proceed.send(()).unwrap();
        }
        states
    });
    let position = reader.change_position().unwrap();
    let states = publisher.join().unwrap();
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
fn already_pinned_reads_do_not_hold_publication_exclusion() {
    use crate::fixture::writer::{WriterWorld, prose};
    let mut world = WriterWorld::new();
    world.write("synthetic", prose("synthetic", 1, 1, "before"));
    world.writer.flush().unwrap();
    let pool = ReadPool::open(&world.archive).unwrap();
    let reader = pool.reader().unwrap();
    let before = reader.change_position().unwrap();
    world.write("synthetic", prose("synthetic", 2, 2, "after"));
    world.writer.flush().unwrap();
    let after = pool.reader().unwrap().change_position().unwrap();
    assert!(after.archive > before.archive);
    assert!(after.index > before.index);
    assert_eq!(reader.change_position().unwrap(), before);
}

#[test]
fn lazy_reservation_can_close_without_waiting_for_publication() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let _publication = File::open(fixture.archive.archive_db()).unwrap();
    _publication.lock().unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    pool.close();
    assert!(matches!(
        reader.change_position(),
        Err(StoreError::ReadPoolClosed)
    ));
}

#[test]
fn interrupted_admission_does_not_wait_for_publication() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let _publication = File::open(fixture.archive.archive_db()).unwrap();
    _publication.lock().unwrap();
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

#[test]
fn rejected_privacy_snapshot_releases_the_admission_lock() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let canonical = open_write(&fixture.archive.archive_db(), cache::SETUP).unwrap();
    canonical.execute("INSERT INTO meta(key,value) VALUES ('excision_version','1'),('excision_revision','synthetic') ON CONFLICT(key) DO UPDATE SET value=excluded.value",[]).unwrap();
    let pool = fixture.reads().unwrap();
    assert!(matches!(
        pool.reader().unwrap().change_position(),
        Err(StoreError::ExcisionPending)
    ));
    let file = File::open(fixture.archive.archive_db()).unwrap();
    file.try_lock().unwrap();
}

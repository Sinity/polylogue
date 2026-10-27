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
            // As the writer does: only register commits take the lock.
            let tx = conn.transaction().unwrap();
            tx.execute("INSERT INTO meta(key,value) VALUES ('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[counter.to_string()]).unwrap();
            if archive_write {
                crate::write_lock::commit(tx).unwrap();
            } else {
                tx.commit().unwrap();
            }
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

fn replace_index(archive: &Archive) {
    for suffix in ["", "-wal", "-shm"] {
        let path = archive.root.join(format!("index.db{suffix}"));
        if path.exists() { std::fs::remove_file(path).unwrap(); }
    }
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

/// A first collection keeps the writer committing back to back for hours;
/// a read must still start within one register commit, not after the
/// writer runs dry.
#[test]
fn reads_answer_promptly_while_the_writer_works_through_a_backlog() {
    use crate::fixture::writer::{prose, sid, step};
    use polylogue_model::{MessageRow, Output, Session, SessionKind};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    const MESSAGES_PER_BATCH: u64 = 8_000;
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = crate::Writer::start(&archive).unwrap();
    let sender = writer.sender().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let feeding = {
        let stop = Arc::clone(&stop);
        // The writer's queue is short, so this keeps it full: every batch
        // the writer takes is replaced at once.
        std::thread::spawn(move || {
            let mut sent = 0_u64;
            while !stop.load(Ordering::Relaxed) {
                let session = format!("backlog-{sent}");
                let messages = (0..MESSAGES_PER_BATCH)
                    .map(|n| MessageRow {
                        offset: n,
                        end: n + 1,
                        message: prose(
                            &session,
                            n as u32,
                            1_000 + n as i64,
                            &format!("synthetic backlog text {sent} {n}, long enough to cost a page write"),
                        )
                        .remove(0),
                    })
                    .collect();
                let output = Output {
                    sessions: vec![Session::new(sid(&session), SessionKind::Main)],
                    messages,
                    ..Output::default()
                };
                let batch = step(
                    &format!("input-{session}"),
                    "claude-code",
                    &format!("/p/{session}"),
                    output,
                    0,
                    MESSAGES_PER_BATCH,
                );
                sender.send(batch).unwrap();
                sent += 1;
            }
            sent
        })
    };
    let pool = ReadPool::open(&archive).unwrap();
    let mut slowest = Duration::ZERO;
    let mut reads = 0;
    let mut positions = Vec::new();
    let reading = Instant::now();
    while reading.elapsed() < Duration::from_secs(5) {
        let started = Instant::now();
        let reader = pool.reader().unwrap();
        reader.status().unwrap();
        positions.push(reader.change_position().unwrap().archive);
        drop(reader);
        slowest = slowest.max(started.elapsed());
        reads += 1;
        std::thread::sleep(Duration::from_millis(50));
    }
    stop.store(true, Ordering::Relaxed);
    let sent = feeding.join().unwrap();
    drop(writer);
    positions.dedup();
    assert!(
        positions.len() >= 3,
        "the writer did not commit throughout the reads: {positions:?}, {sent} batches sent"
    );
    // Reads that waited for whole writer transactions managed about eight
    // in these five seconds, the slowest over a second.
    assert!(
        reads >= 20 && slowest < Duration::from_secs(1),
        "{reads} reads, the slowest took {slowest:?}, while the writer committed {} times",
        positions.len()
    );
}

use super::*;
use crate::{ReadPool, fixture::Fixture};
use std::sync::mpsc::channel;

const SLOW: &str = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n";

fn started(limit: &ReadLimit) {
    let until = Instant::now() + Duration::from_secs(10);
    while limit.sqlite_steps() == 0 {
        assert!(Instant::now() < until, "read never reached SQLite");
        std::thread::yield_now();
    }
}

#[test]
fn slow_statement_stops_at_its_deadline_and_the_slot_can_be_reused() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    // Warm the connection before the statement's deadline starts, then
    // acquire it within the scope so it installs this request's progress hook.
    drop(pool.reader().unwrap());
    let limit = ReadLimit::new(Duration::from_millis(50), Default::default());
    let result = limit.scope(|| pool.reader()?.query(SLOW, &[]));
    assert!(matches!(result, Err(StoreError::ReadDeadlineExceeded)));
    assert!(matches!(
        limit.check(),
        Err(StoreError::ReadDeadlineExceeded)
    ));
    assert!(limit.sqlite_steps() > 0);
    assert_eq!(
        pool.reader().unwrap().query("SELECT 1 AS n", &[]).unwrap()[0]["n"],
        1
    );
}

#[test]
fn cancelling_a_running_statement_frees_its_only_slot() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let limit = ReadLimit::new(Duration::from_secs(30), Default::default());
    std::thread::scope(|scope| {
        let (ready, acquired) = channel();
        let worker_limit = &limit;
        let worker_pool = &pool;
        let worker = scope.spawn(move || {
            worker_limit.scope(|| {
                let reader = worker_pool.reader()?;
                reader.query("SELECT 1", &[])?;
                ready.send(worker_limit.sqlite_steps()).unwrap();
                reader.query(SLOW, &[])
            })
        });
        let before_query = acquired.recv().unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        while limit.sqlite_steps() == before_query {
            assert!(Instant::now() < until, "SLOW query never reached SQLite");
            std::thread::yield_now();
        }
        limit.cancellation.cancel();
        assert!(matches!(
            worker.join().unwrap(),
            Err(StoreError::ReadCancelled)
        ));
        assert!(matches!(limit.check(), Err(StoreError::ReadCancelled)));
        assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
    });
}

#[test]
fn concurrent_reads_keep_separate_deadlines_and_cancellation() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let short = ReadLimit::new(Duration::from_millis(150), Default::default());
    let long = ReadLimit::new(Duration::from_secs(30), Default::default());
    std::thread::scope(|scope| {
        // The short deadline may pass while its reader is still opening; the
        // rule is that it ends that read alone, wherever it is.
        let a = scope.spawn(|| short.scope(|| pool.reader()?.query(SLOW, &[])));
        let b = scope.spawn(|| long.scope(|| pool.reader()?.query(SLOW, &[])));
        started(&long);
        assert!(matches!(
            a.join().unwrap(),
            Err(StoreError::ReadDeadlineExceeded)
        ));
        assert!(matches!(
            short.check(),
            Err(StoreError::ReadDeadlineExceeded)
        ));
        assert!(long.check().is_ok());
        long.cancellation.cancel();
        // SQLite steps start while the reader opens its views, so the
        // cancellation can end the long read before its statement starts.
        assert!(matches!(b.join().unwrap(), Err(StoreError::ReadCancelled)));
    });
}

#[test]
fn deadline_and_cancellation_also_stop_waiting_for_a_reader_slot() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let _held = pool.reader().unwrap();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let limit = ReadLimit::new(Duration::from_millis(30), Default::default());
                assert!(matches!(
                    limit.scope(|| pool.reader()),
                    Err(StoreError::ReadDeadlineExceeded)
                ));
            })
            .join()
            .unwrap();
        let limit = ReadLimit::new(Duration::from_secs(30), Default::default());
        let (ready, wait) = channel();
        let worker_limit = limit.clone();
        let pool = &pool;
        let worker = scope.spawn(move || {
            worker_limit.scope(|| {
                ready.send(()).unwrap();
                assert!(matches!(pool.reader(), Err(StoreError::ReadCancelled)));
            })
        });
        wait.recv().unwrap();
        limit.cancellation.cancel();
        worker.join().unwrap();
    });
}

#[test]
fn a_read_waiting_for_the_write_lock_stops_at_its_deadline() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let connection = crate::archive::open_read(&fixture.archive.archive_db()).unwrap();
    let _held = crate::write_lock::hold(&connection).unwrap();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let limit = ReadLimit::new(Duration::from_millis(50), Default::default());
                let response = limit.scope(|| pool.reader()?.query("SELECT 1", &[]));
                assert!(matches!(response, Err(StoreError::ReadDeadlineExceeded)));
            })
            .join()
            .unwrap();
    });
}

#[test]
fn a_reused_reader_without_a_scope_restores_the_busy_wait() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let limit = ReadLimit::new(Duration::from_secs(1), Default::default());
    limit.scope(|| pool.reader().unwrap().query("SELECT 1", &[]).unwrap());
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT timeout FROM pragma_busy_timeout", &[])
            .unwrap()[0]["timeout"],
        30000
    );
}

#[test]
fn a_read_without_a_deadline_still_observes_cancellation() {
    let limit = ReadLimit::without_deadline(Default::default());
    assert!(limit.check().is_ok());
    limit.cancellation.cancel();
    assert!(matches!(limit.check(), Err(StoreError::ReadCancelled)));
}

#[test]
fn finishing_read_preparation_restores_the_enclosing_scope_after_the_write() {
    let outer = ReadLimit::new(Duration::ZERO, Default::default());
    let inner = ReadLimit::new(Duration::from_secs(30), Default::default());
    outer.scope(|| {
        inner.scope(|| {
            ReadLimit::finish_scope("write").unwrap();
            assert!(ReadLimit::current().is_none());
            assert_eq!(inner.stage(), Some("write"));
        });
        assert!(matches!(
            ReadLimit::current().unwrap().check(),
            Err(StoreError::ReadDeadlineExceeded)
        ));
    });
    assert!(ReadLimit::current().is_none());
}

#[test]
fn absolute_deadline_interrupts_sqlite_and_cannot_be_renewed_by_another_scope() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let deadline = Instant::now() + Duration::from_millis(50);
    let limit = ReadLimit::at(deadline, Default::default());
    assert!(matches!(
        limit.scope(|| pool.reader()?.query(SLOW, &[])),
        Err(StoreError::ReadDeadlineExceeded)
    ));
    assert!(limit.sqlite_steps() > 0);
    assert!(matches!(
        limit.scope(|| pool.reader()?.query("SELECT 1", &[])),
        Err(StoreError::ReadDeadlineExceeded)
    ));
    assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
}

#[test]
fn absolute_limit_cancellation_interrupts_a_real_pool_wait_before_releasing_the_slot() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let held = pool.reader().unwrap();
    let cancellation = Cancellation::default();
    let limit = ReadLimit::at(
        Instant::now() + Duration::from_secs(60),
        cancellation.clone(),
    );
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| limit.scope(|| pool.reader().map(drop)));
        let mut state = pool.state.lock().unwrap();
        while state.waiters == 0 {
            state = pool.changed.wait(state).unwrap();
        }
        drop(state);
        cancellation.cancel();
        assert!(matches!(
            worker.join().unwrap(),
            Err(StoreError::ReadCancelled)
        ));
    });
    drop(held);
    assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
}

#[test]
fn absolute_limit_cancellation_stops_sqlite_progress_and_joins_reader_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let cancellation = Cancellation::default();
    let limit = ReadLimit::at(
        Instant::now() + Duration::from_secs(60),
        cancellation.clone(),
    );
    std::thread::scope(|scope| {
        let (before, observed) = channel();
        let worker_limit = &limit;
        let worker_pool = &pool;
        let worker = scope.spawn(move || {
            worker_limit.scope(|| {
                let reader = worker_pool.reader()?;
                reader.query("SELECT 1", &[])?;
                before.send(worker_limit.sqlite_steps()).unwrap();
                reader.query(SLOW, &[])
            })
        });
        let previous = observed.recv().unwrap();
        while limit.sqlite_steps() == previous {
            assert!(
                !worker.is_finished(),
                "statement ended before reaching its progress hook"
            );
            std::thread::yield_now();
        }
        cancellation.cancel();
        assert!(matches!(
            worker.join().unwrap(),
            Err(StoreError::ReadCancelled)
        ));
    });
    assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
}

#[test]
fn absolute_deadline_stops_a_sqlite_busy_wait() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("busy.db");
    let writer = Connection::open(&database).unwrap();
    writer.execute_batch("PRAGMA journal_mode=DELETE; CREATE TABLE sample(value); INSERT INTO sample VALUES(1); BEGIN EXCLUSIVE;").unwrap();
    let limit = ReadLimit::at(
        Instant::now() + Duration::from_millis(50),
        Default::default(),
    );
    let result: Result<i64> = limit.scope(|| {
        let reader = Connection::open(&database)?;
        limit.install(&reader)?;
        Ok(reader.query_row("SELECT value FROM sample", [], |row| row.get(0))?)
    });
    assert!(matches!(result, Err(StoreError::ReadDeadlineExceeded)));
    writer.execute_batch("ROLLBACK").unwrap();
    assert_eq!(
        Connection::open(&database)
            .unwrap()
            .query_row::<i64, _, _>("SELECT value FROM sample", [], |row| row.get(0))
            .unwrap(),
        1
    );
}

/// A reader that names no request still belongs to its archive: the stop
/// interrupts its endless statement and wakes a checkout waiting for its
/// slot. A newly opened archive is a new process and reads normally.
#[test]
fn the_archive_stop_ends_running_and_waiting_reads() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    std::thread::scope(|scope| {
        let (started, running) = std::sync::mpsc::channel();
        let pool = &pool;
        let endless = scope.spawn(move || {
            let reader = pool.reader()?;
            reader.query("SELECT 1", &[])?;
            started.send(()).unwrap();
            reader.query(SLOW, &[])
        });
        running.recv().unwrap();
        let waiting = scope.spawn(move || pool.reader().map(drop));
        let mut state = pool.state.lock().unwrap();
        while state.waiters == 0 {
            state = pool.changed.wait(state).unwrap();
        }
        drop(state);
        fixture.archive.stop().cancel();
        assert!(matches!(endless.join().unwrap(), Err(StoreError::Stopped)));
        assert!(matches!(waiting.join().unwrap(), Err(StoreError::Stopped)));
    });
    assert!(matches!(pool.reader().map(drop), Err(StoreError::Stopped)));
    let reopened = fixture.archive.reopened();
    let reads = ReadPool::open(&reopened).unwrap();
    assert_eq!(
        reads.reader().unwrap().query("SELECT 1 AS n", &[]).unwrap()[0]["n"],
        1
    );
}

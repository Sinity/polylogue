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
    let limit = ReadLimit::new(Duration::from_millis(50), Default::default());
    let start = Instant::now();
    let result = limit.scope(|| pool.reader().unwrap().query(SLOW, &[]));
    assert!(matches!(result, Err(StoreError::ReadDeadlineExceeded)));
    assert!(matches!(
        limit.check(),
        Err(StoreError::ReadDeadlineExceeded)
    ));
    assert!(limit.sqlite_steps() > 0);
    assert!(start.elapsed() < Duration::from_secs(2));
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
        let worker = scope.spawn(|| limit.scope(|| pool.reader().unwrap().query(SLOW, &[])));
        started(&limit);
        let start = Instant::now();
        limit.cancellation.cancel();
        assert!(matches!(
            worker.join().unwrap(),
            Err(StoreError::ReadCancelled)
        ));
        assert!(start.elapsed() < Duration::from_secs(2));
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
        let a = scope.spawn(|| short.scope(|| pool.reader().unwrap().query(SLOW, &[])));
        let b = scope.spawn(|| long.scope(|| pool.reader().unwrap().query(SLOW, &[])));
        started(&long);
        assert!(a.join().unwrap().is_err());
        assert!(matches!(
            short.check(),
            Err(StoreError::ReadDeadlineExceeded)
        ));
        assert!(long.check().is_ok());
        long.cancellation.cancel();
        assert!(b.join().unwrap().is_err());
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
fn child_cancellation_does_not_cancel_another_request() {
    let connection = ReadCancellation::default();
    let a = connection.child();
    let b = connection.child();
    a.cancel();
    assert!(a.is_cancelled());
    assert!(!b.is_cancelled());
    connection.cancel();
    assert!(b.is_cancelled());
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
                let response = limit.scope(|| pool.reader().unwrap().query("SELECT 1", &[]));
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
    assert!(reader.limit.is_none());
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

use crate::{ReadPool, StoreError, fixture::Fixture, read::limit::ReadLimit};
use std::time::{Duration, Instant};

#[test]
fn reset_invalidates_pinned_reads_and_prepared_delivery_before_cache_changes() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    assert!(reader.query("SELECT 1", &[]).is_ok());
    let delivery = reader.scope_token();
    fixture.archive.read_scope.begin_reset().unwrap();
    assert!(matches!(
        reader.query("SELECT 1", &[]),
        Err(StoreError::ScopeChanged)
    ));
    assert!(matches!(delivery.check(), Err(StoreError::ScopeChanged)));
    assert!(matches!(pool.reader(), Err(StoreError::ScopeChanged)));
    fixture.archive.read_scope.finish_reset();
    assert!(matches!(delivery.check(), Err(StoreError::ScopeChanged)));
    assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
}

#[test]
fn scope_reset_interrupts_a_statement_without_a_deadline() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let limit = ReadLimit::without_deadline(Default::default());
    std::thread::scope(|threads| {
        let (ready, waiting) = std::sync::mpsc::channel();

        let worker_limit = &limit;
        let pool = &pool;
        let work = threads.spawn(move || worker_limit.scope(|| {
            let reader = pool.reader().unwrap();
            reader.query("SELECT 1", &[]).unwrap();
            ready.send(worker_limit.sqlite_steps()).unwrap();
            reader.query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n", &[])
        }));
        let baseline = waiting.recv().unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while limit.sqlite_steps() <= baseline {
            assert!(Instant::now() < until, "statement never reached SQLite");
            std::thread::yield_now();
        }

        fixture.archive.read_scope.begin_reset().unwrap();
        assert!(matches!(
            work.join().unwrap(),
            Err(StoreError::ScopeChanged)
        ));
        fixture.archive.read_scope.finish_reset();
    });
    assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
}

#[test]
fn waiting_for_an_unlimited_reader_slot_stops_when_scope_changes() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let _held = pool.reader().unwrap();
    std::thread::scope(|threads| {
        let work = threads.spawn(|| matches!(pool.reader(), Err(StoreError::ScopeChanged)));
        let until = Instant::now() + Duration::from_secs(5);
        while pool.state.lock().unwrap().waiters == 0 {
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        fixture.archive.read_scope.begin_reset().unwrap();
        assert!(work.join().unwrap());
        fixture.archive.read_scope.finish_reset();
    });
}

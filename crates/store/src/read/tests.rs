use super::*;
use crate::fixture::Fixture;

#[test]
fn closed_pools_refuse_new_readers_and_let_admitted_reads_finish() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader.status().unwrap();
    pool.close();
    assert!(matches!(pool.reader(), Err(StoreError::ReadPoolClosed)));
    assert_eq!(
        reader.query("SELECT 42 AS answer", &[]).unwrap()[0]["answer"],
        42
    );
    drop(reader);
    pool.close();
    assert!(matches!(pool.reader(), Err(StoreError::ReadPoolClosed)));
}

fn wait_for_waiters(pool: &ReadPool, count: usize) {
    let mut state = pool.state.lock().unwrap();
    while state.waiters < count {
        state = pool.changed.wait(state).unwrap();
    }
}

#[test]
fn concurrent_readers_wait_at_the_cap_and_reuse_released_connections() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let (admitted, received) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let mut releases = Vec::new();
        for id in 0..8 {
            let (release, released) = std::sync::mpsc::channel();
            releases.push(release);
            let admitted = admitted.clone();
            let pool = &pool;
            scope.spawn(move || {
                let reader = pool.reader().unwrap();
                reader.status().unwrap();
                admitted.send(id).unwrap();
                released.recv().unwrap();
            });
        }
        let first = received.recv().unwrap();
        let second = received.recv().unwrap();
        wait_for_waiters(&pool, 6);
        assert_eq!(pool.state.lock().unwrap().live, 2);
        releases[first].send(()).unwrap();
        releases[second].send(()).unwrap();
        for _ in 0..6 {
            releases[received.recv().unwrap()].send(()).unwrap();
        }
    });
    let state = pool.state.lock().unwrap();
    assert_eq!(state.live, 2);
    assert_eq!(state.idle.len(), 2);
    assert!(state.owners.is_empty());
}

#[test]
fn close_wakes_all_full_pool_waiters_without_cancelling_the_admitted_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    reader.status().unwrap();
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..4 {
            threads.push(
                scope.spawn(|| assert!(matches!(pool.reader(), Err(StoreError::ReadPoolClosed)))),
            );
        }
        wait_for_waiters(&pool, 4);
        pool.close();
        for thread in threads {
            thread.join().unwrap();
        }
    });
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    drop(reader);
    assert_eq!(pool.state.lock().unwrap().live, 0);
}

#[test]
fn nested_snapshots_are_allowed_but_self_exhaustion_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let first = pool.reader().unwrap();
    let second = pool.reader().unwrap();
    assert!(matches!(pool.reader(), Err(StoreError::ReadPoolNested)));
    drop(first);
    let third = pool.reader().unwrap();
    third.status().unwrap();
    drop((second, third));
    assert!(pool.state.lock().unwrap().owners.is_empty());
}

#[test]
fn open_and_begin_failures_release_the_reservation_for_waiters() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    for begin_failure in [false, true] {
        let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
        std::thread::scope(|scope| {
            let (opening, opened) = std::sync::mpsc::channel();
            let (release, released) = std::sync::mpsc::channel();
            let pool = &pool;
            let dir = &dir;
            let attempt = scope.spawn(move || {
                let result = pool.acquire(|| {
                    opening.send(()).unwrap();
                    released.recv().unwrap();
                    if begin_failure {
                        let conn = pool.connect()?;
                        conn.execute_batch("BEGIN")?;
                        Ok(conn)
                    } else {
                        crate::archive::open_read(&dir.path().join("missing.db"))
                    }
                });
                assert!(matches!(result, Err(StoreError::Sqlite(_))));
            });
            opened.recv().unwrap();
            let waiter = scope.spawn(|| pool.reader().unwrap().status().unwrap());
            wait_for_waiters(pool, 1);
            release.send(()).unwrap();
            attempt.join().unwrap();
            waiter.join().unwrap();
        });
        let state = pool.state.lock().unwrap();
        assert_eq!(state.live, 1);
        assert_eq!(state.idle.len(), 1);
        assert!(state.owners.is_empty());
    }
}

#[test]
fn close_while_connecting_refuses_the_pending_admission_and_releases_its_slot() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    std::thread::scope(|scope| {
        let (opening, opened) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let pool = &pool;
        let attempt = scope.spawn(move || {
            assert!(matches!(
                pool.acquire(|| {
                    opening.send(()).unwrap();
                    released.recv().unwrap();
                    pool.connect()
                }),
                Err(StoreError::ReadPoolClosed)
            ));
        });
        opened.recv().unwrap();
        pool.close();
        release.send(()).unwrap();
        attempt.join().unwrap();
    });
    let state = pool.state.lock().unwrap();
    assert_eq!(state.live, 0);
    assert!(state.idle.is_empty());
    assert!(state.owners.is_empty());
}

#[test]
fn failed_transaction_completion_discards_the_connection_and_releases_its_slot() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    reader.conn().unwrap().execute_batch("ROLLBACK").unwrap();
    drop(reader);
    assert_eq!(pool.state.lock().unwrap().live, 0);
    pool.reader().unwrap().status().unwrap();
}

#[test]
fn main_and_attached_cache_targets_reserve_the_optional_vectors_share() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 3, (8 << 20) + 123).unwrap();
    let readers: Vec<_> = (0..3).map(|_| pool.reader().unwrap()).collect();
    let mut total = 0;
    for reader in &readers {
        for schema in ["main", "a"] {
            let cache: i64 = reader
                .conn()
                .unwrap()
                .pragma_query_value(Some(schema), "cache_size", |r| r.get(0))
                .unwrap();
            assert!(cache < 0);
            total += (-cache) as u64 * 1024;
            let mmap: i64 = reader
                .conn()
                .unwrap()
                .pragma_query_value(Some(schema), "mmap_size", |r| r.get(0))
                .unwrap();
            assert_eq!(mmap, 0);
        }
    }
    assert!(total <= (8 << 20) + 123);
    assert!(total <= (8 << 20) * 2 / 3);
    assert!(ReadPool::with_budget(&fixture.archive, 0, 8 << 20).is_err());
    assert!(ReadPool::with_budget(&fixture.archive, 3, 3 * 3 * 1024 - 1).is_err());
}

#[test]
fn nested_readers_keep_distinct_consistent_snapshots_when_connections_are_reused() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let first = pool.reader().unwrap();
    assert_eq!(first.status().unwrap().sessions, 0);
    fixture
        .session(crate::fixture::SessionFacts {
            id: "test:new",
            ..Default::default()
        })
        .unwrap();
    let second = pool.reader().unwrap();
    assert_eq!(second.status().unwrap().sessions, 1);
    assert_eq!(first.status().unwrap().sessions, 0);
    drop(second);
    let reused = pool.reader().unwrap();
    assert_eq!(reused.status().unwrap().sessions, 1);
    assert_eq!(first.status().unwrap().sessions, 0);
}

mod excision;

mod snapshot_admission;

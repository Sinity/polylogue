use crate::{ReadPool, StoreError, fixture::Fixture, read::limit::ReadLimit};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
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

#[test]
fn an_old_publisher_cannot_reactivate_its_pointer_after_scope_invalidation() {
    let root = tempfile::tempdir().unwrap();
    let archive = crate::Archive::at(root.path());
    let available = Arc::new(AtomicBool::new(true));
    let invalidated = Arc::clone(&available);
    let _registration = crate::Archive::register_scope_invalidator(root.path(), move || {
        invalidated.store(false, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    assert!(!available.load(Ordering::SeqCst));
    let token = archive.scope_token().unwrap();
    std::thread::scope(|threads| {
        let (started, publishing) = std::sync::mpsc::channel();
        let (release, waiting) = std::sync::mpsc::channel();
        let pointer = &available;
        let publisher_token = &token;
        let publisher = threads.spawn(move || {
            publisher_token.with_current(|| {
                started.send(()).unwrap();
                waiting.recv().unwrap();
                pointer.store(true, Ordering::SeqCst);
                Ok(())
            })
        });
        publishing.recv().unwrap();
        let state = &archive.read_scope;
        let reset = threads.spawn(move || {
            let mut reset = state.begin_reset().unwrap();
            reset.invalidate().unwrap();
            drop(reset);
            state.finish_reset();
        });
        let until = Instant::now() + Duration::from_secs(5);
        while token.check().is_ok() {
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        release.send(()).unwrap();
        publisher.join().unwrap().unwrap();
        reset.join().unwrap();
    });
    assert!(!available.load(Ordering::SeqCst));
    assert!(matches!(
        token.with_current(|| {
            available.store(true, Ordering::SeqCst);
            Ok(())
        }),
        Err(StoreError::ScopeChanged)
    ));
    assert!(!available.load(Ordering::SeqCst));
}

#[test]
fn a_preopen_registration_shares_the_later_archive_scope_and_retires_with_its_guard() {
    let root = tempfile::tempdir().unwrap();
    let alias = root.path().join("alias");
    let actual = root.path().join("actual");
    std::fs::create_dir(&actual).unwrap();
    std::os::unix::fs::symlink(&actual, &alias).unwrap();
    let missing = alias.join("cache");
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let registration = crate::Archive::register_scope_invalidator(&missing, move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    std::fs::create_dir(&missing).unwrap();
    let archive = crate::Archive::at(&actual.join("cache"));
    {
        let mut reset = archive.read_scope.begin_reset().unwrap();
        reset.invalidate().unwrap();
    }
    archive.read_scope.finish_reset();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    drop(registration);
    archive
        .read_scope
        .begin_reset()
        .unwrap()
        .invalidate()
        .unwrap();
    archive.read_scope.finish_reset();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn a_second_starter_cannot_invalidate_the_active_daemons_pointer() {
    let root = tempfile::tempdir().unwrap();
    for mode in ["standalone", "primary", "cache"] {
        let path = root.path().join(mode);
        let probe = crate::Archive::at(&path);
        let available = Arc::new(AtomicBool::new(true));
        let pointer = Arc::clone(&available);
        let invalidate = move || {
            if !probe.lock_held() {
                return Err(StoreError::Invalid(
                    "invalidation preceded writer custody".into(),
                ));
            }
            pointer.store(false, Ordering::SeqCst);
            Ok(())
        };
        let (archive, _lock, _registration) = match mode {
            "standalone" => crate::Archive::open_or_create_with_invalidator(&path, invalidate),
            "primary" => crate::Archive::open_primary_with_invalidator(
                &path,
                &path.join("bytes"),
                invalidate,
            ),
            _ => crate::Archive::open_primary_cache_with_invalidator(&path, invalidate),
        }
        .unwrap();
        let current = archive.scope_token().unwrap();
        current
            .with_current(|| {
                available.store(true, Ordering::SeqCst);
                Ok(())
            })
            .unwrap();
        let contender = Arc::clone(&available);
        let attempted = move || {
            contender.store(false, Ordering::SeqCst);
            Ok(())
        };
        let second = match mode {
            "standalone" => crate::Archive::open_or_create_with_invalidator(&path, attempted),
            "primary" => {
                crate::Archive::open_primary_with_invalidator(&path, &path.join("bytes"), attempted)
            }
            _ => crate::Archive::open_primary_cache_with_invalidator(&path, attempted),
        };
        assert!(matches!(second, Err(StoreError::Locked { .. })));
        assert!(available.load(Ordering::SeqCst));
        assert!(current.check().is_ok());
    }
}

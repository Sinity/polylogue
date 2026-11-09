use super::*;
use polylogue_model::read::{FoldProfile, ReadOptions};
use polylogue_store::read::limit::ReadCancellation;
use polylogue_store::{
    ReadPool,
    fixture::writer::{WriterArchive, prose, sid},
};

#[test]
fn repeated_reads_reuse_prepared_positions_and_append_preserves_older_snapshots() {
    let mut archive = WriterArchive::new();
    archive.write("reading", prose("reading", 0, 1000, "first"));
    let cache = ItemCache::default();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let old = cache
        .get(&pool.reader().unwrap(), &sid("reading"))
        .unwrap()
        .unwrap();
    let profile = FoldProfile::of(&ReadOptions::default());
    let before = cache.visible(&old, profile);
    let same = cache.visible(&old, profile);
    assert!(Arc::ptr_eq(&before, &same));
    let old_entries = old.entries.clone();
    let old_sums = (*before).clone();
    archive.write("reading", prose("reading", 1, 2000, "next"));
    let now = cache
        .get(&pool.reader().unwrap(), &sid("reading"))
        .unwrap()
        .unwrap();
    assert!(now.entries.len() > old.entries.len());
    assert_eq!(now.versions.rewrite_version, old.versions.rewrite_version);
    assert_eq!(old.entries, old_entries);
    assert_eq!(*before, old_sums);
    let after = cache.visible(&now, profile);
    assert_eq!(
        *after,
        visible_entries(
            &[Segment {
                entries: &now.entries,
                inherited: false
            }],
            &profile
        )
    );
    for (at, entry) in now.entries.iter().enumerate() {
        assert_eq!(now.places.get(&entry.key), Some(&at));
    }
}

/// Holds a cache's next full load until the test releases it.
#[derive(Debug, Default)]
pub(super) struct Pause {
    /// (load started, load released)
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}

impl Pause {
    pub(super) fn hold(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.changed.notify_all();
        while !state.1 {
            state = self.changed.wait(state).unwrap();
        }
    }

    fn wait_started(&self) {
        let mut state = self.state.lock().unwrap();
        while !state.0 {
            state = self.changed.wait(state).unwrap();
        }
    }

    fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.changed.notify_all();
    }
}

/// Blocks until `waiters` readers wait on the cache's load of `session`.
fn wait_for_waiters(cache: &ItemCache, session: &SessionId, waiters: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let waiting = cache
            .loading
            .lock()
            .unwrap()
            .get(session)
            .map_or(0, |load| load.waiting.load(Ordering::Relaxed));
        if waiting == waiters {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "no reader waited");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn concurrent_reads_of_one_session_share_one_load() {
    let mut archive = WriterArchive::new();
    archive.write("shared", prose("shared", 0, 1000, "first"));
    let cache = Arc::new(ItemCache::default());
    let pause = Arc::new(Pause::default());
    *cache.pause.lock().unwrap() = Some(Arc::clone(&pause));
    let root = archive.archive.clone();
    let read = |cache: Arc<ItemCache>, root: polylogue_store::Archive| {
        std::thread::spawn(move || {
            let pool = ReadPool::open(&root).unwrap();
            cache
                .get(&pool.reader().unwrap(), &sid("shared"))
                .unwrap()
                .unwrap()
        })
    };
    let first = read(Arc::clone(&cache), root.clone());
    pause.wait_started();
    let second = read(Arc::clone(&cache), root);
    wait_for_waiters(&cache, &sid("shared"), 1);
    pause.release();
    let (first, second) = (first.join().unwrap(), second.join().unwrap());
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(cache.loads(), 1);
}

#[test]
fn a_cancelled_read_stops_waiting_for_another_reads_load() {
    let mut archive = WriterArchive::new();
    archive.write("slow", prose("slow", 0, 1000, "first"));
    let cache = Arc::new(ItemCache::default());
    let pause = Arc::new(Pause::default());
    *cache.pause.lock().unwrap() = Some(Arc::clone(&pause));
    let root = archive.archive.clone();
    let leader = {
        let (cache, root) = (Arc::clone(&cache), root.clone());
        std::thread::spawn(move || {
            let pool = ReadPool::open(&root).unwrap();
            cache.get(&pool.reader().unwrap(), &sid("slow")).unwrap()
        })
    };
    pause.wait_started();
    let cancellation = ReadCancellation::default();
    let waiter = {
        let (cache, cancellation) = (Arc::clone(&cache), cancellation.clone());
        std::thread::spawn(move || {
            let pool = ReadPool::open(&root).unwrap();
            let reader = pool.reader().unwrap();
            ReadLimit::without_deadline(cancellation).scope(|| cache.get(&reader, &sid("slow")))
        })
    };
    wait_for_waiters(&cache, &sid("slow"), 1);
    cancellation.cancel();
    let stopped = waiter.join().unwrap();
    assert!(
        matches!(stopped, Err(polylogue_store::StoreError::ReadCancelled)),
        "{stopped:?}"
    );
    assert_eq!(cache.loads(), 1, "the cancelled read loaded nothing itself");
    pause.release();
    assert!(leader.join().unwrap().is_some());
}

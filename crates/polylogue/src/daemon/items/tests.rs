use super::*;
use polylogue_model::read::{FoldProfile, ReadOptions};
use polylogue_store::read::limit::ReadCancellation;
use polylogue_store::{
    ReadPool,
    fixture::writer::{WriterArchive, prose, sid},
};
use std::time::Duration;

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

fn event_entry(event: &str, cleared: Vec<String>) -> IndexEntry {
    IndexEntry {
        key: EntryKey::Event {
            event: event.to_owned(),
        },
        kind: EntryKind::Event,
        written_by: polylogue_model::WrittenBy::Harness,
        prompt_kind: None,
        active_path: None,
        branch_point: false,
        at: None,
        body_bytes: 0,
        tokens_line: 0,
        tokens_full: 0,
        output_lines: 0,
        action: None,
        target: None,
        call_id: None,
        event_kind: Some("microcompaction".to_owned()),
        boundary: None,
        context_tokens: None,
        cleared,
    }
}

/// The cache's size bound counts the strings an index holds, in its entries
/// and again in its key map, not a fixed size per entry: a session whose
/// entries hold large strings takes its real share of the bound.
#[test]
fn a_cached_index_is_sized_by_the_strings_it_holds() {
    let index = |cleared: usize| {
        CachedIndex::new(
            sid("sized"),
            1,
            ItemVersions {
                append_seq: 1,
                rewrite_version: 1,
            },
            (0..100)
                .map(|i| event_entry(&format!("event-{i:04}"), vec!["c".repeat(cleared)]))
                .collect(),
        )
    };
    let small = index(0);
    let large = index(10_000);
    // Each of the 100 entries holds 10,000 more bytes of cleared call IDs.
    assert!(
        large.bytes() - small.bytes() >= 100 * 10_000,
        "{} vs {}",
        large.bytes(),
        small.bytes()
    );
    // Every entry's event ID is held twice: by the entry and by the key map.
    let keyed = |event_bytes: usize| {
        CachedIndex::new(
            sid("sized"),
            1,
            ItemVersions {
                append_seq: 1,
                rewrite_version: 1,
            },
            (0..100)
                .map(|i| event_entry(&format!("{i:0event_bytes$}"), Vec::new()))
                .collect(),
        )
        .bytes()
    };
    assert!(keyed(1_000) - keyed(10) >= 2 * 100 * 990);
}

#[test]
fn older_reader_never_uses_newer_cache() {
    let mut archive = WriterArchive::new();
    archive.write("aging", prose("aging", 0, 1000, "first"));
    let cache = ItemCache::default();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let older = pool.reader().unwrap();
    let before = cache.get(&older, &sid("aging")).unwrap().unwrap();
    archive.write(
        "aging",
        [
            prose("aging", 1, 2000, "second"),
            prose("aging", 2, 3000, "third"),
        ]
        .concat(),
    );
    let newer = cache
        .get(&pool.reader().unwrap(), &sid("aging"))
        .unwrap()
        .unwrap();
    assert!(newer.entries.len() > before.entries.len());
    // The read that began first still sees the session as it began.
    let seen = cache.get(&older, &sid("aging")).unwrap().unwrap();
    assert_eq!(seen.entries, before.entries);
    assert_eq!(seen.versions, before.versions);
    // And the newer index stays cached for newer reads.
    let again = cache
        .get(&pool.reader().unwrap(), &sid("aging"))
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&again, &newer));
}

#[test]
fn cold_and_warm_cache_answers_agree() {
    let mut archive = WriterArchive::new();
    archive.write("both", prose("both", 0, 1000, "first"));
    let warm = ItemCache::default();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let agree = |warm: &ItemCache| {
        let reader = pool.reader().unwrap();
        let held = warm.get(&reader, &sid("both")).unwrap().unwrap();
        let cold = ItemCache::default()
            .get(&reader, &sid("both"))
            .unwrap()
            .unwrap();
        assert_eq!(held.versions, cold.versions);
        assert_eq!(held.entries, cold.entries);
        assert_eq!(held.places, cold.places);
        held.versions
    };
    let first = agree(&warm);
    archive.write("both", prose("both", 1, 2000, "second"));
    let appended = agree(&warm);
    assert_eq!(appended.rewrite_version, first.rewrite_version);
    assert!(appended.append_seq > first.append_seq);
    archive.write_snapshot_at("both", 1, prose("both", 1, 2000, "second, revised"));
    let rewritten = agree(&warm);
    assert!(rewritten.rewrite_version > appended.rewrite_version);
    archive.write(
        "both",
        [
            prose("both", 2, 3000, "third"),
            prose("both", 3, 4000, "fourth"),
        ]
        .concat(),
    );
    agree(&warm);
}

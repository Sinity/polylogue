use super::*;
use polylogue_model::read::{FoldProfile, ReadOptions};
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

use super::*;

#[test]
fn scope_discard_requires_drained_readings_and_preserves_identity_highwater() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let index = crate::archive::open_read(&archive.index_db()).unwrap();
    let resolver = Resolver::new(&archive, &index, IndexUpkeep::PerTransaction).unwrap();
    let key = Key::new(Kind::Session, b"private-session".as_slice());
    let first = resolver.identity(key.clone(), None).unwrap();
    let key_lock = resolver.lock_keys(BTreeSet::from([Key::new(
        Kind::Input,
        b"private-input".as_slice(),
    )]));
    assert!(resolver.discard_cached().is_err());
    assert_eq!(resolver.active.lock().unwrap().values.len(), 1);
    drop(key_lock);
    resolver.discard_cached().unwrap();
    assert!(resolver.active.lock().unwrap().values.is_empty());
    let next = resolver.identity(key, None).unwrap();
    assert!(next.pk > first.pk);
}

#[test]
fn released_reservations_keep_their_row_ids_for_older_snapshots_after_spilling() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let archive = archive.with_memory_budget(crate::archive::MemoryBudget::from_allowance(1 << 20));
    let index = crate::archive::open_read(&archive.index_db()).unwrap();
    let resolver = Resolver::new(&archive, &index, IndexUpkeep::PerTransaction).unwrap();
    let first = resolver
        .identity(Key::new(Kind::Session, b"first".as_slice()), None)
        .unwrap()
        .pk;
    for n in 0..1_000 {
        resolver
            .identity(Key::new(Kind::Session, format!("session-{n}")), None)
            .unwrap();
    }
    let same = resolver
        .identity(Key::new(Kind::Session, b"first".as_slice()), None)
        .unwrap();
    assert_eq!(same.pk, first);
}

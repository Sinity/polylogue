use super::*;

#[test]
fn scope_discard_requires_drained_readings_and_preserves_identity_highwater() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let index = crate::archive::open_read(&archive.index_db()).unwrap();
    let resolver = Resolver::new(&archive, &index, None).unwrap();
    let key = Key::new(Kind::Text, b"private-text-hash".as_slice());
    let first = resolver.identity(key.clone(), None).unwrap();
    let lease = resolver.lease(BTreeSet::from([Key::new(
        Kind::Input,
        b"private-input".as_slice(),
    )]));
    assert!(resolver.discard_cached().is_err());
    assert_eq!(resolver.active.lock().unwrap().values.len(), 1);
    drop(lease);
    resolver.discard_cached().unwrap();
    assert!(resolver.active.lock().unwrap().values.is_empty());
    let next = resolver.identity(key, None).unwrap();
    assert!(next.pk > first.pk);
}

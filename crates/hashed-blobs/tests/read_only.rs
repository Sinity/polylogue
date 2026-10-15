use std::io::Read;
use hashed_blobs::{Reader, Store};

#[test]
fn missing_read_root_is_never_created() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing");
    let reader = Reader::new(&missing, vec![]);
    assert!(reader.read_range(&hashed_blobs::Hash(*blake3::hash(b"x").as_bytes()), 0, 1).is_err());
    assert!(!missing.exists());
}

#[test]
fn existing_root_uses_the_same_checked_decoder() {
    let temp = tempfile::tempdir().unwrap();
    let writer = Store::open(temp.path(), vec![], 0o600).unwrap();
    let hash = writer.put(&b"synthetic bytes"[..], 1).unwrap();
    let reader = Reader::new(temp.path(), vec![]);
    assert_eq!(reader.read_range(&hash, 2, 4).unwrap(), b"nthe");
    let mut full = Vec::new();
    reader.read(&hash).unwrap().read_to_end(&mut full).unwrap();
    assert_eq!(full,b"synthetic bytes");
    assert!(reader.read_range(&hash, 12, 8).is_err());
}

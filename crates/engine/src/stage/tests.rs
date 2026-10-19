use super::*;

#[test]
fn detects_a_changed_tail() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("f");
    std::fs::write(&p, b"hello world\nmore\n").unwrap();
    assert!(tail_intact(&p, 12, b"world\n").unwrap());
    assert!(!tail_intact(&p, 12, b"WORLD\n").unwrap());
    assert_eq!(read_range(&p, 6, 11).unwrap(), b"world");
    assert_eq!(tail_of(b"abc"), b"abc");
}

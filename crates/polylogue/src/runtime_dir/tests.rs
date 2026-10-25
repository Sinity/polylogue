use super::*;

#[test]
fn keys_differ_by_archive_and_are_twelve_hex_digits() {
    let a = archive_key(Path::new("/tmp/a-archive"));
    let b = archive_key(Path::new("/tmp/b-archive"));
    assert_eq!(a.len(), 12);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(a, b);
}

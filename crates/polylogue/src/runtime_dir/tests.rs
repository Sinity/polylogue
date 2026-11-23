use super::*;

#[test]
fn keys_differ_by_archive_and_are_twelve_hex_digits() {
    let a = archive_key(Path::new("/tmp/a-archive"));
    let b = archive_key(Path::new("/tmp/b-archive"));
    assert_eq!(a.len(), 12);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(a, b);
}

#[test]
fn the_key_is_the_blake3_prefix_of_the_resolved_path_for_every_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    std::fs::create_dir(&archive).unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&archive, &link).unwrap();
    let resolved = archive.canonicalize().unwrap();
    let expected = blake3::hash(resolved.as_os_str().as_encoded_bytes()).to_hex()[..12].to_owned();
    assert_eq!(archive_key(&archive), expected);
    assert_eq!(archive_key(&link), expected);
    assert_eq!(archive_key(&archive.join("../archive")), expected);
}

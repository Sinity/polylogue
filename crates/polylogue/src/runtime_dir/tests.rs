use super::*;

#[test]
fn the_key_is_a_known_blake3_prefix_of_a_path_that_does_not_exist_yet() {
    // The first 12 hex digits of `printf %s /synthetic-archive/one | b3sum`.
    assert_eq!(
        archive_key(Path::new("/synthetic-archive/one")),
        "364b629fecf9"
    );
    assert_ne!(
        archive_key(Path::new("/synthetic-archive/one")),
        archive_key(Path::new("/synthetic-archive/two"))
    );
}

#[test]
fn the_key_is_the_blake3_prefix_of_the_canonical_path_for_direct_symlink_and_dotdot_spellings() {
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

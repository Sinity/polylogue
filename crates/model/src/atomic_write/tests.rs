use super::*;
use std::os::unix::fs::PermissionsExt;

#[test]
fn replacement_keeps_open_readers_on_the_previous_complete_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings");
    fs::write(&path, b"before").unwrap();
    let previous = fs::File::open(&path).unwrap();
    write_atomic(&path, b"after", Some(fs::Permissions::from_mode(0o640))).unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"after");
    assert_eq!(std::io::read_to_string(previous).unwrap(), "before");
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn replacement_is_private_and_cleans_up_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("token");
    write_atomic(&path, b"token", None).unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let destination = dir.path().join("directory");
    fs::create_dir(&destination).unwrap();
    assert!(write_atomic(&destination, b"cannot replace a directory", None).is_err());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn private_publication_preserves_a_destination_that_already_exists() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("credential");
    super::write_atomic_if_absent(&path, b"original").unwrap();
    assert_eq!(
        super::write_atomic_if_absent(&path, b"replacement")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

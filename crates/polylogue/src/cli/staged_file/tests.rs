use super::*;
use std::os::unix::fs::{PermissionsExt, symlink};

#[test]
fn file_output_keeps_previous_bytes_until_validation_and_preserves_mode() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("output");
    fs::write(&path, b"previous complete output").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    {
        let mut pending = StagedFile::new(&path).unwrap();
        pending.write_all(b"unverified partial bytes").unwrap();
        let temporary = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|entry| entry != &path)
            .unwrap();
        assert_eq!(
            fs::metadata(temporary).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(fs::read(&path).unwrap(), b"previous complete output");
    }
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    assert_eq!(fs::read(&path).unwrap(), b"previous complete output");
    let payload = b"--secret=synthetic-value\0\xff";
    let mut pending = StagedFile::new(&path).unwrap();
    pending.write_all(payload).unwrap();
    assert_eq!(pending.commit().unwrap(), path);
    assert_eq!(fs::read(&path).unwrap(), payload);
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o640
    );
}

#[test]
fn file_output_follows_existing_and_dangling_symlinks() {
    let directory = tempfile::tempdir().unwrap();
    let link = directory.path().join("link");
    let path = directory.path().join("target");
    symlink("target", &link).unwrap();
    for payload in [b"new bytes".as_slice(), b"replacement bytes"] {
        let mut pending = StagedFile::new(&link).unwrap();
        pending.write_all(payload).unwrap();
        assert_eq!(pending.commit().unwrap(), link);
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&path).unwrap(), payload);
    }
}

#[test]
fn a_new_file_output_gets_the_ordinary_mode_under_the_umask() {
    let umask = std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("Umask:"))
        .map(|value| u32::from_str_radix(value.trim(), 8).unwrap())
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("export.md");
    let mut pending = StagedFile::new(&path).unwrap();
    pending.write_all(b"exported").unwrap();
    pending.commit().unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o666 & !umask
    );
}

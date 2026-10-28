use crate::Archive;
use std::fs;
use std::os::unix::fs::PermissionsExt;

#[test]
fn scanner_key_is_private_and_survives_reopen_and_index_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("scanner.key");
    assert!(!path.exists());
    let key = archive.scanner_key().unwrap();
    assert_eq!(fs::read(&path).unwrap(), key);
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o7777,
        0o600
    );
    drop(lock);
    fs::remove_file(archive.index_db()).unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert_eq!(archive.scanner_key().unwrap(), key);
    let other = tempfile::tempdir().unwrap();
    assert_ne!(Archive::at(other.path()).scanner_key().unwrap(), key);
}

#[test]
fn concurrent_scanner_key_creators_all_read_the_same_complete_key() {
    let dir = tempfile::tempdir().unwrap();
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let barrier = &barrier;
                let root = dir.path();
                scope.spawn(move || {
                    barrier.wait();
                    Archive::at(root).scanner_key().unwrap()
                })
            })
            .collect();
        let keys: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(keys.iter().all(|key| key == &keys[0]));
    });
}

#[test]
fn invalid_scanner_keys_are_refused_without_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    let path = dir.path().join("scanner.key");
    for bytes in [vec![], vec![7; 31], vec![7; 33]] {
        fs::write(&path, &bytes).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(archive.scanner_key().is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
    fs::write(&path, [7; 32]).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(archive.scanner_key().is_err());
    assert_eq!(fs::read(&path).unwrap(), [7; 32]);
    fs::remove_file(&path).unwrap();
    let target = dir.path().join("other.key");
    fs::write(&target, [9; 32]).unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
    std::os::unix::fs::symlink(&target, &path).unwrap();
    assert!(archive.scanner_key().is_err());
    assert_eq!(fs::read(&target).unwrap(), [9; 32]);
}

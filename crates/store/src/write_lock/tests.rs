use super::*;
use crate::archive::{cache, open_write};

#[test]
fn both_database_handles_lock_the_archive_file() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let archive_conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let guard = hold(&index).unwrap();
    let file = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        file.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(guard);
    let guard = hold(&archive_conn).unwrap();
    assert!(matches!(
        file.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(guard);
    file.try_lock_shared().unwrap();
}

#[test]
fn a_register_transaction_takes_the_lock_only_to_commit() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let mut archive_conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let tx = archive_conn.transaction().unwrap();
    tx.execute("INSERT INTO meta(key, value) VALUES ('synthetic', '1')", [])
        .unwrap();
    wait_shared(&archive, || panic!("an open transaction holds the lock")).unwrap();
    commit(tx).unwrap();
    wait_shared(&archive, || panic!("the commit kept the lock")).unwrap();
    // A commit refuses inside a held lock instead of waiting for itself.
    let tx = archive_conn.transaction().unwrap();
    let outer = hold(&tx).unwrap();
    assert!(commit(tx).is_err());
    drop(outer);
}

#[test]
fn in_memory_derivations_have_no_shared_archive_file() {
    let conn = Connection::open_in_memory().unwrap();
    let _guard = hold(&conn).unwrap();
    conn.execute("CREATE TABLE synthetic(value)", []).unwrap();
}

#[test]
fn a_nested_write_lock_and_a_new_read_refuse_without_releasing_the_outer_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let outer = hold(&index).unwrap();
    assert!(hold(&index).is_err());
    assert!(wait_shared(&archive, || panic!("must refuse before waiting")).is_err());
    let probe = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        probe.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(outer);
    probe.try_lock_shared().unwrap();
}

#[test]
fn write_lock_process_child() {
    let Some(path) = std::env::var_os("POLYLOGUE_WRITE_LOCK_TEST_ARCHIVE") else {
        return;
    };
    let conn = open_write(&std::path::PathBuf::from(path), cache::SETUP).unwrap();
    let _lock = hold(&conn).unwrap();
    println!("write-lock-held");
    use std::io::{Read, Write};
    std::io::stdout().flush().unwrap();
    let mut release = [0];
    std::io::stdin().read_exact(&mut release).unwrap();
}

#[test]
fn the_write_lock_excludes_other_processes_and_releases_on_exit() {
    use std::io::{BufRead, Read, Write};
    use std::process::{Command, Stdio};
    struct Child(std::process::Child);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let mut child = Child(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "write_lock::tests::write_lock_process_child",
                "--nocapture",
            ])
            .env("POLYLOGUE_WRITE_LOCK_TEST_ARCHIVE", archive.archive_db())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut output = std::io::BufReader::new(child.0.stdout.take().unwrap());
    let mut found = false;
    for line in output.by_ref().lines() {
        if line.unwrap().contains("write-lock-held") {
            found = true;
            break;
        }
    }
    assert!(found, "the child did not take the write lock");
    let probe = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        probe.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    child.0.stdin.take().unwrap().write_all(b"x").unwrap();
    assert!(child.0.wait().unwrap().success());
    probe.try_lock_shared().unwrap();
}

#[test]
fn a_panic_clears_ownership_before_another_commit() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let mut conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lock = hold(&conn).unwrap();
        panic!("synthetic application failure");
    }));
    assert!(panic.is_err());
    commit(conn.transaction().unwrap()).unwrap();
}

#[test]
fn aliases_share_file_identity_and_other_archives_can_read() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let (other, _other_writer) = Archive::open_or_create(&dir.path().join("other")).unwrap();
    let alias = dir.path().join("alias");
    std::os::unix::fs::symlink(&archive.root, &alias).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let _lock = hold(&conn).unwrap();
    assert!(
        wait_shared(&Archive::at(&alias), || panic!(
            "same-file read must refuse before waiting"
        ))
        .is_err()
    );
    let pool = crate::ReadPool::open(&other).unwrap();
    pool.reader().unwrap().change_position().unwrap();
}

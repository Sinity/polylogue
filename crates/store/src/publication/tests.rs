use super::*;
use crate::archive::{cache, open_write};

#[test]
fn both_database_handles_use_the_existing_canonical_file() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let canonical = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let guard = write(&index).unwrap();
    let file = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        file.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(guard);
    let guard = write(&canonical).unwrap();
    assert!(matches!(
        file.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(guard);
    file.try_lock_shared().unwrap();
}

#[test]
fn failed_or_dropped_transactions_release_publication() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let canonical = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    {
        let tx = transaction(&canonical, rusqlite::TransactionBehavior::Deferred).unwrap();
        assert!(
            tx.execute("INSERT INTO no_such_table VALUES (1)", [])
                .is_err()
        );
    }
    admit(&archive, || panic!("publication was not released")).unwrap();
}

#[test]
fn in_memory_derivations_have_no_shared_archive_file() {
    let conn = Connection::open_in_memory().unwrap();
    let _guard = write(&conn).unwrap();
    conn.execute("CREATE TABLE synthetic(value)", []).unwrap();
}

#[test]
fn nested_publication_and_new_read_refuse_without_releasing_the_outer_guard() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let outer = write(&index).unwrap();
    assert!(write(&index).is_err());
    assert!(admit(&archive, || panic!("must refuse before waiting")).is_err());
    let probe = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        probe.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(outer);
    probe.try_lock_shared().unwrap();
}

#[test]
fn publication_process_child() {
    let Some(path) = std::env::var_os("POLYLOGUE_PUBLICATION_TEST_ARCHIVE") else {
        return;
    };
    let conn = open_write(&std::path::PathBuf::from(path), cache::SETUP).unwrap();
    let _publication = write(&conn).unwrap();
    println!("publication-held");
    use std::io::{Read, Write};
    std::io::stdout().flush().unwrap();
    let mut release = [0];
    std::io::stdin().read_exact(&mut release).unwrap();
}

#[test]
fn publication_excludes_other_processes_and_releases_on_exit() {
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
                "publication::tests::publication_process_child",
                "--nocapture",
            ])
            .env("POLYLOGUE_PUBLICATION_TEST_ARCHIVE", archive.archive_db())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut output = std::io::BufReader::new(child.0.stdout.take().unwrap());
    let mut found = false;
    for line in output.by_ref().lines() {
        if line.unwrap().contains("publication-held") {
            found = true;
            break;
        }
    }
    assert!(found, "child did not acquire the actual publication guard");
    let probe = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        probe.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    child.0.stdin.take().unwrap().write_all(b"x").unwrap();
    assert!(child.0.wait().unwrap().success());
    probe.try_lock_shared().unwrap();
}

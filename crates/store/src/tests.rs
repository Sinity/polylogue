use super::*;
use std::os::unix::fs::{MetadataExt, symlink};

#[test]
fn ddl_is_exact_extraction_from_copied_owning_spec() {
    fn extract(md: &str) -> String {
        let mut blocks = Vec::new();
        let mut current = None;
        for line in md.lines() {
            if line == "```sql" {
                current = Some(String::new());
            } else if line == "```" {
                if let Some(sql) = current.take() {
                    blocks.push(sql);
                }
            } else if let Some(sql) = &mut current {
                sql.push_str(line);
                sql.push('\n');
            }
        }
        blocks.join("\n").trim_end().to_owned() + "\n"
    }
    for (md, sql) in [
        (
            include_str!("../../../docs/rewrite/polylogue-spec/spec/storage/archive-db.md"),
            ARCHIVE_SQL,
        ),
        (
            include_str!("../../../docs/rewrite/polylogue-spec/spec/storage/index-db.md"),
            INDEX_SQL,
        ),
        (
            include_str!("../../../docs/rewrite/polylogue-spec/spec/storage/embeddings.md"),
            EMBEDDINGS_SQL,
        ),
    ] {
        assert_eq!(extract(md), sql);
    }
}

#[test]
fn writer_mutates_and_reader_is_readonly_and_survives_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("archive");
    let mut writer = Writer::open(&path).unwrap();
    assert_eq!(writer.status().unwrap().write_counter, 0);
    let hash = writer.put_file_blob(b"synthetic foundation bytes").unwrap();
    assert_eq!(
        writer.put_file_blob(b"synthetic foundation bytes").unwrap(),
        hash
    );
    let read = writer.read_handle().unwrap();
    assert_eq!(read.read_blob_range(hash, 10, 10).unwrap(), b"foundation");
    assert!(matches!(
        read.read_blob_range(hash, 100, 1),
        Err(Error::Range)
    ));
    assert!(matches!(
        read.read_blob_range(hash, u64::MAX, 2),
        Err(Error::Range)
    ));
    assert!(matches!(
        read.read_blob_range(BlobHash::of(b"absent"), 0, 0),
        Err(Error::UnknownBlob)
    ));
    assert!(
        matches!(read.archive.execute("DELETE FROM blob", []), Err(rusqlite::Error::SqliteFailure(e,_)) if e.code == rusqlite::ErrorCode::ReadOnly)
    );
    assert_eq!(read.status().unwrap().write_counter, 1);
    for name in ["archive.db", "embeddings.db"] {
        assert_eq!(fs::read(path.join(name)).unwrap()[20], 8);
    }
    assert_eq!(fs::read(path.join("index.db")).unwrap()[20], 0);
    drop(read);
    drop(writer);
    let writer = Writer::open(&path).unwrap();
    assert_eq!(writer.status().unwrap().write_counter, 1);
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .read_blob_range(hash, 0, 26)
            .unwrap(),
        b"synthetic foundation bytes"
    );
}

fn snapshot(path: &Path) -> Vec<(PathBuf, Vec<u8>, u32)> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            files.push((
                path.clone(),
                fs::read(&path).unwrap(),
                path.metadata().unwrap().mode(),
            ));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}
#[test]
fn newer_durable_version_refused_before_creating_siblings_or_chmod() {
    for name in ["archive.db", "embeddings.db"] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path();
        let file = path.join(name);
        let conn = Connection::open(&file).unwrap();
        conn.execute_batch("PRAGMA user_version=99; CREATE TABLE protected(value TEXT); INSERT INTO protected VALUES('keep');").unwrap();
        drop(conn);
        fs::set_permissions(&file, fs::Permissions::from_mode(0o640)).unwrap();
        // Embeddings-only state is checked as a version error before missing-archive refusal.
        if name == "embeddings.db" {
            File::create(path.join("archive.db")).unwrap();
        }
        let before = snapshot(path);
        assert!(matches!(
            Writer::open(path),
            Err(Error::NewerSchema { found: 99, .. })
        ));
        assert!(snapshot(path) == before, "refusal changed retained files");
    }
}
#[test]
fn index_version_mismatch_requires_rebuild_without_destroying_it() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    drop(Writer::open(root).unwrap());
    let conn = open(&root.join("index.db"), false, false).unwrap();
    conn.pragma_update(None, "user_version", 99).unwrap();
    drop(conn);
    let before = snapshot(root);
    assert!(matches!(
        Writer::open(root),
        Err(Error::IndexRebuildRequired { found: 99, .. })
    ));
    assert!(snapshot(root) == before, "refusal changed retained files");
}
#[test]
fn unrelated_version_zero_schema_is_not_adopted() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    Connection::open(root.join("archive.db"))
        .unwrap()
        .execute_batch("CREATE TABLE unrelated(value TEXT)")
        .unwrap();
    let before = snapshot(root);
    assert!(matches!(Writer::open(root), Err(Error::Schema(_))));
    assert!(snapshot(root) == before, "refusal changed retained files");
}
#[test]
fn retained_bytes_without_archive_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir(root.join("blobs")).unwrap();
    assert!(matches!(Writer::open(root), Err(Error::MissingArchive(_))));
    assert!(!root.join("archive.db").exists());
}
#[test]
fn same_archive_alias_cannot_get_second_writer() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("real");
    let writer = Writer::open(&root).unwrap();
    let alias = tmp.path().join("alias");
    symlink(&root, &alias).unwrap();
    assert!(matches!(
        Writer::open(&alias),
        Err(Error::WriterHeld { .. })
    ));
    assert!(ReadHandle::open(&alias).is_ok());
    drop(writer);
    assert!(Writer::open(&alias).is_ok());
}
#[test]
fn files_and_directories_are_private() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("archive");
    let mut writer = Writer::open(&root).unwrap();
    writer.put_file_blob(b"private synthetic bytes").unwrap();
    fn walk(path: &Path) {
        let metadata = path.metadata().unwrap();
        assert_eq!(
            metadata.mode() & 0o777,
            if metadata.is_dir() { 0o700 } else { 0o600 },
            "{}",
            path.display()
        );
        if metadata.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                walk(&entry.unwrap().path());
            }
        }
    }
    walk(&root);
}
#[test]
fn corrupted_blob_frame_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut writer = Writer::open(root).unwrap();
    let hash = writer.put_file_blob(&vec![42; 8000]).unwrap();
    let h = hashed_blobs::Hash(*hash.bytes());
    let file = root
        .join("blobs")
        .join(&h.hex()[..2])
        .join(format!("{}.zst", h.hex()));
    let mut bytes = fs::read(&file).unwrap();
    bytes[8] ^= 0x80;
    fs::write(file, bytes).unwrap();
    assert!(
        writer
            .read_handle()
            .unwrap()
            .read_blob_range(hash, 0, 8000)
            .is_err()
    );
}
#[test]
fn durable_page_corruption_is_rejected_by_checksum_vfs() {
    for name in ["archive.db", "embeddings.db"] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        drop(Writer::open(root).unwrap());
        let path = root.join(name);
        let mut bytes = fs::read(&path).unwrap();
        bytes[200] ^= 0x40;
        fs::write(&path, bytes).unwrap();
        let err = Writer::open(root).err().expect("corrupt page must refuse");
        assert!(
            matches!(err,Error::Sqlite {ref file, source:rusqlite::Error::SqliteFailure(e,_)} if file==&path && e.extended_code==rusqlite::ffi::SQLITE_IOERR_DATA),
            "{err}"
        );
    }
}
#[test]
fn readonly_byte_access_does_not_recreate_writer_scratch() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut writer = Writer::open(root).unwrap();
    let hash = writer.put_file_blob(b"read without mutation").unwrap();
    let read = writer.read_handle().unwrap();
    fs::remove_dir(root.join("blobs/tmp")).unwrap();
    assert_eq!(read.read_blob_range(hash, 0, 4).unwrap(), b"read");
    assert!(!root.join("blobs/tmp").exists());
}

#[test]
fn orphan_wal_is_refused_without_recovery_side_effects() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    drop(Writer::open(root).unwrap());
    fs::write(root.join("archive.db-wal"), b"retained orphan WAL marker").unwrap();
    let before = snapshot(root);
    assert!(matches!(Writer::open(root), Err(Error::Initialization(_))));
    assert!(
        snapshot(root) == before,
        "inspection must preserve all retained files"
    );
}

#[test]
fn index_metadata_version_is_checked_even_when_pragma_matches() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    drop(Writer::open(root).unwrap());
    let conn = open(&root.join("index.db"), false, false).unwrap();
    conn.execute("UPDATE meta SET value='98' WHERE key='INDEX_VERSION'", [])
        .unwrap();
    drop(conn);
    let before = snapshot(root);
    assert!(matches!(
        Writer::open(root),
        Err(Error::IndexRebuildRequired { found: 98, .. })
    ));
    assert!(snapshot(root) == before, "refusal must preserve files");
}
#[test]
fn malformed_archive_counter_is_refused_before_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    drop(Writer::open(root).unwrap());
    let conn = open(&root.join("archive.db"), false, true).unwrap();
    conn.execute(
        "UPDATE meta SET value='unknown' WHERE key='write_counter'",
        [],
    )
    .unwrap();
    drop(conn);
    let before = snapshot(root);
    assert!(matches!(Writer::open(root), Err(Error::Schema(_))));
    assert!(snapshot(root) == before, "refusal must preserve files");
}

#[test]
fn non_utf8_path_reads_retained_wal_version_before_mutation() {
    use std::os::unix::ffi::OsStringExt;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp
        .path()
        .join(std::ffi::OsString::from_vec(b"archive-\xff".to_vec()));
    drop(Writer::open(&root).unwrap());
    let file = root.join("archive.db");
    let conn = open(&file, false, true).unwrap();
    conn.pragma_update(None, "user_version", 99).unwrap();
    assert!(sibling(&file, "-wal").metadata().unwrap().len() > 0);
    let before = snapshot(&root);
    assert!(matches!(
        Writer::open(&root),
        Err(Error::NewerSchema { found: 99, .. })
    ));
    let after = snapshot(&root);
    // SQLite read marks in the existing SHM are coordination state, not durable conversation bytes.
    let durable = |entries: Vec<(PathBuf, Vec<u8>, u32)>| {
        entries
            .into_iter()
            .filter(|(p, _, _)| !p.as_os_str().as_bytes().ends_with(b"-shm"))
            .collect::<Vec<_>>()
    };
    assert!(
        durable(before) == durable(after),
        "refusal changed retained files"
    );
    drop(conn);
}
#[test]
fn inaccessible_wal_metadata_is_not_treated_as_no_wal() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    drop(Writer::open(root).unwrap());
    // A self-referential symlink yields ELOOP when followed; it must never select immutable inspection.
    let wal = root.join("archive.db-wal");
    symlink("archive.db-wal", &wal).unwrap();
    assert!(matches!(Writer::open(root), Err(Error::Schema(_))));
}

#[test]
fn missing_or_empty_main_never_overwrites_retained_wal_or_shm() {
    for create_main in [false, true] {
        for sidecar in ["archive.db-wal", "archive.db-shm"] {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            if create_main {
                File::create(root.join("archive.db")).unwrap();
            }
            fs::write(root.join(sidecar), b"retained recovery bytes").unwrap();
            let before = snapshot(root);
            assert!(matches!(Writer::open(root), Err(Error::Initialization(_))));
            assert!(
                snapshot(root) == before,
                "recovery refusal changed retained files"
            );
        }
    }
}

#[test]
fn main_metadata_errors_are_not_treated_as_a_fresh_archive() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("regular-file");
    fs::write(&root, b"preserve this file").unwrap();
    let error = Writer::open(&root).err().expect("invalid root must refuse");
    assert!(matches!(error, Error::Io(ref e) if e.kind() == io::ErrorKind::NotADirectory));
    assert_eq!(fs::read(root).unwrap(), b"preserve this file");
}

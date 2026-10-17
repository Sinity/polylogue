use super::*;
use rusqlite::params;
use std::os::unix::fs::{MetadataExt, symlink};

// A concurrent subprocess test may briefly inherit CLOEXEC file descriptions
// between fork and exec. This harness expects eventual reopen after its own
// writer drop, not zero contention time. A leaked read authority still fails
// at the deadline; product startup keeps its immediate WriterHeld contract.
pub(crate) fn reopen_after_drop(root: &std::path::Path) -> crate::Writer {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        match crate::Writer::open(root) {
            Ok(writer) => return writer,
            Err(crate::Error::WriterHeld { ref holder, .. })
                if holder == &format!("pid={}", std::process::id())
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(error) => panic!("reopen after owned writer drop: {error}"),
        }
    }
}

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
fn context_qualified_archive_uses_explicit_owning_schema_basis() {
    assert_eq!(ARCHIVE_VERSION, 3);
    let provenance = include_str!("../schema/provenance.json");
    assert!(provenance.contains(&format!("\"spec_commit\": \"{ARCHIVE_SPEC_COMMIT}\"")));
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(ARCHIVE_SQL).unwrap();
    assert!(
        conn.execute(
            "INSERT INTO input(id,format,context_json) VALUES('torn','codex-rollout','[]')",
            []
        )
        .is_err()
    );
    assert!(conn.execute("INSERT INTO input(id,format,original_hash,context_json) VALUES('wrong','codex-rollout',zeroblob(31),'[]')",[]).is_err());
    conn.execute("INSERT INTO input(id,format,original_hash,context_json) VALUES('complete','codex-rollout',zeroblob(32),'[]')",[]).unwrap();
}

#[test]
fn forward_index_is_pinned_and_applies_without_predecessor_columns() {
    assert_eq!(
        BlobHash::of(INDEX_SQL.as_bytes()).to_string(),
        INDEX_SQL_BLAKE3
    );
    let provenance = include_str!("../schema/provenance.json");
    assert!(provenance.contains(&format!("\"spec_commit\": \"{INDEX_SPEC_COMMIT}\"")));
    let mut changed = INDEX_SQL.as_bytes().to_vec();
    changed.push(b' ');
    assert_ne!(BlobHash::of(&changed).to_string(), INDEX_SQL_BLAKE3);
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys=ON").unwrap();
    conn.execute_batch(INDEX_SQL).unwrap();
    let mut tables = conn
        .prepare("SELECT name FROM sqlite_schema WHERE type='table'")
        .unwrap();
    let tables = tables
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    for name in [
        "message_body",
        "message_body_node",
        "message_build",
        "tool_item_association",
    ] {
        assert!(tables.iter().any(|v| v == name));
    }
    let mut columns = conn.prepare("PRAGMA table_info(message)").unwrap();
    let columns = columns
        .query_map([], |r| r.get::<_, String>(1))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(columns.iter().any(|v| v == "selected_body_pk"));
    assert!(!columns.iter().any(|v| v == "role"));
    assert!(
        conn.prepare("PRAGMA foreign_key_check")
            .unwrap()
            .query([])
            .unwrap()
            .next()
            .unwrap()
            .is_none()
    );
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
    assert!(matches!(
        Writer::open(root),
        Err(Error::RecoveryRequired { .. })
    ));
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
    assert!(matches!(
        ReadHandle::open(&alias),
        Err(Error::WriterHeld { .. })
    ));
    assert!(writer.read_handle().is_ok());
    drop(writer);
    drop(reopen_after_drop(&alias));
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
            err.to_string().contains("page 1"),
            "checksum failure must name its actual page: {err}"
        );
        assert!(
            matches!(err,Error::CorruptPage {ref file, page:1, offset:0, source:rusqlite::Error::SqliteFailure(e,_), ..} if file==&path && e.extended_code==rusqlite::ffi::SQLITE_IOERR_DATA),
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
    assert!(matches!(
        Writer::open(root),
        Err(Error::RecoveryRequired { .. })
    ));
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
            assert!(matches!(
                Writer::open(root),
                Err(Error::RecoveryRequired { .. })
            ));
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

// Return a closed, checksummed store with a damaged non-header table page.
fn corrupt_table(name: &str, table: &str) -> (tempfile::TempDir, PathBuf, u64, u32) {
    let tmp = tempfile::tempdir().unwrap();
    drop(Writer::open(tmp.path()).unwrap());
    let file = tmp.path().join(name);
    let conn = open(&file, true, true).unwrap();
    let page: u32 = conn
        .query_row(
            "SELECT rootpage FROM sqlite_schema WHERE name=?1",
            [table],
            |r| r.get(0),
        )
        .unwrap();
    let page_size: u32 = conn
        .query_row("PRAGMA page_size", [], |r| r.get(0))
        .unwrap();
    drop(conn);
    let page = u64::from(page);
    assert!(page > 1);
    let mut bytes = fs::read(&file).unwrap();
    bytes[((page - 1) * u64::from(page_size) + 200) as usize] ^= 0x40;
    fs::write(&file, bytes).unwrap();
    (tmp, file, page, page_size)
}

#[test]
fn checksum_failure_names_non_header_page_and_actual_file() {
    for (name, table) in [("archive.db", "meta"), ("embeddings.db", "embedded")] {
        let (_tmp, file, page, page_size) = corrupt_table(name, table);
        let conn = open(&file, true, true).unwrap();
        // Deliberately different context: the VFS filename, not the caller hint,
        // is the authority for which physical file failed.
        let err = db(Path::new("caller-context"), || {
            conn.query_row(
                &format!("SELECT count(*) FROM {table} NOT INDEXED"),
                [],
                |r| r.get::<_, i64>(0),
            )
        })
        .unwrap_err();
        assert!(
            matches!(&err, Error::CorruptPage {file: actual, page: actual_page, offset, page_size: actual_size, source:rusqlite::Error::SqliteFailure(e,_)} if actual == &file && *actual_page == page && *offset == (page-1)*u64::from(page_size) && *actual_size==page_size && e.extended_code==rusqlite::ffi::SQLITE_IOERR_DATA),
            "{err}"
        );
        assert!(native::take_checksum_fault().is_none());
    }
}

#[test]
fn checksum_fault_locations_are_thread_local() {
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let handles: Vec<_> = [("archive.db", "meta"), ("embeddings.db", "embedded")].into_iter().map(|(name, table)| {
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            let (_tmp, file, page, _) = corrupt_table(name, table);
            let conn = open(&file, true, true).unwrap();
            let err = db(&file, || {
                let result = conn.query_row(&format!("SELECT count(*) FROM {table} NOT INDEXED"), [], |r| r.get::<_, i64>(0));
                assert!(result.is_err());
                // Both faults have happened before either wrapper consumes one.
                barrier.wait();
                result
            }).unwrap_err();
            assert!(matches!(&err, Error::CorruptPage { file: actual, page: actual_page, .. } if actual==&file && *actual_page==page), "{err}");
        })
    }).collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn unscoped_or_successful_fault_cannot_taint_later_errors() {
    let (_tmp, file, _, _) = corrupt_table("archive.db", "meta");
    let conn = open(&file, true, true).unwrap();
    assert!(
        conn.query_row("SELECT count(*) FROM meta NOT INDEXED", [], |r| r
            .get::<_, i64>(0))
            .is_err()
    );
    let synthetic = || {
        Err::<(), _>(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR_DATA),
            None,
        ))
    };
    // Stale fault outside the scoped operation cannot supply a page.
    assert!(matches!(db(&file, synthetic), Err(Error::Sqlite { .. })));
    db(&file, || {
        assert!(
            conn.query_row("SELECT count(*) FROM meta NOT INDEXED", [], |r| r
                .get::<_, i64>(0))
                .is_err()
        );
        Ok(())
    })
    .unwrap();
    assert!(native::take_checksum_fault().is_none());
    assert!(matches!(db(&file, synthetic), Err(Error::Sqlite { .. })));
    let err = db(&file, || {
        assert!(
            conn.query_row("SELECT count(*) FROM meta NOT INDEXED", [], |r| r
                .get::<_, i64>(0))
                .is_err()
        );
        Err::<(), _>(rusqlite::Error::InvalidQuery)
    })
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Sqlite {
            source: rusqlite::Error::InvalidQuery,
            ..
        }
    ));
    assert!(native::take_checksum_fault().is_none());
}

#[test]
fn public_reader_reports_corrupt_archive_page_without_repair() {
    let (tmp, file, page, _) = corrupt_table("archive.db", "meta");
    let before = fs::read(&file).unwrap();
    let err = ReadHandle::open(tmp.path())
        .err()
        .expect("reader must refuse corrupt meta");
    assert!(
        matches!(&err, Error::CorruptPage {file: actual, page: actual_page,..} if actual==&file && *actual_page==page),
        "{err}"
    );
    assert_eq!(fs::read(&file).unwrap(), before);
}

// The parent treats marker existence as a complete acknowledgement. Publish
// after the payload write so it cannot observe fs::write's empty creation window.
fn publish_wal_ack(path: &Path, value: &[u8]) {
    let pending = path.with_extension("pending");
    fs::write(&pending, value).unwrap();
    fs::rename(pending, path).unwrap();
}

#[test]
#[ignore = "owned subprocess helper for retained WAL probe"]
fn retained_wal_probe_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_WAL_CHILD_ROOT") else {
        return;
    };
    let ack = PathBuf::from(std::env::var_os("POLYLOGUE_WAL_CHILD_ACK").unwrap());
    let mut writer = Writer::open(Path::new(&root)).unwrap();
    if std::env::var("POLYLOGUE_WAL_CHILD_MODE").as_deref() == Ok("embeddings") {
        writer
            ._embeddings
            .pragma_update(None, "wal_autocheckpoint", 0)
            .unwrap();
        insert_synthetic_embedding(&writer._embeddings, 2);
        insert_synthetic_embedding(&writer._embeddings, 3);
        assert_eq!(
            writer
                ._embeddings
                .query_row("SELECT count(*) FROM embedded", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            3
        );
        publish_wal_ack(&ack, b"3");
        loop {
            std::thread::park();
        }
    }
    writer
        .archive
        .pragma_update(None, "wal_autocheckpoint", 0)
        .unwrap();
    if std::env::var("POLYLOGUE_WAL_CHILD_MODE").as_deref() == Ok("legacy-normal") {
        writer
            .archive
            .pragma_update(None, "user_version", 1)
            .unwrap();
    }
    let first: &[u8] = if std::env::var("POLYLOGUE_WAL_CHILD_MODE").as_deref() == Ok("stale-empty")
    {
        b""
    } else {
        b"first acknowledged WAL commit"
    };
    writer.put_file_blob(first).unwrap();
    writer
        .put_file_blob(b"second acknowledged WAL commit")
        .unwrap();
    assert_eq!(writer.status().unwrap().write_counter, 3);
    if std::env::var("POLYLOGUE_WAL_CHILD_MODE").as_deref() == Ok("stale-newer") {
        writer
            .archive
            .pragma_update(None, "user_version", 99)
            .unwrap();
    }
    match std::env::var("POLYLOGUE_WAL_CHILD_MODE").as_deref() {
        Ok("stale-generation" | "stale-newer" | "stale-empty") => {
            writer
                .archive
                .execute_batch("PRAGMA wal_checkpoint(RESTART)")
                .unwrap();
            writer
                .put_file_blob(b"acknowledged next WAL generation")
                .unwrap();
            assert_eq!(writer.status().unwrap().write_counter, 4);
            publish_wal_ack(&ack, b"4");
        }
        Ok("uncommitted") => {
            let tx = writer.archive.transaction().unwrap();
            tx.execute("UPDATE meta SET value='4' WHERE key='write_counter'", [])
                .unwrap();
            tx.cache_flush().unwrap();
            publish_wal_ack(&ack, b"3");
            loop {
                std::thread::park();
            }
        }
        _ => {
            publish_wal_ack(&ack, b"3");
        }
    }
    loop {
        std::thread::park();
    }
}

fn copy_fixture(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let destination = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_fixture(&entry.path(), &destination);
        } else {
            fs::copy(entry.path(), destination).unwrap();
        }
    }
}

struct WalProbeChild(std::process::Child);
impl Drop for WalProbeChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn retained_wal_cold_reopen_gate() {
    for tier in ["archive.db", "embeddings.db"] {
        let (tmp, root, _) = killed_wal_fixture(if tier == "archive.db" {
            "normal"
        } else {
            "embeddings"
        });
        let expected_counter = if tier == "archive.db" { 3 } else { 1 };
        let wal = fs::read(root.join(format!("{tier}-wal"))).unwrap();
        let word = |bytes: &[u8]| u32::from_be_bytes(bytes.try_into().unwrap());
        let page_size = word(&wal[8..12]) as usize;
        assert!(page_size >= 512 && page_size.is_power_of_two());
        let frame_size = page_size + 24;
        assert_eq!((wal.len() - 32) % frame_size, 0);
        let commits: Vec<_> = (0..(wal.len() - 32) / frame_size)
            .filter(|frame| {
                word(&wal[32 + frame * frame_size + 4..32 + frame * frame_size + 8]) != 0
            })
            .collect();
        assert_eq!(commits.len(), 2, "fixture requires two WAL commits");
        let frame = commits[0] + 1;
        let frame_start = 32 + frame * frame_size;
        println!(
            "WAL_PROBE {tier} fixture acknowledged=3 main_baseline=1 page_size={page_size} commits={commits:?} target_frame={} target_page={}",
            frame + 1,
            word(&wal[frame_start..frame_start + 4])
        );
        if let Some(capture) = std::env::var_os("POLYLOGUE_WAL_PROBE_CAPTURE") {
            copy_fixture(
                &root,
                &PathBuf::from(capture).join(format!("{tier}-original")),
            );
        }
        for mutation in ["none", "payload", "page-number", "frame-checksum"] {
            for shm in ["retained", "zeroed", "absent"] {
                for api in ["reader", "writer"] {
                    let name = format!("{tier}-{mutation}-{shm}-{api}");
                    let case = tmp.path().join(&name);
                    copy_fixture(&root, &case);
                    let wal_path = case.join(format!("{tier}-wal"));
                    let mut damaged = wal.clone();
                    match mutation {
                        "payload" => damaged[frame_start + 24 + 200] ^= 0x40,
                        "page-number" => damaged[frame_start + 3] ^= 0x80,
                        "frame-checksum" => damaged[frame_start + 16] ^= 0x40,
                        _ => {}
                    }
                    fs::write(&wal_path, &damaged).unwrap();
                    let shm_path = case.join(format!("{tier}-shm"));
                    match shm {
                        "zeroed" => fs::write(
                            &shm_path,
                            vec![0; fs::metadata(&shm_path).unwrap().len() as usize],
                        )
                        .unwrap(),
                        "absent" => fs::remove_file(&shm_path).unwrap(),
                        _ => {}
                    }
                    if let Some(capture) = std::env::var_os("POLYLOGUE_WAL_PROBE_CAPTURE") {
                        copy_fixture(&case, &PathBuf::from(capture).join(&name));
                    }
                    let before = snapshot(&case);
                    let outcome = if api == "reader" {
                        ReadHandle::open(&case).and_then(|read| read.status())
                    } else {
                        Writer::open(&case).and_then(|writer| writer.status())
                    };
                    let result = match outcome {
                        Ok(status) => format!("counter={}", status.write_counter),
                        Err(err) => format!("error={err}"),
                    };
                    let main_before = before
                        .iter()
                        .find(|(file, _, _)| file.file_name().unwrap() == tier)
                        .unwrap();
                    let main_unchanged = fs::read(case.join(tier)).unwrap() == main_before.1;
                    let wal_unchanged =
                        fs::read(&wal_path).ok().as_deref() == Some(damaged.as_slice());
                    println!(
                        "WAL_PROBE {name}: {result}; files_unchanged={}; main_unchanged={main_unchanged}; wal_unchanged={wal_unchanged}; wal_exists={}",
                        snapshot(&case) == before,
                        wal_path.exists()
                    );
                    if let Some(capture) = std::env::var_os("POLYLOGUE_WAL_PROBE_CAPTURE") {
                        copy_fixture(&case, &PathBuf::from(capture).join("after").join(&name));
                    }
                    if mutation == "none" && shm != "absent" {
                        assert_eq!(result, format!("counter={expected_counter}"));
                    } else {
                        assert!(
                            result.starts_with("error=controlled recovery required"),
                            "{name}: {result}"
                        );
                        assert!(snapshot(&case) == before, "refusal changed retained bytes");
                    }
                    if shm == "absent" {
                        assert!(result.starts_with("error="));
                        assert!(snapshot(&case) == before);
                    }
                }
            }
        }
    }
}

pub(crate) fn killed_wal_fixture(mode: &str) -> (tempfile::TempDir, PathBuf, u64) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("original");
    let ack = tmp.path().join("ack");
    let expected = if matches!(mode, "stale-generation" | "stale-newer" | "stale-empty") {
        4
    } else {
        3
    };
    let mut initial = Writer::open(&root).unwrap();
    initial.put_file_blob(b"checkpointed baseline").unwrap();
    if mode == "embeddings" {
        insert_synthetic_embedding(&initial._embeddings, 1);
    }
    drop(initial);
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    let mut child = WalProbeChild(
        command
            .args([
                "--exact",
                "tests::retained_wal_probe_child",
                "--ignored",
                "--nocapture",
            ])
            .env("POLYLOGUE_WAL_CHILD_ROOT", &root)
            .env("POLYLOGUE_WAL_CHILD_ACK", &ack)
            .env("POLYLOGUE_WAL_CHILD_MODE", mode)
            .spawn()
            .unwrap(),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while !ack.exists() {
        if let Some(status) = child.0.try_wait().unwrap() {
            panic!("owned fixture exited before acknowledgment: {status}");
        }
        if std::time::Instant::now() > deadline {
            child.0.kill().unwrap();
            child.0.wait().unwrap();
            panic!("owned fixture did not acknowledge");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(fs::read_to_string(&ack).unwrap(), expected.to_string());
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    (tmp, root, expected)
}

#[test]
fn retained_wal_legitimate_tails_gate() {
    for mode in ["stale-generation", "uncommitted"] {
        let (tmp, root, expected) = killed_wal_fixture(mode);
        let original = fs::read(root.join("archive.db-wal")).unwrap();
        let page_size = u32::from_be_bytes(original[8..12].try_into().unwrap()) as usize;
        let frame_size = page_size + 24;
        for tail in ["intact", "partial"] {
            if mode == "stale-generation" && tail == "partial" {
                continue;
            }
            for api in ["reader", "writer"] {
                let name = format!("{mode}-{tail}-{api}");
                let case = tmp.path().join(&name);
                copy_fixture(&root, &case);
                if tail == "partial" {
                    let file = OpenOptions::new()
                        .write(true)
                        .open(case.join("archive.db-wal"))
                        .unwrap();
                    file.set_len((original.len() - frame_size + 24 + 100) as u64)
                        .unwrap();
                }
                let before = fs::read(case.join("archive.db-wal")).unwrap();
                if let Some(capture) = std::env::var_os("POLYLOGUE_WAL_PROBE_CAPTURE") {
                    copy_fixture(&case, &PathBuf::from(capture).join(&name));
                }
                let snapshot_before = snapshot(&case);
                let outcome = if api == "reader" {
                    ReadHandle::open(&case).and_then(|r| r.status())
                } else {
                    Writer::open(&case).and_then(|w| w.status())
                };
                if mode == "stale-generation" || tail == "partial" {
                    assert!(
                        matches!(outcome, Err(Error::RecoveryRequired { .. })),
                        "{name}: {outcome:?}"
                    );
                    assert!(
                        snapshot(&case) == snapshot_before,
                        "{name}: refusal changed files"
                    );
                } else {
                    assert_eq!(outcome.unwrap().write_counter, expected);
                }
                assert!(!before.is_empty());
            }
        }
    }
}

#[test]
fn startup_cancellation_and_mid_wal_scan_preserve_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("never-created");
    let cancelled = AtomicBool::new(true);
    assert!(matches!(
        Writer::open_cancellable(&missing, &cancelled),
        Err(Error::Cancelled)
    ));
    assert!(!missing.exists());
    let (_tmp, root, _) = killed_wal_fixture("normal");
    let before = snapshot(&root);
    let mut polls = 0;
    let error = wal::inspect(&root.join("archive.db-wal"), &mut || {
        polls += 1;
        polls == 3
    })
    .unwrap_err();
    assert!(matches!(error, Error::Cancelled));
    assert_eq!(polls, 3, "cancellation must be checked between pages");
    assert!(snapshot(&root) == before);
}

#[test]
fn owned_reader_keeps_archive_coordination_after_writer_drop() {
    let tmp = tempfile::tempdir().unwrap();
    let writer = Writer::open(tmp.path()).unwrap();
    let reader = writer.read_handle().unwrap();
    drop(writer);
    assert!(matches!(
        Writer::open(tmp.path()),
        Err(Error::WriterHeld { .. })
    ));
    assert!(reader.status().is_ok());
    drop(reader);
    drop(reopen_after_drop(tmp.path()));
}

fn insert_synthetic_embedding(conn: &Connection, n: u8) {
    let hash = BlobHash::of(&[n]);
    conn.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES(?1,'synthetic/test/1/document/unit',0,1,?2,0)", params![hash.bytes().as_slice(), 0f32.to_le_bytes().as_slice()]).unwrap();
}

#[test]
fn active_owned_readers_see_commits_without_cold_start_scans() {
    let tmp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(tmp.path()).unwrap();
    let reader = writer.read_handle().unwrap();
    writer
        .archive
        .pragma_update(None, "wal_autocheckpoint", 0)
        .unwrap();
    for n in 0..20u8 {
        writer.put_file_blob(&[n]).unwrap();
        assert_eq!(reader.status().unwrap().write_counter, u64::from(n) + 1);
        assert!(writer.read_handle().is_ok());
        assert!(matches!(
            ReadHandle::open(tmp.path()),
            Err(Error::WriterHeld { .. })
        ));
    }
}

#[test]
fn missing_lock_refuses_without_creating_coordination_files() {
    let tmp = tempfile::tempdir().unwrap();
    drop(Writer::open(tmp.path()).unwrap());
    fs::remove_file(tmp.path().join("writer.lock")).unwrap();
    let before = snapshot(tmp.path());
    assert!(matches!(
        ReadHandle::open(tmp.path()),
        Err(Error::RecoveryRequired { .. })
    ));
    assert!(matches!(
        Writer::open(tmp.path()),
        Err(Error::RecoveryRequired { .. })
    ));
    assert!(snapshot(tmp.path()) == before);
}

#[test]
fn aligned_prior_commit_is_an_explicit_expected_head_limit() {
    let (tmp, root, _) = killed_wal_fixture("normal");
    let wal = fs::read(root.join("archive.db-wal")).unwrap();
    let size = u32::from_be_bytes(wal[8..12].try_into().unwrap()) as usize;
    let end = (0..(wal.len() - 32) / (size + 24))
        .find(|frame| {
            let pos = 32 + frame * (size + 24) + 4;
            u32::from_be_bytes(wal[pos..pos + 4].try_into().unwrap()) != 0
        })
        .unwrap()
        + 1;
    for api in ["reader", "writer"] {
        let case = tmp.path().join(api);
        copy_fixture(&root, &case);
        fs::write(case.join("archive.db-wal"), &wal[..32 + end * (size + 24)]).unwrap();
        let status = if api == "reader" {
            ReadHandle::open(&case).and_then(|r| r.status())
        } else {
            Writer::open(&case).and_then(|w| w.status())
        };
        // The externally acknowledged head was3, but this earlier prefix is
        // internally valid. No checksum-only guard can infer the missing head.
        assert_eq!(status.unwrap().write_counter, 2);
    }
}

#[test]
fn duplicated_file_description_retains_actual_writer_lock_after_owner_drop() {
    let tmp = tempfile::tempdir().unwrap();
    let writer = Writer::open(tmp.path()).unwrap();
    // A duplicated descriptor has the same kernel custody as an inherited one.
    // This distinguishes custody lifetime without depending on spawn timing.
    let inherited = writer._lock.try_clone().unwrap();
    drop(writer);
    let error = Writer::open(tmp.path())
        .err()
        .expect("duplicated custody must exclude a new writer");
    assert!(
        matches!(error, Error::WriterHeld { ref holder, .. }
        if holder == &format!("pid={}", std::process::id())),
        "{error}"
    );
    let release = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(25));
        drop(inherited);
    });
    let writer = reopen_after_drop(tmp.path());
    release.join().unwrap();
    assert!(writer.status().is_ok());
}

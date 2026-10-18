//! An index from an older version is refused, never rebuilt, because the
//! rebuild would compute different IDs than the notes refer to.
use super::*;

fn old_index(root: &Path, version: u32) {
    let mut writer = crate::Writer::open(root).unwrap();
    writer.put_file_blob(b"preserved synthetic source").unwrap();
    // A note target is what a rebuild with new ID rules would break.
    use polylogue_model::authored::{NoteKind, Target, WriteNote};
    writer
        .write_note(&WriteNote {
            request_id: "01900000-0000-7000-8000-000000000801".into(),
            author: "synthetic-operator".into(),
            machine: "synthetic-machine".into(),
            created_at: Some(1),
            kind: Some(NoteKind::Correction),
            value: "Keep the old target".into(),
            targets: Some(vec![Target {
                reference: "block:claude-code:a:b:c:0".into(),
                seen_hash: Some(BlobHash::of(b"old text")),
            }]),
            relations: vec![],
        })
        .unwrap();
    drop(writer);
    let index = root.join("index.db");
    let conn = crate::open(&index, false, false).unwrap();
    conn.pragma_update(None, "user_version", version).unwrap();
    conn.execute(
        "UPDATE meta SET value=?1 WHERE key='INDEX_VERSION'",
        [version.to_string()],
    )
    .unwrap();
    drop(conn);
    // Empty WAL and shared-memory files must be left untouched too.
    fs::write(root.join("index.db-wal"), b"").unwrap();
    fs::write(root.join("index.db-shm"), b"").unwrap();
}

fn contents(root: &Path) -> Vec<Entry> {
    inventory(root, &AtomicBool::new(false)).unwrap()
}

#[test]
fn old_index_versions_refuse_without_changing_files() {
    for version in [19, 20] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        old_index(&root, version);
        let before = contents(&root);
        assert!(
            matches!(crate::Writer::open(&root), Err(Error::IndexRebuildRequired { found, expected: crate::INDEX_VERSION, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
        assert!(
            matches!(crate::ReadHandle::open(&root), Err(Error::IndexRebuildRequired { found, expected: crate::INDEX_VERSION, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
        assert!(!root.join("migrations").exists());
    }
}

#[test]
fn rebuild_marker_refuses_partial_or_missing_index() {
    for index in [None, Some(b"partial replacement".as_slice())] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        old_index(&root, 18);
        for name in ["index.db", "index.db-wal", "index.db-shm"] {
            fs::remove_file(root.join(name)).unwrap();
        }
        if let Some(index) = index {
            fs::write(root.join("index.db"), index).unwrap();
        }
        let backup = root.join("migrations/index18-to19-preserved");
        fs::create_dir_all(&backup).unwrap();
        fs::write(backup.join("index.db"), b"old index backup").unwrap();
        fs::write(
            root.join("index-rebuild.json"),
            br#"{"version":1,"from":18,"to":19,"backup":"index18-to19-preserved"}"#,
        )
        .unwrap();
        let before = contents(&root);
        assert!(matches!(
            crate::Writer::open(&root),
            Err(Error::RecoveryRefused(_))
        ));
        assert_eq!(contents(&root), before);
        assert!(matches!(
            crate::ReadHandle::open(&root),
            Err(Error::RecoveryRefused(_))
        ));
        assert_eq!(contents(&root), before);
    }
}

#[test]
fn current_index_reopens_without_migration() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("archive");
    let mut writer = crate::Writer::open(&root).unwrap();
    let hash = writer.put_file_blob(b"fresh current source").unwrap();
    drop(writer);
    let writer = crate::Writer::open(&root).unwrap();
    assert_eq!(
        writer
            ._index
            .query_row("PRAGMA user_version", [], |row| row.get::<_, u32>(0))
            .unwrap(),
        crate::INDEX_VERSION
    );
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .read_blob_range(hash, 0, 20)
            .unwrap(),
        b"fresh current source"
    );
    assert!(!root.join("migrations").exists());
}

#[test]
fn old_index_with_wal_refuses_without_touching_read_marks() {
    for version in [19, 20] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        old_index(&root, version);
        let connection = crate::open(&root.join("index.db"), false, false).unwrap();
        connection
            .pragma_update(None, "wal_autocheckpoint", 0)
            .unwrap();
        connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        connection
            .execute(
                "UPDATE meta SET value='old-generation' WHERE key='generation'",
                [],
            )
            .unwrap();
        let siblings = ["index.db", "index.db-wal", "index.db-shm"];
        let saved = siblings.map(|name| fs::read(root.join(name)).unwrap());
        assert!(saved[1].len() > 32);
        assert!(!saved[2].is_empty());
        drop(connection);
        // Put back the files as a crashed old process left them; opening must
        // refuse without updating the WAL read marks.
        for (name, bytes) in siblings.into_iter().zip(saved) {
            fs::write(root.join(name), bytes).unwrap();
        }
        let before = contents(&root);
        assert!(
            matches!(crate::Writer::open(&root), Err(Error::IndexRebuildRequired { found, expected: crate::INDEX_VERSION, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
        assert!(
            matches!(crate::ReadHandle::open(&root), Err(Error::IndexRebuildRequired { found, expected: crate::INDEX_VERSION, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
    }
}

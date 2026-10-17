//! Changed identity recipes must not remint durable references by replaying a
//! predecessor index, even if its schema is recognizable.
use super::*;
use std::collections::BTreeMap;

fn predecessor(root: &Path, version: u32) {
    let mut writer = crate::Writer::open(root).unwrap();
    writer.put_file_blob(b"preserved synthetic source").unwrap();
    // A real durable authored target is exactly what an identity-recipe replay
    // must not silently orphan or retarget.
    use polylogue_model::authored::{NoteKind, Target, WriteNote};
    writer
        .write_note(&WriteNote {
            request_id: "01900000-0000-7000-8000-000000000801".into(),
            author: "synthetic-operator".into(),
            machine: "synthetic-machine".into(),
            created_at: Some(1),
            kind: Some(NoteKind::Correction),
            value: "Keep the exact predecessor target".into(),
            targets: Some(vec![Target {
                reference: "block:claude-code:a:b:c:0".into(),
                seen_hash: Some(BlobHash::of(b"predecessor text")),
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
    // Empty retained siblings are part of the preserved predecessor file set.
    fs::write(root.join("index.db-wal"), b"").unwrap();
    fs::write(root.join("index.db-shm"), b"").unwrap();
}

fn contents(root: &Path) -> BTreeMap<PathBuf, (Vec<u8>, u64, u32)> {
    fn visit(root: &Path, at: &Path, out: &mut BTreeMap<PathBuf, (Vec<u8>, u64, u32)>) {
        for entry in fs::read_dir(at).unwrap() {
            let path = entry.unwrap().path();
            let metadata = path.symlink_metadata().unwrap();
            if metadata.is_dir() {
                visit(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().into(),
                    (fs::read(&path).unwrap(), metadata.ino(), metadata.mode()),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    visit(root, root, &mut out);
    out
}

#[test]
fn predecessor_versions_refuse_before_changing_index_archive_or_blobs() {
    for version in [18, 19] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        predecessor(&root, version);
        let before = contents(&root);
        assert!(
            matches!(crate::Writer::open(&root), Err(Error::IndexRebuildRequired { found, expected: 20, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
        assert!(
            matches!(crate::ReadHandle::open(&root), Err(Error::IndexRebuildRequired { found, expected: 20, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
        assert!(!root.join("migrations").exists());
    }
}

#[test]
fn predecessor_replacement_marker_never_admits_partial_or_missing_index_as_fresh() {
    for index in [None, Some(b"partial replacement".as_slice())] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        predecessor(&root, 18);
        for name in ["index.db", "index.db-wal", "index.db-shm"] {
            fs::remove_file(root.join(name)).unwrap();
        }
        if let Some(index) = index {
            fs::write(root.join("index.db"), index).unwrap();
        }
        let backup = root.join("migrations/index18-to19-preserved");
        fs::create_dir_all(&backup).unwrap();
        fs::write(backup.join("index.db"), b"preserved predecessor").unwrap();
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
fn current_index_still_reopens_without_a_predecessor_transition() {
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
        20
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
fn predecessor_nonempty_wal_and_shm_refuse_without_sqlite_readmark_mutation() {
    for version in [18, 19] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        predecessor(&root, version);
        let connection = crate::open(&root.join("index.db"), false, false).unwrap();
        connection
            .pragma_update(None, "wal_autocheckpoint", 0)
            .unwrap();
        connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        connection
            .execute(
                "UPDATE meta SET value='retained-predecessor' WHERE key='generation'",
                [],
            )
            .unwrap();
        let siblings = ["index.db", "index.db-wal", "index.db-shm"];
        let retained = siblings.map(|name| fs::read(root.join(name)).unwrap());
        assert!(retained[1].len() > 32);
        assert!(!retained[2].is_empty());
        drop(connection);
        // Recreate a valid stopped/crashed predecessor sibling set. No process
        // holds it; startup must preserve the bytes rather than update readmarks.
        for (name, bytes) in siblings.into_iter().zip(retained) {
            fs::write(root.join(name), bytes).unwrap();
        }
        let before = contents(&root);
        assert!(
            matches!(crate::Writer::open(&root), Err(Error::IndexRebuildRequired { found, expected: 20, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
        assert!(
            matches!(crate::ReadHandle::open(&root), Err(Error::IndexRebuildRequired { found, expected: 20, .. }) if found == version)
        );
        assert_eq!(contents(&root), before);
    }
}

//! The known derived-index transition uses the existing missing-index path.
//! A synced marker keeps interrupted replacement separate from a fresh archive.
use super::*;
use serde::Deserialize;

const MARKER: &str = "index-rebuild.json";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexTransition {
    version: u8,
    from: u32,
    to: u32,
    backup: String,
    manifest_hash: BlobHash,
}
pub(crate) struct IndexRebuild;

fn read_small(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = regular(path)?;
    if file.metadata()?.len() > maximum {
        return Err(refused("index transition metadata exceeds admission"));
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(refused("index transition metadata changed during read"));
    }
    Ok(bytes)
}
fn validate_index_backup(
    root: &Path,
    transition: &IndexTransition,
    cancelled: &AtomicBool,
) -> Result<()> {
    if transition.version != 1
        || transition.from != 18
        || transition.to != crate::INDEX_VERSION
        || !transition.backup.starts_with("index18-to19-")
        || !transition
            .backup
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err(refused("unsupported index transition marker"));
    }
    let backup = root.join("migrations").join(&transition.backup);
    for path in [root.join("migrations"), backup.clone()] {
        if !owned_metadata(&path)?.is_some_and(|m| m.is_dir()) {
            return Err(refused("index transition backup directory unavailable"));
        }
    }
    let bytes = read_small(&backup.join("source-manifest.json"), 16384)?;
    if BlobHash::of(&bytes) != transition.manifest_hash {
        return Err(refused("index transition manifest digest mismatch"));
    }
    let entries: Vec<Entry> = serde_json::from_slice(&bytes).map_err(|e| refused(e.to_string()))?;
    if entries.is_empty() || entries.len() > 3 || !entries.iter().any(|e| e.path == "index.db") {
        return Err(refused("index transition manifest membership"));
    }
    let mut names = std::collections::BTreeSet::new();
    for entry in entries {
        if !["index.db", "index.db-wal", "index.db-shm"].contains(&entry.path.as_str())
            || !names.insert(entry.path.clone())
        {
            return Err(refused("index transition manifest path"));
        }
        let actual = digest(&backup.join(&entry.path), cancelled)?;
        if actual.length != entry.length || actual.blake3 != entry.blake3 {
            return Err(refused("index transition backup readback mismatch"));
        }
    }
    Ok(())
}

// Exact schema fingerprint of published8a678519's index18. This validates
// disposal authority only; no predecessor row reader or compatibility path is
// retained. The ordinary archive replay produces every new interpretation.
fn validate_known_index18(index: &Path) -> Result<()> {
    let connection = crate::open(index, true, false)?;
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue-index18-schema-v1\0");
    let mut statement = db(index, || {
        connection.prepare("SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type,name")
    })?;
    let mut rows = db(index, || statement.query([]))?;
    let mut count = 0;
    let mut total = 0usize;
    while let Some(row) = db(index, || rows.next())? {
        count += 1;
        if count > 181 {
            return Err(Error::Schema(index.into()));
        }
        for i in 0..3 {
            let value = db(index, || row.get_ref(i))?;
            let rusqlite::types::ValueRef::Text(bytes) = value else {
                return Err(Error::Schema(index.into()));
            };
            total = total.checked_add(bytes.len() + 8).ok_or(Error::Range)?;
            if total > 100000 {
                return Err(Error::Schema(index.into()));
            }
            hash.update(&(bytes.len() as u64).to_be_bytes());
            hash.update(bytes);
        }
    }
    if count != 181
        || hash.finalize().to_hex().as_str()
            != "451d3f7289804ad217d9d18f50f7e9973003be7c8d16fe94e22ebcf9db54d842"
    {
        return Err(Error::Schema(index.into()));
    }
    let stored: String = db(index, || {
        connection.query_row(
            "SELECT value FROM meta WHERE key='INDEX_VERSION'",
            [],
            |r| r.get(0),
        )
    })?;
    if stored != "18" {
        return Err(Error::Schema(index.into()));
    }
    Ok(())
}

/// Called only under the existing archive writer lock, before any SQLite open.
/// Unknown schemas and unsafe conditions continue to the normal refusal gates.
pub(crate) fn begin_index_rebuild(
    root: &Path,
    cancelled: &AtomicBool,
) -> Result<Option<IndexRebuild>> {
    wal::check_cancel(cancelled)?;
    let marker = root.join(MARKER);
    let transition = if owned_metadata(&marker)?.is_some() {
        let bytes = read_small(&marker, 4096)?;
        let transition: IndexTransition =
            serde_json::from_slice(&bytes).map_err(|e| refused(e.to_string()))?;
        validate_index_backup(root, &transition, cancelled)?;
        transition
    } else {
        let index = root.join("index.db");
        let Some(meta) = owned_metadata(&index)? else {
            return Ok(None);
        };
        if !meta.is_file() {
            return Err(Error::Schema(index));
        }
        if meta.len() == 0 {
            return Ok(None);
        }
        // Eligibility precedes any predecessor-only WAL policy. Current
        // index WAL reuse is ordinary SQLite behavior, not this transition.
        let mut header = [0; 100];
        regular(&index)?.read_exact(&mut header)?;
        if &header[..16] != b"SQLite format 3\0" {
            return Err(Error::Schema(index));
        }
        let main_version = u32::from_be_bytes(header[60..64].try_into().unwrap());
        if main_version != 18 {
            return Ok(None);
        }
        for suffix in ["-wal", "-shm"] {
            let sibling = sibling(&index, suffix);
            if owned_metadata(&sibling)?.is_some_and(|m| !m.is_file()) {
                return Err(Error::Schema(sibling));
            }
        }
        let wal_path = sibling(&index, "-wal");
        if owned_metadata(&wal_path)?.is_some_and(|m| m.len() > 0) {
            if owned_metadata(&sibling(&index, "-shm"))?.is_none() {
                return Err(wal::required(
                    &wal_path,
                    "orphan WAL requires controlled recovery",
                    None,
                ));
            }
            wal::inspect(&wal_path, &mut || {
                cancelled.load(std::sync::atomic::Ordering::Acquire)
            })?;
        }
        if wal::schema_version(&index)? != 18 {
            return Ok(None);
        }
        // An index is disposable only beside its retained archive. Do not make
        // an unrelated/newer durable format look like a recoverable foundation.
        let archive = root.join("archive.db");
        if !owned_metadata(&archive)?.is_some_and(|m| m.is_file() && m.len() > 0) {
            return Err(Error::MissingArchive(root.into()));
        }
        let archive_version = wal::schema_version(&archive)?;
        if ![1, crate::ARCHIVE_VERSION].contains(&archive_version) {
            return Err(Error::NewerSchema {
                file: archive,
                found: archive_version,
                supported: crate::ARCHIVE_VERSION,
            });
        }
        let embeddings = root.join("embeddings.db");
        if owned_metadata(&embeddings)?.is_some_and(|m| m.len() > 0) {
            let found = wal::schema_version(&embeddings)?;
            if found != crate::EMBEDDINGS_VERSION {
                return Err(Error::NewerSchema {
                    file: embeddings,
                    found,
                    supported: crate::EMBEDDINGS_VERSION,
                });
            }
        }
        let migrations = root.join("migrations");
        crate::private_dir(&migrations)?;
        let backup = tempfile::Builder::new()
            .prefix("index18-to19-")
            .tempdir_in(&migrations)?;
        let mut entries = Vec::new();
        for name in ["index.db", "index.db-wal", "index.db-shm"] {
            let source = root.join(name);
            if owned_metadata(&source)?.is_some() {
                let mut entry = digest(&source, cancelled)?;
                entry.path = name.into();
                copy_file(&source, &backup.path().join(name), &entry, cancelled)?;
                entries.push(entry);
            }
        }
        let manifest = json_bytes(&entries)?;
        write_new(&backup.path().join("source-manifest.json"), &manifest)?;
        for entry in &entries {
            let mut current = digest(&root.join(&entry.path), cancelled)?;
            current.path = entry.path.clone();
            if current != *entry {
                return Err(refused("index changed during stopped-startup preservation"));
            }
        }
        sync_tree(backup.path())?;
        let preserved = backup.keep();
        File::open(&migrations)?.sync_all()?;
        File::open(root)?.sync_all()?;
        validate_known_index18(&index)?;
        let transition = IndexTransition {
            version: 1,
            from: 18,
            to: crate::INDEX_VERSION,
            backup: preserved.file_name().unwrap().to_str().unwrap().into(),
            manifest_hash: BlobHash::of(&manifest),
        };
        // Publish only complete metadata, after the exact backup is durable.
        let staged = preserved.join("transition.json");
        write_new(&staged, &json_bytes(&transition)?)?;
        wal::check_cancel(cancelled)?;
        native::publish(&staged, &marker)?;
        File::open(root)?.sync_all()?;
        transition
    };
    // A resumed marker authorizes only disposal of the old or partially built
    // derived set. A source or durable tier is never named by this transition.
    let _ = transition;
    for name in ["index.db", "index.db-wal", "index.db-shm"] {
        wal::check_cancel(cancelled)?;
        let path = root.join(name);
        if let Some(meta) = owned_metadata(&path)? {
            if !meta.is_file() {
                return Err(Error::Schema(path));
            }
            fs::remove_file(path)?;
        }
    }
    File::open(root)?.sync_all()?;
    Ok(Some(IndexRebuild))
}
pub(crate) fn finish_index_rebuild(
    root: &Path,
    _proof: IndexRebuild,
    cancelled: &AtomicBool,
) -> Result<()> {
    // Initialization has committed schema/generation before this call. An
    // interruption here repeats a fresh derived rebuild before any parse work.
    wal::check_cancel(cancelled)?;
    fs::remove_file(root.join(MARKER))?;
    File::open(root)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn legacy(root: &Path) -> Vec<(String, Vec<u8>)> {
        let mut writer = crate::Writer::open(root).unwrap();
        writer.put_file_blob(b"preserved synthetic source").unwrap();
        drop(writer);
        let index = root.join("index.db");
        for suffix in ["", "-wal", "-shm"] {
            let path = sibling(&index, suffix);
            if path.exists() {
                fs::remove_file(path).unwrap();
            }
        }
        let mut sql = crate::INDEX_SQL.to_owned();
        sql = sql
            .lines()
            .filter(|l| {
                !l.contains("source_sequence   INTEGER")
                    && !l.contains("CREATE INDEX message_candidate_eligible_cursor")
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        sql = sql.replace(
            "  UNIQUE(pk,message_pk,session_pk,ready),\n  UNIQUE(pk,ready)",
            "  UNIQUE(pk,message_pk,session_pk,ready)",
        );
        let a = sql.find("-- Derived normalized-body equality").unwrap();
        let b = sql[a..].find("CREATE INDEX message_body_owner").unwrap() + a;
        sql.replace_range(a..b, "");
        let a = sql
            .find("-- Fence every eligible-membership change")
            .unwrap();
        let b = sql[a..].find("-- Representatives retain roots").unwrap() + a;
        sql.replace_range(a..b, "");
        assert_eq!(
            BlobHash::of(sql.as_bytes()).to_string(),
            "b6706f639512e78b17ab0b40f460e6d7885da630674ebdd40ebc22cd66997cc8"
        );
        let conn = crate::open(&index, false, false).unwrap();
        conn.execute_batch(&sql).unwrap();
        conn.execute_batch("INSERT INTO meta(key,value) VALUES('write_counter','0'),('generation','synthetic-old-index'),('parser_generation','0'),('INDEX_VERSION','18');PRAGMA user_version=18;PRAGMA journal_mode=WAL;").unwrap();
        drop(conn);
        fs::write(root.join("index.db-wal"), b"").unwrap();
        fs::write(root.join("index.db-shm"), b"").unwrap();
        ["index.db", "index.db-wal", "index.db-shm"]
            .into_iter()
            .map(|n| (n.into(), fs::read(root.join(n)).unwrap()))
            .collect()
    }
    fn transition(root: &Path) -> IndexTransition {
        serde_json::from_slice(&fs::read(root.join(MARKER)).unwrap()).unwrap()
    }
    #[test]
    fn known_index_transition_restarts_at_each_removed_sibling_and_after_fresh_initialization() {
        for interruption in 0..4 {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("archive");
            let old = legacy(&root);
            let archive = fs::read(root.join("archive.db")).unwrap();
            let _proof = begin_index_rebuild(&root, &AtomicBool::new(false))
                .unwrap()
                .unwrap();
            let marker = transition(&root);
            let backup = root.join("migrations").join(&marker.backup);
            for (name, bytes) in &old {
                assert_eq!(fs::read(backup.join(name)).unwrap(), *bytes)
            }
            // Reconstruct the remaining old suffix after a process stops at an
            // unlink boundary. These are still disposable files under marker.
            if interruption < 3 {
                for (name, bytes) in old.iter().skip(interruption) {
                    fs::write(root.join(name), bytes).unwrap();
                }
            } else {
                drop(
                    crate::initialize_db(
                        &root,
                        "index.db",
                        crate::INDEX_SQL,
                        crate::INDEX_VERSION,
                        false,
                    )
                    .unwrap(),
                );
            }
            let writer = crate::Writer::open(&root).unwrap();
            assert_eq!(
                writer
                    ._index
                    .query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
                    .unwrap(),
                crate::INDEX_VERSION
            );
            assert!(!root.join(MARKER).exists());
            assert_eq!(fs::read(root.join("archive.db")).unwrap(), archive);
            for (name, bytes) in &old {
                assert_eq!(fs::read(backup.join(name)).unwrap(), *bytes)
            }
            assert_eq!(fs::read_dir(root.join("migrations")).unwrap().count(), 1);
        }
    }
    #[test]
    fn cancelled_unknown_unsafe_or_corrupt_backup_never_adopts_partial_index() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        let old = legacy(&root);
        assert!(matches!(
            begin_index_rebuild(&root, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        ));
        assert!(!root.join(MARKER).exists());
        for (name, bytes) in &old {
            assert_eq!(fs::read(root.join(name)).unwrap(), *bytes)
        }
        fs::remove_file(root.join("index.db-shm")).unwrap();
        let foreign = temp.path().join("foreign");
        fs::write(&foreign, b"untouched").unwrap();
        std::os::unix::fs::symlink(&foreign, root.join("index.db-shm")).unwrap();
        assert!(begin_index_rebuild(&root, &AtomicBool::new(false)).is_err());
        assert_eq!(fs::read(&foreign).unwrap(), b"untouched");
        assert!(!root.join(MARKER).exists());
        fs::remove_file(root.join("index.db-shm")).unwrap();
        fs::write(root.join("index.db-shm"), b"").unwrap();
        for version in [17, crate::INDEX_VERSION + 1] {
            let conn = crate::open(&root.join("index.db"), false, false).unwrap();
            conn.pragma_update(None, "user_version", version).unwrap();
            drop(conn);
            assert!(
                matches!(crate::Writer::open(&root),Err(Error::IndexRebuildRequired{found,..}) if found==version)
            );
            assert!(!root.join(MARKER).exists());
            assert!(!root.join("migrations").exists());
        }
        for (name, bytes) in &old {
            fs::write(root.join(name), bytes).unwrap()
        }
        begin_index_rebuild(&root, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        let marker = transition(&root);
        let backup = root.join("migrations").join(marker.backup);
        fs::write(root.join("index.db"), b"partial new derived state").unwrap();
        fs::write(backup.join("index.db"), b"damaged backup").unwrap();
        assert!(crate::Writer::open(&root).is_err());
        assert_eq!(
            fs::read(root.join("index.db")).unwrap(),
            b"partial new derived state"
        );
    }
    #[test]
    fn version18_with_unknown_schema_is_preserved_and_refused_without_a_marker() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("archive");
        legacy(&root);
        let index = root.join("index.db");
        let connection = crate::open(&index, false, false).unwrap();
        connection
            .execute_batch("CREATE TABLE unexpected_cache_data(value TEXT)")
            .unwrap();
        drop(connection);
        let before = fs::read(&index).unwrap();
        assert!(matches!(crate::Writer::open(&root), Err(Error::Schema(_))));
        assert!(!root.join(MARKER).exists());
        assert_eq!(fs::read(&index).unwrap(), before);
    }
}

use super::fixture::{self, Builder};
use super::*;

use fixture::file_hashes;

fn scratch() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// Reads everything the importer reads.
fn read_all(older: &OlderArchive) {
    for id in older.input_ids(None, usize::MAX >> 1).unwrap() {
        older.input(&id).unwrap();
    }
    for hash in older.blob_hashes().unwrap() {
        let blob = older.blob(&hash).unwrap().unwrap();
        let mut out = Vec::new();
        older.copy_blob(&blob, &mut out).unwrap().unwrap();
    }
    older.notes().unwrap();
    older.settings().unwrap();
    older.deliveries().unwrap();
    older.delivery_events().unwrap();
    older.requests().unwrap();
    older.kept_results().unwrap();
    older.vectors(|_| Ok(())).unwrap();
    older.vector_failures().unwrap();
    older.write_rows(&mut Vec::new()).unwrap();
}

#[test]
fn the_older_format_is_recognized_by_its_declared_tables_and_anything_else_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let older = OlderArchive::open(&standard.root, scratch().path()).unwrap();
    assert_eq!(older.version(), NEWEST_VERSION);
    assert_eq!(older.machines().len(), 1);
    assert_eq!(older.machines()[0].host, "workstation");

    let empty = dir.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let error = OlderArchive::open(&empty, scratch().path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no archive.db"), "{error}");

    // An archive written at or after the baseline, by this build.
    let current = dir.path().join("current");
    let (archive, lock) = crate::Archive::open_or_create(&current).unwrap();
    drop((archive, lock));
    let error = OlderArchive::open(&current, scratch().path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("not an archive in the older format"),
        "{error}"
    );

    // A version the older format never had, and a table it never declared.
    let newer = dir.path().join("newer");
    Builder::create(&newer, 5).finish();
    let error = OlderArchive::open(&newer, scratch().path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("version 5"), "{error}");
    let extra = dir.path().join("extra");
    let b = Builder::create(&extra, 4);
    b.archive
        .execute_batch("CREATE TABLE segment(name TEXT PRIMARY KEY, committed_end INTEGER)")
        .unwrap();
    b.finish();
    let error = OlderArchive::open(&extra, scratch().path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("table segment is not in the older format"),
        "{error}"
    );
    let missing = dir.path().join("missing");
    let b = Builder::create(&missing, 4);
    b.archive.execute_batch("DROP TABLE kept_result").unwrap();
    b.finish();
    let error = OlderArchive::open(&missing, scratch().path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("table kept_result of the older format is missing"),
        "{error}"
    );
}

#[test]
fn versions_one_to_three_lack_only_what_later_versions_added() {
    for version in 1..=3 {
        let dir = tempfile::tempdir().unwrap();
        let b = Builder::create(dir.path(), version);
        b.archive
            .execute(
                "INSERT INTO client_request(id, content_hash, created_at) VALUES ('r1', X'00', 5)",
                [],
            )
            .unwrap();
        b.finish();
        let older = OlderArchive::open(dir.path(), scratch().path()).unwrap();
        assert_eq!(older.version(), version);
        let requests = older.requests().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].body, None,
            "version {version} kept no request body"
        );
        let counts = older.row_counts().unwrap();
        assert_eq!(counts.contains_key("backfill_job"), version >= 2);
        assert!(!counts.contains_key("row_count"));
    }
}

#[test]
fn reading_an_older_archive_writes_nothing_in_its_folder() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let before = file_hashes(&standard.root);
    {
        let older = OlderArchive::open(&standard.root, scratch().path()).unwrap();
        read_all(&older);
    }
    assert_eq!(file_hashes(&standard.root), before);
}

#[test]
fn a_write_ahead_log_is_read_from_a_copy_and_the_original_pair_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let live = dir.path().join("live");
    fixture::standard(&live);
    // Leave a committed row only in the log, as a stopped older daemon could.
    let writing = rusqlite::Connection::open(live.join("archive.db")).unwrap();
    writing.pragma_update(None, "journal_mode", "wal").unwrap();
    writing
        .pragma_update(None, "wal_autocheckpoint", 0)
        .unwrap();
    writing
        .execute(
            "INSERT INTO machine(id, host, first_seen) VALUES ('m2', 'laptop', 5)",
            [],
        )
        .unwrap();
    let older_root = dir.path().join("older");
    std::fs::create_dir_all(older_root.join("blobs")).unwrap();
    for name in ["archive.db", "archive.db-wal", "embeddings.db"] {
        std::fs::copy(live.join(name), older_root.join(name)).unwrap();
    }
    drop(writing);
    assert!(
        std::fs::metadata(older_root.join("archive.db-wal"))
            .unwrap()
            .len()
            > 0
    );
    let before = file_hashes(&older_root);
    let copies = scratch();
    {
        let older = OlderArchive::open(&older_root, copies.path()).unwrap();
        assert!(older.machines().iter().any(|m| m.id == "m2"));
    }
    assert_eq!(file_hashes(&older_root), before);
    assert!(!older_root.join("archive.db-shm").exists());
    // The copy goes with the reader.
    assert_eq!(std::fs::read_dir(copies.path()).unwrap().count(), 0);
}

#[test]
fn an_older_archive_a_running_polylogue_holds_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let lock = std::fs::File::create(standard.root.join("lock")).unwrap();
    lock.try_lock().unwrap();
    let error = OlderArchive::open(&standard.root, scratch().path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("a running Polylogue holds"), "{error}");
    lock.unlock().unwrap();
    OlderArchive::open(&standard.root, scratch().path()).unwrap();
}

#[test]
fn inline_and_file_blobs_decode_to_their_bytes_and_a_damaged_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let b = Builder::create(dir.path(), 4);
    let small = b.inline_blob(b"inline bytes");
    let large: Vec<u8> = (0..10_000u32).flat_map(|n| n.to_le_bytes()).collect();
    let file = b.file_blob(&large, 4096);
    let empty = b.file_blob(b"", 4096);
    let path = b.blob_path(&file);
    b.finish();
    let older = OlderArchive::open(dir.path(), scratch().path()).unwrap();
    let copy = |hash: &BlobHash| {
        let blob = older.blob(hash).unwrap().unwrap();
        let mut out = Vec::new();
        older.copy_blob(&blob, &mut out).unwrap().map(|()| out)
    };
    assert_eq!(copy(&small).unwrap(), b"inline bytes");
    assert_eq!(copy(&file).unwrap(), large);
    assert_eq!(copy(&empty).unwrap(), b"");
    drop(older);

    // The same length, other bytes: it decodes but no longer hashes to its name.
    let other: Vec<u8> = large.iter().map(|b| b ^ 1).collect();
    std::fs::write(&path, fixture::seekable(&other, 4096)).unwrap();
    let older = OlderArchive::open(dir.path(), scratch().path()).unwrap();
    let blob = older.blob(&file).unwrap().unwrap();
    let reason = older
        .copy_blob(&blob, &mut Vec::new())
        .unwrap()
        .unwrap_err();
    assert!(reason.contains("do not hash to its name"), "{reason}");
    std::fs::remove_file(&path).unwrap();
    let reason = older
        .copy_blob(&blob, &mut Vec::new())
        .unwrap()
        .unwrap_err();
    assert!(reason.contains("is missing"), "{reason}");
}

#[test]
fn the_rows_input_holds_every_row_but_the_stored_bytes_in_a_fixed_order() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let older = OlderArchive::open(&standard.root, scratch().path()).unwrap();
    let mut first = Vec::new();
    let written = older.write_rows(&mut first).unwrap();
    let mut second = Vec::new();
    older.write_rows(&mut second).unwrap();
    assert_eq!(first, second);
    let counts = older.row_counts().unwrap();
    let expected: u64 = counts
        .iter()
        .filter(|(table, _)| {
            !BYTE_TABLES.contains(&table.as_str())
                && *table != "embedded"
                && *table != "embed_failure"
        })
        .map(|(_, n)| n)
        .sum();
    assert_eq!(written, expected);
    let lines: Vec<serde_json::Value> = first
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(lines.len() as u64, expected);
    let chat = lines
        .iter()
        .find(|line| line["table"] == "chat_action")
        .unwrap();
    assert_eq!(chat["row"]["request_hash"]["sqlite_blob_hex"], "00");
    assert!(lines.iter().all(|line| line["table"] != "input"));
}

#[test]
fn an_input_keeps_every_revision_extent_and_observation_in_insertion_order() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let older = OlderArchive::open(&standard.root, scratch().path()).unwrap();
    let input = older.input(&standard.watched).unwrap();
    assert_eq!(input.format, "claude-code");
    assert_eq!(input.refusal, None);
    let produced: Vec<_> = input
        .revisions
        .iter()
        .map(|r| r.produced.as_str())
        .collect();
    assert_eq!(produced, ["first", "appended", "rewritten"]);
    assert_eq!(input.extents.len(), 3);
    let classes: Vec<_> = input
        .observations
        .iter()
        .map(|o| o.class.as_str())
        .collect();
    assert_eq!(
        classes,
        ["first", "appended", "rewritten", "deleted", "back", "back"]
    );
    let ids: Vec<_> = input.observations.iter().map(|o| o.id.as_str()).collect();
    assert_eq!(ids, ["o1", "o2", "o3", "o4", "o5", "o0"]);
    let first = &input.observations[0];
    assert_eq!(
        first.path_bytes.as_deref(),
        Some(fixture::WATCHED_PATH.as_bytes())
    );
    assert_eq!((first.inode, first.seen_at), (Some(11), 2_000));
    assert_eq!(first.host, "workstation");
    assert_eq!(first.polylogue_version, "0.9.0");
    assert_eq!(input.observations[3].revision, None);
}

#[test]
fn an_excised_or_exported_member_input_carries_its_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let conn = rusqlite::Connection::open(standard.root.join("archive.db")).unwrap();
    conn.execute_batch(&format!(
        "INSERT INTO excision(id, target, reason, created_at, commit_seq) VALUES ('x', 'input:{0}', NULL, 1, 1);
         INSERT INTO observation(id, input, machine, path, class, revision, excision, polylogue_version, seen_at)
           VALUES ('ox', '{0}', '{1}', '{2}', 'excised', NULL, 'x', '0.9.0', 10000);
         INSERT INTO observation_zip(observation_id, zip_input, member_index) VALUES ('o8', '{3}', 0);",
        standard.deleted, standard.machine, fixture::DELETED_PATH, standard.watched
    ))
    .unwrap();
    drop(conn);
    let older = OlderArchive::open(&standard.root, scratch().path()).unwrap();
    let excised = older.input(&standard.deleted).unwrap();
    assert!(excised.refusal.unwrap().contains("excision"));
    let member = older.input(&standard.imported).unwrap();
    assert!(member.refusal.unwrap().contains("member of export"));
    assert_eq!(older.input(&standard.watched).unwrap().refusal, None);
}

#[test]
fn a_path_written_as_lossy_display_text_keeps_no_exact_location() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let conn = rusqlite::Connection::open(standard.root.join("archive.db")).unwrap();
    conn.execute(
        "UPDATE observation SET path = '/synthetic/caf\u{FFFD}.jsonl' WHERE id = 'o6'",
        [],
    )
    .unwrap();
    drop(conn);
    let older = OlderArchive::open(&standard.root, scratch().path()).unwrap();
    let input = older.input(&standard.deleted).unwrap();
    assert_eq!(input.inexact_locations, 1);
    assert_eq!(input.observations[0].path_bytes, None);
    assert_eq!(
        input.observations[1].path_bytes.as_deref(),
        Some(fixture::DELETED_PATH.as_bytes())
    );
}

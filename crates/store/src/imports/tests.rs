use super::*;

fn observation(id: &str) -> ImportObservation<'_> {
    ImportObservation {
        id,
        machine: "synthetic-machine",
        host: "test-host",
        path: "copy:synthetic.jsonl",
        seen_at: 7,
    }
}
fn count(connection: &rusqlite::Connection, table: &str) -> i64 {
    connection
        .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}
fn store_rollout(
    writer: &mut Writer,
    hash: BlobHash,
    id: &str,
    allowance: usize,
) -> Result<StoredInput> {
    writer.store_immutable_input(
        hash,
        InputFormat::CodexRollout,
        &[],
        &observation(id),
        allowance,
    )
}

#[test]
fn retry_replays_before_checking_the_blob() {
    let root = tempfile::tempdir().unwrap();
    let mut w = Writer::open(root.path()).unwrap();
    let hash = w.put_file_blob(b"stored").unwrap();
    let first = store_rollout(&mut w, hash, "retry-before-lookup", 65536).unwrap();
    w.archive
        .execute("UPDATE blob SET state='deleting'", [])
        .unwrap();
    let retry = store_rollout(&mut w, hash, "retry-before-lookup", 65536).unwrap();
    assert_eq!(retry, first);
}

#[test]
fn import_retry_adds_no_revision() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let hash = writer.put_file_blob(b"synthetic\n").unwrap();
    let imported = store_rollout(&mut writer, hash, "import-1", 1024).unwrap();
    let counter = writer.status().unwrap().write_counter;
    assert_eq!(
        store_rollout(&mut writer, hash, "import-1", 1024).unwrap(),
        imported
    );
    assert_eq!(writer.status().unwrap().write_counter, counter);
    let reader = writer.read_handle().unwrap();
    assert_eq!(
        reader.read_blob_range(hash, 0, imported.length).unwrap(),
        b"synthetic\n"
    );
    assert_eq!(count(&writer.archive, "revision"), 1);
    assert_eq!(count(&writer.archive, "observation"), 1);
    assert!(
        writer
            .store_immutable_input(
                hash,
                InputFormat::ClaudeCodeJsonl,
                &[],
                &observation("import-2"),
                1024
            )
            .is_err()
    );
    assert_eq!(writer.status().unwrap().write_counter, counter);
}

#[test]
fn index_failure_keeps_archive_and_retry_repairs_index() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let hash = writer.put_file_blob(b"stored before index\n").unwrap();
    writer._index.execute_batch("CREATE TRIGGER reject_input BEFORE INSERT ON input_ref BEGIN SELECT RAISE(ABORT,'synthetic registration refusal'); END;").unwrap();
    assert!(matches!(
        store_rollout(&mut writer, hash, "import-1", 1024),
        Err(Error::Sqlite { .. })
    ));
    let archive_counter = writer.status().unwrap().write_counter;
    assert_eq!(count(&writer.archive, "extent"), 1);
    writer
        ._index
        .execute_batch("DROP TRIGGER reject_input;")
        .unwrap();
    drop(writer);
    let mut writer = Writer::open(root.path()).unwrap();
    let imported = store_rollout(&mut writer, hash, "import-1", 1024).unwrap();
    assert_eq!(writer.status().unwrap().write_counter, archive_counter);
    assert_eq!(imported.chain, 1);
    assert_eq!(imported.length, 20);
}

#[test]
fn next_stored_source_pages_inputs_and_rebuilds_index_refs() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    for (bytes, id) in [(b"a".as_slice(), "a"), (b"b".as_slice(), "b")] {
        let hash = writer.put_file_blob(bytes).unwrap();
        store_rollout(&mut writer, hash, id, 1024).unwrap();
    }
    writer._index.execute("DELETE FROM input_ref", []).unwrap();
    drop(writer);
    let mut writer = Writer::open(root.path()).unwrap();
    let first = writer.next_stored_source("").unwrap().unwrap();
    assert_eq!(
        first.input.id,
        InputId::imported(&first.binding, first.format.name(), &[]).to_string()
    );
    assert!(
        matches!(first.identity, crate::SourceIdentity::Immutable { original_hash } if original_hash == first.binding)
    );
    assert_eq!(first.input.length, 1);
    assert_eq!(count(&writer._index, "input_ref"), 1);
    let second = writer.next_stored_source(&first.input.id).unwrap().unwrap();
    assert!(second.input.id > first.input.id);
    assert!(
        writer
            .next_stored_source(&second.input.id)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        writer.next_stored_source("").unwrap().unwrap().input,
        first.input
    );
    writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('removal','session:unknown',1,1)",[]).unwrap();
    assert!(matches!(
        writer.check_import_acquisition(),
        Err(Error::ExcisionCheckRequired)
    ));
    assert!(matches!(
        writer.next_stored_source(""),
        Err(Error::ExcisionCheckRequired)
    ));
}

#[test]
fn oversized_context_is_refused_before_any_write() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let hash = writer.put_file_blob(b"synthetic").unwrap();
    let huge = "x".repeat(65536);
    let before = writer.status().unwrap().write_counter;
    assert!(matches!(
        writer.store_immutable_input(
            hash,
            InputFormat::ClaudeCodeJsonl,
            &["claude-main", &huge],
            &observation("large"),
            usize::MAX
        ),
        Err(Error::ImportContextTooLarge { maximum: 65536, .. })
    ));
    assert!(matches!(
        InputFormat::CodexRollout.validate_context(&vec![""; 8192]),
        Err(Error::ImportContextTooLarge { .. })
    ));
    assert!(
        InputFormat::ClaudeCodeJsonl
            .validate_context(&["claude-main", "bad\0stem"])
            .is_err()
    );
    assert_eq!(writer.status().unwrap().write_counter, before);
    assert_eq!(count(&writer.archive, "input"), 0);
}

#[test]
fn same_bytes_with_different_contexts_share_one_blob() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let hash = writer
        .put_file_blob(b"same synthetic provider bytes")
        .unwrap();
    let mut store = |context: &[&str], id: &str, format| {
        writer.store_immutable_input(hash, format, context, &observation(id), 4096)
    };
    let a = store(&["claude-main", "a"], "a", InputFormat::ClaudeCodeJsonl).unwrap();
    let b = store(&["claude-main", "b"], "b", InputFormat::ClaudeCodeJsonl).unwrap();
    let rollout = store(&[], "rollout", InputFormat::CodexRollout).unwrap();
    assert!(store(&[], "bad", InputFormat::ClaudeCodeJsonl).is_err());
    assert!(store(&["unexpected"], "bad", InputFormat::CodexRollout).is_err());
    assert_ne!(a.id, b.id);
    assert_ne!(a.id, rollout.id);
    assert_eq!(count(&writer.archive, "blob"), 1);
    assert_eq!(
        writer.immutable_import(&a.id).unwrap().unwrap().context,
        vec!["claude-main", "a"]
    );
    assert!(
        writer
            .replay_immutable_import(
                InputFormat::ClaudeCodeJsonl,
                &["claude-main", "b"],
                &observation("a"),
                4096
            )
            .is_err()
    );
    writer
        .archive
        .execute(
            "UPDATE input SET context_json='[\"claude-main\",\"different\"]' WHERE id=?1",
            [&a.id],
        )
        .unwrap();
    assert!(writer.immutable_import(&a.id).is_err());
    assert_eq!(writer.immutable_import(&b.id).unwrap().unwrap().input, b);
    // The schema refuses an original hash without its context.
    assert!(writer.archive.execute("INSERT INTO input(id,format,original_hash) VALUES('invalid','codex-rollout',zeroblob(32))",[]).is_err());
}

#[test]
fn retry_returns_stored_result_and_rejects_changed_arguments() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let hash = writer.put_file_blob(b"original").unwrap();
    let imported = store_rollout(&mut writer, hash, "request", 4096).unwrap();
    let counter = writer.status().unwrap().write_counter;
    drop(writer);
    let mut writer = Writer::open(root.path()).unwrap();
    let mut replay = |format, observation: &ImportObservation<'_>| {
        writer.replay_immutable_import(format, &[], observation, 4096)
    };
    assert_eq!(
        replay(InputFormat::CodexRollout, &observation("request")).unwrap(),
        Some(imported.clone())
    );
    let changed = ImportObservation {
        path: "/different-path",
        ..observation("request")
    };
    assert!(replay(InputFormat::CodexRollout, &changed).is_err());
    assert!(replay(InputFormat::ClaudeCodeJsonl, &observation("request")).is_err());
    assert!(
        replay(InputFormat::CodexRollout, &observation("new"))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        writer
            .immutable_import(&imported.id)
            .unwrap()
            .unwrap()
            .input,
        imported
    );
    assert_eq!(writer.status().unwrap().write_counter, counter);
}

#[test]
fn refused_imports_leave_the_archive_unchanged() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let a = writer.put_file_blob(b"a").unwrap();
    let b = writer.put_file_blob(b"b").unwrap();
    let before = writer.status().unwrap().write_counter;
    assert!(matches!(
        store_rollout(&mut writer, a, "id", 1),
        Err(Error::ParserAdmission { .. })
    ));
    assert!(matches!(
        store_rollout(&mut writer, BlobHash::of(b"missing"), "missing", 1024),
        Err(Error::UnknownBlob)
    ));
    assert_eq!(writer.status().unwrap().write_counter, before);
    store_rollout(&mut writer, a, "id", 1024).unwrap();
    let before = writer.status().unwrap().write_counter;
    // Same observation ID, different bytes.
    assert!(store_rollout(&mut writer, b, "id", 1024).is_err());
    assert_eq!(writer.status().unwrap().write_counter, before);
    assert_eq!(count(&writer.archive, "input"), 1);
}

#[test]
fn any_excision_blocks_imports() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let hash = writer.put_file_blob(b"synthetic stored bytes").unwrap();
    writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('synthetic-removal','session:unknown',1,1)", []).unwrap();
    let before = writer.status().unwrap().write_counter;
    assert!(matches!(
        store_rollout(&mut writer, hash, "new-observation", 1024),
        Err(Error::ExcisionCheckRequired)
    ));
    assert_eq!(writer.status().unwrap().write_counter, before);
    assert_eq!(count(&writer.archive, "input"), 0);
    assert_eq!(count(&writer._index, "input_ref"), 0);
}

use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn batch(id: &str, length: u64) -> Batch {
    Batch {
        input: InputKey {
            id: id.into(),
            format: "export-file".into(),
            route: polylogue_model::Route::Export,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: "/synthetic/export.bin".into(),
            path_bytes: b"/synthetic/export.bin".to_vec(),
            facts: None,
            seen_at_ms: 1,
            import: None,
            container: None,
            database: None,
        }),
        continues: false,
        new_blob: None,
        blobs: Vec::new(),
        length,
        parse: Vec::new(),
        capture_read: None,
        release: Default::default(),
    }
}

#[test]
fn dropped_and_other_preparations_leave_no_named_parts_or_committed_blobs() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(first.path()).unwrap();
    let (other, _other_lock) = Archive::open_or_create(second.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let body = b"synthetic private prepared bytes";
    let prepared = archive.prepare_bytes(body.as_slice()).unwrap();
    let hash = prepared.hash();
    let store = archive.blobs().unwrap();
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(hash.0)),
        Err(StoreError::NotFound(_))
    ));
    drop(prepared);
    assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
    let other_bytes = other.prepare_bytes(body.as_slice()).unwrap();
    assert!(matches!(
        writer.store_bytes(other_bytes, batch(&hash.to_string(), body.len() as u64)),
        Err(StoreError::Invalid(_))
    ));
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(hash.0)),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(std::fs::read_dir(other.staging_dir()).unwrap().count(), 0);
}

#[test]
fn an_unreferenced_private_stage_is_refused_before_any_group_installation() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(archive.blobs().unwrap());
    let source = archive
        .prepare_bytes(b"synthetic stored source".as_slice())
        .unwrap();
    let stray = archive
        .prepare_bytes(b"synthetic unreferenced stage".as_slice())
        .unwrap();
    let source_hash = source.hash();
    let stray_hash = stray.hash();
    let mut batch = batch(&source_hash.to_string(), source.len());
    batch.new_blob = Some(NewBlob::whole(source_hash, source.len()));
    let result = writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch,
            files: vec![source, stray],
        }])
        .unwrap();
    assert!(
        result[0].is_err(),
        "a group with an unreferenced private stage was stored"
    );
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(source_hash.0)),
        Err(StoreError::NotFound(_))
    ));
    assert!(matches!(
        store.extent(&hashed_blobs::Hash(stray_hash.0)),
        Err(StoreError::NotFound(_))
    ));
    let db = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn failed_stream_inspection_releases_the_encoded_preparation() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let result: Result<_> = archive.prepare_stream(b"encoded synthetic source".as_slice(), |_| {
        Err::<(), _>(StoreError::Invalid("synthetic inspection refusal".into()))
    });
    assert!(matches!(result, Err(StoreError::Invalid(_))));
    assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
}

#[test]
fn encoded_preparation_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_ENCODED_PREP_ROOT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let path = root.join("archive");
    let (archive, _lock) = Archive::open_or_create(&path).unwrap();
    let _prepared = archive
        .prepare_bytes(b"unaccepted encoded source".as_slice())
        .unwrap();
    std::fs::write(root.join("ready"), b"ready").unwrap();
    loop {
        std::thread::park();
    }
}

#[test]
fn open_releases_encoded_preparations_after_process_death() {
    {
        let directory = tempfile::tempdir().unwrap();
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "writer::capture::tests::encoded_preparation_crash_child",
                "--nocapture",
            ])
            .env("POLYLOGUE_ENCODED_PREP_ROOT", directory.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut child = command.spawn().unwrap();
        let started = std::time::Instant::now();
        while !directory.path().join("ready").exists() {
            if child.try_wait().unwrap().is_some()
                || started.elapsed() > std::time::Duration::from_secs(30)
            {
                let _ = child.kill();
                let _ = child.wait();
                panic!("preparation child did not signal that its bytes were prepared");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let path = directory.path().join("archive");
        assert_eq!(std::fs::read_dir(path.join("staging")).unwrap().count(), 1);
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        let (archive, _lock) = Archive::open_or_create(&path).unwrap();
        assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
    }
}

#[test]
fn late_encoding_refusal_spools_no_group_and_reopen_retries() {
    let directory = tempfile::tempdir().unwrap();
    let bodies = [
        b"first synthetic source".as_slice(),
        b"second synthetic source".as_slice(),
    ];
    {
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let mut group = Vec::new();
        for body in bodies {
            let prepared = archive.prepare_bytes(body).unwrap();
            let mut item = batch(&prepared.hash().to_string(), prepared.len());
            item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
            group.push(crate::staging::PreparedBatch {
                batch: item,
                files: vec![prepared],
            });
        }
        let late = group[1].files[0].staged_path().to_path_buf();
        std::fs::write(late, b"synthetic damaged encoding").unwrap();
        let result = writer.store_group(group).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(Result::is_err));
        for body in bodies {
            assert!(matches!(
                archive
                    .blobs()
                    .unwrap()
                    .extent(&hashed_blobs::Hash::of(body)),
                Err(StoreError::NotFound(_))
            ));
        }
    }
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut group = Vec::new();
    for body in bodies {
        let prepared = archive.prepare_bytes(body).unwrap();
        let mut item = batch(&prepared.hash().to_string(), prepared.len());
        item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
        group.push(crate::staging::PreparedBatch {
            batch: item,
            files: vec![prepared],
        });
    }
    let results = writer.store_group(group).unwrap();
    assert!(results.iter().all(Result::is_ok));
    for body in bodies {
        let mut reader = archive
            .blobs()
            .unwrap()
            .read(&hashed_blobs::Hash::of(body))
            .unwrap();
        let mut restored = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut restored).unwrap();
        assert_eq!(restored, body);
    }
}

#[test]
fn an_equal_length_change_to_a_staged_file_is_refused_at_append() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let body = b"synthetic staged source".as_slice();
    let prepared = archive.prepare_bytes(body).unwrap();
    let mut item = batch(&prepared.hash().to_string(), prepared.len());
    item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
    // One encoded byte changes and the length stays: only the append's
    // encoded digest can see it.
    let path = prepared.staged_path().to_path_buf();
    let mut encoded = std::fs::read(&path).unwrap();
    encoded[0] ^= 1;
    std::fs::write(&path, &encoded).unwrap();
    let results = writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch: item,
            files: vec![prepared],
        }])
        .unwrap();
    assert!(results[0].is_err());
    assert!(matches!(
        archive
            .blobs()
            .unwrap()
            .extent(&hashed_blobs::Hash::of(body)),
        Err(StoreError::NotFound(_))
    ));
    // The same group with its staged bytes intact is stored, so the refusal
    // above comes from the changed byte.
    let control_directory = tempfile::tempdir().unwrap();
    let (control, _control_lock) = Archive::open_or_create(control_directory.path()).unwrap();
    let control_writer = Writer::start(&control).unwrap();
    let prepared = control.prepare_bytes(body).unwrap();
    let mut item = batch(&prepared.hash().to_string(), prepared.len());
    item.new_blob = Some(NewBlob::whole(prepared.hash(), prepared.len()));
    let results = control_writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch: item,
            files: vec![prepared],
        }])
        .unwrap();
    assert!(results[0].is_ok());
    control
        .blobs()
        .unwrap()
        .extent(&hashed_blobs::Hash::of(body))
        .unwrap();
}

fn reading(input: &str, chain: i64, length: u64) -> PreparedBatch {
    PreparedBatch {
        batch: Batch {
            input: InputKey {
                id: input.into(),
                // An imported input can keep export-file as its register
                // format while its reading uses the recognized provider.
                format: "claude-code".into(),
                route: polylogue_model::Route::Live,
            },
            change: Change::Reparse { chain },
            length,
            observation: None,
            new_blob: None,
            blobs: Vec::new(),
            parse: Vec::new(),
            continues: false,
            capture_read: None,
            release: Default::default(),
        },
        files: Vec::new(),
    }
}

fn kept() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&crate::schema::ARCHIVE).unwrap();
    conn.execute("INSERT INTO input VALUES ('input','synthetic')", [])
        .unwrap();
    conn
}

#[test]
fn stored_reading_validation_uses_logical_lengths_without_archive_writes() {
    let conn = kept();
    // Empty, incomplete-prefix and incomplete-tail inputs are valid register
    // states. Their parser diagnosis must not require full byte coverage.
    for (number, chain, length) in [(1, 1, 0), (2, 2, 100), (3, 2, 200), (4, 4, 10)] {
        conn.execute(
            "INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',?1,'rewritten',?2,?3,1)",
            params![number, chain, length],
        )
        .unwrap();
    }
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Insert { .. }
        | AuthAction::Update { .. }
        | AuthAction::Delete { .. }
        | AuthAction::Transaction { .. } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    for (chain, length) in [(1, 0), (2, 0), (2, 100), (2, 200), (4, 10)] {
        let stored = stored_readings(&conn, &[reading("input", chain, length)])
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                stored[0].chain,
                stored[0].length,
                stored[0].already_stored,
                stored[0].bytes_written
            ),
            (chain, length, true, 0)
        );
    }
    for item in [
        reading("other", 2, 100),
        reading("input", 3, 0),
        reading("input", 2, 201),
        reading("input", 4, 200),
        reading("input", 1, 1),
        reading("input", 1, u64::MAX),
    ] {
        assert!(matches!(
            stored_readings(&conn, &[item]),
            Err(StoreError::Invalid(_) | StoreError::Internal(_))
        ));
    }
}

/// Where each batch of a group that only reads stored bytes is, as the
/// archive writer answers it without writing.
fn stored_readings(conn: &Connection, group: &[PreparedBatch]) -> Result<Option<Vec<StoredInput>>> {
    group
        .iter()
        .map(|item| {
            Ok(StoredInput {
                input: item.batch.input.id.clone(),
                chain: crate::archive::writer::captured::stored_reading_chain(conn, &item.batch)?,
                length: item.batch.length,
                already_stored: true,
                bytes_written: 0,
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn validation_steps(revisions: usize) -> usize {
    let mut conn = kept();
    {
        let tx = conn.transaction().unwrap();
        for number in 1..=revisions {
            tx.execute(
                "INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',?1,'appended',1,?1,1)",
                [number as i64],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }
    let count = Arc::new(AtomicUsize::new(0));
    let steps = Arc::clone(&count);
    conn.progress_handler(
        1,
        Some(move || {
            steps.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let stored = stored_readings(&conn, &[reading("input", 1, revisions as u64)])
        .unwrap()
        .unwrap();
    assert_eq!(stored[0].length, revisions as u64);
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    count.load(Ordering::Relaxed)
}

#[test]
fn replay_validation_does_not_walk_a_growing_current_revision_history() {
    let short = validation_steps(1);
    let long = validation_steps(10_000);
    assert_eq!(
        long, short,
        "SQLite VM work must be independent of append history length"
    );
}

use super::*;
use crate::staging::PreparedBatch;
#[cfg(feature = "faults")]
use crate::staging::StoredInput;

#[cfg(feature = "faults")]
fn append(archive: &Archive, writer: &Writer, input: &str, bytes: &[u8]) -> Result<StoredInput> {
    let (new_blob, file) = archive.prepare_blob(bytes)?;
    let batch = Batch {
        input: InputKey {
            id: input.into(),
            format: "test-bytes".into(),
            route: polylogue_model::Route::Live,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "test-machine".into(),
            host: "test-host".into(),
            path: format!("/home/op/{input}"),
            path_bytes: format!("/home/op/{input}").into_bytes(),
            facts: None,
            seen_at_ms: 1,
            import: None,
            container: None,
            database: None,
        }),
        continues: false,
        new_blob: Some(new_blob),
        blobs: Vec::new(),
        length: bytes.len() as u64,
        parse: Vec::new(),
        capture_read: None,
        release: Default::default(),
    };
    writer
        .store_group(vec![PreparedBatch {
            batch,
            files: file.into_iter().collect(),
        }])?
        .into_iter()
        .next()
        .expect("one batch")
}

#[test]
fn group_commits_one_sync_and_one_blob_for_equal_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut group = Vec::new();
    for input in ["first", "second", "third"] {
        let (new_blob, file) = archive.prepare_blob(b"the exact source bytes\n").unwrap();
        let mut batch = batch(input, Route::Live, 0, vec![], &[]);
        batch.input.format = "test-bytes".into();
        batch.length = new_blob.size;
        batch.new_blob = Some(new_blob);
        batch.parse.clear();
        group.push(PreparedBatch {
            batch,
            files: file.into_iter().collect(),
        });
    }
    let results = writer.store_group(group).unwrap();
    assert!(results.iter().all(Result::is_ok));
    let report = writer.flush_rows().unwrap();
    assert_eq!(
        (
            report.segment_syncs,
            report.blobs_appended,
            report.blobs_deduplicated
        ),
        (1, 1, 2)
    );
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        3
    );
    assert_eq!(archive.blobs().unwrap().scan().unwrap().len(), 0);
    let stored = archive
        .blobs()
        .unwrap()
        .read(&hashed_blobs::Hash::of(b"the exact source bytes\n"))
        .unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut { stored }, &mut bytes).unwrap();
    assert_eq!(bytes, b"the exact source bytes\n");
}

#[cfg(feature = "faults")]
#[test]
fn segment_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_SEGMENT_CRASH_ROOT") else {
        return;
    };
    let point = std::env::var("POLYLOGUE_SEGMENT_CRASH_POINT").unwrap();
    let (archive, _lock) = Archive::open_or_create(std::path::Path::new(&root)).unwrap();
    let writer = Writer::start(&archive).unwrap();
    crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
    append(&archive, &writer, "second", b"second immutable source\n").unwrap();
    panic!("the configured durability boundary was not reached");
}

/// The rows an uninterrupted run leaves, apart from observation identities.
#[cfg(feature = "faults")]
fn record(archive: &Archive) -> Vec<String> {
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    let mut rows = Vec::new();
    for sql in [
        "SELECT 'revision', input, number, chain, length, produced FROM revision ORDER BY input, number",
        "SELECT 'extent', input, chain, offset, length, hex(blob) FROM extent ORDER BY input, chain, offset",
        "SELECT 'blob', hex(hash), size, '', '', '' FROM blob ORDER BY hash",
    ] {
        let mut statement = conn.prepare(sql).unwrap();
        let mut query = statement.query([]).unwrap();
        while let Some(row) = query.next().unwrap() {
            rows.push(
                (0..6)
                    .map(|n| match row.get_ref(n).unwrap() {
                        rusqlite::types::ValueRef::Text(text) => {
                            String::from_utf8_lossy(text).into_owned()
                        }
                        rusqlite::types::ValueRef::Integer(n) => n.to_string(),
                        other => format!("{other:?}"),
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }
    rows
}

/// A crash during a put, between a put and the archive commit that cites
/// it, inside that commit or after it restarts to the state of an
/// uninterrupted run: an uncommitted tail is cut, and a blob put before the
/// crash is found by the resumed write and not stored again.
#[cfg(feature = "faults")]
#[test]
fn crash_leaves_only_a_tail_at_each_product_commit_boundary() {
    let first = b"first immutable source\n";
    let second = b"second immutable source\n";
    let uninterrupted = {
        let directory = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        append(&archive, &writer, "first", first).unwrap();
        append(&archive, &writer, "second", second).unwrap();
        record(&archive)
    };
    for (point, put, committed) in [
        ("segment.after_append", false, false),
        ("segment.after_sync", false, false),
        ("capture.between_put_and_commit", true, false),
        ("write.archive_commit", true, false),
        ("capture.after_commit", true, true),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let original = {
            let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
            let writer = Writer::start(&archive).unwrap();
            append(&archive, &writer, "first", first).unwrap();
            archive
                .blobs()
                .unwrap()
                .extent(&hashed_blobs::Hash::of(first))
                .unwrap()
        };
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "writer::tests::segments::segment_crash_child",
                "--nocapture",
            ])
            .env("POLYLOGUE_SEGMENT_CRASH_ROOT", directory.path())
            .env("POLYLOGUE_SEGMENT_CRASH_POINT", point)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id() as libc::pid_t;
        let status = wait_stopped(pid);
        assert!(
            libc::WIFSTOPPED(status),
            "child exited before {point}: {status}"
        );
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let blobs = archive.blobs().unwrap();
        assert_eq!(
            blobs.extent(&hashed_blobs::Hash::of(first)).unwrap(),
            original
        );
        let mut reader = blobs.read(&hashed_blobs::Hash::of(first)).unwrap();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut bytes).unwrap();
        assert_eq!(bytes, first);
        let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
        let end: i64 = conn
            .query_row("SELECT committed_end FROM segment", [], |row| row.get(0))
            .unwrap();
        let segment = directory.path().join("segments/current.hb");
        assert_eq!(
            std::fs::metadata(&segment).unwrap().len(),
            end as u64,
            "{point}"
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM blob", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            1 + i64::from(put),
            "{point}"
        );
        if !committed {
            // The resumed write finds a blob put before the crash.
            append(&archive, &writer, "second", second).unwrap();
            let report = writer.flush_rows().unwrap();
            assert_eq!(
                (report.blobs_appended, report.blobs_deduplicated),
                (u64::from(!put), u64::from(put)),
                "{point}"
            );
        }
        assert_eq!(record(&archive), uninterrupted, "{point}");
        append(
            &archive,
            &writer,
            "third",
            b"next append after ordinary open\n",
        )
        .unwrap();
        assert!(blobs.scan().unwrap().is_empty());
    }
}

#[cfg(feature = "faults")]
#[allow(unsafe_code)]
fn wait_stopped(pid: libc::pid_t) -> i32 {
    let mut status = 0;
    // This test owns pid; status is a live, uniquely borrowed output cell.
    loop {
        let result = unsafe { libc::waitpid(pid, &mut status, libc::WUNTRACED) };
        if result == pid {
            return status;
        }
        let error = std::io::Error::last_os_error();
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::Interrupted,
            "waitpid: {error}"
        );
    }
}

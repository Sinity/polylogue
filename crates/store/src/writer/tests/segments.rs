use super::*;
use crate::staging::PreparedBatch;
#[cfg(feature = "faults")]
use crate::staging::StoredInput;
use std::sync::Arc;

#[cfg(feature = "faults")]
fn append(archive: &Archive, writer: &Writer, input: &str, bytes: &[u8]) -> Result<StoredInput> {
    writer
        .sender()?
        .store_group(
            file_group(archive, input, bytes)?,
            Arc::new(archive.blobs()?),
        )?
        .into_iter()
        .next()
        .expect("one batch")
}

#[cfg(feature = "faults")]
fn file_group(archive: &Archive, input: &str, bytes: &[u8]) -> Result<Vec<PreparedBatch>> {
    let (new_blob, file) = archive.prepare_blob(bytes)?;
    let batch = Batch {
        required_disposition: None,
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
        }),
        continues: false,
        new_blob: Some(new_blob),
        blobs: Vec::new(),
        length: bytes.len() as u64,
        parse: Vec::new(),
        acquisition: None,
        basis: None,
        release: Default::default(),
    };
    Ok(vec![PreparedBatch {
        batch,
        files: file.into_iter().collect(),
    }])
}

/// Queues "second" and, while its append is held, "third", so the writer
/// appends third before second's sync is waited for.
#[cfg(feature = "faults")]
fn queue_two_groups(
    archive: &Archive,
    writer: &Writer,
) -> [std::sync::mpsc::Receiver<Vec<Result<StoredInput>>>; 2] {
    let sender = writer.sender().unwrap();
    let store = Arc::new(archive.blobs().unwrap());
    crate::fault::arm(
        "segment.after_append",
        "second",
        crate::fault::Action::Hold,
        0,
        1,
    );
    let second = sender
        .enqueue_group(
            file_group(archive, "second", SECOND).unwrap(),
            Arc::clone(&store),
        )
        .unwrap();
    while !crate::fault::held() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let third = sender
        .enqueue_group(file_group(archive, "third", THIRD).unwrap(), store)
        .unwrap();
    crate::fault::release();
    [second, third]
}

#[cfg(feature = "faults")]
fn blob_row(conn: &rusqlite::Connection, bytes: &[u8]) -> bool {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM blob WHERE hash=?1)",
        [hashed_blobs::Hash::of(bytes).0.as_slice()],
        |row| row.get(0),
    )
    .unwrap()
}

#[cfg(feature = "faults")]
const SECOND: &[u8] = b"second immutable source\n";
#[cfg(feature = "faults")]
const THIRD: &[u8] = b"third immutable source\n";

#[test]
fn group_commits_one_sync_and_one_extent_for_equal_bytes() {
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
    let results = writer
        .sender()
        .unwrap()
        .store_group(group, Arc::new(archive.blobs().unwrap()))
        .unwrap();
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
    let subject = std::env::var("POLYLOGUE_SEGMENT_CRASH_SUBJECT").unwrap();
    let (archive, _lock) = Archive::open_or_create(std::path::Path::new(&root)).unwrap();
    let writer = Writer::start(&archive).unwrap();
    crate::fault::arm(&point, &subject, crate::fault::Action::StopProcess, 0, 1);
    for results in queue_two_groups(&archive, &writer) {
        let _ = results.recv();
    }
    panic!("the configured durability boundary was not reached");
}

#[cfg(feature = "faults")]
#[test]
fn crash_leaves_only_a_tail_at_each_product_commit_boundary() {
    // The writer appends "third" before it waits for "second"'s sync, so
    // up to the commit of second both groups are uncommitted tails.
    for (point, subject, committed) in [
        ("segment.after_append", "third", 1),
        ("segment.after_sync", "second", 1),
        ("segment.after_commit", "second", 2),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let first = b"first immutable source\n";
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
            .env("POLYLOGUE_SEGMENT_CRASH_SUBJECT", subject)
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
        assert_eq!(
            std::fs::metadata(directory.path().join("segments/current.hb"))
                .unwrap()
                .len(),
            end as u64
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM blob", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            committed,
            "{point} {subject}"
        );
        assert!(!blob_row(&conn, THIRD));
        append(
            &archive,
            &writer,
            "fourth",
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

#[cfg(feature = "faults")]
#[test]
fn a_failed_group_commit_appends_the_next_group_again() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    crate::fault::arm(
        "segment.after_sync",
        "second",
        crate::fault::Action::Fail,
        0,
        1,
    );
    let [second, third] = queue_two_groups(&archive, &writer);
    let second = second.recv().unwrap();
    assert!(second[0].is_err());
    let third = third.recv().unwrap();
    assert_eq!(third[0].as_ref().unwrap().input, "third");
    let blobs = archive.blobs().unwrap();
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert!(!blob_row(&conn, SECOND));
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(
        &mut blobs.read(&hashed_blobs::Hash::of(THIRD)).unwrap(),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(bytes, THIRD);
    let end: i64 = conn
        .query_row("SELECT committed_end FROM segment", [], |row| row.get(0))
        .unwrap();
    let extent = blobs.extent(&hashed_blobs::Hash::of(THIRD)).unwrap();
    assert_eq!(extent.offset + extent.len, end as u64);
    drop(writer);
    assert!(archive.blobs().unwrap().scan().unwrap().is_empty());
}

#[cfg(feature = "faults")]
#[test]
fn a_failed_append_lets_the_group_left_appended_commit_first() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    crate::fault::arm(
        "segment.after_append",
        "third",
        crate::fault::Action::Fail,
        0,
        1,
    );
    let [second, third] = queue_two_groups(&archive, &writer);
    let second = second.recv().unwrap();
    assert_eq!(second[0].as_ref().unwrap().input, "second");
    let third = third.recv().unwrap();
    assert!(third[0].is_err());
    append(&archive, &writer, "fourth", b"fourth immutable source\n").unwrap();
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert!(blob_row(&conn, SECOND));
    assert!(!blob_row(&conn, THIRD));
    assert!(blob_row(&conn, b"fourth immutable source\n"));
    let end: i64 = conn
        .query_row("SELECT committed_end FROM segment", [], |row| row.get(0))
        .unwrap();
    let blobs = archive.blobs().unwrap();
    let extent = blobs
        .extent(&hashed_blobs::Hash::of(b"fourth immutable source\n"))
        .unwrap();
    assert_eq!(extent.offset + extent.len, end as u64);
    drop(writer);
    assert!(archive.blobs().unwrap().scan().unwrap().is_empty());
}

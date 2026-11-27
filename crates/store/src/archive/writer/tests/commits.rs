//! The archive writer's commits: positions, read-back after a position, the
//! request ID rule, and user writes beside the index writer.
use super::*;
use crate::bytes::{Bytes, Directory, Segments};
use crate::{InputKey, Observation};

type Start = fn(&Archive) -> crate::Result<Arc<dyn Bytes>>;

fn homes() -> [(&'static str, Start); 2] {
    [
        ("segments", |archive| {
            Ok(Arc::new(Segments::start(archive)?))
        }),
        ("directory", |archive| {
            Ok(Arc::new(Directory::start(archive)?))
        }),
    ]
}

/// The writers of the archive at `root`, its empty first index installed.
fn open(root: &std::path::Path, start: Start) -> (Archive, crate::ArchiveLock, Writer) {
    let (archive, lock) = Archive::open_or_create(root).unwrap();
    let writer = Writer::start_with(&archive, start(&archive).unwrap()).unwrap();
    if archive.index_build.building() {
        crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
    }
    (archive, lock, writer)
}

/// Captures `bytes` as the whole of a new file `input`.
fn capture(archive: &Archive, writer: &Writer, input: &str, bytes: &[u8]) -> StoredInput {
    let batch = Batch {
        input: InputKey {
            id: input.into(),
            format: "test-bytes".into(),
            route: polylogue_model::Route::Live,
        },
        change: crate::Change::First,
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
        new_blob: None,
        blobs: Vec::new(),
        length: bytes.len() as u64,
        parse: Vec::new(),
        capture_read: None,
        release: Default::default(),
    };
    writer
        .store_bytes(archive.prepare_bytes(bytes).unwrap(), batch)
        .unwrap()
}

fn after(archive: &Archive, position: Position) -> Vec<ArchiveRow> {
    let pool = crate::ReadPool::open(archive).unwrap();
    pool.reader()
        .unwrap()
        .after(position, Followed::ALL)
        .unwrap()
}

/// The inputs of `rows`, as revision or observation, in order.
fn shape(rows: &[ArchiveRow]) -> Vec<(String, &'static str)> {
    rows.iter()
        .map(|row| match row {
            ArchiveRow::Revision { input, .. } => (input.clone(), "revision"),
            ArchiveRow::Observation { input, .. } => (input.clone(), "observation"),
            ArchiveRow::Alternate { id, .. } => (id.clone(), "alternate"),
        })
        .collect()
}

#[test]
fn commit_after_returns_rows_in_commit_order_across_restart() {
    for (name, start) in homes() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("archive");
        let first = {
            let (archive, _lock, writer) = open(&root, start);
            capture(&archive, &writer, "first", b"first synthetic file\n");
            capture(&archive, &writer, "second", b"second synthetic file\n");
            let rows = after(&archive, Position(0));
            assert_eq!(
                shape(&rows),
                [
                    ("first".into(), "revision"),
                    ("first".into(), "observation"),
                    ("second".into(), "revision"),
                    ("second".into(), "observation"),
                ],
                "{name}"
            );
            assert!(rows[0].position() < rows[2].position(), "{name}");
            assert_eq!(rows[0].position(), rows[1].position(), "{name}");
            assert_eq!(
                shape(&after(&archive, rows[1].position())),
                [
                    ("second".into(), "revision"),
                    ("second".into(), "observation")
                ],
                "{name}"
            );
            rows
        };
        let (archive, _lock, writer) = open(&root, start);
        assert_eq!(after(&archive, Position(0)), first, "{name}");
        capture(&archive, &writer, "third", b"third synthetic file\n");
        let later = after(&archive, first[3].position());
        assert_eq!(
            shape(&later),
            [
                ("third".into(), "revision"),
                ("third".into(), "observation")
            ],
            "{name}"
        );
        assert!(later[0].position() > first[3].position(), "{name}");
    }
}

fn setting(id: &str, value: &str) -> UserRequest {
    UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: id.into(),
        machine: "machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: value.into(),
            machine: None,
        },
    }
}

fn write_counter(archive: &Archive) -> i64 {
    crate::write_counter(&crate::archive::open_read(&archive.archive_db()).unwrap()).unwrap()
}

#[test]
fn a_repeated_request_id_answers_the_first_commit_and_writes_nothing() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let first = archive
        .writer
        .archive()
        .submit(setting("repeated", "on"), None)
        .unwrap();
    let counter = write_counter(&archive.archive);
    let again = archive
        .writer
        .archive()
        .submit(setting("repeated", "on"), None)
        .unwrap();
    assert_eq!(again.position, first.position);
    assert_eq!(again.outcome.outcome.id, first.outcome.outcome.id);
    assert!(again.outcome.outcome.repeat);
    assert_eq!(write_counter(&archive.archive), counter);
    let committed = archive
        .writer
        .archive()
        .committed_for("repeated")
        .unwrap()
        .expect("the request is recorded");
    assert_eq!(committed.outcome.id, first.outcome.outcome.id);
}

#[test]
fn accepted_deliveries_are_pending_until_their_commit() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let sender = archive.writer.archive().clone();
    let pause = sender.pause_user_dispatch_for_test("delivery-one", DispatchPhase::Owned);
    let delivery = UserRequest {
        id: "delivery-one".into(),
        write: UserWrite::Deliver(polylogue_model::user::Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some("claude-code:recipient".into()),
            sender: polylogue_model::DeliverySender::Hook,
            event: polylogue_model::DeliveryEvent::SessionStart,
            cwd: None,
            text: "synthetic brief".into(),
            refs: Vec::new(),
            left_out: Vec::new(),
            prepared_position: Some(7),
        }),
        ..setting("delivery-one", "on")
    };
    let committed = sender.accept_delivery(delivery.clone()).unwrap();
    pause.wait().unwrap();
    let pending = |sender: &crate::archive::writer::ArchiveSender| {
        sender
            .pending_deliveries()
            .iter()
            .map(|request| match &request.write {
                UserWrite::Deliver(delivery) => (request.id.clone(), delivery.prepared_position),
                _ => panic!("a delivery"),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(pending(&sender), [("delivery-one".to_owned(), Some(7))]);
    let repeated = sender.accept_delivery(delivery).unwrap();
    assert_eq!(pending(&sender)[0], ("delivery-one".to_owned(), Some(7)));
    pause.release();
    let first = committed.recv().unwrap().unwrap();
    let again = repeated.recv().unwrap().unwrap();
    assert_eq!(again.position, first.position);
    assert!(again.outcome.outcome.repeat);
    assert!(sender.pending_deliveries().is_empty());
}

#[cfg(feature = "faults")]
#[test]
fn a_user_write_commits_while_the_index_writer_holds_a_transaction() {
    use crate::fault::{self, Action};
    let mut archive = crate::fixture::writer::WriterArchive::new();
    fault::clear();
    fault::arm("write.index_commit", "", Action::Hold, 0, 1);
    let writer = archive.writer.sender();
    let held = std::thread::spawn(move || {
        archive.write(
            "held-session",
            crate::fixture::writer::prose("held-session", 1, 1_000, "a synthetic prompt"),
        );
        archive
    });
    fault::wait_held();
    // The index writer is inside its transaction; a user write needs one
    // archive commit only.
    let committed = writer
        .archive()
        .submit(setting("while-held", "on"), None)
        .unwrap();
    assert!(!committed.outcome.outcome.repeat);
    fault::release();
    held.join().unwrap();
    fault::clear();
}

#[test]
fn followed_returns_once_the_index_holds_the_alternate_text() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let committed = archive
        .writer
        .archive()
        .commit(Group::RestoredRequest(super::alternate()))
        .unwrap();
    archive.writer.followed(committed.position, None).unwrap();
    assert!(archive.writer.followed_position() >= committed.position);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let mut found = 0;
    pool.reader()
        .unwrap()
        .visit_alternate_search("pierwszy", Some("pl"), |_| {
            found += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(found, 1);
}

#[test]
fn followed_ends_with_the_writer_and_at_a_past_deadline() {
    let mut archive = crate::fixture::writer::WriterArchive::new();
    let sender = archive.writer.sender();
    let limit = crate::read::limit::ReadLimit::new(
        std::time::Duration::ZERO,
        crate::cancellation::Cancellation::default(),
    );
    assert!(matches!(
        sender.followed(Position(u64::MAX), Some(&limit)),
        Err(StoreError::ReadDeadlineExceeded)
    ));
    archive.writer.stop();
    assert!(matches!(
        sender.followed(Position(u64::MAX), None),
        Err(StoreError::WriterGone)
    ));
}

/// Stores `bytes` as the step `change` of input `input` found at `path`.
fn step(
    archive: &Archive,
    writer: &Writer,
    input: &str,
    path: &str,
    change: crate::Change,
    bytes: &[u8],
    length: u64,
) -> StoredInput {
    let (new_blob, file) = archive.prepare_blob(bytes).unwrap();
    let batch = Batch {
        input: InputKey {
            id: input.into(),
            format: "test-bytes".into(),
            route: polylogue_model::Route::Live,
        },
        change,
        observation: Some(Observation {
            machine: "test-machine".into(),
            host: "test-host".into(),
            path: path.into(),
            path_bytes: path.as_bytes().to_vec(),
            facts: Some(crate::FileFacts {
                device: 1,
                inode: 2,
                size: length,
                mtime_ns: 3,
                ctime_ns: 4,
            }),
            seen_at_ms: 1,
            import: None,
            container: None,
            database: None,
        }),
        continues: false,
        new_blob: Some(new_blob),
        blobs: Vec::new(),
        length,
        parse: Vec::new(),
        capture_read: None,
        release: Default::default(),
    };
    writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch,
            files: file.into_iter().collect(),
        }])
        .unwrap()
        .pop()
        .unwrap()
        .unwrap()
}

fn rows(archive: &Archive, sql: &str) -> i64 {
    crate::archive::open_read(&archive.archive_db())
        .unwrap()
        .query_row(sql, [], |row| row.get(0))
        .unwrap()
}

/// An append at the location the input was last observed at adds a
/// revision and no observation; an append at another location and a
/// rewrite each add one (docs/archive.md, stored inputs).
#[test]
fn appends_at_same_location_add_no_observation() {
    let w = crate::fixture::writer::WriterArchive::new();
    let (archive, writer) = (&w.archive, &w.writer);
    step(
        archive,
        writer,
        "file",
        "/home/op/a",
        crate::Change::First,
        b"one\n",
        4,
    );
    step(
        archive,
        writer,
        "file",
        "/home/op/a",
        crate::Change::Appended { from: 4 },
        b"two\n",
        8,
    );
    assert_eq!(
        rows(
            archive,
            "SELECT count(*) FROM revision WHERE input = 'file'"
        ),
        2
    );
    assert_eq!(
        rows(
            archive,
            "SELECT count(*) FROM observation WHERE input = 'file'"
        ),
        1
    );
    step(
        archive,
        writer,
        "file",
        "/home/op/b",
        crate::Change::Appended { from: 8 },
        b"three\n",
        14,
    );
    assert_eq!(
        rows(
            archive,
            "SELECT count(*) FROM observation WHERE input = 'file' AND class = 'appended'"
        ),
        1
    );
    step(
        archive,
        writer,
        "file",
        "/home/op/b",
        crate::Change::Rewritten,
        b"other\n",
        6,
    );
    assert_eq!(
        rows(
            archive,
            "SELECT count(*) FROM observation WHERE input = 'file'"
        ),
        3
    );
    // Each revision keeps when its bytes were read.
    assert_eq!(
        rows(
            archive,
            "SELECT count(*) FROM revision WHERE input = 'file' AND seen_at = 1"
        ),
        4
    );
}

/// The observation's length is the length of the revision when it was
/// observed: a file stored in two steps of one read keeps the first step's.
#[test]
fn an_observation_keeps_the_length_its_step_stored() {
    let w = crate::fixture::writer::WriterArchive::new();
    let (archive, writer) = (&w.archive, &w.writer);
    step(
        archive,
        writer,
        "file",
        "/home/op/a",
        crate::Change::First,
        b"first half\n",
        11,
    );
    let (new_blob, file) = archive.prepare_blob(b"second half\n").unwrap();
    let batch = Batch {
        input: InputKey {
            id: "file".into(),
            format: "test-bytes".into(),
            route: polylogue_model::Route::Live,
        },
        change: crate::Change::Appended { from: 11 },
        observation: None,
        continues: true,
        new_blob: Some(new_blob),
        blobs: Vec::new(),
        length: 23,
        parse: Vec::new(),
        capture_read: None,
        release: Default::default(),
    };
    writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch,
            files: file.into_iter().collect(),
        }])
        .unwrap()
        .pop()
        .unwrap()
        .unwrap();
    assert_eq!(
        rows(archive, "SELECT length FROM revision WHERE input = 'file'"),
        23
    );
    assert_eq!(
        rows(
            archive,
            "SELECT length FROM observation WHERE input = 'file'"
        ),
        11
    );
}

/// An imported file is one commit of one chain.
#[test]
fn an_imported_file_is_one_commit_of_one_chain() {
    imported_on_two_archives();
}

/// An input named by its content has the same input, revision and extent
/// rows on every archive that imports it (observations aside).
#[test]
fn a_content_named_input_on_two_archives_has_the_same_input_revision_and_extent_rows() {
    let record = imported_on_two_archives();
    assert_eq!(record[0], record[1]);
}

/// Imports one file into two new archives; each import is one commit of
/// one chain. Returns each archive's input, revision and extent rows.
fn imported_on_two_archives() -> Vec<Vec<String>> {
    let bytes = b"synthetic exported conversation\n";
    let mut record = Vec::new();
    for _ in 0..2 {
        let w = crate::fixture::writer::WriterArchive::new();
        let before = write_counter(&w.archive);
        w.writer
            .store_group(vec![super::imported(&w.archive, bytes)])
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(write_counter(&w.archive), before + 1);
        assert_eq!(
            rows(&w.archive, "SELECT count(DISTINCT chain) FROM revision"),
            1
        );
        assert_eq!(
            rows(
                &w.archive,
                "SELECT count(*) FROM chain WHERE begins_like IS NULL AND agrees_to = 0"
            ),
            1
        );
        let conn = crate::archive::open_read(&w.archive.archive_db()).unwrap();
        let mut statement = conn
            .prepare(
                "SELECT 'input', id, origin, '', '' FROM input
                 UNION ALL SELECT 'revision', input, number, chain, length FROM revision
                 UNION ALL SELECT 'extent', input, chain, offset, hex(blob) FROM extent
                 ORDER BY 1, 2, 3",
            )
            .unwrap();
        let these: Vec<String> = statement
            .query_map([], |row| {
                Ok((0..5)
                    .map(|n| format!("{:?}", row.get_ref(n).unwrap()))
                    .collect::<Vec<_>>()
                    .join(" "))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        record.push(these);
    }
    record
}

/// "Latest" is ordered by the observation ID, so a VACUUM, which may
/// renumber rows, changes no answer.
#[test]
fn after_vacuum_the_known_inputs_and_the_import_cursor_read_the_same() {
    let w = crate::fixture::writer::WriterArchive::new();
    let (archive, writer) = (&w.archive, &w.writer);
    for (input, path) in [
        ("one", "/home/op/a"),
        ("two", "/home/op/a"),
        ("three", "/home/op/b"),
    ] {
        step(
            archive,
            writer,
            input,
            path,
            crate::Change::First,
            input.as_bytes(),
            input.len() as u64,
        );
    }
    writer
        .store_group(vec![super::imported(archive, b"exported\n")])
        .unwrap();
    let read = || {
        let pool = crate::ReadPool::open(archive).unwrap();
        let reader = pool.reader().unwrap();
        let known: Vec<(String, Vec<u8>)> = reader
            .known_inputs("test-machine")
            .unwrap()
            .into_iter()
            .map(|known| (known.input_id, known.path_bytes))
            .collect();
        (
            known,
            format!("{:?}", reader.unindexed_import_members().unwrap()),
        )
    };
    let before = read();
    crate::archive::open_write(&archive.archive_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("VACUUM")
        .unwrap();
    assert_eq!(read(), before);
    // Two locations and one import.
    assert_eq!(before.0.len(), 3);
}

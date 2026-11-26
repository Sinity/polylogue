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
        writer.build(Vec::new()).unwrap();
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
        }),
        ..setting("delivery-one", "on")
    };
    let committed = sender.accept_delivery(delivery).unwrap();
    pause.wait().unwrap();
    assert_eq!(
        sender
            .pending_deliveries()
            .iter()
            .map(|request| request.id.as_str())
            .collect::<Vec<_>>(),
        ["delivery-one"]
    );
    pause.release();
    committed.recv().unwrap().unwrap();
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

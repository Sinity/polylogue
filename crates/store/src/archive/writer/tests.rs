use super::*;
use crate::blobs::Change;
use crate::{Archive, Writer};
use crate::{InputKey, Observation};
use polylogue_model::Timestamp;
use polylogue_model::authored::PreparedRequest;
use polylogue_model::user::{UserRequest, UserWrite};
use polylogue_model::{
    AlternateKind, AlternateRelation, Author, BlobHash, NewAlternate, NewAlternatePart,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn prepared(write: UserWrite) -> PreparedRequest {
    prepared_at(write, Timestamp::from_unix_ms(1).unwrap())
}
fn prepared_at(write: UserWrite, at: Timestamp) -> PreparedRequest {
    let archive = crate::fixture::writer::WriterArchive::new();
    let request = UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Agent(None),
            via: polylogue_model::authorship::Channel::Mcp,
        },
        id: "original-request".into(),
        machine: "source-machine".into(),
        at,
        write,
    };
    archive.writer.archive().user(request).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let mut record = None;
    pool.reader()
        .unwrap()
        .visit_authored(&mut |prepared| {
            record = Some(prepared.clone());
            Ok(())
        })
        .unwrap();
    record.unwrap()
}
fn alternate() -> PreparedRequest {
    prepared(UserWrite::Alternate {
        alternate: NewAlternate {
            base: None,
            source: "message:codex:source".into(),
            source_hash: BlobHash([1; 32]),
            kind: AlternateKind::Translation,
            from_lang: Some("en".into()),
            to_lang: Some("pl".into()),
            method: None,
            parts: vec![NewAlternatePart {
                of: "block:codex:source#0".into(),
                of_hash: BlobHash([2; 32]),
                relation: AlternateRelation::Translates,
                body: "pierwszy obraz".into(),
            }],
        },
    })
}
#[test]
fn writer_restore_changes_the_snapshot_once() {
    let original = alternate();
    let destination = crate::fixture::writer::WriterArchive::new();
    let pool = crate::ReadPool::open(&destination.archive).unwrap();
    destination
        .writer
        .archive()
        .restore_authored(original.clone())
        .unwrap();
    // The alternate's text reaches the index once the index writer follows.
    destination.writer.flush_rows().unwrap();
    let first = pool.reader().unwrap().change_position().unwrap();
    assert!(
        destination
            .writer
            .archive()
            .restore_authored(original.clone())
            .unwrap()
            .repeat
    );
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), first);
    // A request carries no record of which bundles delivered it, so a second
    // import of the same request changes nothing.
    assert!(
        destination
            .writer
            .archive()
            .restore_authored(original.clone())
            .unwrap()
            .repeat
    );
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), first);
    let mut copies = Vec::new();
    pool.reader()
        .unwrap()
        .visit_authored(&mut |record| {
            copies.push(record.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(copies, vec![original.clone()]);
    let mut results = Vec::new();
    pool.reader()
        .unwrap()
        .visit_alternate_search("pierwszy", Some("pl"), |found| {
            results.push(found);
            Ok(())
        })
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].part.body, "pierwszy obraz");
    assert_eq!(results[0].alternate.author, Author::Agent(None));
    assert_eq!(results[0].alternate.machine, "source-machine");
}

#[test]
fn commit_subscribers_wake_for_new_commits_and_skip_retries() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let commits = archive.writer.completion_commits();
    let request = UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: "setting".into(),
        machine: "machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    };
    archive.writer.archive().user(request.clone()).unwrap();
    commits.try_recv().unwrap();
    archive.writer.archive().user(request.clone()).unwrap();
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    let mut changed = request;
    if let UserWrite::SetSetting { value, .. } = &mut changed.write {
        *value = "off".into();
    }
    assert!(matches!(
        archive.writer.archive().user(changed),
        Err(StoreError::Conflict(_))
    ));
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
}

fn imported(archive: &Archive, bytes: &[u8]) -> crate::staging::PreparedBatch {
    let (new_blob, file) = archive.prepare_blob(bytes).unwrap();
    crate::staging::PreparedBatch {
        batch: crate::Batch {
            // An import is named by its complete bytes.
            input: InputKey {
                id: new_blob.hash.to_string(),
                format: "test-bytes".into(),
                route: polylogue_model::Route::Export,
            },
            change: Change::First,
            observation: Some(Observation {
                machine: "test-machine".into(),
                host: "test-host".into(),
                path: "/home/op/export.json".into(),
                path_bytes: b"/home/op/export.json".to_vec(),
                facts: None,
                seen_at_ms: 1,
                import: None,
                container: None,
                database: None,
            }),
            continues: false,
            new_blob: Some(new_blob.clone()),
            blobs: Vec::new(),
            length: bytes.len() as u64,
            parse: Vec::new(),
            capture_read: None,
            release: Default::default(),
        },
        files: file.into_iter().collect(),
    }
}

/// Fails the next register sync before it reaches the disk call, so the
/// tests below show what the writer answers and refuses after a failed sync;
/// they cannot show what a real disk error left on the disk.
fn fail_next_sync() {
    crate::fault::arm("write.register_sync", "", crate::fault::Action::Fail, 0, 1);
}

/// What a refused writer start names: the register's log, which failed.
fn assert_refused_start(archive: &Archive) {
    match Writer::start(archive) {
        Err(StoreError::StorageFailed(failure)) => {
            let mut log = archive.archive_db().into_os_string();
            log.push("-wal");
            // The same file, whichever path names it.
            assert_eq!(
                std::fs::canonicalize(&failure.path).unwrap(),
                std::fs::canonicalize(std::path::PathBuf::from(log)).unwrap()
            );
            assert_eq!(crate::storage_failure::recorded(), Some(failure));
        }
        Err(other) => panic!("refused for another reason: {other}"),
        Ok(_) => panic!("a writer started after a failed sync in this process"),
    }
}

/// An authored restore whose sync failed is visible but not known to be on
/// disk: neither it nor its identical retry is answered, and no writer starts
/// again in this process.
#[test]
fn a_failed_sync_refuses_an_authored_retry_and_every_later_writer() {
    let original = alternate();
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    fail_next_sync();
    let first = writer.archive().restore_authored(original.clone());
    assert!(matches!(first, Err(StoreError::WriterGone)), "{first:?}");
    let retry = writer.archive().restore_authored(original.clone());
    assert!(
        matches!(retry, Err(StoreError::WriterGone)),
        "a retry was answered after the failed sync: {retry:?}"
    );
    drop(writer);
    assert_refused_start(&archive);
    crate::fault::clear();
}

/// The same for an import: the repeated import that would find its bytes
/// stored is refused, not answered as already stored.
#[test]
fn a_failed_sync_refuses_a_repeated_import_and_every_later_writer() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let bytes = b"exported conversation bytes\n";
    let writer = Writer::start(&archive).unwrap();
    let sender = writer.sender();
    fail_next_sync();
    let first = sender.store_group(vec![imported(&archive, bytes)]);
    assert!(matches!(first, Err(StoreError::WriterGone)), "{first:?}");
    let retry = sender.store_group(vec![imported(&archive, bytes)]);
    assert!(
        matches!(retry, Err(StoreError::WriterGone)),
        "a repeated import was answered after the failed sync: {retry:?}"
    );
    drop(sender);
    drop(writer);
    assert_refused_start(&archive);
    crate::fault::clear();
}

#[test]
fn literal_deliveries_preserve_variant_reads_without_rebuilding_other_work() {
    let destination = crate::fixture::writer::WriterArchive::new();
    let original = alternate();
    destination
        .writer
        .archive()
        .restore_authored(original.clone())
        .unwrap();
    destination.writer.flush_rows().unwrap();
    let pool = crate::ReadPool::open(&destination.archive).unwrap();
    let read = || {
        let mut alternates = Vec::new();
        pool.reader()
            .unwrap()
            .visit_alternate_search("pierwszy", Some("pl"), |found| {
                alternates.push(found);
                Ok(())
            })
            .unwrap();
        alternates
    };
    let before = serde_json::to_value(read()).unwrap();
    assert_eq!(before.as_array().unwrap().len(), 1);
    let conn = rusqlite::Connection::open(destination.archive.index_db()).unwrap();
    let source_uri = format!(
        "file:{}?mode=ro",
        destination.archive.archive_db().display()
    );
    conn.execute("ATTACH DATABASE ?1 AS a", [&source_uri])
        .unwrap();
    crate::source::views(&conn).unwrap();
    // Variant derivation can lag after an interrupted prior write. Recording an
    // unrelated delivery does not take over that pending derive work.
    conn.execute("DELETE FROM alternate_indexed", []).unwrap();
    conn.execute_batch("CREATE TABLE test_variant_writes(n INTEGER NOT NULL); INSERT INTO test_variant_writes VALUES(0); CREATE TRIGGER test_variant_write AFTER DELETE ON alternate_text BEGIN UPDATE test_variant_writes SET n=n+1; END;").unwrap();
    for advice in [false, true] {
        let body = polylogue_model::user::Delivery {
            recipient_kind: "operator".into(),
            recipient_id: None,
            sender: polylogue_model::DeliverySender::Hook,
            event: polylogue_model::DeliveryEvent::Advice,
            cwd: None,
            text: "A delivery unrelated to alternate history".into(),
            refs: vec![],
            left_out: vec![],
            prepared_position: None,
        };
        let request = crate::fixture::authored::operator_request(
            format!("delivery-{advice}"),
            if advice {
                UserWrite::DeliveredAdvice(body)
            } else {
                UserWrite::Deliver(body)
            },
        );
        destination.writer.archive().user(request).unwrap();
        assert_eq!(serde_json::to_value(read()).unwrap(), before);
        assert_eq!(
            conn.query_row("SELECT n FROM test_variant_writes", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM alternate_indexed", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    // The same derivation routine stays live for the families that need it.
    let steps = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&steps);
    conn.progress_handler(
        1,
        Some(move || {
            observed.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    crate::alternates::project(&conn).unwrap();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    assert!(steps.load(Ordering::Relaxed) > 0);
    assert!(
        conn.query_row("SELECT n FROM test_variant_writes", [], |r| r
            .get::<_, i64>(0))
            .unwrap()
            > 0
    );
    assert_eq!(serde_json::to_value(read()).unwrap(), before);
    println!(
        "alternate derivation positive control: {} VM steps",
        steps.load(Ordering::Relaxed)
    );
}

mod commits;

use super::*;
use crate::primary::{
    Capture, CapturedInput, Confirmed, ConfirmedChange, PendingCapture, PreparedInput,
    PreparedWrite, Product, ProductWrite, ReadingProducts,
};
use polylogue_model::authored::{ExecutionEligibility, PendingAuthored};
use polylogue_model::user::{NewChatAction, UserRequest, UserWrite};
use polylogue_model::{Author, BlobHash, NewVariant, NewVariantPart, VariantKind, VariantRelation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn prepared(write: UserWrite) -> PreparedRequest {
    let archive = crate::fixture::writer::WriterArchive::new();
    let request = UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Agent(None),
            via: polylogue_model::authorship::Channel::Mcp,
        },
        id: "original-request".into(),
        machine: "source-machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        write,
    };
    archive.writer.user(request).unwrap();
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
fn rendition() -> PreparedRequest {
    prepared(UserWrite::Variant {
        variant: NewVariant {
            base: None,
            source: "message:codex:source".into(),
            source_hash: BlobHash([1; 32]),
            kind: VariantKind::Translation,
            from_lang: Some("en".into()),
            to_lang: Some("pl".into()),
            method: None,
            parts: vec![NewVariantPart {
                of: "block:codex:source#0".into(),
                of_hash: BlobHash([2; 32]),
                relation: VariantRelation::Translates,
                body: "pierwszy obraz".into(),
            }],
        },
    })
}
#[test]
fn writer_restore_changes_the_snapshot_once_and_projects_frozen_variant_text() {
    let original = rendition();
    let destination = crate::fixture::writer::WriterArchive::new();
    let pool = crate::ReadPool::open(&destination.archive).unwrap();
    destination
        .writer
        .restore_authored(original.clone())
        .unwrap();
    let first = pool.reader().unwrap().change_position().unwrap();
    assert!(
        destination
            .writer
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
        .visit_variant_search("pierwszy", Some("pl"), |found| {
            results.push(found);
            Ok(())
        })
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].part.body, "pierwszy obraz");
    assert_eq!(results[0].variant.author, Author::Agent(None));
    assert_eq!(results[0].variant.machine, "source-machine");
}

#[derive(Default)]
struct FollowProbe {
    calls: Vec<(std::thread::ThreadId, Option<i64>, usize)>,
    pause: Option<(SyncSender<()>, Receiver<()>)>,
    drains: usize,
}

struct Producer {
    path: std::path::PathBuf,
    online: AtomicBool,
    live_calls: AtomicUsize,
    follow: Mutex<FollowProbe>,
}
impl Producer {
    fn original(&self) -> Option<(String, Vec<ProductWrite>)> {
        self.path
            .exists()
            .then(|| serde_json::from_slice(&std::fs::read(&self.path).unwrap()).unwrap())
    }
    fn unavailable<T>() -> Result<T> {
        Err(StoreError::Sinex("offline".into()))
    }
}
impl crate::primary::Sinex for Producer {
    fn raw_acquisition_cursor(
        &self,
        _: &crate::primary::RawAcquisitionQuery,
    ) -> Result<Option<crate::primary::RawAcquisitionCursor>> {
        Ok(None)
    }
    fn prepare_input_group(
        &self,
        _: &crate::primary::CapturePlan,
        _: &mut crate::primary::CaptureMembers<'_>,
    ) -> Result<crate::primary::CaptureReceipt> {
        Self::unavailable()
    }
    fn pending_source(
        &self,
        _: &crate::primary::SourceSelector,
        _: &str,
    ) -> Result<crate::primary::PendingSource> {
        Ok(Default::default())
    }
    fn capture_body(&self, _: &str) -> Result<Vec<crate::primary::PreparedBody>> {
        Self::unavailable()
    }
    fn prepare_input(
        &self,
        _: &CapturedInput,
        _: crate::primary::PreparedBody,
    ) -> Result<PreparedInput> {
        Self::unavailable()
    }
    fn prepare_blob(
        &self,
        _: BlobHash,
        _: u64,
        _: crate::primary::PreparedBody,
    ) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn pending_group_headers(
        &self,
        _: &str,
        _: Option<u64>,
        _: usize,
    ) -> Result<crate::primary::PendingGroupPage> {
        Err(StoreError::Invalid(
            "synthetic authored adapter has no acquisition journal".into(),
        ))
    }
    fn capture(&self, _: &str) -> Result<Capture> {
        Self::unavailable()
    }
    fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        self.follow.lock().unwrap().drains += 1;
        Ok(vec![])
    }
    fn complete_captures(&self, _: &[String], _: &[ProductWrite]) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_parsed(
        &self,
        _: &crate::primary::ParsedSource,
        _: &[crate::primary::ProductWrite],
    ) -> Result<crate::primary::ParsedPublication> {
        Err(StoreError::Invalid(
            "this synthetic adapter does not admit source interpretations".into(),
        ))
    }
    fn prepare_products(&self, _: &[ProductWrite]) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_request(&self, _: &PreparedRequest) -> Result<UserOutcome> {
        self.live_calls.fetch_add(1, Ordering::SeqCst);
        Self::unavailable()
    }
    fn prepare_reading(
        &self,
        _: &str,
        _: &crate::primary::ExportReadingKey,
        _: &mut ReadingProducts<'_>,
    ) -> Result<crate::primary::ParsedPublication> {
        Self::unavailable()
    }
    fn visit_named_products(
        &self,
        selector: &crate::primary::NamedProductSelector,
        visit: &mut dyn FnMut(&Product) -> Result<()>,
    ) -> Result<()> {
        if let Some((name, writes)) = self.original() {
            let selected = match selector {
                crate::primary::NamedProductSelector::Exact(expected) => name == *expected,
                crate::primary::NamedProductSelector::Prefix(prefix) => name.starts_with(prefix),
            };
            if !selected {
                return Ok(());
            }
            for write in writes {
                visit(&write.product)?;
            }
        }
        Ok(())
    }
    fn visit_pending_requests(
        &self,
        visit: &mut dyn FnMut(&PendingAuthored) -> Result<()>,
    ) -> Result<()> {
        if let Some((_, writes)) = self.original() {
            for write in writes {
                if let Product::RestoredRequest { prepared } = write.product {
                    visit(&PendingAuthored::Restore { prepared })?;
                }
            }
        }
        Ok(())
    }
    fn visit_pending_blobs(&self, _: &mut dyn FnMut(&crate::NewBlob) -> Result<()>) -> Result<()> {
        Ok(())
    }
    fn prepare_named_products(
        &self,
        identity: &str,
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        if let Some((prior, body)) = self.original() {
            assert_eq!(prior, identity);
            assert_eq!(body, writes);
        } else {
            let file = std::fs::File::create(&self.path)?;
            serde_json::to_writer(&file, &(identity, writes))?;
            file.sync_all()?;
        }
        Ok(PreparedWrite {
            identity: identity.into(),
            pending: true,
        })
    }
    fn confirmed(&self, state: &crate::primary::ReplayState, _: &str) -> Result<Vec<Confirmed>> {
        let pause = {
            let mut probe = self.follow.lock().unwrap();
            let drains = probe.drains;
            probe
                .calls
                .push((std::thread::current().id(), state.applied_position, drains));
            probe.pause.take()
        };
        if let Some((ready, release)) = pause {
            ready.send(()).unwrap();
            release.recv().unwrap();
        }
        if !self.online.load(Ordering::SeqCst) {
            return Self::unavailable();
        }
        if state.applied_position.is_some_and(|after| after >= 1) {
            return Ok(vec![]);
        }
        let (_, writes) = self.original().unwrap();
        let mut events = writes
            .into_iter()
            .enumerate()
            .map(|(n, write)| Confirmed {
                native: None,
                position: 1,
                output_index: n as u64,
                event: format!("restoration-{n}"),
                change: ConfirmedChange::Live(Box::new(write.product)),
            })
            .collect::<Vec<_>>();
        events.push(Confirmed {
            native: None,
            position: 1,
            output_index: 0,
            event: String::new(),
            change: ConfirmedChange::Control(Box::new(crate::primary::ReplayControl::Completion {
                position: 1,
                delivery: None,
            })),
        });
        Ok(events)
    }
    fn acknowledge(
        &self,
        _: &crate::primary::ReplayState,
        _: &dyn Fn(&str) -> Result<bool>,
        _: &dyn Fn(
            &crate::primary::ParsedSource,
            &crate::primary::PublicationReceipt,
        ) -> Result<bool>,
    ) -> Result<()> {
        Ok(())
    }
}
#[test]
fn a_pending_scope_reset_cannot_resolve_or_publish_authored_work_from_old_cache() {
    let original = rendition();
    let directory = tempfile::tempdir().unwrap();
    let producer = Arc::new(Producer {
        path: directory.path().join("private-product.json"),
        online: AtomicBool::new(false),
        live_calls: AtomicUsize::new(0),
        follow: Mutex::default(),
    });
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let connection = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
    connection
        .execute(
            "INSERT INTO meta(key,value) VALUES ('confirmed_scope_pending','narrow')",
            [],
        )
        .unwrap();
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    assert!(matches!(
        writer.user(original.request.clone()),
        Err(StoreError::ScopeChanged)
    ));
    assert!(matches!(
        writer.restore_authored(original.clone()),
        Err(StoreError::ScopeChanged)
    ));
    assert_eq!(producer.live_calls.load(Ordering::SeqCst), 0);
    assert!(!producer.path.exists());
    connection
        .execute("DELETE FROM meta WHERE key='confirmed_scope_pending'", [])
        .unwrap();
    archive.read_scope.begin_reset().unwrap();
    assert!(matches!(
        writer.user(original.request),
        Err(StoreError::ScopeChanged)
    ));
    assert_eq!(producer.live_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn primary_restore_stays_pending_across_reopen_and_cannot_publish_a_live_retry() {
    let original = prepared(UserWrite::ChatEnqueue(NewChatAction {
        id: "chat".into(),
        request_key: "original-key".into(),
        request_hash: BlobHash::of(b"chat"),
        request: serde_json::json!({"text":"synthetic"}),
        stage_only: true,
        needs_approval: false,
        attachments: vec![],
    }));
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("cache");
    let bytes = directory.path().join("bytes");
    let producer = Arc::new(Producer {
        path: directory.path().join("durable-product.json"),
        online: AtomicBool::new(false),
        live_calls: AtomicUsize::new(0),
        follow: Mutex::default(),
    });
    {
        let (archive, _lock) = Archive::open_primary(&root, &bytes).unwrap();
        let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        let commits = writer.sender().unwrap().completion_commits();
        assert!(writer.restore_authored(original.clone()).unwrap().pending);
        assert!(writer.restore_authored(original.clone()).unwrap().pending);
        assert!(writer.user(original.request.clone()).unwrap().pending);
        assert_eq!(producer.live_calls.load(Ordering::SeqCst), 0);
        assert!(matches!(
            commits.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        let pool = crate::ReadPool::open(&archive).unwrap();
        assert!(
            pool.reader()
                .unwrap()
                .user_request(&original.request.id)
                .unwrap()
                .is_none()
        );
    }
    producer.online.store(true, Ordering::SeqCst);
    let (archive, _lock) = Archive::open_primary(&root, &bytes).unwrap();
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    let commits = writer.sender().unwrap().completion_commits();
    assert_eq!(writer.synchronize_primary().unwrap(), 1);
    commits.try_recv().unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let stored = pool
        .reader()
        .unwrap()
        .query(
            "SELECT execution FROM source_chat_action WHERE id='chat'",
            &[],
        )
        .unwrap();
    assert_eq!(
        stored[0]["execution"],
        serde_json::json!(ExecutionEligibility::Historical.as_str())
    );
    let mut copies = Vec::new();
    pool.reader()
        .unwrap()
        .visit_authored(&mut |record| {
            assert_eq!(record, &original);
            copies.push(record.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(copies.len(), 1);
    assert!(writer.restore_authored(original).unwrap().repeat);
    assert_eq!(producer.live_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn commit_subscribers_wake_for_new_visible_facts_and_skip_exact_or_failed_user_retries() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let commits = archive.writer.sender().unwrap().completion_commits();
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
    archive.writer.user(request.clone()).unwrap();
    commits.try_recv().unwrap();
    archive.writer.user(request.clone()).unwrap();
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    let mut changed = request;
    if let UserWrite::SetSetting { value, .. } = &mut changed.write {
        *value = "off".into();
    }
    assert!(matches!(
        archive.writer.user(changed),
        Err(StoreError::Conflict(_))
    ));
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
}

#[test]
fn one_follower_owns_confirmation_across_a_blocked_page_and_a_new_write() {
    let directory = tempfile::tempdir().unwrap();
    let (ready, blocked) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let producer = Arc::new(Producer {
        path: directory.path().join("durable-product.json"),
        online: AtomicBool::new(true),
        live_calls: AtomicUsize::new(0),
        follow: Mutex::new(FollowProbe {
            pause: Some((ready, held)),
            ..Default::default()
        }),
    });
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    let original = prepared(UserWrite::ChatEnqueue(NewChatAction {
        id: "interleaved-chat".into(),
        request_key: "interleaved".into(),
        request_hash: BlobHash::of(b"interleaved"),
        request: serde_json::json!({"text":"captured while a page is in flight"}),
        stage_only: true,
        needs_approval: false,
        attachments: vec![],
    }));
    std::thread::scope(|scope| {
        let synchronization = scope.spawn(|| writer.synchronize_primary());
        let started = blocked.recv_timeout(std::time::Duration::from_secs(5));
        if started.is_err() {
            let _ = release.send(());
        }
        started.unwrap();
        // The follower owns a page based on the old applied position. The
        // writer can prepare this operation without fetching another page.
        let restored = writer.restore_authored(original.clone());
        let reads_before_release = producer.follow.lock().unwrap().calls.len();
        release.send(()).unwrap();
        let synchronized = synchronization.join().unwrap();
        assert!(restored.unwrap().pending);
        assert_eq!(reads_before_release, 1);
        assert_eq!(synchronized.unwrap(), 1);
    });
    let probe = producer.follow.lock().unwrap();
    assert!(probe.calls.len() >= 2);
    let owner = probe.calls[0].0;
    assert!(probe.calls.iter().all(|call| call.0 == owner));
    assert_eq!(probe.calls[0].1, None);
    assert_eq!(probe.calls[1].1, Some(1));
    assert!(
        probe.calls[1].2 > probe.calls[0].2,
        "capture drain precedes the next owner page"
    );
    drop(probe);
    let reads = crate::ReadPool::open(&archive).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .user_request(&original.request.id)
            .unwrap()
            .is_some()
    );
}

#[test]
fn a_failed_later_position_reports_earlier_commits_without_inventing_progress() {
    use crate::primary::ReplayControl;
    let directory = tempfile::tempdir().unwrap();
    let producer = Arc::new(Producer {
        path: directory.path().join("unused-product.json"),
        online: AtomicBool::new(false),
        live_calls: AtomicUsize::new(0),
        follow: Mutex::default(),
    });
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let writer = Writer::start_on_sinex(&archive, producer).unwrap();
    let control = |control| Confirmed {
        native: None,
        position: 0,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(control)),
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "partial-page".into(),
        })])
        .unwrap();
    let connection = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
    connection.execute_batch("CREATE TRIGGER stop_projection BEFORE INSERT ON source_input WHEN new.id='second' BEGIN SELECT raise(ABORT,'stop second input'); END").unwrap();
    let input = |position, id: &str| Confirmed {
        native: None,
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: Box::new(crate::input_rows::InputRows {
                acquisition: None,
                input: id.into(),
                format: "synthetic".into(),
                revision: None,
                extents: Vec::new(),
                observation: None,
            }),
            material: None,
        })),
    };
    let commits = writer.sender().unwrap().completion_commits();
    let error = writer
        .apply_confirmed(vec![
            input(1, "first"),
            control(ReplayControl::Completion {
                position: 1,
                delivery: None,
            }),
            input(2, "second"),
            control(ReplayControl::Completion {
                position: 2,
                delivery: None,
            }),
        ])
        .unwrap_err();
    assert!(matches!(error, StoreError::Sqlite(_)), "{error:?}");
    assert!(error.to_string().contains("stop second input"));
    commits.try_recv().unwrap();
    let report = writer.flush_rows().unwrap();
    assert!(
        report.errors.iter().any(|error| {
            error.contains("after position 1") && error.contains("stop second input")
        }),
        "{report:?}"
    );
    assert_eq!(
        crate::primary::projection::checkpoint(&connection).unwrap(),
        Some(1)
    );
    assert!(writer.apply_confirmed(vec![]).is_err());
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    connection
        .execute_batch("DROP TRIGGER stop_projection")
        .unwrap();
    assert_eq!(writer.apply_confirmed(vec![]).unwrap(), 2);
    commits.try_recv().unwrap();
    assert_eq!(
        crate::primary::projection::checkpoint(&connection).unwrap(),
        Some(2)
    );
}

/// One imported file's bytes, prepared for `archive`.
fn imported(archive: &Archive, bytes: &[u8]) -> crate::staging::PreparedBatch {
    let (new_blob, file) = archive.prepare_blob(bytes).unwrap();
    crate::staging::PreparedBatch {
        batch: crate::Batch {
            required_disposition: None,
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
            acquisition: None,
            basis: None,
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
            assert!(
                failure.path.ends_with("archive.db-wal"),
                "{}",
                failure.path.display()
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
    let original = rendition();
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    fail_next_sync();
    let first = writer.restore_authored(original.clone());
    assert!(matches!(first, Err(StoreError::WriterGone)), "{first:?}");
    let retry = writer.restore_authored(original.clone());
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
    let blobs = || Arc::new(archive.blobs().unwrap());
    let bytes = b"exported conversation bytes\n";
    let writer = Writer::start(&archive).unwrap();
    let sender = writer.sender().unwrap();
    fail_next_sync();
    let first = sender.store_group(vec![imported(&archive, bytes)], blobs());
    assert!(matches!(first, Err(StoreError::WriterGone)), "{first:?}");
    let retry = sender.store_group(vec![imported(&archive, bytes)], blobs());
    assert!(
        matches!(retry, Err(StoreError::WriterGone)),
        "a repeated import was answered after the failed sync: {retry:?}"
    );
    drop(sender);
    drop(writer);
    assert_refused_start(&archive);
    crate::fault::clear();
}

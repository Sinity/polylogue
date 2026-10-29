use super::*;
use crate::primary::{
    Capture, CapturedInput, Confirmed, ConfirmedChange, PendingCapture, PreparedInput,
    PreparedWrite, Product, ProductWrite, ReadingProducts,
};
use polylogue_model::authored::{ExecutionEligibility, PendingAuthored};
use polylogue_model::user::{NewChatAction, UserRequest, UserWrite};
use polylogue_model::{Author, BlobHash, NewVariant, NewVariantPart, VariantKind, VariantRelation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn current(input: &str, at: i64) -> ImportAttribution {
    ImportAttribution {
        input: input.into(),
        chain: 1,
        machine: "receiving-machine".into(),
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(at).unwrap(),
    }
}
fn prepared(write: UserWrite) -> PreparedRequest {
    let archive = crate::fixture::writer::WriterArchive::new();
    let request = UserRequest {
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
        .visit_authored(&mut |prepared, _| {
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
            author: Author::Agent(None),
            via: "mcp".into(),
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
fn writer_restore_changes_provenance_snapshot_once_and_projects_frozen_variant_text() {
    let original = rendition();
    let destination = crate::fixture::writer::WriterArchive::new();
    let pool = crate::ReadPool::open(&destination.archive).unwrap();
    destination
        .writer
        .restore_authored(original.clone(), vec![], current("first-import", 10))
        .unwrap();
    let first = pool.reader().unwrap().change_position().unwrap();
    assert!(
        destination
            .writer
            .restore_authored(original.clone(), vec![], current("first-import", 20))
            .unwrap()
            .repeat
    );
    assert_eq!(pool.reader().unwrap().change_position().unwrap(), first);
    let second = current("second-import", 30);
    assert!(
        destination
            .writer
            .restore_authored(original.clone(), vec![], second.clone())
            .unwrap()
            .repeat
    );
    assert_ne!(pool.reader().unwrap().change_position().unwrap(), first);
    let mut copies = Vec::new();
    pool.reader()
        .unwrap()
        .visit_authored(&mut |record, imports| {
            assert_eq!(record, &original);
            copies.extend_from_slice(imports);
            Ok(())
        })
        .unwrap();
    assert_eq!(copies, vec![current("first-import", 10), second]);
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

struct Producer {
    path: std::path::PathBuf,
    online: AtomicBool,
    live_calls: AtomicUsize,
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
    fn prepare_input(&self, _: &CapturedInput, _: &mut dyn std::io::Read) -> Result<PreparedInput> {
        Self::unavailable()
    }
    fn prepare_blob(
        &self,
        _: BlobHash,
        _: u64,
        _: &mut dyn std::io::Read,
    ) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn capture(&self, _: &str) -> Result<Capture> {
        Self::unavailable()
    }
    fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        Ok(vec![])
    }
    fn complete_captures(&self, _: &[String], _: &[ProductWrite]) -> Result<PreparedWrite> {
        Self::unavailable()
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
        _: &str,
        _: &mut ReadingProducts<'_>,
    ) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn visit_pending_requests(
        &self,
        visit: &mut dyn FnMut(&PendingAuthored) -> Result<()>,
    ) -> Result<()> {
        if let Some((_, writes)) = self.original() {
            for write in writes {
                if let Product::RestoredRequest {
                    prepared,
                    claims,
                    current,
                } = write.product
                {
                    visit(&PendingAuthored::Restore {
                        prepared,
                        claims,
                        current,
                    })?;
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
    fn discard_confirmed(&self, _: &[String], _: i64, _: i64) -> Result<()> {
        Ok(())
    }
    fn confirmed(&self, after: i64, _: &str, _: Option<&str>) -> Result<Vec<Confirmed>> {
        if !self.online.load(Ordering::SeqCst) {
            return Self::unavailable();
        }
        if after >= 1 {
            return Ok(vec![]);
        }
        let (_, writes) = self.original().unwrap();
        let mut events = writes
            .into_iter()
            .enumerate()
            .map(|(n, write)| Confirmed {
                position: 1,
                output_index: n as u64,
                event: format!("restoration-{n}"),
                change: ConfirmedChange::Live(Box::new(write.product)),
            })
            .collect::<Vec<_>>();
        events.push(Confirmed {
            position: 1,
            output_index: 0,
            event: String::new(),
            change: ConfirmedChange::PositionComplete,
        });
        Ok(events)
    }
    fn acknowledge(&self, _: i64) -> Result<()> {
        Ok(())
    }
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
    });
    {
        let (archive, _lock) = Archive::open_primary(&root, &bytes).unwrap();
        let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        let commits = writer.sender().unwrap().completion_commits();
        assert!(
            writer
                .restore_authored(original.clone(), vec![], current("import", 10))
                .unwrap()
                .pending
        );
        assert!(
            writer
                .restore_authored(original.clone(), vec![], current("import", 20))
                .unwrap()
                .pending
        );
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
        .visit_authored(&mut |record, imports| {
            assert_eq!(record, &original);
            copies.extend_from_slice(imports);
            Ok(())
        })
        .unwrap();
    assert_eq!(copies, vec![current("import", 10)]);
    assert!(
        writer
            .restore_authored(original, vec![], current("import", 30))
            .unwrap()
            .repeat
    );
    assert_eq!(producer.live_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn commit_subscribers_wake_for_new_visible_facts_and_skip_exact_or_failed_user_retries() {
    let archive = crate::fixture::writer::WriterArchive::new();
    let commits = archive.writer.sender().unwrap().completion_commits();
    let request = UserRequest {
        id: "setting".into(),
        machine: "machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: "key".into(),
            value: "first".into(),
            author: Author::Operator,
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
        *value = "changed".into();
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

use super::*;

fn source(id: &str, bytes: &[u8]) -> PreparedBatch {
    PreparedBatch {
        batch: Batch {
            input: InputKey {
                id: id.into(),
                format: "codex".into(),
                arrival_rank: 0,
            },
            change: Change::First,
            observation: Some(Observation {
                machine: "machine".into(),
                host: "host".into(),
                path: format!("/home/op/{id}"),
                facts: None,
                seen_at_ms: 1,
                import: None,
                container: None,
            }),
            continues: false,
            new_blob: Some(crate::NewBlob {
                hash: BlobHash::of(bytes),
                size: bytes.len() as u64,
            }),
            blobs: Vec::new(),
            length: bytes.len() as u64,
            parse: Vec::new(),
            release: Default::default(),
        },
        files: Vec::new(),
    }
}

#[test]
fn capture_intent_survives_observer_retry_and_preserves_source_order() {
    let mut group = vec![
        source("first", b"first source"),
        source("second", b"second source"),
    ];
    let original = capture_group_identity(&group).unwrap();
    group[0].batch.observation.as_mut().unwrap().seen_at_ms = 9_000;
    group[1].batch.observation.as_mut().unwrap().seen_at_ms = 10_000;
    assert_eq!(capture_group_identity(&group).unwrap(), original);
    group.swap(0, 1);
    assert_ne!(capture_group_identity(&group).unwrap(), original);
    group.swap(0, 1);
    group[0].batch.new_blob.as_mut().unwrap().hash = BlobHash::of(b"other source");
    assert_ne!(capture_group_identity(&group).unwrap(), original);
}

#[test]
fn a_new_reading_and_a_new_source_range_are_new_capture_intents() {
    let mut group = vec![source("session", b"source")];
    let mut step = ParseStep::new(Default::default(), Default::default(), 0..6, true);
    step.reading_version = "codex:1".into();
    group[0].batch.parse.push(step);
    let original = capture_group_identity(&group).unwrap();
    group[0].batch.parse[0].reading_version = "codex:2".into();
    assert_ne!(capture_group_identity(&group).unwrap(), original);
    group[0].batch.parse[0].reading_version = "codex:1".into();
    group[0].batch.change = Change::Appended { from: 6 };
    group[0].batch.length = 12;
    assert_ne!(capture_group_identity(&group).unwrap(), original);
}

struct OfflineProducer {
    root: std::path::PathBuf,
}
impl OfflineProducer {
    fn unavailable<T>() -> Result<T> {
        Err(StoreError::Sinex("synthetic offline transport".into()))
    }
    fn captured(&self) -> CapturedInput {
        serde_json::from_slice(&std::fs::read(self.root.join("capture.json")).unwrap()).unwrap()
    }
}
impl crate::primary::Sinex for OfflineProducer {
    fn prepare_input(
        &self,
        captured: &CapturedInput,
        body: crate::primary::PreparedBody,
    ) -> Result<PreparedInput> {
        std::fs::create_dir_all(&self.root)?;
        let prepared = hashed_blobs::Prepared::from_encoded(
            body.file,
            body.descriptor,
            None,
            hashed_blobs::ReadLimits::default(),
        )?;
        let mut file = prepared.into_file();
        file.rewind()?;
        let mut encoded = std::fs::File::create(self.root.join("source"))?;
        std::io::copy(&mut file, &mut encoded)?;
        encoded.sync_all()?;
        let mut metadata = std::fs::File::create(self.root.join("capture.json"))?;
        serde_json::to_writer(&mut metadata, captured)?;
        metadata.sync_all()?;
        Ok(PreparedInput::Pending {
            identity: "offline-capture".into(),
        })
    }
    fn capture(&self, _: &str) -> Result<Capture> {
        Ok(Capture::Input(Box::new(self.captured())))
    }
    fn prepare_reading(
        &self,
        _: &str,
        _: &str,
        _: &mut crate::primary::ReadingProducts<'_>,
    ) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_named_products(
        &self,
        _: &str,
        _: &[ProductWrite],
    ) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn discard_confirmed(&self, _: &[String], _: i64, _: i64) -> Result<()> {
        Ok(())
    }
    fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        Ok(if self.root.join("capture.json").exists() {
            vec![PendingCapture {
                identity: "offline-capture".into(),
                captured: Capture::Input(Box::new({
                    let mut header = self.captured();
                    header.parsed.clear();
                    header
                })),
                material: None,
                anchor: None,
            }]
        } else {
            Vec::new()
        })
    }
    fn prepare_blob(
        &self,
        _: BlobHash,
        _: u64,
        _: crate::primary::PreparedBody,
    ) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn complete_captures(
        &self,
        _: &[String],
        _: &[ProductWrite],
    ) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_products(&self, _: &[ProductWrite]) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_request(
        &self,
        _: &polylogue_model::authored::PreparedRequest,
    ) -> Result<polylogue_model::user::UserOutcome> {
        Self::unavailable()
    }
    fn visit_pending_requests(
        &self,
        _: &mut dyn FnMut(&polylogue_model::authored::PendingAuthored) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }
    fn visit_pending_blobs(&self, _: &mut dyn FnMut(&crate::NewBlob) -> Result<()>) -> Result<()> {
        Ok(())
    }
    fn confirmed(&self, _: i64, _: &str) -> Result<Vec<crate::primary::Confirmed>> {
        Self::unavailable()
    }
    fn acknowledge(&self, _: i64) -> Result<()> {
        Self::unavailable()
    }
}

#[test]
fn offline_writer_retains_source_and_complete_parsed_facts_before_releasing_preparation() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("cache");
    let byte_root = directory.path().join("bytes");
    let producer_root = directory.path().join("producer");
    let bytes = br#"{"native":"record"}
"#;
    let payload = serde_json::json!({"native":{"values":[1,2,3],"text":"kept event data"}});
    {
        let (archive, _lock) = Archive::open_primary(&cache, &byte_root).unwrap();
        let producer = Arc::new(OfflineProducer {
            root: producer_root.clone(),
        });
        let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        let mut input = source("offline", bytes);
        input.batch.input.arrival_rank = 2;
        let mut counts = RecordCounts {
            records: 1,
            ..Default::default()
        };
        counts.used.insert("native".into(), 1);
        let event = polylogue_model::Event {
            session: None,
            kind: "native_record".into(),
            at: None,
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((0, bytes.len() as u64)),
            data: payload.clone(),
        };
        let mut step = ParseStep::new(
            Output {
                events: vec![event],
                ..Default::default()
            },
            counts,
            0..bytes.len() as u64,
            true,
        );
        step.reading_version = "synthetic:1".into();
        step.prepare();
        input.batch.parse.push(step);
        let prepared = archive.prepare_bytes(bytes.as_slice()).unwrap();
        input.files.push(prepared);
        let mut results = writer
            .sender()
            .unwrap()
            .store_group(vec![input], Arc::new(archive.blobs().unwrap()))
            .unwrap();
        assert!(
            matches!(results.remove(0),Err(StoreError::PrimaryPending {identity}) if identity == "offline-capture")
        );
        let index = crate::archive::open_read(&archive.index_db()).unwrap();
        assert_eq!(
            index
                .query_row("SELECT count(*) FROM source_input", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert!(!archive.root.join("archive.db").exists());
    }
    // The ordinary reopen has neither the source preparation nor a provider
    // file available. The producer's immutable capture contains both facts.
    let (archive, _lock) = Archive::open_primary(&cache, &byte_root).unwrap();
    let reopened = OfflineProducer {
        root: producer_root.clone(),
    };
    let captured = reopened.captured();
    let encoded = std::fs::File::open(producer_root.join("source")).unwrap();
    let prepared = hashed_blobs::Prepared::inspect(encoded, None, Default::default()).unwrap();
    let mut restored = Vec::new();
    prepared
        .reader(None, Default::default())
        .unwrap()
        .read_to_end(&mut restored)
        .unwrap();
    assert_eq!(restored, bytes);
    assert_eq!(captured.arrival_rank, 2);
    assert_eq!(captured.parsed.len(), 1);
    assert_eq!(captured.parsed[0].output.events[0].data, payload);
    assert_eq!(captured.parsed[0].reading_version, "synthetic:1");
    assert_eq!(captured.parsed[0].parsed_upto, bytes.len() as u64);
    assert_eq!(
        std::fs::read_dir(archive.parts_dir().unwrap())
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn import_retry_intent_excludes_its_generated_operation_id() {
    let mut group = vec![source("bundle", b"source")];
    group[0].batch.observation.as_mut().unwrap().import = Some(crate::ImportMember {
        version: 1,
        import_id: "first-operation".into(),
        member: "records.jsonl".into(),
        authorship: Some(polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Script("import-program".into()),
            via: polylogue_model::authorship::Channel::Import,
        }),
        format: Some("polylogue-bundle".into()),
    });
    let first = capture_group_identity(&group).unwrap();
    let member = group[0]
        .batch
        .observation
        .as_mut()
        .unwrap()
        .import
        .as_mut()
        .unwrap();
    member.import_id = "retry-operation".into();
    assert_eq!(capture_group_identity(&group).unwrap(), first);
    group[0]
        .batch
        .observation
        .as_mut()
        .unwrap()
        .import
        .as_mut()
        .unwrap()
        .authorship
        .as_mut()
        .unwrap()
        .author = polylogue_model::Author::Script("other-importer".into());
    assert_ne!(capture_group_identity(&group).unwrap(), first);
}

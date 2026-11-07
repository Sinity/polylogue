use super::*;
use crate::primary::PreparedInput;

fn source(id: &str, bytes: &[u8]) -> PreparedBatch {
    PreparedBatch {
        batch: Batch {
            required_disposition: None,
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
                path_bytes: format!("/home/op/{id}").into_bytes(),
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
            acquisition: None,
            basis: None,
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
    let original = capture_group_identity(&group, false).unwrap();
    group[0].batch.observation.as_mut().unwrap().seen_at_ms = 9_000;
    group[1].batch.observation.as_mut().unwrap().seen_at_ms = 10_000;
    assert_eq!(capture_group_identity(&group, false).unwrap(), original);
    group.swap(0, 1);
    assert_ne!(capture_group_identity(&group, false).unwrap(), original);
    group.swap(0, 1);
    group[0].batch.new_blob.as_mut().unwrap().hash = BlobHash::of(b"other source");
    assert_ne!(capture_group_identity(&group, false).unwrap(), original);
}

#[test]
fn a_new_reading_and_a_new_source_range_are_new_capture_intents() {
    let mut group = vec![source("session", b"source")];
    let mut step = ParseStep::new(Default::default(), Default::default(), 0..6, true);
    step.reading_version = "codex:1".into();
    group[0].batch.parse.push(step);
    let original = capture_group_identity(&group, false).unwrap();
    group[0].batch.parse[0].reading_version = "codex:2".into();
    assert_ne!(capture_group_identity(&group, false).unwrap(), original);
    group[0].batch.parse[0].reading_version = "codex:1".into();
    group[0].batch.change = Change::Appended { from: 6 };
    group[0].batch.length = 12;
    assert_ne!(capture_group_identity(&group, false).unwrap(), original);
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
    fn raw_acquisition_cursor(
        &self,
        _: &crate::primary::RawAcquisitionQuery,
    ) -> Result<Option<crate::primary::RawAcquisitionCursor>> {
        Ok(None)
    }
    fn prepare_input_group(
        &self,
        plan: &crate::primary::CapturePlan,
        members: &mut crate::primary::CaptureMembers<'_>,
    ) -> Result<crate::primary::CaptureReceipt> {
        plan.validate()?;
        if plan.count != 1 {
            return Err(StoreError::Invalid(
                "synthetic offline adapter supports one member".into(),
            ));
        }
        let mut count = 0;
        members(&mut |mut input, body| {
            if count != 0 {
                return Err(StoreError::Invalid(
                    "too many synthetic capture members".into(),
                ));
            }
            input.group = Some(crate::primary::CaptureGroup {
                identity: "offline-capture".into(),
                intent: plan.intent.clone(),
                ordinal: 0,
                count: 1,
            });
            self.prepare_input(&input, body)?;
            count += 1;
            Ok(())
        })?;
        if count != 1 {
            return Err(StoreError::Invalid(
                "missing synthetic capture member".into(),
            ));
        }
        Ok(crate::primary::CaptureReceipt::Accepted {
            write: crate::primary::PreparedWrite {
                identity: "offline-capture".into(),
                pending: true,
            },
        })
    }
    fn capture_body(&self, _: &str) -> Result<crate::primary::PreparedBody> {
        let path = self.root.join("source");
        Ok(crate::primary::PreparedBody {
            file: std::fs::File::open(&path)?,
            descriptor: hashed_blobs::EncodedBlob::decode(&std::fs::read(
                self.root.join("descriptor"),
            )?)?,
            path: Some(path),
        })
    }
    fn pending_source(
        &self,
        source: &crate::primary::SourceSelector,
        version: &str,
    ) -> Result<crate::primary::PendingSource> {
        use crate::primary::{PendingCheckpoint, PendingSource, SourceSelector, SourceStatus};
        if !self.root.join("capture.json").exists() {
            return Ok(PendingSource {
                status: SourceStatus::Ready,
                ..Default::default()
            });
        }
        let mut full = self.captured();
        let selector = full
            .basis
            .as_ref()
            .map(|basis| basis.source.clone())
            .or_else(|| {
                full.rows.observation.as_ref().and_then(|o| {
                    o.path_bytes.as_ref().map(|bytes| SourceSelector {
                        origin: full.rows.format.clone(),
                        machine: o.machine.clone(),
                        path_bytes: bytes.clone(),
                    })
                })
            });
        let Some(selector) = selector else {
            return Ok(PendingSource::default());
        };
        if &selector != source {
            return Ok(PendingSource {
                status: SourceStatus::Ready,
                ..Default::default()
            });
        }
        let checkpoint = full
            .parsed
            .iter()
            .rev()
            .filter_map(|p| p.checkpoint.as_ref())
            .find(|c| c.reading_version == version)
            .cloned();
        full.parsed.clear();
        let head = PendingCapture {
            identity: "offline-capture".into(),
            captured: Capture::Input(Box::new(full.clone())),
            material: None,
            anchor: None,
        };
        Ok(PendingSource {
            status: full
                .basis
                .as_ref()
                .map_or(SourceStatus::MissingEvidence, |basis| basis.status),
            raw: full.raw_range.map(|_| head.clone()),
            head: Some(head),
            continuation: checkpoint.map(|checkpoint| PendingCheckpoint {
                capture: "offline-capture".into(),
                header: full,
                checkpoint,
            }),
        })
    }
    fn prepare_input(
        &self,
        captured: &CapturedInput,
        body: crate::primary::PreparedBody,
    ) -> Result<PreparedInput> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::write(self.root.join("descriptor"), body.descriptor.encode())?;
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
    fn pending_group_headers(
        &self,
        group: &str,
        after: Option<u64>,
        limit: usize,
    ) -> Result<crate::primary::PendingGroupPage> {
        if limit == 0 {
            return Err(StoreError::Invalid("zero group header limit".into()));
        }
        let mut headers = self.pending_captures()?;
        headers.retain(|header| match &header.captured {
            Capture::Input(input) => input.group.as_ref().is_some_and(|g| {
                g.identity == group && after.is_none_or(|ordinal| g.ordinal > ordinal)
            }),
            _ => false,
        });
        for header in &mut headers {
            if let Capture::Input(input) = &mut header.captured {
                input.parsed = Vec::new();
            }
        }
        Ok(crate::primary::PendingGroupPage {
            disposition: Some(crate::primary::CaptureDisposition::Prepared),
            headers,
            next: None,
        })
    }
    fn capture(&self, _: &str) -> Result<Capture> {
        Ok(Capture::Input(Box::new(self.captured())))
    }
    fn prepare_reading(
        &self,
        _: &str,
        _: &crate::primary::ExportReadingKey,
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
                anchor: self
                    .root
                    .join("ready")
                    .exists()
                    .then(|| crate::primary::SourceLocation {
                        material: "confirmed-anchor".into(),
                        anchor: 0,
                    }),
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
        identities: &[String],
        writes: &[ProductWrite],
    ) -> Result<crate::primary::PreparedWrite> {
        if !self.root.join("ready").exists() {
            return Self::unavailable();
        }
        assert_eq!(identities, &["offline-capture".to_owned()]);
        self.prepare_products(writes)
    }
    fn prepare_export_products(
        &self,
        _: &crate::primary::ExportReadingKey,
        _: &[crate::primary::ProductWrite],
    ) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn visit_pending_export_readings(
        &self,
        _: &crate::primary::ExportReadingKey,
        visit: &mut dyn FnMut(&crate::primary::CapturedReading) -> Result<()>,
    ) -> Result<()> {
        let path = self.root.join("pending-export.json");
        if path.exists() {
            let captured = serde_json::from_slice(&std::fs::read(path)?)?;
            visit(&captured)?;
        }
        Ok(())
    }
    /// Spooled products are kept, so a test can read what was sent.
    fn prepare_products(&self, writes: &[ProductWrite]) -> Result<crate::primary::PreparedWrite> {
        std::fs::create_dir_all(&self.root)?;
        let file = std::fs::File::create(self.root.join("products.json"))?;
        serde_json::to_writer(&file, writes)?;
        file.sync_all()?;
        Ok(crate::primary::PreparedWrite {
            identity: "offline-products".into(),
            pending: true,
        })
    }
    fn prepare_request(
        &self,
        _: &polylogue_model::authored::PreparedRequest,
    ) -> Result<polylogue_model::user::UserOutcome> {
        Self::unavailable()
    }
    fn visit_named_products(
        &self,
        _: &crate::primary::NamedProductSelector,
        _: &mut dyn FnMut(&crate::primary::Product) -> Result<()>,
    ) -> Result<()> {
        Ok(())
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
    fn confirmed(
        &self,
        _: &crate::primary::ReplayState,
        _: &str,
    ) -> Result<Vec<crate::primary::Confirmed>> {
        if self.root.join("ready").exists() {
            Ok(vec![])
        } else {
            Self::unavailable()
        }
    }
    fn acknowledge(&self, _: &crate::primary::ReplayState) -> Result<()> {
        if self.root.join("ready").exists() {
            Ok(())
        } else {
            Self::unavailable()
        }
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
        let snapshot = writer
            .source_snapshot(
                crate::primary::SourceSelector {
                    origin: "codex".into(),
                    machine: "machine".into(),
                    path_bytes: b"/home/op/offline".to_vec(),
                },
                "synthetic:1",
            )
            .unwrap();
        assert_eq!(
            snapshot.basis.status,
            crate::primary::SourceStatus::MissingEvidence
        );
        assert_eq!(snapshot.basis.confirmed, None);
        assert_eq!(snapshot.basis.pending.as_deref(), Some("offline-capture"));
        assert!(snapshot.pending.raw.is_some());
        assert!(snapshot.basis.boundary.is_none());
        writer
            .apply_confirmed(vec![crate::primary::Confirmed {
                native: None,
                position: 0,
                output_index: 0,
                event: String::new(),
                change: crate::primary::ConfirmedChange::Control(Box::new(
                    crate::primary::ReplayControl::Completion { position: 0 },
                )),
            }])
            .unwrap();
        let after_checkpoint = writer
            .source_snapshot(
                crate::primary::SourceSelector {
                    origin: "codex".into(),
                    machine: "machine".into(),
                    path_bytes: b"/home/op/offline".to_vec(),
                },
                "synthetic:1",
            )
            .unwrap();
        assert_eq!(
            after_checkpoint.basis.status,
            crate::primary::SourceStatus::MissingEvidence
        );
        assert!(after_checkpoint.basis.boundary.is_none());
        let Capture::Input(hydrated) = writer.capture("offline-capture".into()).unwrap() else {
            panic!("the pending source must hydrate its original input facts");
        };
        assert_eq!(hydrated.parsed[0].output.events[0].data, payload);
        assert_eq!(hydrated.parsed[0].parsed_upto, bytes.len() as u64);
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
    assert_eq!(std::fs::read_dir(archive.staging_dir()).unwrap().count(), 0);
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
    let first = capture_group_identity(&group, false).unwrap();
    let member = group[0]
        .batch
        .observation
        .as_mut()
        .unwrap()
        .import
        .as_mut()
        .unwrap();
    member.import_id = "retry-operation".into();
    assert_eq!(capture_group_identity(&group, false).unwrap(), first);
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
    assert_ne!(capture_group_identity(&group, false).unwrap(), first);
}

fn serialized_binding() -> crate::assets::bindings::AttachmentBindingRecord {
    let message_version = BlobHash::of(b"source message identity");
    let member_hash = BlobHash::of(b"claimed member bytes");
    // A serialized witness is opaque custody, not receiver admission evidence.
    let record: crate::assets::bindings::AttachmentBindingRecord = serde_json::from_value(serde_json::json!({
        "context":{"message":"codex:bound-message","source_version":message_version,"input":"receiving-input","chain":1,"import_id":"receiving-namespace"},
        "claim":{"source_version":message_version,"namespace":{"input":"original-input","chain":2,"import_id":"original-namespace"},"exported_from":{"input":"export-input","chain":3,"import_id":"export-namespace"},"block":0,"attachment":0,"source_id":"claimed-attachment","candidates":[{"hash":member_hash,"size":20,"sources":[]}]},
        "members":[{"candidate":0,"witness":{"version":1,"member":{"import_id":"receiving-namespace","input":"receiving-member","member":"blobs/member","chain":1,"source_version":"frozen-source-version","hash":member_hash,"size":20},"observation":"original-receiving-observation"}}]
    })).unwrap();
    record
}

#[test]
fn an_imported_attachment_binding_is_sent_to_sinex_and_reaches_the_index_only_from_the_stream() {
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("cache");
    let bytes = directory.path().join("bytes");
    let producer_root = directory.path().join("producer");
    let record = serialized_binding();
    let (archive, _lock) = Archive::open_primary(&cache, &bytes).unwrap();
    let producer = Arc::new(OfflineProducer {
        root: producer_root.clone(),
    });
    let writer = Writer::start_on_sinex(&archive, producer).unwrap();
    let mut staged = crate::export_items::ExportItems::new().unwrap();
    staged
        .push(
            0..0,
            Vec::new(),
            None,
            &serde_json::to_vec(&crate::assets::AssetRow::Binding {
                record: record.clone(),
            })
            .unwrap(),
        )
        .unwrap();
    writer.import_assets(staged).unwrap();
    let sent: Vec<crate::primary::ProductWrite> =
        serde_json::from_reader(std::fs::File::open(producer_root.join("products.json")).unwrap())
            .unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].product,
        crate::primary::Product::AttachmentBinding {
            record: Box::new(record),
        }
    );
    assert!(sent[0].parents.is_empty());
    let bindings = || {
        Connection::open(archive.index_db())
            .unwrap()
            .query_row("SELECT count(*) FROM attachment_binding", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
    };
    assert_eq!(bindings(), 0);
    writer
        .apply_confirmed(vec![
            crate::primary::Confirmed {
                native: None,
                position: 1,
                output_index: 0,
                event: "binding-event".into(),
                change: crate::primary::ConfirmedChange::Live(Box::new(sent[0].product.clone())),
            },
            crate::primary::Confirmed {
                native: None,
                position: 0,
                output_index: 0,
                event: String::new(),
                change: crate::primary::ConfirmedChange::Control(Box::new(
                    crate::primary::ReplayControl::Completion { position: 1 },
                )),
            },
        ])
        .unwrap();
    assert_eq!(bindings(), 1);
}

#[test]
fn pending_export_fork_is_refused_before_semantic_preparation_or_publication() {
    use polylogue_model::batch::{ExportDelivery, ExportDeliverySource};
    let directory = tempfile::tempdir().unwrap();
    let producer_root = directory.path().join("producer");
    std::fs::create_dir_all(&producer_root).unwrap();
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let producer = Arc::new(OfflineProducer {
        root: producer_root.clone(),
    });
    let writer = Writer::start_on_sinex(&archive, producer).unwrap();
    let key = crate::primary::ExportReadingKey {
        input: InputKey {
            id: "export".into(),
            format: "fixture".into(),
            arrival_rank: 0,
        },
        chain: 1,
        reading_version: "crate::export::VERSION=2".into(),
        length: 2,
    };
    let mut retained = ParseStep::new(Default::default(), Default::default(), 0..0, false);
    retained.reading_version = key.reading_version.clone();
    retained
        .seal_export(
            0,
            ExportDeliverySource::ContainerHeader,
            ExportDelivery::seed(),
        )
        .unwrap();
    retained.prepare();
    let captured = CapturedReading::from_step(&retained, Some(key.clone())).unwrap();
    std::fs::write(
        producer_root.join("pending-export.json"),
        serde_json::to_vec(&captured).unwrap(),
    )
    .unwrap();
    let mut changed = retained;
    changed.counts.records = 1;
    changed.counts.skipped.insert("changed header".into(), 1);
    let cursor = changed
        .seal_export(
            0,
            ExportDeliverySource::ContainerHeader,
            ExportDelivery::seed(),
        )
        .unwrap();
    // Scratch requires complete values, as the actual Engine replacement does.
    changed = CapturedReading::from_step(&changed, Some(key.clone()))
        .unwrap()
        .into_step();
    let mut reading = crate::writer::ChainReading::new().unwrap();
    reading.push(&mut changed).unwrap();
    let mut complete = ParseStep::new(Default::default(), Default::default(), 0..2, true);
    complete.reading_version = key.reading_version.clone();
    complete.reset = false;
    complete
        .seal_export(
            1,
            ExportDeliverySource::Complete {
                length: 2,
                item_count: 0,
                delivery_count: 1,
            },
            cursor.prefix_hash,
        )
        .unwrap();
    reading.push(&mut complete).unwrap();
    let error = writer
        .sender()
        .unwrap()
        .replace_chain(key.input, key.chain, key.length, reading)
        .unwrap_err();
    assert!(
        matches!(&error, StoreError::Invalid(reason) if reason.contains("replayed export prefix")),
        "{error}"
    );
    assert_eq!(
        std::fs::read(producer_root.join("pending-export.json")).unwrap(),
        serde_json::to_vec(&captured).unwrap()
    );
    let index = Connection::open(archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_pending", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert!(!producer_root.join("capture.json").exists());
}

#[test]
fn an_idle_primary_writer_records_a_failed_stream_read() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let producer = Arc::new(OfflineProducer {
        root: directory.path().join("producer"),
    });
    let writer = Writer::start_on_sinex(&archive, producer).unwrap();
    for _ in 0..10 {
        // Each flush is work, so the writer must stay idle past its poll first.
        std::thread::sleep(std::time::Duration::from_secs(3));
        let report = writer.flush().unwrap();
        if let Some(error) = report.errors.first() {
            assert!(error.starts_with("confirmed stream:"), "{error}");
            assert!(error.contains("synthetic offline transport"), "{error}");
            return;
        }
    }
    panic!("the idle writer never recorded its failed stream read");
}

#[test]
fn frozen_unknown_empty_capture_publishes_only_after_scoped_stream_completion() {
    use crate::primary::{Confirmed, ConfirmedChange, ReplayControl, SourceSelector, SourceStatus};
    let directory = tempfile::tempdir().unwrap();
    let producer_root = directory.path().join("producer");
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let producer = Arc::new(OfflineProducer {
        root: producer_root.clone(),
    });
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    let mut capture = source("offline", b"");
    capture.batch.new_blob = None;
    let outcome = writer
        .sender()
        .unwrap()
        .store_group(vec![capture], Arc::new(archive.blobs().unwrap()))
        .unwrap();
    assert!(matches!(
        &outcome[0],
        Err(StoreError::PrimaryPending { .. })
    ));
    let original = std::fs::read(producer_root.join("capture.json")).unwrap();
    let bytes = std::fs::read(producer_root.join("source")).unwrap();
    let source = SourceSelector {
        origin: "codex".into(),
        machine: "machine".into(),
        path_bytes: b"/home/op/offline".to_vec(),
    };
    let control = |control| Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        native: None,
        change: ConfirmedChange::Control(Box::new(control)),
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "scope".into(),
        })])
        .unwrap();
    std::fs::write(producer_root.join("ready"), b"confirmed native anchor").unwrap();
    writer.synchronize_primary().unwrap();
    assert!(
        !producer_root.join("products.json").exists(),
        "raw custody and an anchor do not prove absent source coordinates"
    );
    writer
        .apply_confirmed(vec![
            control(ReplayControl::Head { position: 1 }),
            control(ReplayControl::Completion { position: 0 }),
        ])
        .unwrap();
    let snapshot = writer.source_snapshot(source.clone(), "").unwrap();
    assert_eq!(snapshot.basis.status, SourceStatus::MissingEvidence);
    assert!(snapshot.basis.boundary.is_some());
    writer.synchronize_primary().unwrap();
    assert!(
        !producer_root.join("products.json").exists(),
        "known unapplied commits leave current source coordinates pending"
    );
    writer
        .apply_confirmed(vec![control(ReplayControl::Completion { position: 1 })])
        .unwrap();
    writer.synchronize_primary().unwrap();
    let writes: Vec<ProductWrite> =
        serde_json::from_slice(&std::fs::read(producer_root.join("products.json")).unwrap())
            .unwrap();
    let products: Vec<_> = writes
        .into_iter()
        .enumerate()
        .map(|(index, write)| Confirmed {
            position: 2,
            event: format!("published-{index}"),
            output_index: index as u64,
            native: None,
            change: ConfirmedChange::Live(Box::new(write.product)),
        })
        .chain(std::iter::once(control(ReplayControl::Completion {
            position: 2,
        })))
        .collect();
    writer.apply_confirmed(products).unwrap();
    let reads = crate::ReadPool::open(&archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .source_observation(&source)
            .unwrap()
            .1
            .unwrap()
            .input,
        "offline"
    );
    std::fs::remove_file(producer_root.join("products.json")).unwrap();
    writer
        .apply_confirmed(vec![
            Confirmed {
                position: 3,
                event: "unreadable-input".into(),
                output_index: 0,
                native: None,
                change: ConfirmedChange::Refused {
                    kind: "opaque-owner-input".into(),
                    reason: "unavailable".into(),
                    unreadable_input: true,
                },
            },
            control(ReplayControl::Completion { position: 3 }),
        ])
        .unwrap();
    writer.synchronize_primary().unwrap();
    assert!(!producer_root.join("products.json").exists());
    let (status, positive) = reads.reader().unwrap().source_observation(&source).unwrap();
    assert_eq!(status, SourceStatus::MissingEvidence);
    assert_eq!(positive.unwrap().input, "offline");
    assert_eq!(
        std::fs::read(producer_root.join("capture.json")).unwrap(),
        original
    );
    assert_eq!(std::fs::read(producer_root.join("source")).unwrap(), bytes);
    assert!(producer.captured().basis.unwrap().resolved.is_none());
}

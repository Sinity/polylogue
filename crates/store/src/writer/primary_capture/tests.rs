use super::*;
use crate::primary::PreparedInput;
use crate::primary::Sinex as _;

fn source(id: &str, bytes: &[u8]) -> PreparedBatch {
    PreparedBatch {
        batch: Batch {
            required_disposition: None,
            input: InputKey {
                id: id.into(),
                format: "codex".into(),
                route: polylogue_model::Route::Live,
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
                database: None,
            }),
            continues: false,
            new_blob: Some(crate::NewBlob::whole(
                BlobHash::of(bytes),
                bytes.len() as u64,
            )),
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
fn raw_capture_identity_is_independent_of_reading_and_tracks_new_source_ranges() {
    let mut group = vec![source("session", b"source")];
    let mut step = ParseStep::new(Default::default(), Default::default(), 0..6, true);
    step.reading_version = "codex:1".into();
    group[0].batch.parse.push(step);
    let original = capture_group_identity(&group, false).unwrap();
    group[0].batch.parse[0].reading_version = "codex:2".into();
    assert_eq!(capture_group_identity(&group, false).unwrap(), original);
    group[0].batch.parse[0].reading_version = "codex:1".into();
    group[0].batch.change = Change::Appended { from: 6 };
    group[0].batch.length = 12;
    assert_ne!(capture_group_identity(&group, false).unwrap(), original);
}

struct OfflineProducer {
    root: std::path::PathBuf,
}

#[test]
fn projection_start_replaces_an_equal_scope_and_preserves_original_and_paid_stores() {
    use crate::primary::{Confirmed, ConfirmedChange, ProjectionBasis, ReplayControl};
    let directory = tempfile::tempdir().unwrap();
    let producer_root = directory.path().join("producer");
    std::fs::create_dir(&producer_root).unwrap();
    std::fs::write(producer_root.join("pending"), b"original queued bytes").unwrap();
    let (archive, _lock) = Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    Connection::open(archive.embeddings_db())
        .unwrap()
        .execute_batch(
            "CREATE TABLE paid_fixture(bytes BLOB); INSERT INTO paid_fixture VALUES (X'01020304')",
        )
        .unwrap();
    let writer = Writer::start_on_sinex(
        &archive,
        Arc::new(OfflineProducer {
            root: producer_root.clone(),
        }),
    )
    .unwrap();
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
    writer
        .apply_confirmed(vec![control(ReplayControl::Completion {
            position: 4,
            delivery: None,
        })])
        .unwrap();
    let read_state = || {
        let pool = crate::ReadPool::open(&archive).unwrap();
        pool.reader().unwrap().confirmed_replay_state().unwrap()
    };
    assert_eq!(read_state().applied_position, Some(4));
    writer
        .apply_confirmed(vec![control(ReplayControl::ProjectionStart {
            scope: "scope".into(),
            origin: 10,
            stream_instance: "new-stream".into(),
            reason: None,
        })])
        .unwrap();
    let state = read_state();
    assert_eq!(state.applied_position, None);
    assert_eq!(state.projection.as_ref().unwrap().origin, 10);
    assert_eq!(state.projection.unwrap().stream_instance, "new-stream");
    writer
        .apply_confirmed(vec![
            Confirmed {
                position: 11,
                event: "pending-projection-fact".into(),
                output_index: 0,
                native: None,
                change: ConfirmedChange::Refused {
                    kind: "synthetic".into(),
                    reason: "retained pending fact".into(),
                    unreadable_input: true,
                },
            },
            control(ReplayControl::ProjectionPage {
                position: 12,
                cursor: "unfinished".into(),
            }),
        ])
        .unwrap();
    let staged = read_state();
    let pending_bytes = || {
        Connection::open(archive.index_db())
            .unwrap()
            .query_row(
                "SELECT product FROM primary_pending WHERE event='pending-projection-fact'",
                [],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap()
    };
    let bytes = pending_bytes();
    assert!(
        writer
            .synchronize_primary()
            .unwrap_err()
            .to_string()
            .contains("synthetic offline transport")
    );
    assert_eq!(read_state(), staged);
    assert_eq!(pending_bytes(), bytes);
    let error = writer
        .apply_confirmed(vec![control(ReplayControl::ProjectionStart {
            scope: "scope".into(),
            origin: 10,
            stream_instance: "new-stream".into(),
            reason: Some("owner snapshot invalidated".into()),
        })])
        .unwrap_err();
    assert!(error.to_string().contains("owner snapshot invalidated"));
    assert_eq!(read_state().projection.unwrap().cursor, None);
    assert_eq!(read_state().applied_position, None);
    assert_eq!(
        std::fs::read(producer_root.join("pending")).unwrap(),
        b"original queued bytes"
    );
    let paid: Vec<u8> = Connection::open(archive.embeddings_db())
        .unwrap()
        .query_row("SELECT bytes FROM paid_fixture", [], |row| row.get(0))
        .unwrap();
    assert_eq!(paid, [1, 2, 3, 4]);
    let basis = ProjectionBasis {
        stamp: "owner-signature".into(),
        position: 12,
        scope: "scope".into(),
        catalog: "catalog".into(),
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::ProjectionComplete {
            basis: basis.clone(),
        })])
        .unwrap();
    let state = read_state();
    assert_eq!(state.applied_position, Some(12));
    assert_eq!(state.basis, Some(basis));
    assert_eq!(state.projection, None);
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
            let [body] = <[_; 1]>::try_from(body).map_err(|_| {
                StoreError::Invalid("synthetic offline adapter keeps one body".into())
            })?;
            self.prepare_input(&input, body)?;
            count += 1;
            Ok(())
        })?;
        if count != 1 {
            return Err(StoreError::Invalid(
                "missing synthetic capture member".into(),
            ));
        }
        if plan.required_disposition
            == Some(crate::primary::CaptureDisposition::UnresolvedBasisConflict)
        {
            return Ok(crate::primary::CaptureReceipt::UnresolvedBasisConflict {
                custody: crate::primary::RawCustodyReceipt {
                    identity: "offline-capture".into(),
                    count: plan.count,
                },
            });
        }
        Ok(crate::primary::CaptureReceipt::Accepted {
            write: crate::primary::PreparedWrite {
                identity: "offline-capture".into(),
                pending: true,
            },
        })
    }
    fn capture_body(&self, _: &str) -> Result<Vec<crate::primary::PreparedBody>> {
        let path = self.root.join("source");
        Ok(vec![crate::primary::PreparedBody {
            file: std::fs::File::open(&path)?,
            descriptor: hashed_blobs::EncodedBlob::decode(&std::fs::read(
                self.root.join("descriptor"),
            )?)?,
            path: Some(path),
        }])
    }
    fn pending_source(
        &self,
        source: &crate::primary::SourceSelector,
    ) -> Result<crate::primary::PendingSource> {
        use crate::primary::{PendingSource, SourceSelector, SourceStatus};
        if !self.root.join("capture.json").exists() {
            return Ok(PendingSource {
                status: SourceStatus::Ready,
                ..Default::default()
            });
        }
        let full = self.captured();
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
    ) -> Result<crate::primary::ParsedPublication> {
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
                captured: Capture::Input(Box::new(self.captured())),
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
    /// Spooled products are kept, so a test can read what was sent. Like the
    /// real producer's spool, a write becomes visible whole: a publication
    /// pass may rewrite the file while the test reads it.
    fn prepare_parsed(
        &self,
        source: &crate::primary::ParsedSource,
        writes: &[crate::primary::ProductWrite],
    ) -> Result<crate::primary::ParsedPublication> {
        let prepared = self.prepare_products(writes)?;
        Ok(crate::primary::ParsedPublication::Admitted(
            crate::primary::PublicationReceipt {
                producer: "fixture-producer".into(),
                frame: prepared.identity,
                number: 1,
                commit_seq: None,
                root_material: source.root_material.clone(),
            },
        ))
    }
    fn prepare_products(&self, writes: &[ProductWrite]) -> Result<crate::primary::PreparedWrite> {
        std::fs::create_dir_all(&self.root)?;
        let preparing = self.root.join("products.json.preparing");
        let file = std::fs::File::create(&preparing)?;
        serde_json::to_writer(&file, writes)?;
        file.sync_all()?;
        std::fs::rename(&preparing, self.root.join("products.json"))?;
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
    fn acknowledge(
        &self,
        _: &crate::primary::ReplayState,
        _: &dyn Fn(&str) -> Result<bool>,
        _: &dyn Fn(
            &crate::primary::ParsedSource,
            &crate::primary::PublicationReceipt,
        ) -> Result<bool>,
    ) -> Result<()> {
        if self.root.join("ready").exists() {
            Ok(())
        } else {
            Self::unavailable()
        }
    }
}

#[test]
fn offline_writer_retains_original_source_before_releasing_preparation() {
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
        input.batch.input.route = polylogue_model::Route::Export;
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
            .source_snapshot(crate::primary::SourceSelector {
                origin: "codex".into(),
                machine: "machine".into(),
                path_bytes: b"/home/op/offline".to_vec(),
            })
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
                    crate::primary::ReplayControl::Completion {
                        position: 0,
                        delivery: None,
                    },
                )),
            }])
            .unwrap();
        let after_checkpoint = writer
            .source_snapshot(crate::primary::SourceSelector {
                origin: "codex".into(),
                machine: "machine".into(),
                path_bytes: b"/home/op/offline".to_vec(),
            })
            .unwrap();
        assert_eq!(
            after_checkpoint.basis.status,
            crate::primary::SourceStatus::MissingEvidence
        );
        assert!(after_checkpoint.basis.boundary.is_none());
        let Capture::Input(hydrated) = writer.capture("offline-capture".into()).unwrap() else {
            panic!("the pending source must hydrate its original input facts");
        };
        assert_eq!(hydrated.hash, BlobHash::of(bytes));
        assert_eq!(hydrated.length, bytes.len() as u64);
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
    // Reopening recovers the original bytes without the provider file or
    // the parser preparation. The ordinary parser reconstructs their model.
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
    assert_eq!(captured.route, polylogue_model::Route::Export);
    assert_eq!(captured.hash, BlobHash::of(bytes));
    assert_eq!(captured.length, bytes.len() as u64);
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
                    crate::primary::ReplayControl::Completion {
                        position: 1,
                        delivery: None,
                    },
                )),
            },
        ])
        .unwrap();
    assert_eq!(bindings(), 1);
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
            control(ReplayControl::Completion {
                position: 0,
                delivery: None,
            }),
        ])
        .unwrap();
    let snapshot = writer.source_snapshot(source.clone()).unwrap();
    assert_eq!(snapshot.basis.status, SourceStatus::MissingEvidence);
    assert!(snapshot.basis.boundary.is_some());
    writer.synchronize_primary().unwrap();
    assert!(
        !producer_root.join("products.json").exists(),
        "known unapplied commits leave current source coordinates pending"
    );
    writer
        .apply_confirmed(vec![control(ReplayControl::Completion {
            position: 1,
            delivery: None,
        })])
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
            delivery: None,
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
            control(ReplayControl::Completion {
                position: 3,
                delivery: None,
            }),
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

#[test]
fn original_source_basis_survives_normal_progress_and_retains_real_conflicts() {
    use crate::primary::{Confirmed, ConfirmedChange, ReplayControl, SourceSelector};
    for scenario in [
        "unknown-to-ready",
        "later-head",
        "changed-scope",
        "copied-alias",
        "forged-alias",
    ] {
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
        let selector = SourceSelector {
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
        let unknown = writer.source_snapshot(selector.clone()).unwrap().basis;
        writer
            .apply_confirmed(vec![control(ReplayControl::Reset {
                scope: "original".into(),
            })])
            .unwrap();
        writer
            .apply_confirmed(vec![control(ReplayControl::Completion {
                position: 0,
                delivery: None,
            })])
            .unwrap();
        let mut sampled = if scenario == "unknown-to-ready" {
            unknown
        } else {
            writer.source_snapshot(selector.clone()).unwrap().basis
        };
        if scenario == "later-head" {
            let observation = serde_json::from_value(serde_json::json!({"id":"own-confirmation","machine":"machine","host":"host","path":"/home/op/offline","path_bytes":selector.path_bytes.clone(),"class":"seen","polylogue_version":"test","seen_at":1})).unwrap();
            writer
                .apply_confirmed(vec![
                    Confirmed {
                        position: 1,
                        event: "own-input".into(),
                        output_index: 0,
                        native: None,
                        change: ConfirmedChange::Live(Box::new(Product::Input {
                            rows: Box::new(InputRows {
                                acquisition: None,
                                input: "offline".into(),
                                format: "codex".into(),
                                revision: None,
                                extents: Vec::new(),
                                observation: Some(observation),
                            }),
                            material: None,
                        })),
                    },
                    control(ReplayControl::Completion {
                        position: 1,
                        delivery: None,
                    }),
                ])
                .unwrap();
        } else if scenario == "changed-scope" {
            writer
                .apply_confirmed(vec![control(ReplayControl::Reset {
                    scope: "changed".into(),
                })])
                .unwrap();
            writer
                .apply_confirmed(vec![control(ReplayControl::Completion {
                    position: 0,
                    delivery: None,
                })])
                .unwrap();
        }
        if matches!(scenario, "copied-alias" | "forged-alias") {
            let observation = serde_json::from_value(serde_json::json!({"id":"alias-head","machine":"machine","host":"host","path":"/home/op/original","path_bytes":b"/home/op/original".to_vec(),"class":"seen","polylogue_version":"test","seen_at":1})).unwrap();
            writer
                .apply_confirmed(vec![
                    Confirmed {
                        position: 1,
                        event: "alias-input".into(),
                        output_index: 0,
                        native: None,
                        change: ConfirmedChange::Live(Box::new(Product::Input {
                            rows: Box::new(InputRows {
                                acquisition: None,
                                input: "alias-target".into(),
                                format: "codex".into(),
                                revision: None,
                                extents: Vec::new(),
                                observation: Some(observation),
                            }),
                            material: None,
                        })),
                    },
                    control(ReplayControl::Completion {
                        position: 1,
                        delivery: None,
                    }),
                ])
                .unwrap();
            sampled = writer.source_snapshot(selector.clone()).unwrap().basis;
            sampled.resolved = Some(crate::primary::SourceSubject {
                input: "alias-target".into(),
                observation: if scenario == "copied-alias" {
                    "alias-head"
                } else {
                    "missing-head"
                }
                .into(),
            });
        }
        let bytes = b"retained source bytes";
        let mut capture = source("offline", bytes);
        capture.batch.basis = Some(sampled.clone());
        if sampled.resolved.is_some() {
            capture.batch.input.id = "alias-target".into();
        }
        capture
            .files
            .push(archive.prepare_bytes(bytes.as_slice()).unwrap());
        let mut outcomes = writer
            .sender()
            .unwrap()
            .store_group(vec![capture], Arc::new(archive.blobs().unwrap()))
            .unwrap();
        let outcome = outcomes.remove(0);
        if matches!(scenario, "changed-scope" | "forged-alias") {
            assert!(
                matches!(outcome, Err(StoreError::UnresolvedBasisConflict { .. })),
                "{scenario}: {outcome:?}"
            );
        } else {
            assert!(
                matches!(outcome, Err(StoreError::PrimaryPending { .. })),
                "{scenario}: {outcome:?}"
            );
        }
        let original = producer.captured();
        assert_eq!(original.basis.as_ref(), Some(&sampled));
        assert_eq!(original.hash, BlobHash::of(bytes));
        assert_eq!(original.basis.unwrap().resolved, sampled.resolved);
        let [body] = <[_; 1]>::try_from(producer.capture_body("offline-capture").unwrap())
            .ok()
            .unwrap();
        let prepared = hashed_blobs::Prepared::from_encoded(
            body.file,
            body.descriptor,
            None,
            Default::default(),
        )
        .unwrap();
        let mut retained = vec![];
        prepared
            .reader(None, Default::default())
            .unwrap()
            .read_to_end(&mut retained)
            .unwrap();
        assert_eq!(retained, bytes);
    }
}

#[test]
fn stored_readings_use_their_exact_confirmed_chain_instead_of_a_database_path_head() {
    use crate::primary::{Confirmed, ConfirmedChange, ReplayControl};
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
    let writer = Writer::start_on_sinex(&archive, producer).unwrap();
    let control = |control| Confirmed {
        position: 0,
        output_index: 0,
        event: String::new(),
        native: None,
        change: ConfirmedChange::Control(Box::new(control)),
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "scope".into(),
        })])
        .unwrap();
    let hash = BlobHash::of(b"rows");
    let extent = hashed_blobs::Extent {
        segment: "segments/confirmed.hb".into(),
        offset: 0,
        len: 100,
        decoded_len: 4,
        hash: hashed_blobs::Hash(hash.0),
        kind: hashed_blobs::Kind::Plain,
        index_hash: hashed_blobs::Hash([1; 32]),
        encoded_hash: hashed_blobs::Hash([2; 32]),
    };
    for (position, id) in [(1, "first-session"), (2, "second-session")] {
        let observation = serde_json::from_value(serde_json::json!({
            "id":format!("observation-{id}"), "machine":"machine", "host":"host",
            "path":"/home/op/state.sqlite", "path_bytes":b"/home/op/state.sqlite".to_vec(),
            "class":"seen", "revision":1, "length":4, "polylogue_version":"test", "seen_at":1
        }))
        .unwrap();
        writer
            .apply_confirmed(vec![
                Confirmed {
                    position,
                    output_index: 0,
                    event: format!("input-{id}"),
                    native: None,
                    change: ConfirmedChange::Live(Box::new(Product::Input {
                        rows: Box::new(InputRows {
                            acquisition: None,
                            input: id.into(),
                            format: "codex-database".into(),
                            revision: Some(Revision {
                                number: 1,
                                produced: "first".into(),
                                chain: 1,
                                length: 4,
                            }),
                            extents: vec![crate::input_rows::Extent {
                                chain: 1,
                                offset: 0,
                                length: 4,
                                blob: hash,
                                blob_offset: 0,
                            }],
                            observation: Some(observation),
                        }),
                        material: Some(Material {
                            id: format!("material-{id}"),
                            hash,
                            length: 4,
                            extents: vec![extent.encode()],
                        }),
                    })),
                },
                control(ReplayControl::Completion {
                    position,
                    delivery: None,
                }),
            ])
            .unwrap();
    }
    let mut reading = source("first-session", b"rows");
    reading.batch.input.format = "codex-database".into();
    reading.batch.change = Change::Reparse { chain: 1 };
    reading.batch.new_blob = None;
    reading.batch.observation = None;
    let mut step = ParseStep::new(Default::default(), Default::default(), 0..4, true);
    step.reading_version = "database-reading".into();
    step.prepare();
    reading.batch.parse.push(step);
    let store = Arc::new(archive.blobs().unwrap());
    for _ in 0..2 {
        let results = writer
            .sender()
            .unwrap()
            .store_group(
                vec![PreparedBatch {
                    batch: reading.batch.clone(),
                    files: vec![],
                }],
                Arc::clone(&store),
            )
            .unwrap();
        assert!(
            matches!(&results[0], Ok(stored) if stored.input == "first-session" && stored.chain == 1 && stored.already_stored),
            "stored reading result: {results:?}"
        );
        assert!(
            !producer_root.join("capture.json").exists(),
            "stored parsing must not prepare another acquisition"
        );
        assert_eq!(
            crate::ReadPool::open(&archive)
                .unwrap()
                .reader()
                .unwrap()
                .parser_checkpoint("first-session", 1, "database-reading")
                .unwrap()
                .unwrap()
                .parsed_upto,
            4
        );
    }
    let writes: Vec<ProductWrite> =
        serde_json::from_slice(&std::fs::read(producer_root.join("products.json")).unwrap())
            .unwrap();
    assert_eq!(
        writes.len(),
        1,
        "one retained parse-progress product: {writes:?}"
    );
    assert!(
        writes
            .iter()
            .filter_map(|write| write.product.reading())
            .all(|reading| reading.input.id == "first-session" && reading.chain == 1)
    );
    assert!(writes.iter().all(|write| {
        write
            .location
            .as_ref()
            .is_some_and(|location| location.material == "material-first-session")
    }));
    for invalid in ["format", "chain", "range"] {
        let mut batch = reading.batch.clone();
        match invalid {
            "format" => batch.input.format = "other-origin".into(),
            "chain" => batch.change = Change::Reparse { chain: 2 },
            _ => batch.length = 5,
        }
        let error = writer
            .sender()
            .unwrap()
            .store_group(
                vec![PreparedBatch {
                    batch,
                    files: vec![],
                }],
                Arc::clone(&store),
            )
            .unwrap_err();
        match (invalid, error) {
            ("format", StoreError::Invalid(reason)) => {
                assert_eq!(reason, "stored reading input identity differs")
            }
            ("chain", StoreError::NotFound(reason)) => {
                assert_eq!(reason, "stored input first-session chain 2 range 0..4")
            }
            ("range", StoreError::NotFound(reason)) => {
                assert_eq!(reason, "stored input first-session chain 1 range 0..5")
            }
            (invalid, error) => panic!("unexpected {invalid} refusal: {error}"),
        }
    }
    assert!(!producer_root.join("capture.json").exists());
    let mut fresh = source("fresh-provider", b"new source");
    fresh
        .files
        .push(archive.prepare_bytes(b"new source".as_slice()).unwrap());
    let mixed = writer
        .sender()
        .unwrap()
        .store_group(
            vec![
                fresh,
                PreparedBatch {
                    batch: reading.batch.clone(),
                    files: vec![],
                },
            ],
            Arc::clone(&store),
        )
        .unwrap();
    assert!(
        matches!(&mixed[0], Err(StoreError::PrimaryPending { .. })),
        "mixed source acquisition: {mixed:?}"
    );
    assert!(
        matches!(&mixed[1], Ok(stored) if stored.input == "first-session" && stored.chain == 1),
        "mixed stored reading: {mixed:?}"
    );
}

#[test]
fn a_failed_scope_reset_reports_its_original_error_before_reply_without_a_commit_cue() {
    use crate::primary::{Confirmed, ConfirmedChange, ReplayControl};
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
    let commits = writer.sender().unwrap().completion_commits();
    let reset = |scope: &str| {
        vec![Confirmed {
            native: None,
            position: 0,
            output_index: 0,
            event: String::new(),
            change: ConfirmedChange::Control(Box::new(ReplayControl::Reset {
                scope: scope.into(),
            })),
        }]
    };
    writer.apply_confirmed(reset("original")).unwrap();
    while commits.try_recv().is_ok() {}
    let index = rusqlite::Connection::open(archive.index_db()).unwrap();
    index.execute_batch("CREATE TRIGGER fail_scope BEFORE INSERT ON meta WHEN new.key='confirmed_scope' AND new.value='changed' BEGIN SELECT raise(ABORT,'original scope reset failure'); END").unwrap();
    let error = writer.apply_confirmed(reset("changed")).unwrap_err();
    assert!(
        error.to_string().contains("original scope reset failure"),
        "{error}"
    );
    assert!(matches!(
        archive.scope_token(),
        Err(StoreError::ScopeChanged)
    ));
    let report = writer.flush_rows().unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.starts_with("confirmed stream:")
                && error.contains("original scope reset failure")),
        "{report:?}"
    );
    assert!(matches!(
        commits.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    index.execute_batch("DROP TRIGGER fail_scope").unwrap();
    writer.apply_confirmed(reset("changed")).unwrap();
    archive.scope_token().unwrap().check().unwrap();
    assert_eq!(
        crate::ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .confirmed_scope()
            .unwrap()
            .as_deref(),
        Some("changed")
    );
}

use super::*;
use crate::{
    Archive, ArchiveLock, Batch, Change, ImportMember, InputKey, Observation, ParseStep, ReadPool,
    Writer,
    assets::AssetRow,
    export_items::ExportItems,
    primary::{AcquisitionReceipt, AcquisitionRoute, RawRange, SourceSelector},
    staging::PreparedBatch,
};
use polylogue_model::{
    Message, MessageRow, Output, RecordCounts, RecordOutcome, Session, SessionKind,
    attachment_binding::{AttachmentCandidate, AttachmentSourceClaim},
};
use serde_json::json;

struct Fixture {
    writer: Writer,
    reads: ReadPool,
    archive: Archive,
    _lock: ArchiveLock,
    _dir: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let reads = ReadPool::open(&archive).unwrap();
        Self {
            writer,
            reads,
            archive,
            _lock: lock,
            _dir: dir,
        }
    }
    fn capture(&self, mut batch: Batch, bytes: &[u8]) {
        let (blob, file) = self.archive.prepare_blob(bytes).unwrap();
        batch.new_blob = Some(blob);
        let mut stored = self
            .writer
            .sender()
            .unwrap()
            .store_group(
                vec![PreparedBatch {
                    batch,
                    files: file.into_iter().collect(),
                }],
                Arc::new(self.archive.blobs().unwrap()),
            )
            .unwrap();
        stored.pop().expect("one actual capture").unwrap();
        assert!(self.writer.flush().unwrap().errors.is_empty());
    }
    fn message(
        &self,
        input: &str,
        namespace: &str,
        rank: u8,
        record: &MessageRecord,
    ) -> BindingContext {
        let bytes = serde_json::to_vec(record).unwrap();
        let mut batch = raw(
            input,
            Some(namespace),
            "records.jsonl",
            Change::First,
            bytes.len(),
        );
        batch.input.format = "polylogue-bundle".into();
        batch.input.arrival_rank = rank;
        let mut counts = RecordCounts {
            records: 1,
            ..Default::default()
        };
        counts.add("message", &RecordOutcome::Used);
        let mut step = ParseStep::new(
            Output {
                sessions: vec![Session::new(
                    record.message.session.clone(),
                    SessionKind::Main,
                )],
                messages: vec![MessageRow {
                    offset: 0,
                    end: bytes.len() as u64,
                    message: record.message.clone(),
                }],
                ..Default::default()
            },
            counts,
            0..bytes.len() as u64,
            true,
        );
        step.reading_version = "binding-fixture=1".into();
        batch.parse.push(step);
        self.capture(batch, &bytes);
        BindingContext {
            message: record.message.id.clone(),
            source_version: record.source_version,
            input: input.into(),
            chain: 1,
            import_id: namespace.into(),
        }
    }
    /// A received member as an explicit import stores it: with the
    /// acquisition receipt that names its source and its whole byte range.
    fn member(&self, input: &str, namespace: &str, bytes: &[u8]) {
        let mut batch = raw(
            input,
            Some(namespace),
            "blobs/received",
            Change::First,
            bytes.len(),
        );
        batch.acquisition = Some(AcquisitionReceipt {
            version: AcquisitionReceipt::VERSION,
            route: AcquisitionRoute::ExplicitImport,
            source: SourceSelector {
                origin: batch.input.format.clone(),
                machine: "synthetic".into(),
                path_bytes: format!("/synthetic/{input}").into_bytes(),
            },
            range: RawRange {
                start: 0,
                end: bytes.len() as u64,
            },
            hash: BlobHash::of(bytes),
            observed_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            generation: None,
            raw_total: None,
            raw_predecessor: None,
        });
        self.capture(batch, bytes);
    }
    fn refresh(&self) -> Vec<ObservedMember> {
        let (members, through) = self
            .reads
            .reader()
            .unwrap()
            .unindexed_import_members()
            .unwrap();
        let mut staged = ExportItems::new().unwrap();
        for source in &members {
            stage(
                &mut staged,
                &AssetRow::Member {
                    source: source.clone(),
                    source_id: None,
                    sidecar: None,
                },
            );
        }
        if let Some(through) = through {
            stage(&mut staged, &AssetRow::Scan { through });
        }
        self.writer.import_assets(staged).unwrap();
        members
    }
    fn install(&self, record: AttachmentBindingRecord) -> Result<()> {
        let mut staged = ExportItems::new()?;
        stage(&mut staged, &AssetRow::Binding { record });
        self.writer.import_assets(staged)
    }
}
fn stage(staged: &mut ExportItems, row: &AssetRow) {
    staged
        .push(0..0, Vec::new(), None, &serde_json::to_vec(row).unwrap())
        .unwrap();
}
fn raw(input: &str, namespace: Option<&str>, member: &str, change: Change, length: usize) -> Batch {
    Batch {
        required_disposition: None,
        input: InputKey {
            id: input.into(),
            format: "export-file".into(),
            arrival_rank: 0,
        },
        change,
        observation: Some(Observation {
            machine: "synthetic".into(),
            host: "synthetic".into(),
            path: format!("/synthetic/{input}"),
            path_bytes: format!("/synthetic/{input}").into_bytes(),
            facts: None,
            seen_at_ms: 1,
            import: namespace.map(|namespace| ImportMember {
                version: 1,
                import_id: namespace.into(),
                member: member.into(),
                authorship: None,
                format: None,
            }),
            container: None,
        }),
        continues: false,
        new_blob: None,
        blobs: Vec::new(),
        length: length as u64,
        parse: Vec::new(),
        acquisition: None,
        basis: None,
        release: Default::default(),
    }
}
fn record(bytes: &[u8]) -> MessageRecord {
    let message: Message = serde_json::from_value(json!({"id":"codex:bound-message","session":"codex:bound-session","role":"assistant","blocks":[{"type":"file","source_id":"opaque:λ/#id","direction":"unknown","name":"unchanged source name"}]})).unwrap();
    let version = message.content_hash().unwrap();
    serde_json::from_value(json!({"ref":"message:codex:bound-message","commit":1,"order":0,"source_version":version,"block_count":1,"block_ordinals":[0],"message":message,
        "bindings":[{"source_version":version,
            "namespace":{"input":"original-message-source","chain":4,"import_id":"original-scope"},
            "exported_from":{"input":"export-message-source","chain":5,"import_id":"export-scope"},
            "prior_exports":[{"input":"supplied-prior-source","chain":2,"import_id":"supplied-prior-scope"}],"block":0,"attachment":0,"source_id":"opaque:λ/#id","candidates":[AttachmentCandidate { hash:BlobHash::of(bytes),size:bytes.len() as u64,sources:vec![AttachmentSourceClaim { namespace:None,input:"historical-source".into(),chain:4,source_version:"historical-version".into(),members:vec!["old/member".into()],names:vec!["original name".into()],media_types:vec!["application/octet-stream".into()],declared_sizes:vec![999] }] }]}]})).unwrap()
}
fn source<'a>(sources: &'a [ObservedMember], input: &str) -> &'a ObservedMember {
    sources
        .iter()
        .find(|source| source.input == input)
        .expect("actual captured member")
}

#[test]
fn claims_without_received_bytes_never_borrow_a_later_same_hash_member() {
    let fixture = Fixture::new();
    let bytes = b"actual member bytes";
    let record = record(bytes);
    let context = fixture.message("message-input", "scope-a", 0, &record);
    fixture.refresh();
    let fact = AttachmentBindingRecord::new(context, &record, 0, Vec::new()).unwrap();
    let selection = fact.selection();
    fixture.install(fact.clone()).unwrap();
    assert_eq!(
        fixture
            .reads
            .reader()
            .unwrap()
            .attachment_binding(&selection)
            .unwrap()
            .unwrap()
            .state,
        AttachmentState::Unavailable
    );
    fixture.member("late-member", "scope-a", bytes);
    fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    let resolution = reader.attachment_binding(&selection).unwrap().unwrap();
    assert_eq!(resolution.state, AttachmentState::Unavailable);
    assert!(resolution.members.is_empty());
    assert_eq!(resolution.claim, record.bindings[0]);
    assert_eq!(
        resolution.claim.candidates[0].sources[0].input,
        "historical-source"
    );
    assert!(matches!(
        reader.pin_attachment(&selection).unwrap(),
        AttachmentPin::Unavailable
    ));
}

#[test]
fn validated_multi_extent_member_pins_survive_a_later_input_rewrite() {
    let fixture = Fixture::new();
    let bytes = "prefix λ🙂 suffix".as_bytes();
    let record = record(bytes);
    let context = fixture.message("message-input", "scope-a", 0, &record);
    fixture.capture(
        raw("received-member", None, "blobs/received", Change::First, 8),
        &bytes[..8],
    );
    let mut tail = raw(
        "received-member",
        Some("scope-a"),
        "blobs/received",
        Change::First,
        bytes.len(),
    );
    tail.continues = true;
    fixture.capture(tail, &bytes[8..]);
    // Continuation chunks do not record a new observation. Establish the
    // receiving namespace only after the complete stored endpoint exists.
    fixture
        .writer
        .send(raw(
            "received-member",
            Some("scope-a"),
            "blobs/received",
            Change::Observed {
                revision: 1,
                chain: 1,
            },
            bytes.len(),
        ))
        .unwrap();
    assert!(fixture.writer.flush().unwrap().errors.is_empty());
    let sources = fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    let witness = reader
        .validate_attachment_member(
            source(&sources, "received-member"),
            BlobHash::of(bytes),
            bytes.len() as u64,
        )
        .unwrap();
    assert!(
        reader
            .asset_source("received-member")
            .unwrap()
            .unwrap()
            .blob
            .is_none()
    );
    let fact =
        AttachmentBindingRecord::new(context, &record, 0, vec![witness.for_candidate(0)]).unwrap();
    let restored: AttachmentBindingRecord =
        serde_json::from_slice(&serde_json::to_vec(&fact).unwrap()).unwrap();
    assert_eq!(restored, fact);
    let selection = fact.selection();
    drop(reader);
    fixture.install(restored).unwrap();
    fixture.capture(
        raw(
            "received-member",
            Some("scope-a"),
            "blobs/received",
            Change::Rewritten,
            3,
        ),
        b"new",
    );
    fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    let AttachmentPin::Available(pin) = reader.pin_attachment(&selection).unwrap() else {
        panic!("retained exact member lost its binding")
    };
    assert_eq!(pin.identity().chain, 1);
    assert_eq!(pin.read_range(2..13).unwrap(), bytes[2..13]);
    assert_eq!(pin.content_hash().unwrap(), BlobHash::of(bytes));
    assert!(pin.read_range(0..bytes.len() as u64 + 1).is_err());
    assert_eq!(
        reader
            .attachment_binding(&selection)
            .unwrap()
            .unwrap()
            .claim,
        record.bindings[0]
    );
}

#[test]
fn same_message_hash_in_other_source_and_namespace_cannot_authorize_a_binding() {
    let fixture = Fixture::new();
    let bytes = b"same actual bytes";
    let record = record(bytes);
    let first = fixture.message("message-a", "scope-a", 0, &record);
    let second = fixture.message("message-b", "scope-b", 1, &record);
    fixture.member("member-b", "scope-b", bytes);
    let sources = fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    let witness = reader
        .validate_attachment_member(
            source(&sources, "member-b"),
            BlobHash::of(bytes),
            bytes.len() as u64,
        )
        .unwrap();
    assert!(
        AttachmentBindingRecord::new(
            first.clone(),
            &record,
            0,
            vec![witness.clone().for_candidate(0)]
        )
        .is_err()
    );
    let unknown = AttachmentBindingRecord::new(first, &record, 0, Vec::new()).unwrap();
    let known =
        AttachmentBindingRecord::new(second, &record, 0, vec![witness.for_candidate(0)]).unwrap();
    let (first, second) = (unknown.selection(), known.selection());
    drop(reader);
    fixture.install(unknown).unwrap();
    fixture.install(known).unwrap();
    let reader = fixture.reads.reader().unwrap();
    assert_eq!(
        reader.attachment_binding(&first).unwrap().unwrap().state,
        AttachmentState::Unavailable
    );
    assert_eq!(
        reader.attachment_binding(&second).unwrap().unwrap().state,
        AttachmentState::Available
    );
    let shown = reader
        .attachment_binding_contexts(&record.message.id, 0, 0)
        .unwrap();
    assert_eq!(shown.namespace_count, 1);
    assert_eq!(shown.contexts[0].selection.context.input, "message-a");
    assert_eq!(shown.state, AttachmentState::Unavailable);
}

#[test]
fn wrong_observation_namespace_hash_and_size_cannot_mint_an_ingress_witness() {
    let fixture = Fixture::new();
    let bytes = b"verified body";
    fixture.member("member-a", "scope-a", bytes);
    let sources = fixture.refresh();
    let observed = source(&sources, "member-a");
    let reader = fixture.reads.reader().unwrap();
    assert!(
        reader
            .validate_attachment_member(
                observed,
                BlobHash::of(b"different body"),
                bytes.len() as u64
            )
            .is_err()
    );
    assert!(
        reader
            .validate_attachment_member(observed, BlobHash::of(bytes), bytes.len() as u64 + 1)
            .is_err()
    );
    let mut forged = observed.clone();
    forged.membership.import_id = "another-scope".into();
    assert!(
        reader
            .validate_attachment_member(&forged, BlobHash::of(bytes), bytes.len() as u64)
            .is_err()
    );
    forged = observed.clone();
    forged.observation = "not-a-received-observation".into();
    assert!(
        reader
            .validate_attachment_member(&forged, BlobHash::of(bytes), bytes.len() as u64)
            .is_err()
    );
}

#[test]
fn a_frozen_witness_does_not_recertify_an_unrelated_member_and_restores_only_its_exact_association()
{
    let fixture = Fixture::new();
    let bytes = b"same bytes";
    let record = record(bytes);
    let context = fixture.message("message-input", "scope-a", 0, &record);
    fixture.member("member-a", "scope-a", bytes);
    let sources = fixture.refresh();
    let observed = source(&sources, "member-a").clone();
    let reader = fixture.reads.reader().unwrap();
    let witness = reader
        .validate_attachment_member(&observed, BlobHash::of(bytes), bytes.len() as u64)
        .unwrap();
    let fact =
        AttachmentBindingRecord::new(context, &record, 0, vec![witness.for_candidate(0)]).unwrap();
    let selection = fact.selection();
    drop(reader);
    fixture.install(fact.clone()).unwrap();
    let index =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    index
        .execute(
            "DELETE FROM import_member WHERE observation=?1",
            [&observed.observation],
        )
        .unwrap();
    let mut seen_again = raw(
        "member-a",
        Some("scope-a"),
        "blobs/received",
        Change::Unchanged,
        bytes.len(),
    );
    seen_again.observation.as_mut().unwrap().seen_at_ms = 2;
    fixture.writer.send(seen_again).unwrap();
    assert!(fixture.writer.flush().unwrap().errors.is_empty());
    fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    assert_eq!(
        reader
            .attachment_binding(&selection)
            .unwrap()
            .unwrap()
            .state,
        AttachmentState::Unavailable
    );
    drop(reader);
    let mut staged = ExportItems::new().unwrap();
    stage(
        &mut staged,
        &AssetRow::Member {
            source: observed,
            source_id: None,
            sidecar: None,
        },
    );
    fixture.writer.import_assets(staged).unwrap();
    fixture.install(fact).unwrap();
    let reader = fixture.reads.reader().unwrap();
    assert_eq!(
        reader
            .attachment_binding(&selection)
            .unwrap()
            .unwrap()
            .state,
        AttachmentState::Available
    );
}

#[test]
fn zero_length_received_bytes_are_known_and_conflicting_candidates_stay_ambiguous() {
    let fixture = Fixture::new();
    let mut record = record(b"");
    let context = fixture.message("message-input", "scope-a", 0, &record);
    fixture.member("empty-member", "scope-a", b"");
    let sources = fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    let witness = reader
        .validate_attachment_member(source(&sources, "empty-member"), BlobHash::of(b""), 0)
        .unwrap();
    let fact = AttachmentBindingRecord::new(
        context.clone(),
        &record,
        0,
        vec![witness.clone().for_candidate(0)],
    )
    .unwrap();
    let selection = fact.selection();
    drop(reader);
    fixture.install(fact).unwrap();
    let reader = fixture.reads.reader().unwrap();
    let AttachmentPin::Available(pin) = reader.pin_attachment(&selection).unwrap() else {
        panic!("actual empty member became unknown")
    };
    assert!(pin.read_range(0..0).unwrap().is_empty());
    assert_eq!(pin.content_hash().unwrap(), BlobHash::of(b""));
    drop(pin);
    drop(reader);
    let mut alternative = record.bindings[0].candidates[0].clone();
    alternative.sources[0].input = "different-history".into();
    record.bindings[0].candidates.push(alternative);
    let other = fixture.message("other-message-input", "scope-a", 1, &record);
    fixture.refresh();
    let fact =
        AttachmentBindingRecord::new(other, &record, 0, vec![witness.for_candidate(0)]).unwrap();
    let selection = fact.selection();
    fixture.install(fact).unwrap();
    let reader = fixture.reads.reader().unwrap();
    let resolution = reader.attachment_binding(&selection).unwrap().unwrap();
    assert_eq!(resolution.state, AttachmentState::Ambiguous);
    assert_eq!(resolution.claim.candidates.len(), 2);
    assert_ne!(
        resolution.claim.candidates[0].sources,
        resolution.claim.candidates[1].sources
    );
    assert!(matches!(
        reader.pin_attachment(&selection).unwrap(),
        AttachmentPin::Ambiguous
    ));
}

#[test]
fn different_opaque_source_ids_can_bind_one_actual_member_without_overwriting_member_identity() {
    let fixture = Fixture::new();
    let bytes = b"one received object";
    let mut record = record(bytes);
    let mut other = record.message.blocks[0].clone();
    let polylogue_model::Block::File(attachment) = &mut other else {
        unreachable!()
    };
    attachment.source_id = Some("another:opaque/#id".into());
    record.message.blocks.push(other);
    record.source_version = record.message.content_hash().unwrap();
    record.block_count = 2;
    record.block_ordinals.push(1);
    record.bindings[0].source_version = record.source_version;
    let mut other = record.bindings[0].clone();
    other.block = 1;
    other.source_id = "another:opaque/#id".into();
    record.bindings.push(other);
    let context = fixture.message("message-input", "scope-a", 0, &record);
    fixture.member("member-a", "scope-a", bytes);
    let sources = fixture.refresh();
    let reader = fixture.reads.reader().unwrap();
    let witness = reader
        .validate_attachment_member(
            source(&sources, "member-a"),
            BlobHash::of(bytes),
            bytes.len() as u64,
        )
        .unwrap();
    drop(reader);
    for slot in 0..2 {
        fixture
            .install(
                AttachmentBindingRecord::new(
                    context.clone(),
                    &record,
                    slot,
                    vec![witness.clone().for_candidate(0)],
                )
                .unwrap(),
            )
            .unwrap();
    }
    let reader = fixture.reads.reader().unwrap();
    for block in 0..2 {
        let contexts = reader
            .attachment_binding_contexts(&record.message.id, block, 0)
            .unwrap();
        assert_eq!(contexts.state, AttachmentState::Available);
        assert_eq!(
            contexts.contexts[0].claim.source_id,
            record.bindings[block as usize].source_id
        );
        assert_eq!(
            reader
                .attachment(&record.message.id, block as usize, 0)
                .unwrap()
                .state,
            AttachmentState::Available
        );
    }
    let row = reader
        .query(
            "SELECT source_id FROM import_member WHERE input='member-a'",
            &[],
        )
        .unwrap();
    assert_eq!(row.len(), 1);
    assert!(row[0]["source_id"].is_null());
}

#[test]
fn one_selected_source_with_two_import_namespaces_remains_ambiguous_even_for_equal_claim_hashes() {
    let fixture = Fixture::new();
    let record = record(b"not received");
    let first = fixture.message("message-input", "scope-a", 0, &record);
    fixture.refresh();
    fixture
        .install(AttachmentBindingRecord::new(first.clone(), &record, 0, Vec::new()).unwrap())
        .unwrap();
    let length = serde_json::to_vec(&record).unwrap().len();
    let mut observed_again = raw(
        "message-input",
        Some("scope-b"),
        "records.jsonl",
        Change::Unchanged,
        length,
    );
    observed_again.input.format = "polylogue-bundle".into();
    fixture.writer.send(observed_again).unwrap();
    assert!(fixture.writer.flush().unwrap().errors.is_empty());
    fixture.refresh();
    let second = BindingContext {
        import_id: "scope-b".into(),
        ..first
    };
    fixture
        .install(AttachmentBindingRecord::new(second, &record, 0, Vec::new()).unwrap())
        .unwrap();
    let contexts = fixture
        .reads
        .reader()
        .unwrap()
        .attachment_binding_contexts(&record.message.id, 0, 0)
        .unwrap();
    assert_eq!(contexts.namespace_count, 2);
    assert_eq!(contexts.state, AttachmentState::Ambiguous);
    assert_ne!(
        contexts.contexts[0].selection.context.import_id,
        contexts.contexts[1].selection.context.import_id
    );
    assert_eq!(
        contexts.contexts[0].claim.candidates[0].hash,
        contexts.contexts[1].claim.candidates[0].hash
    );
}

#[test]
fn a_slot_with_unknown_candidates_stays_explicitly_unknown() {
    let fixture = Fixture::new();
    let mut record = record(b"unknown bytes");
    record.bindings[0].candidates.clear();
    let context = fixture.message("message-input", "scope-a", 0, &record);
    fixture.refresh();
    let fact = AttachmentBindingRecord::new(context, &record, 0, Vec::new()).unwrap();
    let selection = fact.selection();
    fixture.install(fact).unwrap();
    let reader = fixture.reads.reader().unwrap();
    assert_eq!(
        reader
            .attachment_binding(&selection)
            .unwrap()
            .unwrap()
            .state,
        AttachmentState::Unknown
    );
    assert!(matches!(
        reader.pin_attachment(&selection).unwrap(),
        AttachmentPin::Unknown
    ));
}

#[test]
fn exact_member_classification_fills_unknown_fields_and_refuses_conflicting_known_facts() {
    let fixture = Fixture::new();
    fixture.member("member-a", "scope-a", b"actual bytes");
    let members = fixture.refresh();
    let observed = source(&members, "member-a");
    for (source_id, sidecar) in [
        (Some("classified:opaque"), None),
        (None, Some("metadata.json")),
        (None, None),
    ] {
        let mut staged = ExportItems::new().unwrap();
        stage(
            &mut staged,
            &AssetRow::Member {
                source: observed.clone(),
                source_id: source_id.map(str::to_owned),
                sidecar: sidecar.map(str::to_owned),
            },
        );
        fixture.writer.import_assets(staged).unwrap();
    }
    let before = fixture.reads.reader().unwrap().query("SELECT import_id,input,member,source_id,sidecar FROM import_member WHERE input='member-a'",&[]).unwrap();
    assert_eq!(before[0]["source_id"], "classified:opaque");
    assert_eq!(before[0]["sidecar"], "metadata.json");
    for conflict in ["source_id", "sidecar", "identity"] {
        let mut source = observed.clone();
        if conflict == "identity" {
            source.membership.import_id = "another-namespace".into();
        }
        let mut staged = ExportItems::new().unwrap();
        stage(
            &mut staged,
            &AssetRow::Member {
                source,
                source_id: Some(
                    if conflict == "source_id" {
                        "conflicting-id"
                    } else {
                        "classified:opaque"
                    }
                    .into(),
                ),
                sidecar: Some(
                    if conflict == "sidecar" {
                        "conflicting.json"
                    } else {
                        "metadata.json"
                    }
                    .into(),
                ),
            },
        );
        assert!(matches!(
            fixture.writer.import_assets(staged),
            Err(StoreError::Conflict(_))
        ));
        assert_eq!(fixture.reads.reader().unwrap().query("SELECT import_id,input,member,source_id,sidecar FROM import_member WHERE input='member-a'",&[]).unwrap(),before);
    }
}

#[test]
fn distinct_declared_namespaces_and_full_claims_survive_one_receiving_context() {
    let fixture = Fixture::new();
    let bytes = b"received bound bytes";
    let mut record = record(bytes);
    for difference in ["namespace", "export", "prior", "bytes"] {
        let mut another = record.bindings[0].clone();
        match difference {
            "namespace" => another.namespace.import_id = "another-original-scope".into(),
            "export" => another.exported_from.import_id = "another-export-scope".into(),
            "prior" => another.prior_exports[0].import_id = "another-supplied-prior".into(),
            "bytes" => another.candidates[0].hash = BlobHash::of(b"unreceived conflicting bytes"),
            _ => unreachable!(),
        }
        record.bindings.push(another);
    }
    let context = fixture.message("message-input", "receiving-scope", 0, &record);
    fixture.member("member-input", "receiving-scope", bytes);
    let observed = fixture.refresh();
    let witness = fixture
        .reads
        .reader()
        .unwrap()
        .validate_attachment_member(
            source(&observed, "member-input"),
            BlobHash::of(bytes),
            bytes.len() as u64,
        )
        .unwrap();
    let mut facts = Vec::new();
    for slot in 0..record.bindings.len() {
        let members = if slot == record.bindings.len() - 1 {
            Vec::new()
        } else {
            vec![witness.clone().for_candidate(0)]
        };
        let fact = AttachmentBindingRecord::new(context.clone(), &record, slot, members).unwrap();
        fixture.install(fact.clone()).unwrap();
        fixture.install(fact.clone()).unwrap();
        facts.push(fact);
    }
    let reader = fixture.reads.reader().unwrap();
    let contexts = reader
        .attachment_binding_contexts(&record.message.id, 0, 0)
        .unwrap();
    assert_eq!(contexts.namespace_count, 5);
    assert_eq!(contexts.state, AttachmentState::Ambiguous);
    for (slot, fact) in facts.iter().enumerate() {
        let resolution = reader
            .attachment_binding(&fact.selection())
            .unwrap()
            .unwrap();
        assert_eq!(resolution.claim, record.bindings[slot]);
        assert_eq!(resolution.selection.context, context);
        assert_eq!(resolution.claim.namespace.input, "original-message-source");
        assert_ne!(
            resolution.claim.namespace.import_id,
            resolution.selection.context.import_id
        );
        assert_eq!(
            resolution.claim.prior_exports,
            record.bindings[slot].prior_exports
        );
        assert_eq!(
            resolution.state,
            if slot == facts.len() - 1 {
                AttachmentState::Unavailable
            } else {
                AttachmentState::Available
            }
        );
    }
    let mut altered = facts[0].selection();
    altered.claim.namespace.chain += 1;
    assert!(reader.attachment_binding(&altered).unwrap().is_none());
}

#[test]
fn unknown_declared_namespaces_remain_distinct_and_never_borrow_foreign_membership() {
    let fixture = Fixture::new();
    let bytes = b"foreign bytes";
    let mut record = record(bytes);
    record.bindings[0].candidates.clear();
    let mut another = record.bindings[0].clone();
    another.namespace.import_id = "another-original-scope".into();
    record.bindings.push(another);
    let context = fixture.message("message-input", "receiving-scope", 0, &record);
    fixture.member("foreign-member", "original-scope", bytes);
    let observed = fixture.refresh();
    let witness = fixture
        .reads
        .reader()
        .unwrap()
        .validate_attachment_member(
            source(&observed, "foreign-member"),
            BlobHash::of(bytes),
            bytes.len() as u64,
        )
        .unwrap();
    for slot in 0..record.bindings.len() {
        let fact =
            AttachmentBindingRecord::new(context.clone(), &record, slot, Vec::new()).unwrap();
        fixture.install(fact).unwrap();
    }
    let reader = fixture.reads.reader().unwrap();
    let contexts = reader
        .attachment_binding_contexts(&record.message.id, 0, 0)
        .unwrap();
    assert_eq!(contexts.namespace_count, 2);
    assert_eq!(contexts.state, AttachmentState::Ambiguous);
    assert!(
        contexts
            .contexts
            .iter()
            .all(|context| context.claim.candidates.is_empty()
                && context.members.is_empty()
                && context.state == AttachmentState::Unknown)
    );
    let mut known = record.clone();
    known.bindings[0].candidates = vec![AttachmentCandidate {
        hash: BlobHash::of(bytes),
        size: bytes.len() as u64,
        sources: Vec::new(),
    }];
    assert!(
        AttachmentBindingRecord::new(context, &known, 0, vec![witness.for_candidate(0)]).is_err()
    );
}

#[test]
fn retained_nonshown_claims_export_exactly_without_granting_their_members_to_the_shown_source() {
    let fixture = Fixture::new();
    let bytes = b"retained namespace bytes";
    let record = record(bytes);
    let first = fixture.message("message-a", "scope-a", 0, &record);
    let second = fixture.message("message-b", "scope-b", 1, &record);
    fixture.member("member-b", "scope-b", bytes);
    let observed = fixture.refresh();
    let witness = fixture
        .reads
        .reader()
        .unwrap()
        .validate_attachment_member(
            source(&observed, "member-b"),
            BlobHash::of(bytes),
            bytes.len() as u64,
        )
        .unwrap();
    let unavailable = AttachmentBindingRecord::new(first, &record, 0, Vec::new()).unwrap();
    let retained =
        AttachmentBindingRecord::new(second, &record, 0, vec![witness.for_candidate(0)]).unwrap();
    assert_eq!(unavailable.claim(), retained.claim());
    for fact in [&unavailable, &retained] {
        fixture.install(fact.clone()).unwrap();
        fixture.install(fact.clone()).unwrap();
    }
    let reader = fixture.reads.reader().unwrap();
    let shown = reader
        .attachment_binding_contexts(&record.message.id, 0, 0)
        .unwrap();
    assert_eq!(shown.namespace_count, 1);
    assert_eq!(shown.state, AttachmentState::Unavailable);
    assert_eq!(shown.contexts[0].selection.context.input, "message-a");
    assert!(shown.contexts[0].members.is_empty());
    let expected = vec![unavailable, retained];
    let claims = reader
        .attachment_binding_claims(&record.message.id, record.source_version)
        .unwrap();
    assert_eq!(claims, expected);
    assert_ne!(claims[0].context(), claims[1].context());
    assert_eq!(claims[0].claim(), claims[1].claim());
    assert_eq!(claims[1].members().len(), 1);
    assert_eq!(
        reader
            .attachment_binding_claims(&record.message.id, record.source_version)
            .unwrap(),
        expected
    );
    assert!(
        reader
            .attachment_binding_claims(&record.message.id, BlobHash::of(b"another source version"))
            .unwrap()
            .is_empty()
    );
}

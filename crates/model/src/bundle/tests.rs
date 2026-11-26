use super::*;

fn manifest() -> Manifest {
    Manifest {
        format: FORMAT.into(),
        version: VERSION,
        query: format!("query:{}", "a".repeat(64)),
        query_text: "sessions".into(),
        ran_at: Timestamp::from_unix_ms(1).unwrap(),
        read_at: ChangePosition {
            generation: "generation".into(),
            index: 2,
            archive: 3,
        },
        freshness: crate::freshness::ReadFreshness {
            local: crate::freshness::ReadCompleteness {
                search: crate::freshness::CompletenessDimension::Incomplete(vec![
                    crate::freshness::CountedReason::search_pending(Some(2)),
                ]),
                ..crate::freshness::ReadCompleteness::empty()
            },
            sinex: crate::freshness::IndexFreshness::Standalone,
        },
        counts: Counts::default(),
        flags: Flags {
            format: Format::Json,
            with_blobs: false,
            read: crate::read::ReadOptions::default(),
        },
        enabled_alternates: vec![],
        author: Author::Operator,
        inventory: None,
        notes: vec![],
        incomplete: vec![],
        coverage: vec![],
    }
}

#[test]
fn bundle_header_refuses_earlier_and_newer_formats() {
    let mut value = manifest();
    assert!(value.validate().is_ok());
    for version in [0, 1, 2, 3, 4, 6, u32::MAX] {
        value.version = version;
        assert_eq!(
            value.validate(),
            Err(BundleError::UnsupportedVersion { found: version })
        );
    }
    value.format = "polylogue-export".into();
    assert_eq!(
        value.validate(),
        Err(BundleError::WrongFormat("polylogue-export".into()))
    );
}

#[test]
fn manifest_refuses_a_query_that_is_no_question_or_run_and_a_bad_position() {
    for query in ["session:test:one", "not a reference", ""] {
        let mut value = manifest();
        value.query = query.into();
        assert_eq!(value.validate(), Err(BundleError::InvalidQuery), "{query}");
    }
    let mut value = manifest();
    value.read_at.generation.clear();
    assert_eq!(value.validate(), Err(BundleError::InvalidPosition));
    let mut value = manifest();
    value.read_at.index = -1;
    assert_eq!(value.validate(), Err(BundleError::InvalidPosition));
    let mut value = manifest();
    value.read_at.archive = -1;
    assert_eq!(value.validate(), Err(BundleError::InvalidPosition));
}

#[test]
fn event_identity_refuses_each_malformed_claim() {
    let claim = StatedSource {
        input: "a".into(),
        source: "codex".into(),
        source_chain: 1,
        source_start: Some(10),
        source_end: Some(20),
    };
    assert!(claim.validate().is_ok());
    for bad in [
        StatedSource {
            source_chain: 0,
            ..claim.clone()
        },
        StatedSource {
            source_end: None,
            ..claim.clone()
        },
        StatedSource {
            source_start: None,
            ..claim.clone()
        },
        StatedSource {
            source_start: Some(21),
            ..claim.clone()
        },
        StatedSource {
            input: String::new(),
            ..claim.clone()
        },
    ] {
        assert_eq!(bad.validate(), Err(BundleError::InvalidEventIdentity));
    }
    let identity = |id: &str, sources: Vec<StatedSource>| ImportedEventIdentity {
        id: id.into(),
        sources,
    };
    assert!(identity("e", vec![claim.clone()]).validate().is_ok());
    assert_eq!(
        identity("", vec![claim.clone()]).validate(),
        Err(BundleError::InvalidEventIdentity)
    );
    assert_eq!(
        identity("e", vec![]).validate(),
        Err(BundleError::InvalidEventIdentity)
    );
    let mut event: Event = serde_json::from_value(
        serde_json::json!({ "kind": "quota_report", "data": {"remaining": 7} }),
    )
    .unwrap();
    event.after = Some(crate::MessageId::parse("test:one:m1").unwrap());
    let record = InputEventRecord {
        reference: "event:e".into(),
        commit: 1,
        identity: identity("e", vec![claim]),
        event,
    };
    assert_eq!(record.validate(), Err(BundleError::InvalidEventAssociation));
}

#[test]
fn inventory_names_are_relative_and_cannot_escape_or_alias_members() {
    for path in [
        "",
        "/absolute",
        "../outside",
        "a/../b",
        "./member",
        "a//b",
        "a/",
        "a\\b",
        "a\0b",
    ] {
        let file = FileDescriptor {
            path: path.into(),
            hash: BlobHash::of(b"member"),
            size: 6,
        };
        assert_eq!(file.validate(), Err(BundleError::InvalidPath), "{path:?}");
    }
    let file = FileDescriptor {
        path: "codex/native%2Fid.json".into(),
        hash: BlobHash::of(b"member"),
        size: 6,
    };
    assert!(file.validate().is_ok());
}

#[test]
fn event_identity_keeps_every_stated_source() {
    let stated = |input: &str| StatedSource {
        input: input.into(),
        source: "codex".into(),
        source_chain: 1,
        source_start: Some(10),
        source_end: Some(20),
    };
    let mut identity = ImportedEventIdentity {
        id: "event-id".into(),
        sources: vec![stated("b"), stated("a"), stated("b")],
    };
    identity.canonicalize();
    assert_eq!(identity.sources, vec![stated("a"), stated("b")]);
    let event: Event = serde_json::from_value(
        serde_json::json!({ "kind": "quota_report", "data": {"remaining": 7} }),
    )
    .unwrap();
    let mut record = InputEventRecord {
        reference: "event:event-id".into(),
        commit: 4,
        identity,
        event,
    };
    assert!(record.validate().is_ok());
    record.event.at = Timestamp::from_unix_ms(1);
    assert_eq!(
        record.validate(),
        Err(BundleError::Time(crate::TimeAccuracyError::UnknownQuality))
    );
    record
        .event
        .set_time(
            record.event.at,
            crate::TimeAccuracy::intrinsic(crate::Precision::Instant),
        )
        .unwrap();
    assert!(record.validate().is_ok());
    record.reference = "event:another".into();
    assert_eq!(record.validate(), Err(BundleError::EventReferenceMismatch));
}

fn message_record() -> MessageRecord {
    let message: Message = serde_json::from_value(serde_json::json!({
        "id":"codex:message", "session":"codex:session", "role":"assistant",
        "blocks":[{"type":"text","text":"first"},{"type":"text","text":"second"}]
    }))
    .unwrap();
    MessageRecord {
        reference: "message:codex:message".into(),
        commit: 1,
        order: 0,
        source_version: message.content_hash().unwrap(),
        block_count: 2,
        block_ordinals: vec![0, 1],
        injections: Vec::new(),
        bindings: Vec::new(),
        copied_for: Vec::new(),
        message,
    }
}

#[test]
fn sparse_message_round_trip_keeps_block_positions() {
    let full = message_record();
    let mut partial = full.clone();
    partial.message.blocks.remove(0);
    partial.block_ordinals.remove(0);
    assert!(partial.validate().is_ok());
    let restored: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&partial).unwrap()).unwrap();
    assert_eq!(restored, partial);
    assert_eq!(restored.block_ordinals, vec![1]);
    assert_eq!(restored.source_version, full.source_version);
    assert_eq!(restored.block_count, 2);
}

#[test]
fn compatible_sparse_observations_union_without_downgrading_full_content() {
    let full = message_record();
    let mut first = full.clone();
    first.message.blocks.pop();
    first.block_ordinals.pop();
    let mut second = full.clone();
    second.message.blocks.remove(0);
    second.block_ordinals.remove(0);
    let (message, block_selection) = first
        .block_selection()
        .merge(&first.message, &second.block_selection(), &second.message)
        .unwrap()
        .unwrap();
    assert_eq!(message, full.message);
    assert_eq!(block_selection, full.block_selection());
    let merged = full
        .block_selection()
        .merge(&full.message, &first.block_selection(), &first.message)
        .unwrap()
        .unwrap();
    assert_eq!(merged, (full.message.clone(), full.block_selection()));
    let repeated = block_selection
        .merge(&message, &block_selection, &message)
        .unwrap()
        .unwrap();
    assert_eq!(repeated, (message, block_selection));
}

#[test]
fn conflicting_overlap_is_neither_equal_nor_a_new_version() {
    let full = message_record();
    let mut conflict = full.clone();
    conflict.message.blocks[0] = crate::Block::Text {
        text: "different".into(),
    };
    assert_eq!(
        full.block_selection().merge(
            &full.message,
            &conflict.block_selection(),
            &conflict.message
        ),
        Err(BundleError::BlockSelectionConflict)
    );
    conflict.source_version = conflict.message.content_hash().unwrap();
    assert_eq!(
        full.block_selection()
            .merge(
                &full.message,
                &conflict.block_selection(),
                &conflict.message
            )
            .unwrap(),
        None
    );
    let mut invalid = full;
    invalid.block_ordinals = vec![1, 1];
    assert_eq!(invalid.validate(), Err(BundleError::InvalidBlockSelection));
}

#[test]
fn current_bundle_refuses_unknown_outer_record_fields() {
    let malformed = include_str!("tests/extra-wrapper-field.json");
    let mut valid: serde_json::Value = serde_json::from_str(malformed).unwrap();
    valid.as_object_mut().unwrap().remove("future");
    assert!(serde_json::from_value::<Record>(valid).is_ok());
    assert!(serde_json::from_str::<Record>(malformed).is_err());
}

#[test]
fn presentation_transfer_cannot_carry_an_omitted_body() {
    use crate::read::ReadRecord;
    let record: ReadRecord = serde_json::from_str(r#"{"kind":"presentation","parts":[]}"#).unwrap();
    assert!(record.record().is_none());
    assert!(record.parts().is_empty());
    assert_eq!(
        serde_json::to_string(&record).unwrap(),
        r#"{"kind":"presentation","parts":[]}"#
    );
    let forbidden =
        r#"{"kind":"presentation","parts":[],"record":{"record":"document","data":{}}}"#;
    assert!(serde_json::from_str::<ReadRecord>(forbidden).is_err());
}

#[test]
fn original_transfer_keeps_its_typed_record() {
    use crate::read::ReadRecord;
    let transfer = ReadRecord::Original {
        record: Box::new(Record::Message(Box::new(message_record()))),
        parts: vec![],
    };
    let restored: ReadRecord =
        serde_json::from_slice(&serde_json::to_vec(&transfer).unwrap()).unwrap();
    assert_eq!(transfer, restored);
    assert!(matches!(restored.record(), Some(Record::Message(_))));
}

#[test]
fn hook_injection_keeps_message_and_imports_as_declared() {
    let mut record = message_record();
    let hash = record.message.content_hash().unwrap();
    let injection = crate::injection::InjectionEvidence {
        message: record.message.id.clone(),
        source_version: record.source_version,
        block: 1,
        observed: Some(crate::injection::StatedHookField {
            input: "synthetic-input".into(),
            chain: 1,
            start: 3,
            end: 30,
            pointer: "/hookAdditionalContext/0".into(),
        }),
        declared: Vec::new(),
    };
    record.injections.push(injection.clone());
    record.validate().unwrap();
    assert_eq!(record.message.content_hash().unwrap(), hash);
    let decoded: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
    assert_eq!(decoded, record);
    let declared = injection.clone().into_declared();
    assert_eq!(declared.observed, None);
    assert_eq!(declared.declared, vec![injection.observed.clone().unwrap()]);
    assert_eq!(declared.clone().into_declared(), declared);
    for case in 0..5 {
        let mut invalid = record.clone();
        let injection = &mut invalid.injections[0];
        match case {
            0 => injection.message = crate::MessageId::parse("codex:other").unwrap(),
            1 => injection.source_version = crate::BlobHash([7; 32]),
            2 => injection.block = 2,
            3 => injection.observed.as_mut().unwrap().pointer = "/bad~2escape".into(),
            _ => injection.observed.as_mut().unwrap().end = 3,
        }
        assert_eq!(
            invalid.validate(),
            Err(BundleError::InvalidInjectionEvidence),
            "case {case}"
        );
    }
}

#[test]
fn declared_blob_records_require_blob_transfer() {
    let mut value: Manifest = serde_json::from_str(
        r#"{
        "format":"polylogue-bundle","version":5,
        "query":"query:0000000000000000000000000000000000000000000000000000000000000000",
        "query_text":"synthetic","ran_at":"2026-01-01T00:00:00Z",
        "read_at":{"generation":"synthetic","index":0,"archive":0},
        "counts":{"sessions":0,"messages":0,"documents":0,"events":0,"links":0,
                  "authored":0,"annotations":0,"text_fragments":0,"alternates":0,"alternate_parts":0,"blobs":1},
        "flags":{"format":"jsonl","with_blobs":false,
                 "read":{"view":"transcript","hide":["annotations"]}},
        "enabled_alternates":[],"author":"script:fixture","freshness":{"local":{"parsing":{"state":"complete"},"derivation":{"state":"complete"},"search":{"state":"complete"},"confirmed_stream":{"state":"not_applicable"},"kept_chains":0,"raw_only_chains":0},"sinex":{"mode":"standalone"}}
    }"#,
    )
    .unwrap();
    assert_eq!(value.validate(), Err(BundleError::BlobFlagMismatch));
    value.flags.with_blobs = true;
    assert!(value.validate().is_ok());
    value.counts.blobs = 0;
    assert!(value.validate().is_ok());
    value.flags.with_blobs = false;
    assert!(value.validate().is_ok());
}

#[test]
fn annotation_record_carries_no_invented_request() {
    use crate::annotation::{
        AnnotationFact, AnnotationKind, AnnotationLink, AnnotationRecord, LinkRole,
    };
    let fact = AnnotationFact {
        id: "imported-revision".into(),
        root: "imported-root".into(),
        kind: AnnotationKind::Note,
        key: None,
        value: Some(
            serde_json::json!({"text":"original text","predecessor":{"id":"old-17","extra":["preserved",2]}}),
        ),
        author: Author::Operator,
        via: crate::authorship::Channel::Import,
        machine: "importing-host".into(),
        created_at: Some(crate::Timestamp::from_unix_ms(1).unwrap()),
        valid_until: None,
        links: vec![AnnotationLink {
            role: LinkRole::About,
            reference: crate::Reference::parse("session:codex:subject").unwrap(),
            seen_hash: None,
            quote: None,
        }],
    };
    let record = Record::Annotation(Box::new(AnnotationRecord {
        reference: "annotation:imported-root@imported-revision".into(),
        annotation: fact.clone(),
    }));
    record.validate().unwrap();
    assert_eq!(record.commit(), None);
    let encoded = serde_json::to_value(&record).unwrap();
    assert!(encoded["data"].get("prepared").is_none());
    assert!(encoded["data"].get("commit").is_none());
    let decoded: Record = serde_json::from_value(encoded).unwrap();
    assert_eq!(record, decoded);
    let Record::Annotation(mut invalid) = decoded else {
        unreachable!()
    };
    assert_eq!(invalid.annotation, fact);
    invalid.reference = "annotation:another@imported-revision".into();
    assert_eq!(
        Record::Annotation(invalid).validate(),
        Err(BundleError::ReferenceMismatch)
    );
}

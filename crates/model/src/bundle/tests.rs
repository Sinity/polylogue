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
        caught_up: false,
        currency: crate::currency::IndexCurrency::Standalone,
        counts: Counts::default(),
        flags: Flags {
            format: Format::Json,
            with_blobs: false,
            read: crate::read::ReadOptions::default(),
        },
        enabled_variants: vec![],
        author: Author::Operator,
        inventory: None,
        notes: vec![],
        incomplete: vec![],
        coverage: vec![],
    }
}

#[test]
fn bundle_header_refuses_earlier_and_newer_formats_before_contents() {
    let mut value = manifest();
    assert!(value.validate().is_ok());
    for version in [0, 2, u32::MAX] {
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
fn event_identity_keeps_every_claim_and_validates_exact_reference_and_time() {
    let claim = |input: &str| EventSourceClaim {
        input: input.into(),
        source: "codex".into(),
        source_chain: 1,
        source_start: Some(10),
        source_end: Some(20),
    };
    let mut identity = ImportedEventIdentity {
        id: "event-id".into(),
        sources: vec![claim("b"), claim("a"), claim("b")],
    };
    identity.canonicalize();
    assert_eq!(identity.sources, vec![claim("a"), claim("b")]);
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
        Err(BundleError::Time(crate::TimeEvidenceError::UnknownQuality))
    );
    record
        .event
        .set_time(
            record.event.at,
            crate::TimeEvidence::intrinsic(crate::Precision::Instant),
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
        message,
    }
}

#[test]
fn sparse_message_round_trip_keeps_original_block_refs_and_source_identity() {
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
    let (message, projection) = first
        .projection()
        .merge(&first.message, &second.projection(), &second.message)
        .unwrap()
        .unwrap();
    assert_eq!(message, full.message);
    assert_eq!(projection, full.projection());
    let merged = full
        .projection()
        .merge(&full.message, &first.projection(), &first.message)
        .unwrap()
        .unwrap();
    assert_eq!(merged, (full.message.clone(), full.projection()));
    let repeated = projection
        .merge(&message, &projection, &message)
        .unwrap()
        .unwrap();
    assert_eq!(repeated, (message, projection));
}

#[test]
fn overlapping_conflicting_content_is_not_equal_or_an_implicit_new_version() {
    let full = message_record();
    let mut conflict = full.clone();
    conflict.message.blocks[0] = crate::Block::Text {
        text: "different".into(),
    };
    assert_eq!(
        full.projection()
            .merge(&full.message, &conflict.projection(), &conflict.message),
        Err(BundleError::ProjectionConflict)
    );
    conflict.source_version = conflict.message.content_hash().unwrap();
    assert_eq!(
        full.projection()
            .merge(&full.message, &conflict.projection(), &conflict.message)
            .unwrap(),
        None
    );
    let mut invalid = full;
    invalid.block_ordinals = vec![1, 1];
    assert_eq!(invalid.validate(), Err(BundleError::InvalidBlockSelection));
}

#[test]
fn current_bundle_refuses_unknown_outer_record_fields() {
    let malformed = include_str!("tests/extra-envelope-field.json");
    let mut valid: serde_json::Value = serde_json::from_str(malformed).unwrap();
    valid.as_object_mut().unwrap().remove("future");
    assert!(serde_json::from_value::<Record>(valid).is_ok());
    assert!(serde_json::from_str::<Record>(malformed).is_err());
}

#[test]
fn presentation_transfer_cannot_carry_an_omitted_canonical_body() {
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
    let schema = serde_json::to_value(schemars::schema_for!(ReadRecord)).unwrap();
    assert!(
        schema["oneOf"]
            .as_array()
            .is_some_and(|variants| variants.len() == 2)
    );
}

#[test]
fn canonical_transfer_retains_its_typed_record_after_round_trip() {
    use crate::read::ReadRecord;
    let transfer = ReadRecord::Canonical {
        record: Box::new(Record::Message(Box::new(message_record()))),
        parts: vec![],
    };
    let restored: ReadRecord =
        serde_json::from_slice(&serde_json::to_vec(&transfer).unwrap()).unwrap();
    assert_eq!(transfer, restored);
    assert!(matches!(restored.record(), Some(Record::Message(_))));
}

#[test]
fn published_bundle_schema_matches_the_canonical_wire_types() {
    let published: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/reference/polylogue-bundle-v1.schema.json"
    )).unwrap();
    let current = serde_json::to_value(schemars::schema_for!(Record)).unwrap();
    assert_eq!(published, current);
}

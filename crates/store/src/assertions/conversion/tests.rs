use super::*;
use polylogue_model::historical::{AcceptedHashRule, Intent, NoteKind, RecordedKind};

const REQUEST:&[u8]=br#"{"id":"old-note","machine":"sender","at":"2026-01-02T03:04:05Z","write":{"write":"add_note","note":{"kind":"note","value":"exact old text","author":"agent","targets":["repo:synthetic"]}}}"#;
const INTENT:&[u8]=br#"["sender",{"write":"add_note","note":{"kind":"note","value":"exact old text","author":"agent","targets":["repo:synthetic"]}}]"#;
const OUTCOME: &[u8] = br#"{"id":"old-note","repeat":false,"pending":false}"#;
const EFFECT:&[u8]=br#"{"variants":{"variants":[],"variant_parts":[]},"notes":[{"note":{"id":"old-note","first_note":"old-note","kind":"note","value":"exact old text","author":"agent","created_at":"2026-01-02T03:04:05Z","targets":["repo:synthetic"]},"machine":"sender","revises":[]}],"withdrawals":[],"settings":[],"kept_results":[],"deliveries":[],"chats":[],"backfills":[]}"#;

#[test]
fn numbered_conversion_verifies_original_hash_and_conserves_typed_intent_and_effects() {
    let source = SourceSchema::PolylogueArchive { version: 4 };
    let converted = conserve_request(
        source.clone(),
        REQUEST,
        Some(OUTCOME),
        Some(EFFECT),
        Some(BlobHash::of(INTENT)),
        None,
    )
    .unwrap();
    let ConservedAccepted::Historical(record) = converted else {
        panic!("retired actions remain historical");
    };
    assert_eq!(
        record.original.as_ref().unwrap().request.as_deref(),
        Some(REQUEST)
    );
    assert_eq!(
        record.original.as_ref().unwrap().intent.as_deref(),
        Some(INTENT)
    );
    assert_eq!(
        record.original.as_ref().unwrap().outcome.as_deref(),
        Some(OUTCOME)
    );
    assert_eq!(
        record.original.as_ref().unwrap().effect.as_deref(),
        Some(EFFECT)
    );
    assert_eq!(record.hash_evidence, AcceptedHashEvidence::Observed);
    assert_eq!(record.authorship, None);
    let Some(Intent::Note { note }) = &record.intent else {
        panic!("typed historical intent remains available");
    };
    assert_eq!(note.claimed_author.as_deref(), Some("agent"));
    let facts = &record
        .conserved_effects
        .as_ref()
        .unwrap()
        .historical_assertions;
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].kind, RecordedKind::Known(NoteKind::Note));
    assert_eq!(facts[0].recorded_author.as_deref(), Some("agent"));
    assert_eq!(facts[0].value.as_deref(), Some("exact old text"));
    assert!(
        conserve_request(
            source,
            REQUEST,
            Some(OUTCOME),
            Some(EFFECT),
            Some(BlobHash([1; 32])),
            None
        )
        .is_err()
    );
}

#[test]
fn compatible_encoding_keeps_original_empty_retired_effect_fields_without_reexecution() {
    let request = crate::fixture::authored::operator_request(
        "kept".into(),
        polylogue_model::user::UserWrite::KeepResult {
            name: None,
            query: "sessions".into(),
            rows_json: "[]".into(),
            read_at: "{}".into(),
        },
    );
    let request_bytes = serde_json::to_vec(&request).unwrap();
    let outcome = UserOutcome {
        id: "kept".into(),
        repeat: false,
        pending: false,
    };
    let outcome_bytes = serde_json::to_vec(&outcome).unwrap();
    let mut effect = serde_json::to_value(AuthoredEffect::default()).unwrap();
    effect["notes"] = serde_json::json!([]);
    effect["withdrawals"] = serde_json::json!([]);
    effect["settings"] = serde_json::json!([]);
    effect.as_object_mut().unwrap().remove("assertions");
    let effect_bytes = serde_json::to_vec(&effect).unwrap();
    let converted = conserve_request(
        SourceSchema::PolylogueArchive { version: 4 },
        &request_bytes,
        Some(&outcome_bytes),
        Some(&effect_bytes),
        Some(BlobHash(
            crate::primary::authored::request_hash(&request).unwrap(),
        )),
        None,
    )
    .unwrap();
    let ConservedAccepted::Compatible(prepared) = converted else {
        panic!("unchanged actions keep their actual prepared form");
    };
    assert_eq!(prepared.request, request);
    assert_eq!(prepared.outcome, outcome);
    assert_eq!(prepared.effect, AuthoredEffect::default());
    assert_eq!(
        prepared.original.as_ref().unwrap().original.effect.as_ref(),
        Some(&effect_bytes)
    );
    let restored: PreparedRequest =
        serde_json::from_slice(&serde_json::to_vec(&prepared).unwrap()).unwrap();
    assert_eq!(*prepared, restored);
}

#[test]
fn unknown_nonempty_effects_refuse_conversion_instead_of_becoming_empty_success() {
    let mut effect: serde_json::Value = serde_json::from_slice(EFFECT).unwrap();
    effect["unmapped_effect"] = serde_json::json!([{"id":"must-survive"}]);
    assert!(
        conserve_request(
            SourceSchema::PolylogueArchive { version: 4 },
            REQUEST,
            Some(OUTCOME),
            Some(&serde_json::to_vec(&effect).unwrap()),
            Some(BlobHash::of(INTENT)),
            None
        )
        .is_err()
    );
}

#[test]
fn native_projection_retains_available_envelope_and_marks_semantic_hash_as_derived() {
    let mut request: serde_json::Value = serde_json::from_slice(REQUEST).unwrap();
    request["authorship"] = serde_json::json!({"author":"operator","via":"cli"});
    request["write"]["note"]
        .as_object_mut()
        .unwrap()
        .remove("author");
    let request_bytes = serde_json::to_vec(&request).unwrap();
    let mut effect: serde_json::Value = serde_json::from_slice(EFFECT).unwrap();
    effect["notes"][0]["note"]["author"] = serde_json::json!("operator");
    let effect_bytes = serde_json::to_vec(&effect).unwrap();
    let payload=serde_json::to_vec(&serde_json::json!({"record":{"product":"request","request":request,"outcome":serde_json::from_slice::<serde_json::Value>(OUTCOME).unwrap(),"effect":effect}})).unwrap();
    let native = NativeProjectionProvenance {
        event: "synthetic-event".into(),
        run: "synthetic-run".into(),
        commit_seq: 17,
        output_index: 2,
        record_hash: "original-reading-hash".into(),
        projection_epoch: 23,
        projection_payload_hash: "current-projection-hash".into(),
        location: Some(polylogue_model::historical::NativeRecordLocation {
            material: "immutable-material".into(),
            anchor: 37,
        }),
        parents: vec!["actual-parent".into()],
        producer_family: Some("polylogue".into()),
        producer_identity: Some("actual-producer".into()),
        ts_orig: None,
        ts_quality: "unknown".into(),
        ts_coided: None,
        ts_persisted: Timestamp::from_unix_ms(123).unwrap(),
        record_state: "present".into(),
        class: "work".into(),
        validation: "full".into(),
        equivalence_key: None,
        reading_order: 4,
        projected_payload: payload,
    };
    let source = SourceSchema::NativeProjection {
        source: "polylogue".into(),
        source_version: 3,
        source_fingerprint: "frozen-source3-fingerprint".into(),
        kind: "polylogue.note".into(),
        kind_version: 4,
        kind_definition_hash: "frozen-kind-definition".into(),
    };
    let converted = conserve_request(
        source.clone(),
        &request_bytes,
        Some(OUTCOME),
        Some(&effect_bytes),
        None,
        Some(native.clone()),
    )
    .unwrap();
    let ConservedAccepted::Historical(record) = converted else {
        panic!("an old action remains historical");
    };
    assert_eq!(record.source, source);
    assert_eq!(record.native.observed.len(), 1);
    assert_eq!(record.native.observed[0].provenance, native);
    assert!(record.native.declared.is_empty());
    assert_eq!(
        record.hash_evidence,
        AcceptedHashEvidence::Derived {
            rule: AcceptedHashRule::MachineAuthorshipAndIntent
        }
    );
    assert_eq!(
        record.original.as_ref().unwrap().request.as_deref(),
        Some(request_bytes.as_slice())
    );
    assert!(
        conserve_request(
            source,
            &request_bytes,
            Some(OUTCOME),
            Some(&effect_bytes),
            Some(BlobHash::of(INTENT)),
            Some(record.native.observed[0].provenance.clone())
        )
        .is_err()
    );
}

#[test]
fn missing_original_request_keeps_actual_outcome_and_facts_without_invented_intent() {
    let recorded_hash = BlobHash::of(b"unavailable old intent");
    let ConservedAccepted::Historical(record) = conserve_without_request(
        SourceSchema::PolylogueArchive { version: 4 },
        "old-note".into(),
        recorded_hash,
        None,
        Some(OUTCOME),
        Some(EFFECT),
    )
    .unwrap() else {
        panic!("missing intent cannot become a prepared request")
    };
    assert_eq!(
        record.validate().unwrap(),
        polylogue_model::historical::Integrity::Unrecorded
    );
    assert_eq!(record.content_hash, recorded_hash);
    assert_eq!(record.intent, None);
    assert_eq!(record.machine, None);
    assert_eq!(record.authorship, None);
    assert_eq!(record.original.as_ref().unwrap().request, None);
    assert_eq!(record.original.as_ref().unwrap().intent, None);
    assert_eq!(
        record.original.as_ref().unwrap().outcome.as_deref(),
        Some(OUTCOME)
    );
    assert_eq!(
        record.original.as_ref().unwrap().effect.as_deref(),
        Some(EFFECT)
    );
    assert_eq!(
        record
            .conserved_effects
            .as_ref()
            .unwrap()
            .historical_assertions[0]
            .id,
        "old-note"
    );
    let mut changed = record.clone();
    changed.original.as_mut().unwrap().intent = Some(Vec::new());
    assert!(
        changed.validate().is_err(),
        "known hash cannot certify invented empty semantic bytes"
    );
}

#[test]
fn retained_publication_uses_the_same_decoder_without_fetched_event_or_acceptance() {
    let mut request: serde_json::Value = serde_json::from_slice(REQUEST).unwrap();
    request["authorship"] = serde_json::json!({"author":"operator","via":"cli"});
    request["write"]["note"]
        .as_object_mut()
        .unwrap()
        .remove("author");
    let bytes = serde_json::to_vec(&request).unwrap();
    let source = SourceSchema::RetainedPublication {
        source: "polylogue".into(),
        source_version: 3,
        source_fingerprint: "verified-source3-fingerprint".into(),
        parser: "polylogue-request".into(),
        parser_version: 3,
    };
    let accepted = conserve_request(
        source.clone(),
        &bytes,
        Some(OUTCOME),
        Some(EFFECT),
        None,
        None,
    )
    .unwrap();
    assert_eq!(retained_source3_kind(&accepted).unwrap(), "polylogue.note");
    let ConservedAccepted::Historical(record) = &accepted else {
        panic!("retired action stays historical")
    };
    assert_eq!(record.source, source);
    assert_eq!(
        record.hash_evidence,
        AcceptedHashEvidence::Derived {
            rule: AcceptedHashRule::MachineAuthorshipAndIntent
        }
    );
    assert!(record.native.is_empty());
    assert_eq!(
        record.original.as_ref().unwrap().request.as_deref(),
        Some(bytes.as_slice())
    );
    let mut conn = Connection::open_in_memory().unwrap();
    let tx = conn.transaction().unwrap();
    assert!(
        crate::primary::authored::conserve(&tx, &accepted, &[], 1).is_err(),
        "retained custody is not a fetched canonical event"
    );
    let mut unsupported = source;
    if let SourceSchema::RetainedPublication { parser_version, .. } = &mut unsupported {
        *parser_version = 4;
    }
    assert!(
        conserve_request(unsupported, &bytes, Some(OUTCOME), Some(EFFECT), None, None).is_err()
    );
}

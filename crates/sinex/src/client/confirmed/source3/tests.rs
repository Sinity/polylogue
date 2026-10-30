use super::*;
use serde_json::json;
use sinex_model::{Class, EventId, RunId, Timestamp, Validation};
use sinex_protocol::{ConfirmedEvent, ProjectionIdentity};

pub(super) fn event(payload: serde_json::Value) -> ConfirmedEvent {
    let declaration = contract::declaration().unwrap();
    let kind = declaration
        .kinds
        .iter()
        .find(|kind| kind.kind.name == "polylogue.note")
        .unwrap();
    ConfirmedEvent {
        source_ref: Some(declaration.source.clone()),
        source_fingerprint: Some(contract::SOURCE_FINGERPRINT.into()),
        producer_family: Some("polylogue".into()),
        producer_identity: Some("recorded-receiver".into()),
        kind_version: 4,
        kind_definition_hash: kind.definition_hash(),
        validation: Validation::Validated,
        commit_seq: 17,
        id: EventId(uuid::Uuid::from_bytes([1; 16])),
        event_type: "polylogue.note".into(),
        source: "polylogue".into(),
        run: RunId(uuid::Uuid::from_bytes([2; 16])),
        location: Some(sinex_model::RecordLocation {
            material: sinex_model::MaterialId(uuid::Uuid::from_bytes([3; 16])),
            anchor: 9,
        }),
        output_index: 2,
        parents: vec![EventId(uuid::Uuid::from_bytes([4; 16]))],
        ts_orig: None,
        ts_quality: "unknown".into(),
        ts_coided: None,
        ts_persisted: Timestamp::from_unix_secs(1).unwrap(),
        record_state: "present".into(),
        class: Class::Open,
        equivalence_key: None,
        reading_order: 17,
        record_hash: "5".repeat(64),
        projection: ProjectionIdentity::for_payload(7, &payload),
        payload,
    }
}

#[test]
fn historical_source3_contract_refuses_wrong_schema_source_or_changed_projection() {
    let original = event(json!({"record":{"product":"request"}}));
    assert!(contract::matches(&original).unwrap());
    for field in [
        "source",
        "fingerprint",
        "definition",
        "family",
        "producer",
        "payload",
    ] {
        let mut changed = original.clone();
        match field {
            "source" => changed.source_ref.as_mut().unwrap().version = 4,
            "fingerprint" => changed.source_fingerprint = None,
            "definition" => changed.kind_definition_hash = "6".repeat(64),
            "family" => changed.producer_family = None,
            "producer" => changed.producer_identity = None,
            _ => changed.payload["extra"] = true.into(),
        }
        let before = serde_json::to_vec(&changed).unwrap();
        assert!(contract::matches(&changed).is_err(), "{field}");
        assert_eq!(serde_json::to_vec(&changed).unwrap(), before);
    }
    let mut current = original;
    current.kind_version = 5;
    assert!(!contract::matches(&current).unwrap());
}

#[test]
fn native_projection_envelope_keeps_available_nested_encoding_before_request_serde() {
    let payload = br#"{"record":{"product":"request","request": { "id" : "recorded-request", "write":{"write":"add_note"} },"outcome": {"id":"recorded-fact","repeat":false,"pending":false},"effect":{"notes":[]}}}"#;
    let fields = envelope::extract("polylogue.note", payload).unwrap();
    assert_eq!(
        fields.request,
        br#"{ "id" : "recorded-request", "write":{"write":"add_note"} }"#
    );
    assert_eq!(
        fields.outcome,
        br#"{"id":"recorded-fact","repeat":false,"pending":false}"#
    );
    assert_eq!(fields.effect, br#"{"notes":[]}"#);
    assert!(fields.claims.is_empty());
    assert!(envelope::extract("polylogue.request_restore", payload).is_err());
}

fn note_payload() -> serde_json::Value {
    json!({"record_key":"original-natural-key", "record": {
        "product":"request",
        "request": {"id":"recorded-note-request", "machine":"recorded-host",
            "at":"2026-01-02T03:04:05Z", "authorship":{"author":"operator","via":"cli"},
            "write":{"write":"add_note","note":{"kind":"note","value":"recorded assertion","targets":["repo:synthetic"]}}},
        "outcome":{"id":"recorded-note","repeat":false,"pending":false},
        "effect":{"variants":{"variants":[],"variant_parts":[]},
            "notes":[{"note":{"id":"recorded-note","first_note":"recorded-note","kind":"note",
                "value":"recorded assertion","author":"operator","created_at":"2026-01-02T03:04:05Z",
                "targets":["repo:synthetic"]},"machine":"recorded-host","revises":[]}],
            "withdrawals":[],"settings":[],"kept_results":[],"deliveries":[],"chats":[],"backfills":[]}
    }})
}

#[test]
fn retired_native_request_conserves_available_bytes_and_actual_envelope_without_reaccepting_it() {
    use polylogue_model::historical::{AcceptedHashEvidence, ConservedAccepted, SourceSchema};
    use polylogue_store::primary::Product;
    let mut original = event(note_payload());
    original.ts_persisted = Timestamp::from_unix_ns(1_000_000_123).unwrap();
    assert!(serde_json::from_value::<Product>(original.payload["record"].clone()).is_err());
    let before = serde_json::to_vec(&original).unwrap();
    let encoded = serde_json::to_vec(&original.payload).unwrap();
    let fields = envelope::extract(&original.event_type, &encoded).unwrap();
    let confirmed = super::super::decode_event(&original).unwrap();
    assert_eq!(confirmed.position, original.commit_seq);
    assert_eq!(confirmed.output_index, original.output_index);
    assert_eq!(confirmed.native.as_ref().unwrap().kind.version, 4);
    assert_eq!(
        confirmed.native.as_ref().unwrap().projection.policy_epoch,
        7
    );
    assert_eq!(
        confirmed
            .native
            .as_ref()
            .unwrap()
            .origin
            .source
            .as_ref()
            .unwrap()
            .version,
        3
    );
    let polylogue_store::primary::ConfirmedChange::Live(product) = confirmed.change else {
        panic!("a fetched original record remains one live historical projection");
    };
    let Product::ConservedRequest { accepted, .. } = *product else {
        panic!("the old action must use restoration-only conservation");
    };
    let ConservedAccepted::Historical(history) = accepted else {
        panic!("retired note intent must remain historical");
    };
    assert!(matches!(
        history.hash_evidence,
        AcceptedHashEvidence::Derived { .. }
    ));
    assert!(matches!(
        &history.source,
        SourceSchema::NativeProjection {
            source_version: 3,
            kind_version: 4,
            ..
        }
    ));
    let body = history.original.as_ref().unwrap();
    assert_eq!(body.request.as_deref(), Some(fields.request));
    assert_eq!(body.outcome.as_deref(), Some(fields.outcome));
    assert_eq!(body.effect.as_deref(), Some(fields.effect));
    assert_eq!(history.native.observed.len(), 1);
    let actual = &history.native.observed[0].provenance;
    assert_eq!(actual.event, original.id.to_string());
    assert_eq!(actual.record_hash, original.record_hash);
    assert_eq!(actual.projection_epoch, original.projection.policy_epoch);
    assert_eq!(
        actual.projection_payload_hash,
        original.projection.payload_hash
    );
    assert_eq!(actual.projected_payload, encoded);
    assert_eq!(
        actual.location.as_ref().unwrap().material,
        original.location.as_ref().unwrap().material.to_string()
    );
    assert_eq!(
        actual.parents,
        original
            .parents
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    assert_eq!(actual.ts_persisted.unix_ns(), 1_000_000_123);
    assert!(history.imports.observed.is_empty());
    assert_eq!(serde_json::to_vec(&original).unwrap(), before);
}

#[test]
fn compatible_source3_request_still_enters_restoration_instead_of_native_acceptance() {
    use polylogue_model::historical::{AcceptedHashEvidence, ConservedAccepted};
    let mut payload = note_payload();
    payload["record"]["request"]["write"] =
        json!({"write":"keep_result","query":"sessions","rows_json":"[]","read_at":"{}"});
    payload["record"]["effect"]["notes"] = json!([]);
    let mut original = event(payload);
    original.event_type = "polylogue.kept_result".into();
    original.kind_definition_hash = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == original.event_type)
        .unwrap()
        .definition_hash();
    let Product::ConservedRequest { accepted, claims } = decode(&original).unwrap().unwrap() else {
        panic!("a compatible predecessor is not a new native request");
    };
    let ConservedAccepted::Compatible(prepared) = accepted else {
        panic!("the unchanged action remains representable");
    };
    let original_encoding = prepared.original.as_ref().unwrap();
    assert!(matches!(
        &original_encoding.hash_evidence,
        AcceptedHashEvidence::Derived { .. }
    ));
    assert_eq!(original_encoding.native.observed.len(), 1);
    assert_eq!(prepared.outcome.id, "recorded-note");
    assert!(claims.is_empty());
}

#[test]
fn historical_request_definitions_are_read_only_and_do_not_enter_write_registration() {
    let historical = crate::kinds::historical_request_declarations().unwrap();
    assert_eq!(historical.len(), 7);
    let writes = crate::kinds::declarations("polylogue");
    let current_source = crate::kinds::source_declaration();
    for old in historical {
        assert_eq!(old.kind.version, 4);
        assert!(contract::request_kind(&old.kind.name));
        assert!(!writes.iter().any(|current| current.kind == old.kind));
        assert!(
            !current_source
                .kinds
                .iter()
                .any(|current| current.kind == old.kind)
        );
    }
    assert_eq!(current_source.source.version, 4);
}

#[test]
fn source3_restoration_keeps_carried_imports_declared_without_new_receiver_metadata() {
    let mut payload = note_payload();
    let mut prepared = payload["record"].take();
    prepared.as_object_mut().unwrap().remove("product");
    let import = json!({"machine":"old-importer","input":"old-bundle","chain":2,"actor":"operator","at":"2026-01-02T03:04:06Z"});
    payload["record"] =
        json!({"product":"restored_request", "prepared":prepared,"claims":[],"current":import});
    let mut original = event(payload);
    original.event_type = "polylogue.request_restore".into();
    original.kind_definition_hash = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == original.event_type)
        .unwrap()
        .definition_hash();
    let confirmed = super::super::decode_event(&original).unwrap();
    let polylogue_store::primary::ConfirmedChange::Live(product) = confirmed.change else {
        panic!("live projection expected");
    };
    let Product::ConservedRequest { accepted, claims } = *product else {
        panic!("restoration-only product expected");
    };
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].machine, "old-importer");
    assert_eq!(claims[0].input, "old-bundle");
    let polylogue_model::historical::ConservedAccepted::Historical(history) = accepted else {
        panic!("retired note remains historical");
    };
    assert!(history.imports.observed.is_empty());
    assert_eq!(history.native.observed.len(), 1);
    assert_eq!(
        history.native.observed[0].provenance.event,
        original.id.to_string()
    );
    assert_eq!(
        history.native.observed[0]
            .provenance
            .producer_identity
            .as_deref(),
        Some("recorded-receiver")
    );
}

#[test]
fn unchanged_source3_fact_requires_its_actual_frozen_source_and_current_kind_hash() {
    let product = Product::Blob(polylogue_store::primary::Material {
        id: uuid::Uuid::from_bytes([8; 16]).to_string(),
        hash: polylogue_model::BlobHash::of(b"retained bytes"),
        length: 14,
        extents: Vec::new(),
    });
    let mut original = event(crate::products::payload(&product, 0).unwrap());
    let old = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == "polylogue.blob")
        .unwrap();
    let current = crate::kinds::declarations("polylogue")
        .into_iter()
        .find(|kind| kind.kind == old.kind)
        .unwrap();
    assert_eq!(old.definition_hash(), current.definition_hash());
    original.event_type = old.kind.name.clone();
    original.kind_version = old.kind.version;
    original.kind_definition_hash = old.definition_hash();
    let confirmed = super::super::decode_event(&original).unwrap();
    let polylogue_store::primary::ConfirmedChange::Live(decoded) = confirmed.change else {
        panic!("retained fact expected");
    };
    assert_eq!(*decoded, product);
    assert_eq!(confirmed.native.unwrap().origin.source.unwrap().version, 3);
    original.source_fingerprint = None;
    let before = serde_json::to_vec(&original).unwrap();
    assert!(super::super::decode_event(&original).is_err());
    assert_eq!(serde_json::to_vec(&original).unwrap(), before);
}

#[test]
fn conserved_projection_refuses_all_native_emit_routes_before_spool_staging() {
    let product = decode(&event(note_payload())).unwrap().unwrap();
    assert!(crate::products::kind(&product).is_err());
    assert!(crate::products::key(&product, 0).is_err());
    assert!(crate::products::payload(&product, 0).is_err());
    let write = polylogue_store::primary::ProductWrite {
        product,
        location: None,
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    };
    assert!(crate::products::events(std::slice::from_ref(&write)).is_err());
    let root = tempfile::tempdir().unwrap();
    let client = crate::tests::client(root.path());
    assert!(
        client
            .prepare_products(std::slice::from_ref(&write))
            .is_err()
    );
    assert!(
        client
            .prepare_named_products("not-a-native-republication", std::slice::from_ref(&write))
            .is_err()
    );
    let proof = polylogue_store::primary::RawInputProof {
        stamp: "irrelevant-proof".into(),
        position: 17,
        policy_epoch: 7,
        scope: "fixture".into(),
    };
    assert!(
        client
            .prepare_canonical_products(
                "not-a-canonical-republication",
                std::slice::from_ref(&write),
                &proof
            )
            .is_err()
    );
    assert_eq!(client.status().unwrap().prepared_materials, 0);
    assert_eq!(client.status().unwrap().prepared_writes, 0);
}

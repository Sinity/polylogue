use super::*;
use polylogue_model::query_objects::{HistoricalRunReplay, QueryRun};
use serde_json::{Value, json};

fn original() -> QueryRun {
    crate::tests::named::phases()
        .into_iter()
        .find_map(|(_, product)| match product {
            Product::QueryRun { run } => Some(*run),
            _ => None,
        })
        .unwrap()
}
fn payload() -> Value {
    let run = original();
    json!({"record_key":"original-run-natural-key","run":run.id,"question":run.question,
        "binding_hash":run.binding_hash().unwrap(),"run_hash":run.record_hash().unwrap(),
        "record":{"product":"query_run","run":run}})
}
fn event(payload: Value) -> ConfirmedEvent {
    let mut event = super::super::tests::event(payload);
    let kind = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == "polylogue.query_run" && kind.kind.version == 2)
        .unwrap();
    event.event_type = kind.kind.name.clone();
    event.kind_version = kind.kind.version;
    event.kind_definition_hash = kind.definition_hash();
    event
}
fn historical(event: &ConfirmedEvent) -> polylogue_model::query_objects::HistoricalQueryRun {
    let Product::HistoricalQueryRun { run } = super::decode(event).unwrap().unwrap() else {
        panic!("historical run")
    };
    *run
}

#[test]
fn source3_run2_keeps_available_json_claims_and_actual_owner_envelope_without_rehashing_provenance()
{
    let event = event(payload());
    let kind = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == "polylogue.query_run")
        .unwrap();
    jsonschema::validator_for(kind.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&event.payload)
        .unwrap();
    let run = historical(&event);
    assert_eq!(
        crate::spool::decode_json::<Value>(run.encoded_run.as_slice()).unwrap(),
        event.payload["record"]["run"]
    );
    assert!(
        matches!(&run.replay,HistoricalRunReplay::Compatible{run} if run.question==Some(original().question.unwrap()))
    );
    assert_eq!(run.native.observed.len(), 1);
    assert!(run.native.declared.is_empty());
    let observed = &run.native.observed[0].provenance;
    assert_eq!(observed.event, event.id.to_string());
    assert_eq!(observed.record_hash, event.record_hash);
    assert_eq!(
        observed.projection_payload_hash,
        event.projection.payload_hash
    );
    assert_eq!(
        observed.location.as_ref().unwrap().material,
        event.location.as_ref().unwrap().material.to_string()
    );
    assert_eq!(
        observed.parents,
        event
            .parents
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        observed.projected_payload,
        serde_json::to_vec(&event.payload).unwrap()
    );
    assert_eq!(
        run.declared_run_hash,
        Some(original().record_hash().unwrap())
    );
    assert_ne!(
        run.declared_run_hash.unwrap().to_string(),
        observed.record_hash
    );
}

#[test]
fn missing_run_summary_hashes_stay_unknown_while_original_available_body_remains_readable() {
    let mut payload = payload();
    payload.as_object_mut().unwrap().remove("binding_hash");
    payload.as_object_mut().unwrap().remove("run_hash");
    let run = historical(&event(payload));
    assert_eq!(run.declared_binding_hash, None);
    assert_eq!(run.declared_run_hash, None);
    assert!(matches!(run.replay, HistoricalRunReplay::Compatible { .. }));
}

#[test]
fn retired_note_reference_remains_read_only_old_json_without_a_current_reference_alias() {
    let mut payload = payload();
    payload["record"]["run"]["binding"]["execution"]["query_options"]["direct_reference"] =
        "note:retained-note".into();
    let event = event(payload);
    let run = historical(&event);
    assert!(matches!(
        run.replay,
        HistoricalRunReplay::Unavailable { .. }
    ));
    let body: Value = crate::spool::decode_json(run.encoded_run.as_slice()).unwrap();
    assert_eq!(
        body["binding"]["execution"]["query_options"]["direct_reference"],
        "note:retained-note"
    );
    assert!(polylogue_model::Reference::parse("note:retained-note").is_err());
    assert_eq!(
        run.native.observed[0].provenance.projected_payload,
        serde_json::to_vec(&event.payload).unwrap()
    );
}

#[test]
fn a_recorded_read_continuation_is_historical_and_kept_byte_for_byte() {
    let mut body = serde_json::to_value(original()).unwrap();
    body["binding"] = json!({
        "kind": "continuation",
        "root": "00000000-0000-7000-8000-000000000002",
        "binding_hash": BlobHash::of(b"retained root binding"),
        "seek": {
            "direction": "before",
            "anchor": {
                "selection": {"row": 2, "record": {"kind": "authored", "ordinal": 3}},
                "item": 4,
                "character": 5,
                "item_hash": BlobHash::of(b"original encoded item"),
            }
        },
        "accepted_request": null,
    });
    let payload = json!({"record_key":"original-key","record":{"product":"query_run","run":body}});
    let historical = historical(&event(payload));
    assert!(matches!(
        historical.replay,
        HistoricalRunReplay::Unavailable { .. }
    ));
    assert_eq!(
        crate::spool::decode_json::<Value>(historical.encoded_run.as_slice()).unwrap(),
        body
    );
}

#[test]
fn wrong_source_kind_hash_or_projection_cannot_mutate_a_historical_run() {
    let original = event(payload());
    for field in ["source", "fingerprint", "kind_hash", "projection"] {
        let mut changed = original.clone();
        match field {
            "source" => changed.source_ref.as_mut().unwrap().version = 4,
            "fingerprint" => changed.source_fingerprint = None,
            "kind_hash" => changed.kind_definition_hash = "different".into(),
            _ => changed.payload["copied"] = true.into(),
        }
        let before = serde_json::to_vec(&changed).unwrap();
        assert!(super::decode(&changed).is_err(), "{field}");
        assert_eq!(serde_json::to_vec(&changed).unwrap(), before);
    }
}

#[test]
fn new_exact_object_binding_cannot_enter_through_the_frozen_source3_run2_contract() {
    for field in [
        "missing_question",
        "null_question",
        "invalid_question",
        "binding",
    ] {
        let mut payload = payload();
        match field {
            "missing_question" => {
                payload["record"]["run"]
                    .as_object_mut()
                    .unwrap()
                    .remove("question");
            }
            "null_question" => payload["record"]["run"]["question"] = Value::Null,
            "invalid_question" => {
                payload["record"]["run"]["question"] = json!({"unsupported":"question"});
            }
            _ => payload["record"]["run"]["binding"]["kind"] = "original_object".into(),
        }
        assert!(super::decode(&event(payload)).is_err(), "{field}");
    }
}

#[test]
fn historical_run_keeps_deep_accepted_envelope_using_the_existing_unlimited_decoder() {
    let mut deep = json!("retained");
    for _ in 0..200 {
        deep = json!({"nested":deep});
    }
    let mut payload = payload();
    payload.as_object_mut().unwrap().remove("binding_hash");
    payload.as_object_mut().unwrap().remove("run_hash");
    payload["record"]["run"]["binding"]["execution"]["accepted_request"] = deep;
    let event = event(payload);
    let run = historical(&event);
    assert!(matches!(run.replay, HistoricalRunReplay::Compatible { .. }));
    assert_eq!(
        crate::spool::decode_json::<Value>(run.encoded_run.as_slice()).unwrap(),
        event.payload["record"]["run"]
    );
}

#[test]
fn historical_run_receive_only_product_has_no_emission_key_payload_or_current_named_write() {
    let product = Product::HistoricalQueryRun {
        run: Box::new(historical(&event(payload()))),
    };
    assert!(crate::products::kind(&product).is_err());
    assert!(crate::products::key(&product, 0).is_err());
    assert!(crate::products::payload(&product, 0).is_err());
    let root = tempfile::tempdir().unwrap();
    let client = crate::tests::client(root.path());
    let write = polylogue_store::primary::ProductWrite {
        location: None,
        product,
        parents: vec![],
        machine: None,
        output_index: 0,
    };
    assert!(
        client
            .prepare_named_products("historical-run", &[write])
            .is_err()
    );
    assert!(
        client
            .queue()
            .unwrap()
            .pending_after(None, 64)
            .unwrap()
            .is_empty()
    );
}

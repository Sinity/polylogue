use super::*;
use polylogue_store::primary::{ConfirmedChange, Reading};
use serde_json::{Value, json};

fn event(name: &str, version: u64, payload: Value) -> ConfirmedEvent {
    let mut event = super::super::tests::event(payload);
    let kind = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == name && kind.kind.version == version)
        .unwrap();
    event.event_type = name.into();
    event.kind_version = version;
    event.kind_definition_hash = kind.definition_hash();
    event
}
fn old_notes() -> Product {
    Product::Query {
        object: Box::new(
            polylogue_store::fixture::query_objects::synthetic_source3_notes().unwrap(),
        ),
    }
}
fn query_event() -> ConfirmedEvent {
    event(
        "polylogue.query",
        1,
        crate::products::payload(&old_notes(), 0).unwrap(),
    )
}
fn reading() -> Reading {
    Reading {
        delivery: None,
        source_length: None,
        input: polylogue_store::InputKey {
            id: "old-input".into(),
            format: "codex".into(),
            arrival_rank: 0,
        },
        chain: 1,
        reading_version: "reader-3".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start: 0,
        parsed_upto: 8,
        progress: polylogue_model::batch::StepProgress::AtEnd,
    }
}
fn raw_products() -> Vec<Product> {
    let file = json!({"type":"file","name":"synthetic.txt","media_type":"text/plain","size":8,"direction":"unknown"});
    let message=Product::Message {
        native_fields:None,declared_injections:None,declared_attachment_bindings:None,
        reading:reading(),order:Some(0),projection:None,
        facts:Box::new(serde_json::from_value(json!({"offset":0,"end":8,"message":{"id":"codex:old-message","session":"codex:old-session","role":"assistant","blocks":[file.clone()]}})).unwrap()),
    };
    let document=Product::Document {reading:reading(),facts:serde_json::from_value(json!({"offset":0,"end":8,"document":{"id":"codex:old-document","purpose":"context","blocks":[file]}})).unwrap()};
    let recorded = Product::Event {
        reading: reading(),
        id: "old-recorded-event".into(),
        facts: serde_json::from_value(
            json!({"kind":"source_observation","range":[0,8],"data":{"native":{"value":false}}}),
        )
        .unwrap(),
        imported: None,
    };
    vec![message, document, recorded]
}

#[test]
fn frozen_query1_retains_a_synthetic_old_notes_definition_without_current_catalog_support() {
    let product = old_notes();
    let event = query_event();
    let definition = contract::declaration()
        .unwrap()
        .kinds
        .iter()
        .find(|kind| kind.kind.name == "polylogue.query")
        .unwrap();
    jsonschema::validator_for(definition.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&event.payload)
        .unwrap();
    let confirmed = super::super::super::decode_event(&event).unwrap();
    assert!(matches!(&confirmed.change,ConfirmedChange::Live(actual) if actual.as_ref()==&product));
    let Product::Query { object } = product else {
        unreachable!()
    };
    assert_eq!(
        object.normal_form,
        json!({"source":{"kind":"unit","unit":"notes","condition":null},"attach":[],"stages":[]})
    );
    assert_eq!(
        serde_json::to_value(&object.identity_ast).unwrap(),
        json!({"source":{"kind":"unit","unit":"notes","condition":null,"implicit":false},"attach":[],"stages":[]})
    );
    assert!(object.parameters.is_empty());
    assert!(object.source_outputs.is_empty());
    assert!(object.definitions.is_empty());
    assert!(object.dependencies.is_empty());
    assert!(object.edges.is_empty());
    let native = confirmed.native.unwrap();
    assert_eq!(native.kind.name, event.event_type);
    assert_eq!(native.kind.version, 1);
    assert_eq!(native.kind_definition_hash, event.kind_definition_hash);
    assert_eq!(
        native.projection.payload_hash,
        event.projection.payload_hash
    );
    assert_eq!(native.origin.source_fingerprint, event.source_fingerprint);
    assert_eq!(confirmed.event, event.id.to_string());
    assert_eq!(confirmed.output_index, event.output_index);
}

#[test]
fn the_query_contract_is_unchanged_and_reading_kinds_take_new_versions() {
    let old = contract::declaration().unwrap();
    let current = crate::kinds::declarations("polylogue");
    let find = |kinds: &[sinex_model::KindDeclaration], name: &str| {
        kinds
            .iter()
            .find(|kind| kind.kind.name == name)
            .unwrap()
            .clone()
    };
    let before = find(&old.kinds, "polylogue.query");
    let after = find(&current, "polylogue.query");
    assert_eq!(before.kind, after.kind, "polylogue.query");
    assert_eq!(
        before.definition_hash(),
        after.definition_hash(),
        "polylogue.query"
    );
    assert_eq!(
        serde_json::to_value(&before).unwrap(),
        serde_json::to_value(&after).unwrap(),
        "polylogue.query"
    );
    // A kind embedding a parser reading changed its definition when the
    // reading's two flags became one progress value, so it carries a new
    // version and never a changed definition under the frozen one.
    for (name, frozen, now) in [
        ("polylogue.message", 5, 6),
        ("polylogue.document", 4, 5),
        ("polylogue.event", 4, 5),
    ] {
        let before = find(&old.kinds, name);
        let after = find(&current, name);
        assert_eq!(before.kind.version, frozen, "{name}");
        assert_eq!(after.kind.version, now, "{name}");
        assert_ne!(before.definition_hash(), after.definition_hash(), "{name}");
    }
}

#[test]
fn source3_raw_facts_at_their_frozen_versions_are_refused_not_misread() {
    for product in raw_products() {
        let name = crate::products::kind(&product).unwrap();
        let version = match name {
            "polylogue.message" => 5,
            _ => 4,
        };
        let mut payload = crate::products::payload(&product, 0).unwrap();
        let reading = payload["record"]["reading"].as_object_mut().unwrap();
        assert_eq!(reading.remove("progress"), Some(json!("at_end")));
        reading.insert("at_end".into(), json!(true));
        reading.insert("no_parser_yet".into(), json!(false));
        let event = event(name, version, payload);
        let old = contract::declaration()
            .unwrap()
            .kinds
            .iter()
            .find(|kind| kind.kind.name == name && kind.kind.version == version)
            .unwrap();
        jsonschema::validator_for(old.payload_schema.as_ref().unwrap())
            .unwrap()
            .validate(&event.payload)
            .unwrap();
        let before = serde_json::to_vec(&event).unwrap();
        let refused = super::super::super::decode_event(&event).unwrap_err();
        assert!(
            matches!(&refused, crate::Error::Invalid(message)
                if *message == format!("unsupported indexed kind {name}@{version}")),
            "{refused:?}"
        );
        assert_eq!(serde_json::to_vec(&event).unwrap(), before);
    }
}

#[test]
fn historical_query_unknown_source_definition_or_changed_projection_refuses_before_decode() {
    let original = query_event();
    for field in [
        "source",
        "fingerprint",
        "kind_hash",
        "family",
        "producer",
        "projection",
    ] {
        let mut changed = original.clone();
        match field {
            "source" => changed.source_ref.as_mut().unwrap().version = 2,
            "fingerprint" => changed.source_fingerprint = None,
            "kind_hash" => changed.kind_definition_hash = "changed".into(),
            "family" => changed.producer_family = None,
            "producer" => changed.producer_identity = None,
            _ => changed.payload["changed"] = true.into(),
        }
        let before = serde_json::to_vec(&changed).unwrap();
        assert!(
            super::super::super::decode_event(&changed).is_err(),
            "{field}"
        );
        assert_eq!(serde_json::to_vec(&changed).unwrap(), before);
    }
}

#[test]
fn a_current_source4_query1_uses_the_current_codec_and_keeps_its_current_origin() {
    let product = crate::tests::named::phases()
        .into_iter()
        .find_map(|(_, product)| matches!(product, Product::Query { .. }).then_some(product))
        .unwrap();
    let mut event = event(
        "polylogue.query",
        1,
        crate::products::payload(&product, 0).unwrap(),
    );
    event.source_ref = Some(crate::source_reference());
    event.source_fingerprint = Some(crate::kinds::source_declaration().fingerprint());
    let confirmed = super::super::super::decode_event(&event).unwrap();
    assert!(matches!(confirmed.change,ConfirmedChange::Live(actual) if *actual==product));
    assert_eq!(confirmed.native.unwrap().origin.source.unwrap().version, 4);
}

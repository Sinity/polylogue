use super::*;
use crate::products;
use polylogue_store::primary::{Product, Reading};
use serde_json::json;

fn reading(input: &str, start: u64) -> Reading {
    Reading {
        input: polylogue_store::InputKey {
            id: input.into(),
            format: "codex".into(),
            arrival_rank: 0,
        },
        chain: 1,
        reading_version: "reader-3".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start,
        parsed_upto: 100,
        at_end: true,
        no_parser_yet: false,
    }
}
fn message(input: &str, start: u64) -> Product {
    Product::Message {reading:reading(input,start),order:Some(6),projection:None,facts:Box::new(serde_json::from_value(json!({"offset":50,"end":100,"message":{"id":"codex:m","session":"codex:s","role":"assistant","attrs":{"nested":{"citation":["source"]}},"production":{"phase":"final","server_tool_use":{"web_search_requests":0}},"blocks":[{"type":"tool_result","call_id":"native","outcome":{"kind":"ok"},"output":"done","reported_duration_ms":17}]}})).unwrap())}
}

#[test]
fn source_copies_have_distinct_keys_and_chunk_boundaries_do_not_rename_a_record() {
    let first = message("input-a", 0);
    let later = message("input-a", 25);
    let copy = message("input-b", 0);
    assert_eq!(
        products::key(&first, 0).unwrap(),
        products::key(&later, 0).unwrap()
    );
    assert_ne!(
        products::key(&first, 0).unwrap(),
        products::key(&copy, 0).unwrap()
    );
    let payload = products::payload(&first, 0).unwrap();
    assert_eq!(
        products::decode("polylogue.message", payload.clone()).unwrap(),
        first
    );
    let declaration = declarations("polylogue")
        .into_iter()
        .find(|declaration| declaration.kind.name == "polylogue.message")
        .unwrap();
    declaration
        .evaluate(&payload, sinex_model::Class::Open)
        .unwrap();
    let validator = jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap()).unwrap();
    validator.validate(&payload).unwrap();
    let mut malformed = payload;
    malformed["record"]["facts"]["message"]["usage"] = json!("invalid usage");
    assert!(validator.validate(&malformed).is_err());
}

#[test]
fn complete_snapshot_schema_keys_session_and_reducer_and_carries_honest_unknown_time() {
    let snapshot = polylogue_model::work_products::WorkSnapshot {
        reducer_version: 1,
        position: 42,
        session_id: polylogue_model::SessionId::parse("codex:s").unwrap(),
        session: None,
        retired: true,
        unavailable: Vec::new(),
        messages: Vec::new(),
        calls: Vec::new(),
        turns: Vec::new(),
        sources: Vec::new(),
        native: Vec::new(),
    };
    let fact = Product::WorkSnapshot { snapshot };
    let payload = products::payload(&fact, 0).unwrap();
    let declarations = declarations("polylogue");
    let declaration = declarations
        .iter()
        .find(|declaration| declaration.kind.name == "polylogue.work_snapshot")
        .unwrap();
    assert_eq!(
        declaration.key_rule,
        Some(sinex_model::kinds::KeyRule::Fields {
            fields: vec!["session".into(), "reducer_version".into()]
        })
    );
    assert_eq!(
        declaration.time_rule,
        Some(sinex_model::kinds::TimeRule::Unknown)
    );
    declaration
        .evaluate(&payload, sinex_model::Class::Open)
        .unwrap();
    assert_eq!(
        products::decode("polylogue.work_snapshot", payload.clone()).unwrap(),
        fact
    );
    let mut malformed = payload;
    malformed["source_position"] = 43.into();
    assert!(products::decode("polylogue.work_snapshot", malformed).is_err());
    assert!(declarations.iter().all(|declaration| !matches!(
        declaration.kind.name.as_str(),
        "polylogue.tool_call" | "polylogue.turn"
    )));
}

#[test]
fn declarations_guard_complete_records_and_reject_unknown_kind_names() {
    for declaration in declarations("polylogue") {
        assert_eq!(declaration.kind.version, 3);
        assert!(
            declaration
                .field_hints
                .iter()
                .any(|hint| hint.field == "/record" && hint.class == sinex_model::Class::Guarded)
        );
        assert_eq!(declaration.search, sinex_model::kinds::Search::None);
    }
    assert!(kind_reference("polylogue.unknown").is_err());
}

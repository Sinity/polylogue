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
    Product::Message {native_fields:Vec::new(),declared_injections:Vec::new(),declared_attachment_bindings:Vec::new(),reading:reading(input,start),order:Some(6),projection:None,facts:Box::new(serde_json::from_value(json!({"offset":50,"end":100,"message":{"id":"codex:m","session":"codex:s","role":"assistant","attrs":{"nested":{"citation":["source"]}},"production":{"phase":"final","server_tool_use":{"web_search_requests":0}},"blocks":[{"type":"tool_result","call_id":"native","outcome":{"kind":"ok"},"output":"done","reported_duration_ms":17}]}})).unwrap())}
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
    let validator =
        jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap()).unwrap();
    validator.validate(&payload).unwrap();
    let mut malformed = payload;
    malformed["record"]["facts"]["message"]["usage"] = json!("invalid usage");
    assert!(validator.validate(&malformed).is_err());
}

#[test]
fn complete_snapshot_schema_keys_session_and_reducer_and_carries_honest_unknown_time() {
    let snapshot = polylogue_model::work_products::WorkSnapshot {
        reducer_version: polylogue_model::work_products::VERSION,
        position: 42,
        session_id: polylogue_model::SessionId::parse("codex:s").unwrap(),
        membership: polylogue_model::work_products::SnapshotMembership::Retired {
            basis: polylogue_model::work_products::component::MembershipBasis {
                membership_version: polylogue_model::work_products::membership::VERSION,
                selection_version: polylogue_model::work_products::SelectionVersion::Confirmed1,
                completed_position: 42,
                scope: "scope".into(),
                membership_hash: "retired-known-root".into(),
            },
        },
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
        assert_eq!(
            declaration.kind.version,
            match declaration.kind.name.as_str() {
                "polylogue.work_snapshot" => 5,
                "polylogue.input_observation"
                | "polylogue.message"
                | "polylogue.note"
                | "polylogue.setting"
                | "polylogue.kept_result"
                | "polylogue.delivery"
                | "polylogue.delivery_outcome"
                | "polylogue.request"
                | "polylogue.request_restore" => 4,
                "polylogue.fragment" => 1,
                "polylogue.query"
                | "polylogue.query_run"
                | "polylogue.query_run_outcome"
                | "polylogue.work_component_status" => 1,
                _ => 3,
            }
        );
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

#[test]
fn partial_fragment_has_its_own_kind_and_retains_the_receiving_record_range() {
    use polylogue_model::bundle::fragments::{
        TextCoverage, TextFragment, TextFragmentProvenance, TextFragmentRecord, TextSourceVersion,
        TextSpanOrigin,
    };
    use polylogue_model::{BlobHash, Reference};
    let container =
        Reference::Message(polylogue_model::MessageId::parse("codex:morrow-fragment").unwrap());
    let mut fragment = TextFragment {
        container: container.clone(),
        source_version: TextSourceVersion::ContentHash {
            record: container,
            hash: BlobHash::of(b"original source value"),
        },
        coverage: TextCoverage {
            start: 7,
            end: 10,
            total_bytes: None,
        },
        text: "abc".into(),
        text_hash: BlobHash::of(b"abc"),
        provenance: TextFragmentProvenance {
            spans: vec![],
            sources: vec![],
            imports: vec![],
        },
    };
    fragment.provenance.spans.push(TextSpanOrigin {
        reference: fragment.reference(),
        start: 7,
        end: 10,
    });
    let record = TextFragmentRecord {
        reference: fragment.reference().to_string(),
        commit: 9000,
        fragment,
    };
    record.validate().unwrap();
    let product = Product::Fragment {
        reading: reading("input-fragment", 12),
        facts: Box::new(polylogue_model::batch::FragmentRow {
            offset: 12,
            end: 47,
            record,
        }),
    };
    let payload = products::payload(&product, 0).unwrap();
    let declaration = declarations("polylogue")
        .into_iter()
        .find(|declaration| declaration.kind.name == "polylogue.fragment")
        .unwrap();
    assert_eq!(declaration.kind.version, 1);
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&payload)
        .unwrap();
    assert_eq!(payload["record"]["facts"]["offset"], 12);
    assert_eq!(payload["record"]["facts"]["end"], 47);
    assert_eq!(payload["record"]["facts"]["record"]["commit"], 9000);
    assert!(payload.get("message").is_none());
    assert_eq!(
        products::decode("polylogue.fragment", payload).unwrap(),
        product
    );
}

#[test]
fn component_replacement_keeps_its_physical_key_across_copy_arrival_and_resolution() {
    use polylogue_model::work_products::component::*;
    let mut status = ComponentStatus {
        version: VERSION,
        reducer_version: polylogue_model::work_products::VERSION,
        position: 42,
        component_kind: ComponentKind::LogicalWork,
        physical: vec![PhysicalRef {
            session: polylogue_model::SessionId::parse("codex:physical").unwrap(),
            evidence: vec![PhysicalEvidence {
                event: "00000000-0000-7000-8000-000000000001".into(),
                field: PhysicalField::MessageSession,
            }],
        }],
        links: vec![],
        status: ComponentState::Unavailable {
            reasons: vec!["logical_membership_root_unavailable".into()],
        },
    };
    let unavailable = Product::WorkComponentStatus {
        status: Box::new(status.clone()),
    };
    status.position = 43;
    status.physical[0].evidence.push(PhysicalEvidence {
        event: "00000000-0000-7000-8000-000000000002".into(),
        field: PhysicalField::SessionId,
    });
    status.status = ComponentState::Resolved {
        superseding_root_refs: vec![RootRef {
            session: polylogue_model::SessionId::parse("codex:established-root").unwrap(),
            reducer_version: polylogue_model::work_products::VERSION,
        }],
        basis: MembershipBasis {
            membership_version: polylogue_model::work_products::membership::VERSION,
            selection_version: polylogue_model::work_products::SelectionVersion::Confirmed1,
            completed_position: 43,
            scope: "owner-scope".into(),
            membership_hash: "confirmed-membership".into(),
        },
    };
    let resolved = Product::WorkComponentStatus {
        status: Box::new(status),
    };
    assert_eq!(
        products::key(&unavailable, 0).unwrap(),
        products::key(&resolved, 0).unwrap()
    );
    let declaration = declarations("polylogue")
        .into_iter()
        .find(|declaration| declaration.kind.name == "polylogue.work_component_status")
        .unwrap();
    assert_eq!(declaration.kind.version, 1);
    assert_eq!(
        declaration.key_rule,
        Some(sinex_model::kinds::KeyRule::Fields {
            fields: vec!["component_kind".into(), "component_sessions".into()]
        })
    );
    let validator =
        jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap()).unwrap();
    for product in [unavailable, resolved] {
        let payload = products::payload(&product, 0).unwrap();
        validator.validate(&payload).unwrap();
        assert!(payload.get("session").is_none());
        assert!(payload.get("complete").is_none());
        assert_eq!(
            products::decode("polylogue.work_component_status", payload).unwrap(),
            product
        );
    }
}

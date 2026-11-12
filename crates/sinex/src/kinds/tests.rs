use super::*;
use crate::products;
use polylogue_store::primary::{Product, Reading};
use serde_json::json;
mod ai;

fn reading(input: &str, start: u64) -> Reading {
    Reading {
        delivery: None,
        source_length: None,
        input: polylogue_store::InputKey {
            id: input.into(),
            format: "codex".into(),
            route: polylogue_model::Route::Live,
        },
        chain: 1,
        reading_version: "reader-3".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start,
        parsed_upto: 100,
        progress: polylogue_model::batch::StepProgress::AtEnd,
    }
}
fn message(input: &str, start: u64) -> Product {
    Product::Message {native_fields:None,declared_injections:None,declared_attachment_bindings:None,reading:reading(input,start),order:Some(6),projection:None,facts:Box::new(serde_json::from_value(json!({"offset":50,"end":100,"message":{"id":"codex:m","session":"codex:s","role":"assistant","attrs":{"nested":{"citation":["source"]}},"production":{"phase":"final","server_tool_use":{"web_search_requests":0}},"blocks":[{"type":"tool_result","call_id":"native","outcome":{"kind":"ok"},"output":"done","reported_duration_ms":17}]}})).unwrap())}
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
        products::decode(
            &crate::kinds::kind_reference("polylogue.message").unwrap(),
            payload.clone()
        )
        .unwrap(),
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
    for (field, malformed) in [("text", json!(17)), ("at", json!(17))] {
        let mut malformed_payload = payload.clone();
        malformed_payload[field] = malformed;
        assert!(validator.validate(&malformed_payload).is_err(), "{field}");
    }
    let mut malformed = payload;
    malformed["record"]["facts"]["message"]["usage"] = json!("invalid usage");
    assert!(validator.validate(&malformed).is_err());
}

#[test]
fn richer_and_weaker_link_contributions_keep_their_own_retry_stable_keys() {
    // A Claude subagent's fork-context-ref supplies the branch message, then
    // its user record names the same parent without repeating that message.
    let strong: polylogue_model::Link = serde_json::from_value(json!({
        "kind":"spawn", "parent":"claude-code:parent", "child":"claude-code:agent-child",
        "at":"claude-code:branch", "evidence":"fork-context-ref"
    }))
    .unwrap();
    let mut weak = strong.clone();
    weak.at = None;
    weak.evidence = "agentId".into();
    let contribution = |facts, start| Product::Link {
        reading: reading("subagent-source", start),
        facts,
    };
    let rich = contribution(strong.clone(), 0);
    let weak = contribution(weak, 0);
    let rich_key = products::key(&rich, 0).unwrap();
    assert_ne!(rich_key, products::key(&weak, 1).unwrap());
    assert_eq!(rich_key, products::key(&rich, 99).unwrap());
    assert_eq!(
        rich_key,
        products::key(&contribution(strong, 75), 0).unwrap()
    );
    for product in [rich, weak] {
        assert_eq!(
            products::decode(
                &crate::kinds::kind_reference("polylogue.link").unwrap(),
                products::payload(&product, 0).unwrap()
            )
            .unwrap(),
            product
        );
    }
}

#[test]
fn declarations_guard_complete_records_and_reject_unknown_kind_names() {
    for declaration in declarations("polylogue") {
        let schema = declaration.payload_schema.as_ref().unwrap();
        for hint in &declaration.field_hints {
            let field = hint.field.strip_prefix('/').unwrap();
            assert!(
                schema["properties"].get(field).is_some(),
                "{} hints undeclared {field}",
                declaration.kind.name
            );
        }
        if let Some(sinex_model::kinds::TimeRule::FirstPresent { fields, .. }) =
            &declaration.time_rule
        {
            for field in fields {
                assert!(
                    schema["properties"].get(field).is_some(),
                    "{} reads undeclared time field {field}",
                    declaration.kind.name
                );
            }
        }
        assert_eq!(
            declaration.kind.version,
            match declaration.kind.name.as_str() {
                "polylogue.message" => 7,
                "polylogue.kept_result"
                | "polylogue.delivery"
                | "polylogue.delivery_outcome"
                | "polylogue.request"
                | "polylogue.request_restore" => 7,
                "polylogue.session" => 6,
                "polylogue.document"
                | "polylogue.event"
                | "polylogue.link"
                | "polylogue.background"
                | "polylogue.repository" => 5,
                "polylogue.assertion_restore" => 2,
                "polylogue.parse_progress" => 6,
                "polylogue.input_observation" => 6,
                "polylogue.query_run" => 4,
                "polylogue.blob" | "polylogue.coverage" | "polylogue.fragment" => 3,
                "polylogue.assertion" => 3,
                "polylogue.query" => 2,
                "polylogue.tool_call"
                | "polylogue.historical_restore"
                | "polylogue.attachment_binding"
                | "polylogue.query_run_outcome"
                | "polylogue.prefix_copy"
                | "polylogue.prefix_member" => 1,
                other => panic!("undeclared successor kind/version expectation: {other}"),
            }
        );
        let guarded = if declaration.kind.name == "polylogue.tool_call" {
            "/input"
        } else {
            "/record"
        };
        assert!(
            declaration
                .field_hints
                .iter()
                .any(|hint| hint.field == guarded && hint.class == sinex_model::Class::Guarded)
        );
        assert_eq!(
            declaration.search,
            if matches!(
                declaration.kind.name.as_str(),
                "polylogue.message" | "polylogue.session" | "polylogue.tool_call"
            ) {
                sinex_model::kinds::Search::Words
            } else {
                sinex_model::kinds::Search::None
            }
        );
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
    assert_eq!(declaration.kind.version, 3);
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&payload)
        .unwrap();
    assert_eq!(payload["record"]["facts"]["offset"], 12);
    assert_eq!(payload["record"]["facts"]["end"], 47);
    assert_eq!(payload["record"]["facts"]["record"]["commit"], 9000);
    assert!(payload.get("message").is_none());
    assert_eq!(
        products::decode(
            &crate::kinds::kind_reference("polylogue.fragment").unwrap(),
            payload
        )
        .unwrap(),
        product
    );
}

#[test]
fn message_hash_survives_sinex_round_trip() {
    let facts: polylogue_model::MessageRow = serde_json::from_str(
        r#"{"offset":0,"end":10,"message":{"id":"codex:m","session":"codex:s","role":"assistant","attrs":{"x":1.50},"blocks":[{"type":"text","text":"done"}]}}"#,
    )
    .unwrap();
    let hash = facts.message.content_hash().unwrap();
    let product = Product::Message {
        native_fields: None,
        declared_injections: None,
        declared_attachment_bindings: None,
        reading: reading("input-a", 0),
        order: None,
        projection: None,
        facts: Box::new(facts),
    };
    let wire = serde_json::to_vec(&products::payload(&product, 0).unwrap()).unwrap();
    let returned: serde_json::Value = serde_json::from_slice(&wire).unwrap();
    let Product::Message { facts, .. } = products::decode(
        &crate::kinds::kind_reference("polylogue.message").unwrap(),
        returned,
    )
    .unwrap() else {
        panic!("a message product decodes as a message");
    };
    assert_eq!(facts.message.attrs["x"].to_string(), "1.50");
    assert_eq!(facts.message.content_hash().unwrap(), hash);
}

#[test]
fn an_input_payload_at_an_earlier_kind_version_is_refused() {
    let product = Product::Input {
        rows: Box::new(polylogue_store::input_rows::InputRows {
            acquisition: None,
            input: "input-a".into(),
            format: "codex".into(),
            revision: None,
            extents: vec![polylogue_store::input_rows::Extent {
                chain: 1,
                offset: 0,
                length: 10,
                blob: polylogue_model::BlobHash::of(b"0123456789"),
                blob_offset: 0,
            }],
            observation: None,
        }),
        material: None,
    };
    let current = kind_reference(crate::kinds::INPUT_KIND).unwrap();
    let payload = products::payload(&product, 0).unwrap();
    assert_eq!(
        products::decode(&current, payload.clone()).unwrap(),
        product
    );
    // Version 4 named its one extent `extent`; read through the current
    // shape it would have none.
    let mut earlier = payload;
    let rows = earlier["record"]["rows"].as_object_mut().unwrap();
    let extent = rows.remove("extents").unwrap()[0].clone();
    rows.insert("extent".into(), extent);
    let kind = sinex_model::KindRef {
        version: 4,
        ..current
    };
    let error = products::decode(&kind, earlier.clone()).unwrap_err();
    assert!(
        matches!(&error, crate::Error::Invalid(reason) if reason == "unsupported Polylogue kind polylogue.input_observation@4"),
        "{error}"
    );
    // A spool body of product writes carries no kind version: its rows in
    // the earlier shape are refused, not read as rows without extents.
    let write = json!([{
        "product": earlier["record"],
        "location": null,
        "parents": [],
        "machine": null,
        "output_index": 0
    }]);
    let body = serde_json::to_vec(&write).unwrap();
    let error =
        crate::spool::decode_json::<Vec<polylogue_store::primary::ProductWrite>>(body.as_slice())
            .unwrap_err();
    assert!(
        error.to_string().contains("unknown field `extent`"),
        "{error}"
    );
    let mut current_write = write;
    current_write[0]["product"] = products::payload(&product, 0).unwrap()["record"].clone();
    let body = serde_json::to_vec(&current_write).unwrap();
    assert_eq!(
        crate::spool::decode_json::<Vec<polylogue_store::primary::ProductWrite>>(body.as_slice())
            .unwrap()[0]
            .product,
        product
    );
}

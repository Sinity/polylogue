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
        parse_version: "reader-3".into(),
        start,
        parsed_upto: 100,
        progress: polylogue_model::batch::StepProgress::AtEnd,
    }
}
fn message(input: &str, start: u64) -> Product {
    Product::Message {stated_fields:None,declared_injections:None,declared_attachment_bindings:None,reading:reading(input,start),order:Some(6),projection:None,facts:Box::new(serde_json::from_value(json!({"offset":50,"end":100,"message":{"id":"codex:m","session":"codex:s","role":"assistant","attrs":{"nested":{"citation":["source"]}},"production":{"phase":"final","server_tool_use":{"web_search_requests":0}},"blocks":[{"type":"tool_result","call_id":"native","outcome":{"kind":"ok"},"output":"done","reported_duration_ms":17}]}})).unwrap())}
}

fn interpreted_record(product: &Product) -> Product {
    Product::Interpreted {
        reading: product.reading().unwrap().clone(),
        record: polylogue_store::primary::SourceRecord::of(product)
            .unwrap()
            .unwrap(),
    }
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
        interpreted_record(&first)
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
    malformed["usage"] = json!("invalid usage");
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
    weak.branch_point = None;
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
        let payload = products::payload(&product, 0).unwrap();
        let declaration = declarations("polylogue")
            .into_iter()
            .find(|declaration| declaration.kind.name == "polylogue.link")
            .unwrap();
        assert_eq!(
            declaration
                .evaluate(&payload, sinex_model::Class::Open)
                .unwrap()
                .ts_orig,
            sinex_model::TsOrig::unknown()
        );
        if let Product::Link { facts, .. } = &product {
            assert_eq!(payload["branch_point"], json!(facts.branch_point));
        }
        assert_eq!(
            products::decode(
                &crate::kinds::kind_reference("polylogue.link").unwrap(),
                products::payload(&product, 0).unwrap()
            )
            .unwrap(),
            interpreted_record(&product)
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
                "polylogue.message" => 10,
                "polylogue.delivery"
                | "polylogue.delivery_outcome"
                | "polylogue.request"
                | "polylogue.request_restore" => 11,
                "polylogue.session" | "polylogue.link" | "polylogue.parse_progress" => 8,
                "polylogue.input_observation" => 7,
                "polylogue.document"
                | "polylogue.event"
                | "polylogue.background"
                | "polylogue.repository" => 7,
                "polylogue.query_run" => 9,
                "polylogue.query_run_outcome" => 2,
                "polylogue.blob" | "polylogue.coverage" => 3,
                "polylogue.fragment" => 5,
                "polylogue.annotation" | "polylogue.attachment_binding" => 2,
                "polylogue.annotation_restore" | "polylogue.historical_restore" => 1,
                "polylogue.query"
                | "polylogue.tool_call"
                | "polylogue.prefix_copy"
                | "polylogue.prefix_member" => 3,
                "polylogue.line_resolution" | "polylogue.line_resolution_restore" => 7,
                "polylogue.line_save" | "polylogue.line_save_restore" => 5,
                other => panic!("undeclared successor kind/version expectation: {other}"),
            }
        );
        let guarded = if schema["properties"].get("source").is_some() {
            "/source"
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
        TextCoverage, TextFragment, TextFragmentRecord, TextFragmentSource, TextSourceVersion,
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
        provenance: TextFragmentSource {
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
    assert_eq!(declaration.kind.version, 5);
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&payload)
        .unwrap();
    assert_eq!(payload["source_start"], 12);
    assert_eq!(payload["source_end"], 47);
    assert_eq!(payload["commit"], 9000);
    assert!(payload.get("message").is_none());
    assert_eq!(
        products::decode(
            &crate::kinds::kind_reference("polylogue.fragment").unwrap(),
            payload
        )
        .unwrap(),
        interpreted_record(&product)
    );
}

#[test]
fn sinex_message_keeps_numbers_and_source_identity() {
    let facts: polylogue_model::MessageRow = serde_json::from_str(
        r#"{"offset":0,"end":10,"message":{"id":"codex:m","session":"codex:s","role":"assistant","attrs":{"x":1.50},"blocks":[{"type":"text","text":"done"}]}}"#,
    )
    .unwrap();
    let product = Product::Message {
        stated_fields: None,
        declared_injections: None,
        declared_attachment_bindings: None,
        reading: reading("input-a", 0),
        order: None,
        projection: None,
        facts: Box::new(facts),
    };
    let wire = serde_json::to_vec(&products::payload(&product, 0).unwrap()).unwrap();
    let returned: serde_json::Value = serde_json::from_slice(&wire).unwrap();
    assert_eq!(returned["attrs"]["x"].to_string(), "1.50");
    assert_eq!(
        products::decode(
            &crate::kinds::kind_reference("polylogue.message").unwrap(),
            returned
        )
        .unwrap(),
        interpreted_record(&product),
    );
}

#[test]
fn an_input_payload_at_an_earlier_kind_version_is_refused() {
    let product = Product::Input {
        rows: Box::new(polylogue_store::input_rows::InputRows {
            capture_read: None,
            input: "input-a".into(),
            origin: "codex".into(),
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
}

#[test]
fn current_declarations_keep_every_source7_kind_version() {
    let old: sinex_model::SourceDeclaration =
        serde_json::from_str(include_str!("tests/source7.json")).unwrap();
    let mut changed = Vec::new();
    for current in declarations("polylogue") {
        if let Some(previous) = old
            .kinds
            .iter()
            .find(|previous| previous.kind == current.kind)
            && previous.definition_hash() != current.definition_hash()
        {
            let previous = serde_json::to_value(previous).unwrap();
            let current = serde_json::to_value(&current).unwrap();
            let fields = current
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, value)| previous.get(*key) != Some(*value))
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            changed.push((current["kind"].clone(), fields));
        }
    }
    assert!(
        changed.is_empty(),
        "immutable declaration changes: {changed:?}"
    );
}

#[test]
fn markdown_link_roles_versioned_only_the_changed_authored_declarations() {
    let old: serde_json::Value =
        serde_json::from_str(include_str!("tests/source11-kind-hashes.json")).unwrap();
    let previous = old["kinds"].as_array().unwrap();
    let changed = [
        "polylogue.assertion",
        "polylogue.delivery",
        "polylogue.delivery_outcome",
        "polylogue.request",
        "polylogue.request_restore",
        "polylogue.line_resolution",
        "polylogue.line_resolution_restore",
        "polylogue.line_save",
        "polylogue.line_save_restore",
        "polylogue.assertion_restore",
    ];
    let source12: serde_json::Value =
        serde_json::from_str(include_str!("tests/source12-kind-hashes.json")).unwrap();
    let introduced = source12["kinds"].as_array().unwrap();
    assert_eq!(introduced.len(), previous.len());
    assert_eq!(
        source12["source"]["version"].as_u64().unwrap(),
        old["source"]["version"].as_u64().unwrap() + 1
    );
    for introduction in introduced {
        let name = introduction["name"].as_str().unwrap();
        let previous = previous.iter().find(|kind| kind["name"] == name).unwrap();
        let version = previous["version"].as_u64().unwrap();
        let hash = previous["definition_hash"].as_str().unwrap();
        let bump = u64::from(changed.contains(&name));
        assert_eq!(
            introduction["version"].as_u64().unwrap(),
            version + bump,
            "{name}"
        );
        assert_eq!(
            introduction["definition_hash"].as_str().unwrap() != hash,
            bump == 1,
            "{name}"
        );
    }
}

#[test]
fn trace_view_versioned_only_its_three_reachable_declarations() {
    let source12: serde_json::Value =
        serde_json::from_str(include_str!("tests/source12-kind-hashes.json")).unwrap();
    let source13: serde_json::Value =
        serde_json::from_str(include_str!("tests/source13-kind-hashes.json")).unwrap();
    let changed = [
        "polylogue.line_resolution",
        "polylogue.line_resolution_restore",
        "polylogue.query_run",
    ];
    let old = source12["kinds"].as_array().unwrap();
    let new = source13["kinds"].as_array().unwrap();
    assert_eq!(new.len(), old.len());
    for kind in new {
        let name = kind["name"].as_str().unwrap();
        let old = old.iter().find(|old| old["name"] == name).unwrap();
        let bump = u64::from(changed.contains(&name));
        assert_eq!(
            kind["version"].as_u64().unwrap(),
            old["version"].as_u64().unwrap() + bump,
            "{name}"
        );
        assert_eq!(
            kind["definition_hash"] != old["definition_hash"],
            bump == 1,
            "{name}"
        );
    }
}

/// Every kind of `current` keeps its version from `previous` when its
/// definition is unchanged and is raised by one when it changed; a kind new in
/// `current` starts at version 1, and `retired` kinds are gone.
fn assert_versioned_transition(
    previous: &serde_json::Value,
    current: &[(String, u64, String)],
    introduced: &[&str],
    retired: &[&str],
) {
    let old = previous["kinds"].as_array().unwrap();
    assert_eq!(current.len(), old.len());
    let mut wrong = Vec::new();
    for (name, version, hash) in current {
        let expected = match old.iter().find(|old| old["name"] == name.as_str()) {
            None => {
                assert!(introduced.contains(&name.as_str()), "{name}");
                1
            }
            Some(old) => {
                old["version"].as_u64().unwrap()
                    + u64::from(old["definition_hash"] != hash.as_str())
            }
        };
        if *version != expected {
            wrong.push(format!("{name}: {version} should be {expected}"));
        }
    }
    for name in retired {
        assert!(old.iter().any(|old| old["name"] == *name), "{name}");
        assert!(current.iter().all(|(kind, ..)| kind != name), "{name}");
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

fn recorded_kinds(file: &serde_json::Value) -> Vec<(String, u64, String)> {
    file["kinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kind| {
            (
                kind["name"].as_str().unwrap().to_owned(),
                kind["version"].as_u64().unwrap(),
                kind["definition_hash"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

/// Source 14 gave annotations their own kind names and raised every other
/// declaration whose definition changed.
#[test]
fn annotation_names_versioned_every_changed_declaration() {
    let source13: serde_json::Value =
        serde_json::from_str(include_str!("tests/source13-kind-hashes.json")).unwrap();
    let source14: serde_json::Value =
        serde_json::from_str(include_str!("tests/source14-kind-hashes.json")).unwrap();
    assert_versioned_transition(
        &source13,
        &recorded_kinds(&source14),
        &["polylogue.annotation", "polylogue.annotation_restore"],
        &["polylogue.assertion", "polylogue.assertion_restore"],
    );
}

/// Source 15 renamed the parse version, time accuracy and other fields and
/// raised every declaration whose definition changed.
#[test]
fn field_names_version_every_changed_declaration() {
    let previous: serde_json::Value =
        serde_json::from_str(include_str!("tests/source14-kind-hashes.json")).unwrap();
    assert_eq!(
        u64::from(crate::source_reference().version),
        previous["source"]["version"].as_u64().unwrap() + 1
    );
    let current: Vec<_> = declarations("polylogue")
        .iter()
        .map(|kind| {
            (
                kind.kind.name.clone(),
                kind.kind.version,
                kind.definition_hash(),
            )
        })
        .collect();
    assert_versioned_transition(&previous, &current, &[], &[]);
}

#[test]
fn historical_annotation_positions_keep_the_published_execution_string_schema() {
    // This is the at field from the preserved SOURCE12 declaration. SOURCE13
    // changed View only; at's nullable-string codec is still unchanged.
    let old = serde_json::json!({
        "description":"The message a context reading is taken at; the latest when absent.",
        "anyOf":[{"$ref":"#/$defs/MessageId"},{"type":"null"}]
    });
    for kind in declarations("polylogue") {
        let schema = kind.payload_schema.as_ref().unwrap();
        if let Some(options) = schema["$defs"].get("ReadOptions") {
            assert_eq!(options["properties"]["at"], old, "{}", kind.kind.name);
            assert_eq!(
                schema["$defs"]["MessageId"],
                serde_json::json!({"type":"string"})
            );
            assert!(
                schema["$defs"].get("ReadAt").is_none(),
                "{}",
                kind.kind.name
            );
        }
    }
    assert_eq!(crate::source_reference().version, 15);
}

use super::*;
use serde_json::json;
use std::collections::BTreeMap;

fn historical_query() -> Value {
    let normal_form = json!({
        "source": {"kind":"unit","unit":"sessions","condition":null},
        "attach":[], "stages":[],
    });
    let question =
        query_language::identity::question_id_from_normal_form(&normal_form, &BTreeMap::new())
            .unwrap();
    let mut syntax = normal_form.clone();
    syntax["source"]["implicit"] = false.into();
    json!({
        "record_key":"query-fixture", "question":question,
        "record": {"product":"query","object":{
            "id":question,"normal_form":normal_form,"parameters":{},
            "identity_ast":syntax,"source_outputs":[],"definitions":[],
            "dependencies":[],"edges":[],
        }},
    })
}

#[test]
fn exact_historical_source_evolves_without_reidentifying_the_question() {
    let source = historical_query();
    let declaration = historical_declarations()
        .iter()
        .find(|declaration| declaration.kind.name == "polylogue.query")
        .unwrap();
    let validator =
        jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap()).unwrap();
    validator.validate(&source).unwrap();
    let Product::Query { object } = decode_source(
        &declaration.kind,
        &declaration.definition_hash(),
        source.clone(),
    )
    .unwrap() else {
        panic!("query source returned another product");
    };
    object.validate_evolution().unwrap();
    assert_eq!(
        serde_json::to_value(object.id).unwrap(),
        source["record"]["object"]["id"]
    );
    assert_eq!(
        object.normal_form,
        source["record"]["object"]["normal_form"]
    );
    assert_eq!(
        object.historical_source.as_ref().unwrap().record,
        source["record"]["object"]
    );
    assert_eq!(
        serde_json::to_value(object.executable_ast().unwrap()).unwrap()["transforms"],
        json!([])
    );
    assert!(decode("polylogue.query", source).is_err());
}

#[test]
fn source_dispatch_requires_the_complete_frozen_definition() {
    for declaration in historical_declarations() {
        assert!(decode_source(&declaration.kind, &"0".repeat(64), json!({})).is_err());
        let mut unrecognized = declaration.kind.clone();
        unrecognized.version += 1000;
        assert!(decode_source(&unrecognized, &declaration.definition_hash(), json!({})).is_err());
    }
    let declaration = crate::kinds::declarations("polylogue")
        .into_iter()
        .find(|declaration| declaration.kind.name == "polylogue.query")
        .unwrap();
    assert_eq!(declaration.kind.version, 2);
    assert!(
        decode_source(
            &declaration.kind,
            &declaration.definition_hash(),
            historical_query()
        )
        .is_err()
    );
}

#[test]
fn historical_declarations_are_read_only_and_current_writes_select_successors() {
    let current = crate::kinds::declarations("polylogue");
    for historical in historical_declarations() {
        assert!(!current.contains(historical));
        assert_ne!(
            crate::kinds::kind_reference(&historical.kind.name).unwrap(),
            historical.kind
        );
    }
    assert_eq!(crate::source_reference().version, 6);
}

#[test]
fn historical_run_source_keeps_original_hashes_and_uses_current_executable_syntax() {
    let run = crate::tests::named::phases()
        .into_iter()
        .find_map(|(_, product)| match product {
            Product::QueryRun { run } => Some(run),
            _ => None,
        })
        .unwrap();
    let object = historical_query()["record"]["object"].clone();
    let mut source = serde_json::to_value(run).unwrap();
    source["question"] = object["id"].clone();
    source["binding"]["execution"]["replay"]["bound_ast"] = object["identity_ast"].clone();
    source["binding"]["execution"]["replay"]["context"]["question"] = object["id"].clone();
    source["binding"]["execution"]["replay"]["context"]["versions"]["catalog"] =
        "1".repeat(64).into();
    source["binding"]["execution"]["accepted_request"] = Value::Null;
    let binding_hash = original_hash(
        b"polylogue-execution-binding-v1\0",
        &source["binding"]["execution"],
    );
    let run_hash = original_hash(b"polylogue-query-run-begin-v1\0", &source);
    let declaration = historical_declarations()
        .iter()
        .find(|declaration| declaration.kind.name == "polylogue.query_run")
        .unwrap();
    let payload = json!({"record_key":"synthetic-original-run","record":{"product":"query_run","run":source}});
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&payload)
        .unwrap();
    let Product::QueryRun { run } =
        decode_source(&declaration.kind, &declaration.definition_hash(), payload).unwrap()
    else {
        panic!("run source returned another product");
    };
    assert_eq!(run.binding_hash().unwrap(), binding_hash);
    assert_eq!(run.record_hash().unwrap(), run_hash);
    assert_eq!(run.historical_source.as_ref().unwrap().record, source);
    let execution = run.binding.original().unwrap();
    assert_eq!(
        serde_json::to_value(&execution.replay.bound_ast).unwrap()["transforms"],
        json!([])
    );
    assert_ne!(execution.replay.context.versions["catalog"], "1".repeat(64));
}

fn original_hash(domain: &[u8], source: &Value) -> polylogue_model::BlobHash {
    fn ordered(value: &mut Value) {
        match value {
            Value::Object(fields) => {
                for child in fields.values_mut() {
                    ordered(child);
                }
                fields.sort_keys();
            }
            Value::Array(values) => {
                for child in values {
                    ordered(child);
                }
            }
            _ => (),
        }
    }
    let mut value = source.clone();
    ordered(&mut value);
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&serde_json::to_vec(&value).unwrap());
    polylogue_model::BlobHash::parse(&hasher.finalize().to_hex()).unwrap()
}

#[test]
fn source_three_retired_continuation_preserves_source_position_without_current_replay() {
    use polylogue_model::query_objects::{RetiredReadInvocation, RunBinding};
    let mut source = crate::tests::named::phases()
        .into_iter()
        .find_map(|(_, product)| match product {
            Product::QueryRun { run } => Some(serde_json::to_value(run).unwrap()),
            _ => None,
        })
        .unwrap();
    let root = "00000000-0000-0000-0000-000000000001";
    source["binding"] = json!({
        "kind":"continuation", "root":root, "binding_hash":"a".repeat(64),
        "seek":{"direction":"after","anchor":{"selection":{"row":3,"record":{"kind":"message","ordinal":2}},"item":1,"character":7,"item_hash":"b".repeat(64)}},
        "accepted_request":null,
    });
    let record_hash = original_hash(b"polylogue-query-run-begin-v1\0", &source);
    let payload = json!({"record_key":"synthetic-retired-continuation","record":{"product":"query_run","run":source},"binding_hash":"a".repeat(64),"run_hash":record_hash});
    let declaration = historical_declarations()
        .iter()
        .find(|d| d.kind.name == "polylogue.query_run" && d.kind.version == 2)
        .unwrap();
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&payload)
        .unwrap();
    let Product::QueryRun { run } = decode_source(
        &declaration.kind,
        &declaration.definition_hash(),
        payload.clone(),
    )
    .unwrap() else {
        panic!("wrong product")
    };
    assert!(
        matches!(&run.binding,RunBinding::RetiredRead {invocation:RetiredReadInvocation::Continuation {root:original,..}} if original.as_str()==root)
    );
    assert!(run.binding.original().is_none());
    assert_eq!(run.record_hash().unwrap(), record_hash);
    assert_eq!(
        run.historical_source.as_ref().unwrap().record,
        payload["record"]["run"]
    );
    assert!(decode("polylogue.query_run", payload.clone()).is_err());
    let mut wrong = payload;
    wrong["run_hash"] = json!("0".repeat(64));
    assert!(decode_source(&declaration.kind, &declaration.definition_hash(), wrong).is_err());
}

#[test]
fn parse_progress_v5_retains_exact_source_fields_and_unknown_attempt() {
    let old = progress_v5::declaration().clone();
    let mut successor = crate::kinds::declarations("polylogue")
        .into_iter()
        .find(|d| d.kind.name == old.kind.name)
        .unwrap();
    assert_eq!(successor.kind.version, 6);
    successor.kind.version = 5;
    let schema = successor.payload_schema.as_mut().unwrap();
    schema["properties"]["record"]["oneOf"][0]["properties"]
        .as_object_mut()
        .unwrap()
        .remove("stopped_attempt");
    let defs = schema["$defs"].clone();
    schema["$defs"] = super::super::reachable_definitions(&schema["properties"]["record"], &defs);
    assert_eq!(old, successor);
    let product = Product::ParseProgress {
        reading: polylogue_store::primary::Reading {
            source_length: Some(1),
            delivery: None,
            input: polylogue_store::InputKey {
                id: "synthetic".into(),
                format: "claude-code".into(),
                arrival_rank: 0,
            },
            chain: 1,
            reading_version: "original-reading".into(),
            order_from: None,
            repeats_whole_conversations: false,
            start: 0,
            parsed_upto: 0,
            progress: polylogue_model::batch::StepProgress::Failed {
                error: "synthetic failure".into(),
            },
        },
        counts: Default::default(),
        reset: false,
        comparison: None,
        checkpoint: None,
        stopped_attempt: None,
        semantic_event_ids: None,
    };
    let payload = crate::products::payload(&product, 0).unwrap();
    let restored = decode_source(&old.kind, &old.definition_hash(), payload.clone()).unwrap();
    assert_eq!(restored, product);
    let mut extension = payload.clone();
    extension["record"]["stopped_attempt"] =
        serde_json::to_value(polylogue_model::batch::StoppedAttempt {
            executable: polylogue_model::BlobHash::of(b"not stated by version five"),
            parser_settings: None,
        })
        .unwrap();
    assert_eq!(
        decode_source(&old.kind, &old.definition_hash(), extension).unwrap(),
        product
    );
    assert!(decode_source(&old.kind, &"0".repeat(64), payload).is_err());
}

#[test]
fn reconstructed_progress_candidate_matches_the_independent_old_type_snapshot() {
    let candidate: sinex_model::KindDeclaration =
        serde_json::from_str(include_str!("progress_v5.candidate.json")).unwrap();
    assert_eq!(candidate, *progress_v5::declaration());
}

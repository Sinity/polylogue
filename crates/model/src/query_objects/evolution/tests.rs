use super::*;
use query_language::catalog::{Field, FieldKind, Unit, UnknownRule};

fn scalar_value(s: &str) -> Value {
    json!({"kind":"scalar","text":s,"quoted":false})
}
fn cond(k: Value) -> Value {
    json!({"span":{"start":3,"end":20},"kind":k})
}
fn ast(condition: Value, stages: Vec<Value>) -> Value {
    json!({"source":{"kind":"unit","unit":"rows","implicit":false,"condition":condition},"attach":[],"stages":stages})
}
fn stage(k: Value) -> Value {
    json!({"span":{"start":25,"end":50},"kind":k})
}
fn object(ast: Value) -> Value {
    json!({"id":vec![7;32],"normal_form":{"custody":"original"},"parameters":{"x":"text"},"identity_ast":ast,"source_outputs":[],"definitions":[],"dependencies":[vec![8;32]],"edges":[{"how":"result","result":"synthetic-kept-result"}]})
}
fn catalog() -> Catalog {
    let mut c = Catalog::new("rows");
    c.units.push(Unit {
        name: "rows".into(),
        fields: vec![
            Field {
                name: "n".into(),
                kind: FieldKind::Number,
                ..Field::default()
            },
            Field {
                name: "opaque".into(),
                kind: FieldKind::Exact,
                ..Field::default()
            },
            Field {
                name: "elapsed".into(),
                kind: FieldKind::Duration,
                ..Field::default()
            },
            Field {
                name: "closed".into(),
                kind: FieldKind::Exact,
                values: vec!["Alpha".into()],
                unknown: UnknownRule::Word("missing".into()),
                ..Field::default()
            },
        ],
        ..Unit::default()
    });
    c
}
#[test]
fn conversion_preserves_identity_graph_and_nested_query_syntax() {
    let nested = ast(
        cond(json!({"kind":"field_exists","operand":field("n")})),
        vec![],
    );
    let before = object(ast(
        cond(json!({"kind":"in","operand":field("n"),"query":nested})),
        vec![
            stage(json!({"kind":"group","keys":[field("a.b")]})),
            stage(json!({"kind":"aggregate","aggregates":[{"function":"sum","fields":["a.b"]}]})),
            stage(json!({"kind":"fields","fields":["sum_a_b"]})),
        ],
    ));
    let after = query_object(&before).unwrap();
    for k in ["id", "normal_form", "parameters", "dependencies", "edges"] {
        assert_eq!(after[k], before[k]);
    }
    let q = &after["body"]["query"];
    assert!(q.get("stages").is_none());
    assert_eq!(
        q["transforms"][0]["kind"]["keys"][0],
        column(field("a.b"), "a_b".into())
    );
    assert_eq!(
        q["transforms"][1]["kind"]["aggregates"][0]["name"],
        "sum_a_b"
    );
    assert_eq!(
        q["source"]["condition"]["kind"]["query"]["source"]["condition"]["kind"]["value"]["kind"],
        "known"
    );
}
#[test]
fn run_conversion_changes_executable_catalog_only() {
    let clock = query_language::time::Clock::new(
        jiff::Timestamp::from_millisecond(0).unwrap(),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get("UTC").unwrap(),
            jiff::civil::time(4, 0, 0, 0),
        )
        .unwrap(),
    );
    let before = json!({
        "id":"00000000-0000-0000-0000-000000000001","question":vec![1;32],
        "at":crate::Timestamp::from_unix_ms(0).unwrap(),
        "read_at":{"generation":"synthetic","index":1,"archive":1},"machine":"synthetic",
        "authorship":{"author":"unattributed","via":"mcp"},
        "binding":{"kind":"original","execution":{
            "query_options":{"direct_reference":null,"fields":[],"self_session":null,"embeddings":false,"why":false,"ranking_identity":null},
            "presentation":{"kind":"query"},"accepted_request":{"untouched":"n:001.0"},
            "replay":{"bound_ast":ast(Value::Null,vec![]),"source_outputs":[],"definitions":[],
                "context":{"question":vec![1;32],"clock":clock.wire(),"resolved_times":[],"random_seed":7,"read_positions":{},"versions":{"catalog":"old","other":"kept"}},
                "scope_fingerprint":null,"original_text":"n:001.0"}
        }}
    });
    let after = query_run(&before, "current").unwrap();
    assert_eq!(
        after["binding"]["execution"]["replay"]["context"]["versions"]["catalog"],
        "current"
    );
    assert_eq!(
        after["binding"]["execution"]["accepted_request"],
        before["binding"]["execution"]["accepted_request"]
    );
    assert_eq!(
        before["binding"]["execution"]["replay"]["context"]["versions"]["catalog"],
        "old"
    );
}
#[test]
fn oracle_canonicalizes_by_historical_type_and_preserves_opaque_text() {
    let q = ast(
        cond(json!({"kind":"and","conditions":[
            cond(json!({"kind":"match","field":"n","value":scalar_value("001.200")})),
            cond(json!({"kind":"match","field":"opaque","value":scalar_value("001.200")})),
            cond(json!({"kind":"match","field":"elapsed","value":scalar_value("1s")})),
            cond(json!({"kind":"match","field":"closed","value":scalar_value("aLPHA")})),
        ]})),
        vec![],
    );
    let actual = legacy_normal_form(&q, &catalog()).unwrap();
    let cs = actual["source"]["condition"]["kind"]["conditions"]
        .as_array()
        .unwrap();
    let text = |name: &str| {
        cs.iter().find(|c| c["kind"]["field"] == name).unwrap()["kind"]["value"]["text"].clone()
    };
    assert_eq!(text("n"), "12e-1");
    assert_eq!(text("opaque"), "001.200");
    assert_eq!(text("elapsed"), "1000000000ns");
    assert_eq!(text("closed"), "Alpha");
    assert!(actual["source"].get("implicit").is_none());
    assert!(!actual.to_string().contains("span"));
}
#[test]
fn oracle_does_not_derive_original_identity_from_current_ast() {
    let a = ast(
        cond(json!({"kind":"compare","operand":field("n"),"op":"eq","value":scalar_value("1.00")})),
        vec![],
    );
    let b = ast(
        cond(json!({"kind":"compare","operand":field("n"),"op":"eq","value":scalar_value("2.00")})),
        vec![],
    );
    assert_ne!(
        legacy_normal_form(&a, &catalog()).unwrap(),
        legacy_normal_form(&b, &catalog()).unwrap()
    );
    assert!(
        legacy_normal_form(&a, &catalog()).unwrap()["source"]["condition"]["kind"]
            .get("operand")
            .is_some()
    );
}
#[test]
fn refuses_semantically_unmapped_old_relations() {
    for kind in ["matches", "temporal"] {
        assert!(
            query_object(&object(ast(Value::Null, vec![stage(json!({"kind":kind}))]))).is_err()
        );
    }
}

#[test]
fn exists_matches_preserves_predecessor_child_columns() {
    let source = cond(json!({"kind":"exists","unit":"messages","condition":null}));
    let mut input = object(ast(source, vec![stage(json!({"kind":"matches"}))]));
    input["identity_ast"]["source"]["unit"] = json!("sessions");
    let result = query_object(&input).unwrap();
    let ts = result["body"]["query"]["transforms"].as_array().unwrap();
    assert_eq!(ts.len(), 2);
    let cs = ts[1]["kind"]["columns"].as_array().unwrap();
    assert_eq!(cs[0], column(field("match"), "match".into()));
    assert!(
        cs.iter()
            .any(|c| c == &column(field("step.written_by"), "written_by".into()))
    );
    assert!(!cs.iter().any(|c| c["name"] == "position"));
}
#[test]
fn sequence_matches_reconstructs_tuple_and_timeline_gaps() {
    let source = cond(
        json!({"kind":"sequence","unit":"tool_calls","all":false,"steps":[cond(json!({"kind":"match","field":"command","value":scalar_value("one")})),cond(json!({"kind":"match","field":"command","value":scalar_value("two")}))],"gaps":["later"]}),
    );
    let result =
        query_object(&object(ast(source, vec![stage(json!({"kind":"matches"}))]))).unwrap();
    let ts = result["body"]["query"]["transforms"].as_array().unwrap();
    assert_eq!(ts.len(), 5);
    let aggs = ts[2]["kind"]["aggregates"].as_array().unwrap();
    assert_eq!(aggs.len(), 10);
    assert_eq!(
        aggs[4]["arguments"][0]["arguments"][2],
        field("step.position")
    );
    let cs = ts[3]["kind"]["columns"].as_array().unwrap();
    assert_eq!(cs.len(), 12);
    assert_eq!(cs[10]["name"], "gap_steps");
    assert_eq!(
        cs[10]["expression"]["left"]["left"],
        field("__format1_step2_position")
    );
    assert_eq!(cs[10]["expression"]["right"], numeric(1));
    assert_eq!(cs[11]["name"], "gap_time");
    assert!(!cs.iter().any(|c| c["name"] == "count"));
}

fn retired_run_source(binding: Value) -> Value {
    json!({
        "id":"00000000-0000-0000-0000-000000000002", "question":vec![1;32],
        "at":crate::Timestamp::from_unix_ms(0).unwrap(),
        "read_at":{"generation":"synthetic","index":1,"archive":1},
        "machine":"synthetic", "authorship":{"author":"unattributed","via":"mcp"},
        "binding":binding,
    })
}

#[test]
fn retired_original_and_continuation_keep_exact_invocation_and_root_custody() {
    use crate::query_objects::{QueryRun, RetiredReadInvocation, RunBinding};
    let execution = json!({"retained":"original-source-execution","presentation":{"kind":"read"}});
    let original = retired_run_source(json!({"kind":"original","execution":execution}));
    let root = QueryRun::from_retired_read_v2(original.clone()).unwrap();
    assert!(root.binding.original().is_none());
    assert!(matches!(
        root.binding,
        RunBinding::RetiredRead {
            invocation: RetiredReadInvocation::Original { .. }
        }
    ));
    let mut child_source = retired_run_source(json!({
        "kind":"continuation", "root":root.id,
        "binding_hash":root.binding_hash().unwrap(),
        "seek":{"direction":"after","anchor":{"selection":{"row":3,"record":{"kind":"message","ordinal":2}},"item":1,"character":7,"item_hash":"a".repeat(64)}},
        "accepted_request":{"operation":"read","arguments":{"cursor":"retired-source-token"}},
    }));
    child_source["id"] = json!("00000000-0000-0000-0000-000000000003");
    let child = QueryRun::from_retired_read_v2(child_source.clone()).unwrap();
    assert_eq!(child.binding_hash().unwrap(), root.binding_hash().unwrap());
    assert_eq!(
        child.record_hash().unwrap(),
        super::super::metadata_hash(b"polylogue-query-run-begin-v1\0", &child_source).unwrap()
    );
    assert_ne!(child.record_hash().unwrap(), root.record_hash().unwrap());
    assert_eq!(
        child.historical_source.as_ref().unwrap().record,
        child_source
    );
    assert!(child.binding.original().is_none());
    let RunBinding::RetiredRead {
        invocation:
            RetiredReadInvocation::Continuation {
                root: root_id,
                seek,
                ..
            },
    } = &child.binding
    else {
        panic!("continuation became an original")
    };
    assert_eq!(root_id, &root.id);
    assert_eq!(seek, &child_source["binding"]["seek"]);
    let displayed = serde_json::to_value(&child).unwrap();
    assert_eq!(displayed["binding"]["kind"], "retired_read");
    assert_eq!(
        displayed["historical_source"]["record"]["binding"]["accepted_request"],
        child_source["binding"]["accepted_request"]
    );
    let mut tampered = child.clone();
    if let RunBinding::RetiredRead {
        invocation: RetiredReadInvocation::Continuation { seek, .. },
    } = &mut tampered.binding
    {
        seek["anchor"]["character"] = json!(8);
    }
    assert!(tampered.validate_evolution().is_err());
    let mut forged = child;
    forged.historical_source = None;
    assert!(forged.validate_evolution().is_err());
}

#[test]
fn retired_continuation_rejects_self_root_and_other_binding_shapes() {
    use crate::query_objects::QueryRun;
    let source = retired_run_source(
        json!({"kind":"continuation","root":"00000000-0000-0000-0000-000000000002","binding_hash":"a".repeat(64),"seek":{}}),
    );
    assert!(QueryRun::from_retired_read_v2(source).is_err());
    assert!(
        QueryRun::from_retired_read_v2(retired_run_source(
            json!({"kind":"original_object","execution":{}})
        ))
        .is_err()
    );
}

#[test]
fn grouped_saved_outputs_evolve_row_domain_without_changing_original_declarations() {
    let question = vec![2u8; 32];
    let name = "02".repeat(32);
    let saved = json!({"question":question,"normal_form":{},"parameters":{},"identity_ast":ast(Value::Null,vec![stage(json!({"kind":"group","keys":[field("n")]}))]),"source_outputs":[]});
    let field_json = |name: &str, kind: &str| json!({"name":name,"kind":kind,"values":[],"unknown":"unknown","help":"","read_as":"","unit":null});
    let output = json!({"source":{"kind":"saved","name":name,"arguments":[]},"schema":{"unit":"rows","fields":[field_json("n","number"),field_json("count","integer")],"main_time":null}});
    let mut before = object(
        json!({"source":{"kind":"saved","name":name,"arguments":[],"condition":null,"output":"known"},"attach":[],"stages":[]}),
    );
    before["definitions"] = json!([saved]);
    before["source_outputs"] = json!([output]);
    let after = query_object(&before).unwrap();
    assert!(after["source_outputs"][0]["schema"]["unit"].is_null());
    assert_eq!(before["source_outputs"][0]["schema"]["unit"], "rows");
    assert_eq!(
        after["source_outputs"][0]["schema"]["fields"],
        before["source_outputs"][0]["schema"]["fields"]
    );
    // Original proof uses its own captured schema, independently of the new
    // grouped output's absent native row domain.
    let original = legacy_definition_normal_form(&before, &catalog(), &[]).unwrap();
    assert_eq!(original["source"]["name"], name);
}

#[test]
fn notes_source_and_nested_scope_use_exact_imported_rows() {
    let mut original = ast(
        Value::Null,
        vec![stage(
            json!({"kind":"scope","unit":"notes","condition":null}),
        )],
    );
    original["source"]["unit"] = json!("notes");
    let after = query_object(&object(original)).unwrap();
    assert_eq!(after["body"]["query"]["source"]["unit"], "imported_notes");
    assert_eq!(
        after["body"]["query"]["transforms"][0]["kind"]["unit"],
        "imported_notes"
    );
}

#[test]
fn note_reference_rewrite_preserves_root_currency_revision_prefix_and_default_overrides() {
    for reference in ["note:synthetic-root", "note:synthetic-root@REV%_\\suffix"] {
        let mut syntax = ast(
            cond(json!({"kind":"reference","reference":reference})),
            vec![],
        );
        syntax["source"]["unit"] = json!("notes");
        let after = query_object(&object(syntax)).unwrap();
        let conditions = after["body"]["query"]["source"]["condition"]["kind"]["conditions"]
            .as_array()
            .unwrap();
        assert_eq!(conditions[0]["kind"]["left"], field("first_note"));
        assert_eq!(
            conditions[0]["kind"]["right"]["value"]["text"],
            "synthetic-root"
        );
        if reference.contains('@') {
            assert_eq!(conditions[1]["kind"]["field"], "id");
            assert_eq!(conditions[1]["kind"]["value"]["text"], "REV%_\\suffix");
        } else {
            assert_eq!(conditions[1]["kind"]["field"], "current");
            assert_eq!(conditions[1]["kind"]["value"]["text"], "true");
        }
        for (condition, name) in conditions[2..].iter().zip(["kind", "current", "withdrawn"]) {
            assert_eq!(condition["kind"]["field"], name);
            assert_eq!(condition["kind"]["value"]["kind"], "any");
        }
    }
    let invalid = ast(
        cond(json!({"kind":"reference","reference":"note:synthetic-root"})),
        vec![],
    );
    assert!(query_object(&object(invalid)).is_err());
}

#[test]
fn never_executable_and_retired_key_questions_keep_explicit_source_only_bodies() {
    use crate::query_objects::{QueryObject, QueryUnavailable, QuestionBody};
    for (unit, condition, expected) in [
        (
            "rows",
            cond(
                json!({"kind":"overlaps","from":scalar_value("2026-01-01"),"to":scalar_value("2026-01-02")}),
            ),
            QueryUnavailable::SpanOverlap,
        ),
        (
            "rows",
            cond(
                json!({"kind":"field_exists","operand":{"kind":"function","name":"day","arguments":[field("elapsed")]}}),
            ),
            QueryUnavailable::ComputedExistence,
        ),
    ] {
        let mut syntax = ast(condition, vec![]);
        syntax["source"]["unit"] = json!(unit);
        let mut source = object(syntax);
        source["dependencies"] = json!([]);
        source["edges"] = json!([]);
        source["parameters"] = json!({});
        let form = legacy_definition_normal_form(&source, &catalog(), &[]).unwrap();
        source["id"] = serde_json::to_value(
            query_language::identity::question_id_from_normal_form(&form, &Default::default())
                .unwrap(),
        )
        .unwrap();
        source["normal_form"] = form;
        let question = QueryObject::from_structured_v1(source.clone()).unwrap();
        assert!(matches!(&question.body,QuestionBody::Unavailable {reason} if reason==&expected));
        assert_eq!(question.executable_ast().unwrap_err(), expected);
        assert_eq!(question.historical_source.as_ref().unwrap().record, source);
        let mut forged = question.clone();
        forged.historical_source = None;
        assert!(forged.validate_evolution().is_err());
        let mut fictional = question;
        fictional.body = QuestionBody::Executable {
            query: query_language::ast::Query {
                source: query_language::ast::Source::Unit {
                    unit: "rows".into(),
                    implicit: false,
                    condition: None,
                },
                attach: vec![],
                transforms: vec![],
            },
        };
        assert!(fictional.validate_evolution().is_err());
    }
    let mut retired = ast(
        cond(json!({"kind":"exists","unit":"tool_calls","condition":null})),
        vec![stage(json!({"kind":"matches"}))],
    );
    retired["source"]["unit"] = json!("blocks");
    let result = query_object(&object(retired)).unwrap();
    assert_eq!(result["body"]["kind"], "unavailable");
    assert_eq!(result["body"]["reason"]["kind"], "retired_storage_key");
}

#[test]
fn original_refusal_classes_and_dependency_unavailability_are_explicit() {
    let nested = ast(Value::Null, vec![]);
    let computed = json!({"kind":"function","name":"day","arguments":[field("elapsed")]});
    for kind in ["since", "until"] {
        let syntax = ast(
            cond(json!({"kind":kind,"operand":computed,"value":scalar_value("2026-01-01")})),
            vec![],
        );
        let upgraded = query_object(&object(syntax)).unwrap();
        assert_eq!(upgraded["body"]["reason"]["kind"], "computed_time_bound");
    }
    let temporal = ast(
        Value::Null,
        vec![stage(
            json!({"kind":"temporal","relation":"near","query":nested,"at":"when","within":null,"all":false}),
        )],
    );
    assert_eq!(
        query_object(&object(temporal)).unwrap()["body"]["reason"]["kind"],
        "temporal_stage"
    );
    for (source, tag) in [
        (
            json!({"kind":"walk","direction":"trace","reference":"session:synthetic"}),
            "walk",
        ),
        (json!({"kind":"recall","time":null,"window":null}), "recall"),
        (
            json!({"kind":"recall_last","unit":"sessions","condition":null}),
            "recall_last",
        ),
        (
            json!({"kind":"recall_mark","number":1,"window":null}),
            "recall_mark",
        ),
    ] {
        let mut syntax = ast(Value::Null, vec![]);
        syntax["source"] = source;
        let upgraded = query_object(&object(syntax)).unwrap();
        assert_eq!(upgraded["body"]["reason"]["kind"], "native_source");
        assert_eq!(upgraded["body"]["reason"]["source_kind"], tag);
    }
    let child_id = json!(vec![8; 32]);
    let child_ast = ast(
        cond(json!({"kind":"overlaps","from":scalar_value("2026-01-01"),"to":null})),
        vec![],
    );
    let child = json!({"question":child_id,"normal_form":{},"parameters":{},"identity_ast":child_ast,"source_outputs":[]});
    let mut syntax = ast(Value::Null, vec![]);
    syntax["source"] = json!({"kind":"saved","name":question_key(&child_id).unwrap(),"arguments":[],"condition":null,"output":"known"});
    let mut original = object(syntax);
    original["definitions"] = json!([child]);
    let upgraded = query_object(&original).unwrap();
    assert_eq!(upgraded["body"]["reason"]["kind"], "dependency");
    assert_eq!(upgraded["body"]["reason"]["question"], child_id);
    assert_eq!(
        upgraded["definitions"][0]["body"]["reason"]["kind"],
        "span_overlap"
    );
}

#[test]
fn real_parent_identity_projection_preserves_original_text_semantics() {
    let mut original = ast(
        cond(json!({"kind":"exists","unit":"tool_calls","condition":null})),
        vec![stage(json!({"kind":"matches"}))],
    );
    original["source"]["unit"] = json!("messages");
    let upgraded = query_object(&object(original)).unwrap();
    assert_eq!(
        upgraded["body"]["query"]["transforms"][1]["kind"]["columns"][0],
        column(
            json!({"kind":"function","name":"text","arguments":[field("parent.id")]}),
            "parent_id".into()
        ),
    );
}

#[test]
fn historical_result_declarations_keep_missing_matching_evidence_unproven() {
    let mut before = object(json!({
        "source":{"kind":"result","reference":"result:legacy","part":"rows","condition":null,"output":"known"},
        "attach":[],"stages":[]
    }));
    before["source_outputs"] = json!([{
        "source":{"kind":"result","reference":"result:legacy","part":"rows"},
        "schema":{"unit":"sessions","fields":[{
            "name":"body","kind":"text","values":[],"unknown":"unknown","help":"","read_as":"","unit":null
        }],"main_time":null}
    }]);
    let after = query_object(&before).unwrap();
    assert_eq!(after["id"], before["id"]);
    assert_eq!(after["normal_form"], before["normal_form"]);
    assert_eq!(after["source_outputs"], before["source_outputs"]);
    let typed: crate::query_objects::QueryObject = serde_json::from_value(after).unwrap();
    assert!(typed.source_outputs[0].unproven_fields.is_none());
}

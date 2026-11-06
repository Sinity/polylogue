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
    let q = &after["identity_ast"];
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
    let ts = result["identity_ast"]["transforms"].as_array().unwrap();
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
    let ts = result["identity_ast"]["transforms"].as_array().unwrap();
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

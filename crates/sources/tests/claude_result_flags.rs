//! Exact supplied execution flags, including the grouped-parent first failure.
use polylogue_model::BlockContent;
use polylogue_sources::{
    claude_code::{
        self, FileCtx, KeyedRecords,
        keyed::{Contribution, MemoryReadView, ReadView, ResponseKey},
        projection::{CountingObligation, Field, RecordCursor, Unit},
        selection::Transform,
    },
    frame::drive_jsonl,
    selection::{PathPart, Recipe},
};
use serde_json::{Value, json};
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";

fn parse(meta: Value) -> (Vec<u8>, Contribution) {
    let mut bytes = Vec::new();
    for (uuid, content) in [
        ("a0", json!([{"type":"text","text":"neutral"}])),
        ("a1", json!([{"type":"text","text":"neutral"}])),
        (
            "a2",
            json!([{"type":"tool_use","id":"call","name":"Bash","input":{"command":"true"}}]),
        ),
    ] {
        bytes.extend(serde_json::to_vec(&json!({"type":"assistant","uuid":uuid,"sessionId":SID,
            "requestId":"request","message":{"id":"response","role":"assistant","content":content}})).unwrap());
        bytes.push(b'\n');
    }
    bytes.extend(
        serde_json::to_vec(&json!({"type":"user","uuid":"000-result","sessionId":SID,
        "parentUuid":"a2","message":{"role":"user","content":[{"type":"tool_result",
        "tool_use_id":"call","content":"neutral output","is_error":false}]},"toolUseResult":meta}))
        .unwrap(),
    );
    bytes.push(b'\n');
    let mut memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main(SID), &memory).unwrap(),
        &bytes,
        0,
    );
    let output = framed.output.unwrap();
    memory.apply(output.header, output.delta);
    let response = memory
        .response(SID, &ResponseKey::user("000-result"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    assert!(
        response.message.parent.is_some(),
        "later fragment resolves its grouped owner"
    );
    (bytes, c)
}
fn evaluate(recipe: &impl Recipe<Error = claude_code::selection::Error>, bytes: &[u8]) -> Value {
    let mut output = Vec::new();
    polylogue_sources::selection::write_value(
        recipe,
        |r| -> Result<Vec<u8>, std::io::Error> {
            Ok(bytes[r.start as usize..r.end as usize].to_vec())
        },
        1 << 20,
        &mut output,
    )
    .unwrap();
    serde_json::from_slice(&output).unwrap()
}
fn result(c: &Contribution) -> polylogue_model::ToolResult {
    let mut cursor = RecordCursor::default();
    let mut blocks = Vec::new();
    while let Some(unit) = c.next_output(0, &mut cursor).unwrap() {
        if let Unit::Block { content, .. } = unit {
            blocks.push(content);
        }
    }
    assert_eq!(blocks.len(), 1, "flags cannot fabricate image blocks");
    match blocks.pop().unwrap() {
        BlockContent::ToolResult(r) => r,
        other => panic!("{other:?}"),
    }
}
#[test]
fn supplied_flags_preserve_presence_values_origins_and_version_evidence_without_inference() {
    let mut evidence = Vec::new();
    let (_, absent) = parse(json!({"stdout":"neutral output","stderr":"","interrupted":false}));
    let mut baseline = result(&absent);
    baseline.data = None;
    for supplied in [None, Some(false), Some(true)] {
        let mut meta = json!({"stdout":"neutral output","stderr":"","interrupted":false});
        if let Some(value) = supplied {
            for key in ["isImage", "noOutputExpected"] {
                meta[key] = json!(value);
            }
        }
        let (bytes, c) = parse(meta);
        assert!(
            c.projection.obligations.is_empty(),
            "{:?}",
            c.projection.obligations
        );
        evidence.push(c.record_unit(0));
        let mut mapped = result(&c);
        let data = serde_json::to_value(mapped.data.take().unwrap()).unwrap();
        assert_eq!(
            mapped, baseline,
            "flags cannot change output, status or exit information"
        );
        for key in ["isImage", "noOutputExpected"] {
            let facts: Vec<_> = c
                .projection
                .facts
                .iter()
                .filter(|f| matches!(&f.field, Field::ResultData {local:0,key:k} if k==key))
                .collect();
            if let Some(value) = supplied {
                assert_eq!(data[key], value);
                assert_eq!(facts.len(), 1);
                let path = vec!["toolUseResult".to_owned(), key.to_owned()];
                assert_eq!(facts[0].source_path, path);
                let recipe = facts[0].selection.as_ref().unwrap();
                assert_eq!(recipe.transform, Transform::JsonValueV1);
                assert_eq!(recipe.inputs.len(), 1);
                assert_eq!(
                    recipe.inputs[0].path,
                    path.into_iter().map(PathPart::Key).collect::<Vec<_>>()
                );
                let record = recipe.inputs[0].record;
                let donor: Value =
                    serde_json::from_slice(&bytes[record.start as usize..record.end as usize])
                        .unwrap();
                assert_eq!(donor["uuid"], "000-result");
                assert_eq!(donor["toolUseResult"][key], value);
                assert_eq!(evaluate(recipe, &bytes), value);
            } else {
                assert!(data.get(key).is_none());
                assert!(facts.is_empty());
            }
        }
    }
    assert_ne!(
        evidence[0], evidence[1],
        "absence differs from false in Message version input"
    );
    assert_ne!(
        evidence[1], evidence[2],
        "false differs from true in Message version input"
    );
}
#[test]
fn wrong_types_and_unknown_metadata_keep_exact_typed_obligations() {
    for key in ["isImage", "noOutputExpected", "unknownFlag"] {
        for value in [
            json!(null),
            json!(0),
            json!("false"),
            json!([]),
            json!({}),
            json!(true),
        ] {
            if key != "unknownFlag" && value.is_boolean() {
                continue;
            }
            let (_, c) = parse(json!({key:value}));
            assert_eq!(
                c.projection.obligations,
                vec![CountingObligation::UnmappedField {
                    path: vec!["toolUseResult".into(), key.into()]
                }]
            );
            assert!(
                !c.projection
                    .facts
                    .iter()
                    .any(|f| matches!(&f.field,Field::ResultData{key:k,..} if k==key))
            );
        }
    }
}

//! Named opaque result sidecars preserve supplied evidence without tool semantics.
use polylogue_model::{BlockContent, RetainedJson};
use polylogue_sources::{
    claude_code::{
        self, FileCtx, KeyedRecords,
        keyed::{Contribution, MemoryReadView, ReadView, ResponseKey},
        projection::{CountingObligation, Field},
    },
    frame::drive_jsonl,
};
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const NAMES: &[&str] = &[
    "canReadOutputFile",
    "content",
    "description",
    "file",
    "filePath",
    "gitOperation",
    "matches",
    "message",
    "newString",
    "oldString",
    "outputFile",
    "pin",
    "prompt",
    "query",
    "replaceAll",
    "resolvedModel",
    "resumedAgentId",
    "staleReadFileStateHint",
    "staleRecovered",
    "success",
    "total_deferred_tools",
    "type",
    "userModified",
];
fn row(meta: Option<&str>, content: Option<&str>) -> String {
    let message = content.map(|v| format!(r#", "message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"call","is_error":false,"content":{v}}}]}}"#)).unwrap_or_default();
    format!(
        r#"{{"type":"user","uuid":"result","sessionId":"{SID}"{message}{}}}
"#,
        meta.map(|v| format!(",\"toolUseResult\":{v}"))
            .unwrap_or_default()
    )
}
fn parse(raw: &str) -> Contribution {
    let mut memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main(SID), &memory).unwrap(),
        raw.as_bytes(),
        0,
    );
    let output = framed.output.unwrap();
    memory.apply(output.header, output.delta);
    let response = memory
        .response(SID, &ResponseKey::user("result"))
        .unwrap()
        .unwrap();
    memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap()
}
fn data(c: &Contribution) -> &RetainedJson {
    let BlockContent::ToolResult(r) = &c.blocks[0] else {
        panic!("result")
    };
    assert_eq!(
        r.output,
        Some(polylogue_model::ToolOutput::Text("output".into()))
    );
    assert_eq!(r.outcome, polylogue_model::ToolOutcome::Ok);
    r.data.as_ref().unwrap()
}
fn evaluate(c: &Contribution, key: &str, raw: &str) -> (RetainedJson, u64) {
    let f = c
        .projection
        .facts
        .iter()
        .find(|f| matches!(&f.field, Field::ResultData{local:0,key:k} if k==key))
        .unwrap();
    let recipe = f.selection.as_ref().unwrap();
    assert_eq!(
        recipe.transform,
        claude_code::selection::Transform::JsonValueV1
    );
    let mut out = Vec::new();
    recipe
        .write_value(
            |r| -> Result<Vec<u8>, std::io::Error> {
                Ok(raw.as_bytes()[r.start as usize..r.end as usize].to_vec())
            },
            65536,
            &mut out,
        )
        .unwrap();
    (
        RetainedJson::parse(&out).unwrap(),
        recipe.inputs[0].record.start,
    )
}
#[test]
fn finite_named_members_preserve_all_supplied_shapes_exactly_with_no_output_or_status_inference() {
    for key in NAMES {
        for value in [
            "null",
            "false",
            "true",
            "0",
            "1.2300",
            r#""\ud800""#,
            r#"{"n":9007199254740993,"z":-0}"#,
            "[]",
            "{}",
        ] {
            let meta = format!("{{\"{key}\":{value}}}");
            let raw = row(Some(&meta), Some(r#""output""#));
            let c = parse(&raw);
            assert!(
                c.projection.obligations.is_empty(),
                "{key}: {:?}",
                c.projection.obligations
            );
            let expected = RetainedJson::parse(value.as_bytes()).unwrap();
            assert_eq!(data(&c).get(key), Some(&expected));
            assert_eq!(evaluate(&c, key, &raw), (expected, 0));
        }
    }
    let absent = parse(&row(None, Some(r#""output""#)));
    assert!(
        !absent
            .projection
            .facts
            .iter()
            .any(|f| matches!(f.field, Field::ResultData { .. }))
    );
}
#[test]
fn whole_sidecar_replacement_and_omission_keep_separate_exact_suppliers() {
    let a = row(
        Some(
            r#"{"gitOperation":{"pr":{"action":"create","number":7,"url":"https://example.invalid/7"}},"success":false}"#,
        ),
        Some(r#""output""#),
    );
    let b = row(Some(r#"{"content":null}"#), None);
    let omitted = row(None, None);
    let raw = a.clone() + &b + &omitted;
    let c = parse(&raw);
    assert!(c.projection.obligations.is_empty());
    assert_eq!(data(&c).get("content"), Some(&RetainedJson::Null));
    assert!(data(&c).get("gitOperation").is_none());
    assert!(data(&c).get("success").is_none());
    assert_eq!(evaluate(&c, "content", &raw).1, a.len() as u64);
    let duplicate = parse(&(raw.clone() + &omitted));
    assert_eq!(
        duplicate, c,
        "identical effective replay has no supplier revision"
    );
    let string = row(Some(r#""raw sidecar\ud800""#), None);
    let raw = raw + &string;
    let changed = parse(&raw);
    assert_eq!(data(&changed).as_object().unwrap().len(), 1);
    assert_eq!(
        evaluate(&changed, "toolUseResult", &raw).0,
        RetainedJson::parse(br#""raw sidecar\ud800""#).unwrap()
    );
    assert_ne!(changed.record_unit(0), c.record_unit(0));
}
#[test]
fn string_carrier_requires_exactly_one_result_and_unknown_status_and_sidecar_containers_still_refuse()
 {
    for meta in [
        "null",
        "true",
        "0",
        "[]",
        r#"{"unknown":true}"#,
        r#"{"gitOperation":{},"isImage":"false"}"#,
        r#"{"success":true,"is_error":false}"#,
    ] {
        let c = parse(&row(Some(meta), Some(r#""output""#)));
        assert!(!c.projection.obligations.is_empty(), "{meta}");
        assert!(
            c.projection
                .obligations
                .iter()
                .all(|v| matches!(v, CountingObligation::UnmappedField { .. }))
        );
    }
    let raw = row(Some(r#""sidecar""#), Some(r#""output""#));
    let c = parse(&raw);
    assert_eq!(
        evaluate(&c, "toolUseResult", &raw).0,
        RetainedJson::parse(br#""sidecar""#).unwrap()
    );
    for content in [
        "[]",
        r#"[{"type":"tool_result","content":"output"},{"type":"tool_result","content":"output"}]"#,
    ] {
        let raw = format!(
            r#"{{"type":"user","uuid":"result","sessionId":"{SID}","message":{{"content":{content}}},"toolUseResult":"carrier"}}
"#
        );
        let c = parse(&raw);
        // A result cannot claim the root carrier when attribution is ambiguous.
        assert!(
            !c.projection
                .facts
                .iter()
                .any(|f| matches!(&f.field,Field::ResultData{key,..} if key=="toolUseResult"))
        );
        if content != "[]" {
            assert!(!c.projection.obligations.is_empty());
        }
    }
}
#[test]
fn opaque_success_metadata_never_waives_invalid_protocol_status() {
    let raw = row(
        Some(r#"{"success":true,"gitOperation":null}"#),
        Some(r#""output""#),
    )
    .replace("\"is_error\":false", "\"is_error\":\"false\"");
    let c = parse(&raw);
    assert!(c.projection.obligations.iter().any(|o|matches!(o,CountingObligation::UnmappedField{path} if path==&["message","content","0","is_error"])));
}

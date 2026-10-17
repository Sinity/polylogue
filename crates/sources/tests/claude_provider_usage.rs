//! One existing selected usage owner supplies raw provider evidence and counters.
use polylogue_model::RetainedJson;
use polylogue_sources::{
    claude_code::{
        self, FileCtx, KeyedRecords,
        keyed::{MemoryReadView, ReadView, ResponseKey},
        projection::Field,
    },
    frame::drive_jsonl,
};
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
fn row(uuid: &str, text: &str, usage: Option<&str>) -> String {
    format!(
        r#"{{"type":"assistant","uuid":"{uuid}","sessionId":"{SID}","requestId":"r","message":{{"id":"response","role":"assistant","content":[{{"type":"text","text":"{text}"}}]{}}}}}
"#,
        usage.map(|v| format!(",\"usage\":{v}")).unwrap_or_default()
    )
}
fn parse(raw: &str) -> (MemoryReadView, claude_code::keyed::ResponseCell) {
    let mut memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main(SID), &memory).unwrap(),
        raw.as_bytes(),
        0,
    );
    let output = framed.output.unwrap();
    memory.apply(output.header, output.delta);
    let r = memory
        .response(SID, &ResponseKey::new("response", "r"))
        .unwrap()
        .unwrap();
    (memory, r)
}
fn selected(r: &claude_code::keyed::ResponseCell, raw: &str) -> RetainedJson {
    let unit = r.selected_unit(0).unwrap();
    let facts = unit
        .facts
        .iter()
        .filter(|f| matches!(&f.field,Field::SelectedExtra{key} if key=="provider_usage"))
        .collect::<Vec<_>>();
    assert_eq!(facts.len(), 1);
    let f = facts[0];
    assert_eq!(f.source_path, ["message", "usage"]);
    let recipe = f.selection.as_ref().unwrap();
    assert_eq!(
        recipe.transform,
        claude_code::selection::Transform::JsonValueV1
    );
    assert_eq!(
        recipe.inputs[0].path,
        vec![
            polylogue_sources::selection::PathPart::Key("message".into()),
            polylogue_sources::selection::PathPart::Key("usage".into())
        ]
    );
    let mut out = Vec::new();
    polylogue_sources::selection::write_value(
        recipe,
        |r| -> Result<Vec<u8>, std::io::Error> {
            Ok(raw.as_bytes()[r.start as usize..r.end as usize].to_vec())
        },
        65536,
        &mut out,
    )
    .unwrap();
    RetainedJson::parse(&out).unwrap()
}
#[test]
fn four_named_members_preserve_each_supplied_shape_and_absence_without_counter_inference() {
    let (_, absent) = parse(&row("a", "a", Some(r#"{"input_tokens":0}"#)));
    assert!(
        absent
            .message
            .attrs
            .as_ref()
            .is_none_or(|v| v.get("provider_usage").is_none())
    );
    for name in ["inference_geo", "iterations", "server_tool_use", "speed"] {
        for value in [
            "null",
            r#""""#,
            "[]",
            "{}",
            "false",
            "1.2300",
            r#""\ud800""#,
        ] {
            let usage = format!(r#"{{"input_tokens":0,"{name}":{value}}}"#);
            let raw = row("a", "a", Some(&usage));
            let (_, r) = parse(&raw);
            let expected = RetainedJson::parse(usage.as_bytes()).unwrap();
            assert_eq!(selected(&r, &raw), expected);
            assert_eq!(
                r.message.attrs.as_ref().unwrap()["provider_usage"],
                expected
            );
            assert_eq!(r.message.usage.as_ref().unwrap().input, Some(0));
        }
    }
}
#[test]
fn usage_supplier_order_omission_and_replacement_reuse_existing_raw_correction_evidence() {
    let a = row(
        "a",
        "a",
        Some(r#"{"input_tokens":1,"inference_geo":"first"}"#),
    );
    let b = row("b", "b", Some(r#"{"input_tokens":2,"speed":"later"}"#));
    let omitted = row("a", "changed", None);
    let raw = a.clone() + &b + &omitted;
    let (_, r) = parse(&raw);
    assert_eq!(
        selected(&r, &raw),
        RetainedJson::parse(br#"{"input_tokens":2,"speed":"later"}"#).unwrap()
    );
    assert_eq!(
        r.selected_origins[0].as_ref().unwrap().record.start,
        a.len() as u64
    );
    let replacement = row(
        "a",
        "changed",
        Some(r#"{"input_tokens":3,"server_tool_use":null}"#),
    );
    let raw = a.clone() + &b + &replacement;
    let (memory, r) = parse(&raw);
    assert_eq!(
        selected(&r, &raw),
        RetainedJson::parse(br#"{"input_tokens":3,"server_tool_use":null}"#).unwrap()
    );
    assert_eq!(
        r.selected_origins[0].as_ref().unwrap().record.start,
        (a.len() + b.len()) as u64
    );
    let mut baseline: [Option<Vec<u8>>; 8] = Default::default();
    for item in 0..r.items {
        let c = memory.contribution(&r.message.id, item).unwrap().unwrap();
        for (old, current) in std::iter::zip(&mut baseline, c.final_fields().unwrap()) {
            if current.is_some() {
                *old = current;
            }
        }
    }
    assert!(
        matches!(r.metadata_units(&baseline).as_slice(),[claude_code::keyed::EvidenceUnit::Json{tag,..}]if tag=="response/usage")
    );
    let removed = row("a", "changed", Some(r#"{"input_tokens":4}"#));
    let (_, r) = parse(&(a + &b + &removed));
    assert!(
        r.message
            .attrs
            .as_ref()
            .unwrap()
            .get("provider_usage")
            .is_none()
    );
    assert_eq!(r.message.usage.as_ref().unwrap().input, Some(4));
}

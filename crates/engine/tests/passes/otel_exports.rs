use super::*;
use serde_json::json;

fn span(id: usize, text: &str) -> serde_json::Value {
    json!({"traceId":"synthetic", "spanId":format!("span-{id}"),
    "startTimeUnixNano":(1_700_000_000_000_000_000u64 + id as u64).to_string(),
    "attributes":[
        {"key":"gen_ai.operation.name","value":{"stringValue":"chat"}},
        {"key":"gen_ai.conversation.id","value":{"stringValue":format!("case-{id}")}},
        {"key":"gen_ai.input.messages","value":{"stringValue":json!([
            {"role":"user","parts":[{"type":"text","content":text}]}
        ]).to_string()}}
    ]})
}
fn wrap(spans: Vec<serde_json::Value>) -> serde_json::Value {
    json!({"resourceSpans":[{"scopeSpans":[{"spans":spans}]}]})
}

#[test]
fn otlp_disk_ordering_and_refused_lines_survive_original_and_index_deletion() {
    let w = ClaudeHome::new();
    let original = w.dir.path().join("traces.jsonl");
    let good = wrap(
        (0..40)
            .rev()
            .map(|i| span(i, &format!("message {i}")))
            .collect(),
    );
    let bad = wrap(vec![span(999, "refused prefix")]);
    let repeated = wrap(vec![span(0, "later repeated history")]);
    fs::write(&original, format!("{good}\n{bad}!\n{repeated}\n")).unwrap();
    let expected;
    {
        let (mut e, _lock) = w.engine(64);
        let imported = e.import(std::slice::from_ref(&original)).unwrap();
        assert!(
            imported.pass.errors.is_empty(),
            "{:?}",
            imported.pass.errors
        );
        assert!(imported.pass.bytes_read >= fs::metadata(&original).unwrap().len());
        assert_eq!(imported.pass.messages_written, 40, "{imported:?}");
        for i in 0..40 {
            let messages = messages_of(&w.root, &format!("otel-genai:case-{i}"));
            assert_eq!(messages.len(), 1, "case {i}; import {imported:?}");
            assert_eq!(messages[0].text(), format!("message {i}"));
        }
        assert!(messages_of(&w.root, "otel-genai:case-999").is_empty());
        expected = (0..40)
            .map(|i| w.export(&format!("otel-genai:case-{i}")))
            .collect::<Vec<_>>();
        let repeated = e.import(std::slice::from_ref(&original)).unwrap();
        assert!(repeated.imported.is_empty());
        assert!(repeated.pass.errors.is_empty());
    }
    fs::remove_file(original).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = w.engine(64);
    let rebuilt = e.run_pass().unwrap();
    assert!(rebuilt.errors.is_empty(), "{:?}", rebuilt.errors);
    let actual = (0..40)
        .map(|i| w.export(&format!("otel-genai:case-{i}")))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(messages_of(&w.root, "otel-genai:case-999").is_empty());
    assert_eq!(e.run_pass().unwrap().bytes_read, 0);
}

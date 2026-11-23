use super::*;

#[test]
fn append_output_keeps_stated_hook_field_positions() {
    let parser =
        polylogue_sources::claude_code::open(std::path::Path::new("projects/p/s1.jsonl")).unwrap();
    let mut combined = polylogue_sources::Output::default();
    for (index, id) in ["first", "second"].into_iter().enumerate() {
        let record = serde_json::json!({"type":"system","subtype":"stop_hook_summary",
            "sessionId":"s1","uuid":id,"hookAdditionalContext":["context"]});
        let mut bytes = serde_json::to_vec(&record).unwrap();
        bytes.push(b'\n');
        let (step, _) = parser
            .step(None, &bytes, index as u64 * 1000, &Default::default())
            .unwrap();
        let output = step.output;
        append_output(&mut combined, output);
    }
    assert_eq!(combined.messages.len(), 2);
    assert_eq!(combined.hook_fields[&1][0].start, 1000);
    assert_eq!(
        combined.messages[1].message.id.to_string(),
        "claude-code:second"
    );
}

use super::*;
use serde_json::json;

#[test]
fn plain_and_json_reports_preserve_change_details_and_sample_limits() {
    let data = json!({"rule_version":1,"comparable":false,"before":{"origin":"synthetic","sample":true,"records":2,"unreadable_inputs":0,"unreadable_records":0},"after":{"origin":"synthetic","sample":true,"records":1,"unreadable_inputs":1,"unreadable_records":0},"reasons":["different_samples"],"changes":[{"path":"/text","kind":"types","before":{"types":{"string":2},"required":true},"after":{"types":{"null":1},"required":false}}]});
    let mut bytes = Vec::new();
    render(&data, Format::Terminal, &mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    for part in [
        "different_samples",
        "/text",
        "string",
        "null",
        "required true -> false",
    ] {
        assert!(text.contains(part), "{text}");
    }
    let mut bytes = Vec::new();
    render(&data, Format::Json, &mut bytes).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), data);
}

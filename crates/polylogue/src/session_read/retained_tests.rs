use super::*;
#[test]
fn terminal_reads_opaque_json_without_repairing_units_or_numbers() {
    let document = polylogue_model::RetainedJson::parse(br#"{"session":{"id":"session:one"},"messages":[{"id":"message:one","role":"assistant","blocks":[{"kind":"tool_call","tool":"sample","input":{"n":1e+02,"s":"\ud800"}},{"kind":"tool_result","outcome":"ok","call_id":{"value":"~u:eda080","encoding":"wtf8"},"data":{"s":"\ud801"}}]}]}"#).unwrap();
    let mut out = Vec::new();
    render_terminal(&document, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains(r#"{"n":1e+02,"s":"\ud800"}"#));
    assert!(text.contains(r#"{"s":"\ud801"}"#));
    assert!(text.contains(r#"{"encoding":"wtf8","value":"~u:eda080"}"#));
    assert!(!text.contains('\u{fffd}'));
}

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

#[test]
fn complete_repaired_text_keeps_advisory_and_diagnostics_without_missing_content() {
    let qualification = json!({"outcome":"ok","notes":["1 unpaired UTF-16 code unit repaired"],
        "diagnostics":{"text_repairs":1},"unknown":{}});
    let end = completion(&qualification, &[]);
    assert_eq!(end["missing"], json!([]));
    assert_eq!(end["notes"], qualification["notes"]);
    assert_eq!(end["diagnostics"], qualification["diagnostics"]);
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let send = std::thread::spawn(move || {
        frame(&mut writer, json!({"type":"session","id":"claude-code:repaired","format":"polylogue-session","version":1})).unwrap();
        frame(&mut writer, json!({"type":"text","data":"{}"})).unwrap();
        frame(&mut writer, end).unwrap();
    });
    let mut out = Vec::new();
    let result = decode(
        BufReader::new(reader),
        "claude-code:repaired",
        &mut out,
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    send.join().unwrap();
    assert_eq!(out, b"{}\n");
    assert_eq!(result["canonical_complete"], true);
    assert_eq!(result["diagnostics"]["text_repairs"], 1);
    let degraded = completion(
        &json!({"outcome":"degraded","notes":["source_catalog_coverage"],"unknown":{"source_catalog_coverage":true}}),
        &["source_catalog_coverage".into()],
    );
    assert_eq!(degraded["canonical_complete"], false);
    assert_eq!(degraded["missing"], json!(["source_catalog_coverage"]));
}

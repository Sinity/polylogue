use super::*;
fn exchange(frames: Vec<Value>, expected: &str) -> (io::Result<Value>, Vec<u8>) {
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let send = std::thread::spawn(move || {
        for item in frames {
            if frame(&mut writer, item).is_err() {
                break;
            }
        }
    });
    let mut out = Vec::new();
    let result = decode(
        reader,
        expected,
        &mut out,
        Instant::now() + Duration::from_secs(24 * 60 * 60),
    );
    send.join().unwrap();
    (result, out)
}
fn header() -> Value {
    json!({"type":"session","id":"opaque:session:with:colons","format":"polylogue-session","version":1})
}
fn end() -> Value {
    json!({"type":"end","outcome":"degraded","canonical_complete":false,"missing":["source_catalog_coverage"]})
}
#[test]
fn session_header_must_match_before_text_and_end() {
    let expected = "opaque:session:with:colons";
    let frames = vec![
        header(),
        json!({"type":"text","data":"{\"text\":\"λ🧪\"}"}),
        end(),
    ];
    let (result, out) = exchange(frames.clone(), expected);
    assert_eq!(result.unwrap()["canonical_complete"], false);
    assert_eq!(String::from_utf8(out).unwrap(), "{\"text\":\"λ🧪\"}\n");
    let mut wrong = frames;
    wrong[0]["id"] = json!("other:opaque:session");
    let (result, out) = exchange(wrong, expected);
    assert!(result.is_err());
    assert!(out.is_empty());
}
#[test]
fn malformed_end_or_frame_order_is_an_error() {
    let expected = "opaque:session:with:colons";
    let mut complete = end();
    complete["canonical_complete"] = json!(true);
    let mut false_complete = end();
    false_complete["outcome"] = json!("ok");
    false_complete["canonical_complete"] = json!(true);
    let mut missing = end();
    missing["missing"] = json!([]);
    for frames in [
        vec![header()],
        vec![end()],
        vec![header(), header(), end()],
        vec![header(), complete],
        vec![header(), false_complete],
        vec![header(), missing],
        vec![json!({"type":"text","data":"private"}), end()],
    ] {
        let (result, _) = exchange(frames, expected);
        assert!(result.is_err());
    }
    let failure = failed(ErrorKind::NotFound, "Session absent");
    let (result, out) = exchange(vec![failure], expected);
    assert!(out.is_empty());
    assert_eq!(result.unwrap()["outcome"], "error");
}
#[test]
fn only_full_peer_close_counts_as_gone() {
    let (server, client) = UnixStream::pair().unwrap();
    assert!(!peer_gone(&server));
    client.shutdown(std::net::Shutdown::Write).unwrap();
    assert!(!peer_gone(&server));
    drop(client);
    assert!(peer_gone(&server));
}
#[test]
fn terminal_prints_opaque_json_without_repairing_it() {
    let document = polylogue_model::RawJson::parse(br#"{"session":{"id":"session:one"},"messages":[{"id":"message:one","role":"assistant","blocks":[{"kind":"tool_call","tool":"sample","input":{"n":1e+02,"s":"\ud800"}},{"kind":"tool_result","outcome":"ok","call_id":{"value":"~u:eda080","encoding":"wtf8"},"data":{"s":"\ud801"}}]}]}"#).unwrap();
    let mut out = Vec::new();
    render_terminal(&document, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains(r#"{"n":1e+02,"s":"\ud800"}"#));
    assert!(text.contains(r#"{"s":"\ud801"}"#));
    assert!(text.contains(r#"{"encoding":"wtf8","value":"~u:eda080"}"#));
    assert!(!text.contains('\u{fffd}'));
}

#[test]
fn complete_read_with_text_repair_keeps_notes_and_diagnostics() {
    let status = json!({"outcome":"ok","notes":["1 unpaired UTF-16 code unit repaired"],
        "diagnostics":{"text_repairs":1},"unknown":{}});
    let end = completion(&status, &[]);
    assert_eq!(end["missing"], json!([]));
    assert_eq!(end["notes"], status["notes"]);
    assert_eq!(end["diagnostics"], status["diagnostics"]);
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let send = std::thread::spawn(move || {
        frame(&mut writer, json!({"type":"session","id":"claude-code:repaired","format":"polylogue-session","version":1})).unwrap();
        frame(&mut writer, json!({"type":"text","data":"{}"})).unwrap();
        frame(&mut writer, end).unwrap();
    });
    let mut out = Vec::new();
    let result = decode(
        reader,
        "claude-code:repaired",
        &mut out,
        Instant::now() + Duration::from_secs(24 * 60 * 60),
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

mod deadline;

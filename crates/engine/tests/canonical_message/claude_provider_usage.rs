//! Raw provider usage is kept apart from the normalized counters.
use super::*;
use polylogue_model::RawJson;
fn row(uuid: &str, text: &str, usage: Option<&str>) -> String {
    format!(
        r#"{{"type":"assistant","uuid":"{uuid}","requestId":"r","message":{{"id":"response","content":[{{"type":"text","text":"{text}"}}]{}}}}}
"#,
        usage.map(|v| format!(",\"usage\":{v}")).unwrap_or_default()
    )
}
fn materialize(raw: &str) -> (World, SourceProjection, MessageView) {
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[0]);
    let h = begin(&mut w, &req);
    let source = w.complete(&h);
    let view = selected(&mut w, &source);
    (w, source, view)
}
fn bag(view: &MessageView) -> Option<(RawJson, Value)> {
    match view
        .field(
            &Slot::Extra {
                block: None,
                key: "provider_usage".into(),
            },
            65536,
        )
        .unwrap()
    {
        None => None,
        Some(FieldRead::Value(f)) => {
            assert!(f.supplied);
            Some((
                RawJson::parse(&f.range(0, 65536).unwrap().bytes).unwrap(),
                serde_json::from_slice(&f.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap(),
            ))
        }
        _ => panic!("conflict"),
    }
}
#[test]
fn provider_usage_cites_its_whole_subtree_and_keeps_normalized_counts_separate() {
    let usage = r#"{"input_tokens":2,"output_tokens":3,"inference_geo":"provided","iterations":[1.2300,"\ud800"],"server_tool_use":{"n":9007199254740993,"none":null},"speed":"provider value"}"#;
    let raw = row("a", "a", Some(usage));
    let (_world, _, view) = materialize(&raw);
    let (v, origin) = bag(&view).unwrap();
    assert_eq!(v, RawJson::parse(usage.as_bytes()).unwrap());
    assert_eq!(
        origin["selection"]["inputs"][0]["path"],
        json!([{"Key":"message"},{"Key":"usage"}])
    );
    assert_eq!(
        origin["selection"]["inputs"][0]["record"],
        json!({"start":0,"end":raw.len()-1})
    );
    assert_eq!(
        value(
            &view,
            &Slot::Usage(polylogue_model::merge::UsageField::Input)
        )
        .0,
        2
    );
    assert_eq!(
        value(
            &view,
            &Slot::Usage(polylogue_model::merge::UsageField::Output)
        )
        .0,
        3
    );
}
#[test]
fn usage_replacement_and_omission_preserve_supplier_and_remove_stale_owned_bag() {
    let a = row("a", "a", Some(r#"{"input_tokens":1,"inference_geo":null}"#));
    let b = row("b", "b", Some(r#"{"input_tokens":2,"iterations":[]}"#));
    let omitted = row("a", "changed", None);
    let (_omitted_world, _, view) = materialize(&(a.clone() + &b + &omitted));
    let (v, o) = bag(&view).unwrap();
    assert_eq!(
        v,
        RawJson::parse(br#"{"input_tokens":2,"iterations":[]}"#).unwrap()
    );
    assert_eq!(o["selection"]["inputs"][0]["record"]["start"], a.len());
    let update = row("a", "changed", Some(r#"{"input_tokens":3,"speed":{}}"#));
    let raw = a.clone() + &b + &update;
    let (_corrected_world, corrected, view) = materialize(&raw);
    assert_eq!(
        bag(&view).unwrap().1["selection"]["inputs"][0]["record"]["start"],
        a.len() + b.len()
    );
    let (_base_world, base, _base_view) = materialize(&(update + &b));
    assert_ne!(corrected.version_hash(), base.version_hash());
    let removed = row("a", "changed", Some(r#"{"input_tokens":4}"#));
    let (_removed_world, _, view) = materialize(&(a + &b + &removed));
    assert!(bag(&view).is_none());
    assert_eq!(
        value(
            &view,
            &Slot::Usage(polylogue_model::merge::UsageField::Input)
        )
        .0,
        4
    );
}
#[test]
fn provider_evidence_does_not_waive_unknown_members_invalid_counters_or_literal_collision() {
    for usage in [
        r#"{"inference_geo":null,"unknown":0}"#,
        r#"{"speed":"raw","input_tokens":"wrong"}"#,
        r#"{"speed":"raw","output_tokens":-1}"#,
        r#"{"speed":"raw","cache_read_input_tokens":null}"#,
        r#"{"speed":"raw","cache_creation_input_tokens":true}"#,
        r#"{"speed":"raw","cache_creation":null}"#,
        r#"{"speed":"raw","cache_creation":{"ephemeral_5m_input_tokens":"wrong"}}"#,
        r#"{"speed":"raw","output_tokens_details":{"thinking_tokens":1.5}}"#,
        r#"{"speed":"raw","output_tokens_details":{"unknown":1}}"#,
    ] {
        refuse(&row("a", "a", Some(usage)));
    }
    let raw = r#"{"type":"assistant","uuid":"a","requestId":"r","provider_usage":null,"message":{"id":"response","content":[{"type":"text","text":"a"}],"usage":{"inference_geo":null}}}
"#;
    refuse(raw);
}
fn refuse(raw: &str) {
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[0]);
    let h = begin(&mut w, &req);
    for _ in 0..4000 {
        match w.compute_with(&h, work_budget()) {
            Ok(action) => assert!(!matches!(
                m::apply(&mut w.writer, action).unwrap(),
                m::Applied::Source(_)
            )),
            Err(m::Error::UnsupportedField { .. }) => return,
            Err(e) => panic!("{e}"),
        }
    }
    panic!("invalid usage certified")
}

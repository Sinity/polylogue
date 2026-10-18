//! Exact provider evidence through public JSON/export and bounded metadata.
use super::*;
use polylogue_model::RawJson;
#[test]
fn provider_usage_is_opaque_in_reads_and_separate_from_counters() {
    let world = World::new();
    let usage = r#"{"input_tokens":2,"output_tokens":3,"cache_creation":{"ephemeral_5m_input_tokens":0},"inference_geo":null,"iterations":[1.2300,"\ud800"],"server_tool_use":{"n":9007199254740993,"empty":{}},"speed":"provider speed"}"#;
    let raw = format!(
        r#"{{"type":"assistant","uuid":"usage","sessionId":"{SID}","requestId":"r","message":{{"id":"response","role":"assistant","content":[{{"type":"text","text":"Neutral usage carrier"}}],"usage":{usage},"stop_reason":"end_turn"}}}}
"#
    );
    let path = world.input("usage.jsonl", raw.as_bytes());
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000009701");
    let until = Instant::now() + WAIT;
    loop {
        let p = world.progress(receipt["input"]["id"].as_str().unwrap());
        if p["source_rows"]["state"] == "complete" {
            break;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(10));
    }
    let reference = reference();
    let out = world.cli(&[&reference, "read", "--format", "json"]);
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let doc = RawJson::parse(&out.stdout).unwrap();
    let message = &doc["messages"][0];
    let expected = RawJson::parse(usage.as_bytes()).unwrap();
    assert_eq!(message["attrs"]["provider_usage"], expected);
    assert_eq!(
        message["usage"],
        RawJson::parse(br#"{"input":2,"output":3,"cache_write_5m":0}"#).unwrap()
    );
    let origins = message["field_origins"].as_array().unwrap();
    let origin = origins
        .iter()
        .find(|f| f["field"].as_str() == Some("attrs.provider_usage"))
        .unwrap();
    assert_eq!(
        origin["origin"]["selection"]["inputs"][0]["path"],
        RawJson::parse(br#"[{"Key":"message"},{"Key":"usage"}]"#).unwrap()
    );
    assert_eq!(
        origin["origin"]["binding"]["input"].as_str(),
        receipt["input"]["id"].as_str()
    );
    let destination = world.work.path().join("usage-export");
    let exported = world.cli(&[
        &reference,
        "export",
        "--format",
        "json",
        "--to",
        destination.to_str().unwrap(),
    ]);
    assert!(
        matches!(exported.status.code(), Some(0 | 5)),
        "{exported:?}"
    );
    let mut dirs = vec![destination];
    let mut documents = vec![];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|e| e == "json")
                && p.file_name().unwrap() != "manifest.json"
            {
                documents.push(fs::read(p).unwrap());
            }
        }
    }
    assert_eq!(documents.len(), 1);
    let export = RawJson::parse(&documents[0]).unwrap();
    assert_eq!(export["messages"], doc["messages"]);
    let page = model_call(&world, &known_catalog_call(&reference), Some(16000));
    assert_ne!(page["outcome"], "error", "{page}");
    assert_eq!(page["primary_content_complete"], true);
    let metadata = &page["shown"][0]["message_metadata"];
    let attrs = RawJson::parse(metadata["attrs"]["json"].as_str().unwrap().as_bytes()).unwrap();
    assert_eq!(attrs["provider_usage"], expected);
    assert_eq!(
        metadata["usage"],
        json!({"input":2,"output":3,"cache_write_5m":0})
    );
    assert!(
        page["shown"][0]["sources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["input"] == receipt["input"]["id"]
                && s["start"] == 0
                && s["end"] == raw.len() - 1)
    );
}

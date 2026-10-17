//! Exact provider evidence through public JSON/export and bounded metadata.
use super::*;
use polylogue_model::RetainedJson;
#[test]
fn repeated_record_extra_selects_one_supplier_in_public_routes() {
    let world = World::new();
    let usage = r#"{"number":9007199254740993,"text":"\ud800","none":null}"#;
    let row = |uuid: &str, text: &str, extra: &str| {
        format!(
            r#"{{"type":"assistant","uuid":"{uuid}","sessionId":"{SID}","requestId":"r","effort":{extra},"message":{{"id":"response","role":"assistant","content":[{{"type":"text","text":"{text}"}}],"usage":{{"input_tokens":2,"output_tokens":3}},"stop_reason":"end_turn"}}}}
"#
        )
    };
    let first = row("a", "Neutral first", "1");
    let second = row("b", "Neutral second", "2");
    let correction = row("a", "Neutral correction", usage);
    let supplier_start = first.len() + second.len();
    let raw = first + &second + &correction;
    let path = world.input("usage.jsonl", raw.as_bytes());
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000009701");
    let until = Instant::now() + WAIT;
    loop {
        let p = world.progress(receipt["retained"]["id"].as_str().unwrap());
        if p["source_rows"]["state"] == "settled" {
            break;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(10));
    }
    let reference = reference();
    let out = world.cli(&[&reference, "read", "--format", "json"]);
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let doc = RetainedJson::parse(&out.stdout).unwrap();
    let message = &doc["messages"][0];
    let expected = RetainedJson::parse(usage.as_bytes()).unwrap();
    assert_eq!(message["attrs"]["effort"], expected);
    assert_eq!(
        message["usage"],
        RetainedJson::parse(br#"{"input":2,"output":3}"#).unwrap()
    );
    let origins = message["field_origins"].as_array().unwrap();
    let origin = origins
        .iter()
        .find(|f| f["field"].as_str() == Some("attrs.effort"))
        .unwrap();
    assert_eq!(
        origin["origin"]["selection"]["inputs"][0]["path"],
        RetainedJson::parse(br#"[{"Key":"effort"}]"#).unwrap()
    );
    assert_eq!(
        origin["origin"]["binding"]["input"].as_str(),
        receipt["retained"]["id"].as_str()
    );
    assert_eq!(
        origin["origin"]["selection"]["inputs"][0]["record"]["start"]
            .as_u64()
            .unwrap(),
        supplier_start as u64
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
    let export = RetainedJson::parse(&documents[0]).unwrap();
    assert_eq!(export["messages"], doc["messages"]);
    let page = model_call(&world, &known_catalog_call(&reference), Some(16000));
    assert_ne!(page["outcome"], "error", "{page}");
    assert_eq!(page["primary_content_complete"], true);
    let metadata = &page["shown"][0]["message_metadata"];
    let attrs =
        RetainedJson::parse(metadata["attrs"]["json"].as_str().unwrap().as_bytes()).unwrap();
    assert_eq!(attrs["effort"], expected);
    assert_eq!(metadata["usage"], json!({"input":2,"output":3}));
    assert!(
        page["shown"][0]["sources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["input"] == receipt["retained"]["id"]
                && s["start"] == supplier_start
                && s["end"] == raw.len() - 1)
    );
}

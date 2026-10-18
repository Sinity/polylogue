//! Exact supplied sidecar and protocol references survive every public consumer.
use super::*;
use polylogue_model::RawJson;
#[test]
fn sidecar_and_reference_content_have_distinct_exact_public_data_leaves_and_origins() {
    let w = World::new();
    let meta = r#"{"canReadOutputFile":false,"content":"sidecar","description":null,"file":{},"filePath":"synthetic/path","gitOperation":{"pr":{"action":"create","number":9007199254740993,"raw":[1.2300,"\ud800"]}},"matches":[],"message":"metadata","newString":"n","oldString":"o","outputFile":"synthetic/output","pin":{},"prompt":null,"query":"q","replaceAll":false,"resolvedModel":"supplied","resumedAgentId":null,"staleReadFileStateHint":{},"staleRecovered":false,"success":true,"total_deferred_tools":0,"type":"opaque","userModified":false}"#;
    let content = r#"[{"type":"text","text":"actual text"},{"type":"tool_reference","tool_name":"synthetic"}]"#;
    let raw = format!(
        r#"{{"type":"user","uuid":"result","sessionId":"{SID}","toolUseResult":{meta},"message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"call","content":{content},"is_error":false}}]}}}}
{{"type":"user","uuid":"string-result","sessionId":"{SID}","toolUseResult":"supplied carrier\ud800","message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"other-call","content":"actual output","is_error":false}}]}}}}
"#
    );
    let path = w.input("result-metadata.jsonl", raw.as_bytes());
    let receipt = w.import(&path, "01900000-0000-7000-8000-000000009801");
    let until = Instant::now() + WAIT;
    loop {
        let p = w.progress(receipt["retained"]["id"].as_str().unwrap());
        if p["source_rows"]["state"] == "settled" {
            break;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(10));
    }
    let reference = reference();
    let out = w.cli(&[&reference, "read", "--format", "json"]);
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let doc = RawJson::parse(&out.stdout).unwrap();
    let block = &doc["messages"][0]["blocks"][0];
    let mut expected = RawJson::parse(meta.as_bytes()).unwrap();
    expected.as_object_mut().unwrap().insert(
        polylogue_model::NativeId::unicode("content_items"),
        RawJson::parse(content.as_bytes()).unwrap(),
    );
    assert_eq!(block["data"], expected);
    assert!(block.get("output").is_none());
    assert!(block.get("text").is_none());
    assert_eq!(block["outcome"].as_str(), Some("ok"));
    let carrier = RawJson::parse(br#""supplied carrier\ud800""#).unwrap();
    assert_eq!(
        doc["messages"][1]["blocks"][0]["data"]["toolUseResult"],
        carrier
    );
    assert_eq!(
        doc["messages"][1]["blocks"][0]["text"].as_str(),
        Some("actual output")
    );
    let origins = doc["messages"][0]["field_origins"].as_array().unwrap();
    for (key, path) in [
        (
            "gitOperation",
            r#"[{"Key":"toolUseResult"},{"Key":"gitOperation"}]"#,
        ),
        (
            "content_items",
            r#"[{"Key":"message"},{"Key":"content"},{"Index":0},{"Key":"content"}]"#,
        ),
    ] {
        let origin = origins
            .iter()
            .find(|f| f["field"].as_str() == Some(&format!("blocks[0].data.{key}")))
            .unwrap();
        assert_eq!(
            origin["origin"]["selection"]["inputs"][0]["path"],
            RawJson::parse(path.as_bytes()).unwrap()
        );
        assert_eq!(
            origin["origin"]["binding"]["input"].as_str(),
            receipt["retained"]["id"].as_str()
        );
    }
    let destination = w.work.path().join("result-export");
    let export = w.cli(&[
        &reference,
        "export",
        "--format",
        "json",
        "--to",
        destination.to_str().unwrap(),
    ]);
    assert!(matches!(export.status.code(), Some(0 | 5)), "{export:?}");
    let mut dirs = vec![destination];
    let mut docs = vec![];
    while let Some(dir) = dirs.pop() {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|e| e == "json")
                && p.file_name().unwrap() != "manifest.json"
            {
                docs.push(fs::read(p).unwrap());
            }
        }
    }
    assert_eq!(docs.len(), 1);
    assert_eq!(
        RawJson::parse(&docs[0]).unwrap()["messages"],
        doc["messages"]
    );
    let page = model_call(&w, &known_catalog_call(&reference), Some(24000));
    assert_ne!(page["outcome"], "error", "{page}");
    assert_eq!(page["primary_content_complete"], true);
    let shown = &page["shown"][0];
    let bag = RawJson::parse(
        shown["block_fields"]["data"]["json"]
            .as_str()
            .unwrap()
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(bag, expected);
    let carrier_bag = RawJson::parse(
        page["shown"][1]["block_fields"]["data"]["json"]
            .as_str()
            .unwrap()
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(carrier_bag["toolUseResult"], carrier);
    assert!(
        shown["field_origins"]
            .as_array()
            .unwrap()
            .iter()
            .any(|o| o.to_string().contains("data.content_items"))
    );
}

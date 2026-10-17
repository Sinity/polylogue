//! Actual CLI JSON/export and bounded metadata share one block-extra policy.
use super::*;
use polylogue_model::RetainedJson;

fn wait_selected(world: &World, input: &str) {
    let until = Instant::now() + WAIT;
    loop {
        let p = world.progress(input);
        if p["source_rows"]["state"] != "pending" {
            assert!(
                p["source_rows"]["state"] == "settled" || p["source_rows"]["usable_result"] == true,
                "{p}"
            );
            return;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(10));
    }
}
fn exported(world: &World, reference: &str) -> Vec<u8> {
    let destination = world.work.path().join("metadata-export");
    let out = world.cli(&[
        reference,
        "export",
        "--format",
        "json",
        "--to",
        destination.to_str().unwrap(),
    ]);
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let mut paths = vec![destination];
    let mut documents = vec![];
    while let Some(path) = paths.pop() {
        for entry in fs::read_dir(path).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                paths.push(p);
            } else if p.extension().is_some_and(|x| x == "json")
                && p.file_name().unwrap() != "manifest.json"
            {
                documents.push(fs::read(p).unwrap());
            }
        }
    }
    assert_eq!(documents.len(), 1);
    documents.pop().unwrap()
}
#[test]
fn claude_block_metadata_keeps_tool_data_and_other_attrs_in_cli_export_and_bounded_pages() {
    let world = World::new();
    let raw = format!(
        r#"{{"type":"assistant","uuid":"a","sessionId":"{SID}","requestId":"r","message":{{"id":"response","content":[{{"type":"tool_use","id":"call","name":"synthetic","input":{{}},"caller":{{"type":"direct","n":1.2300,"s":"\ud800","nullable":null}}}},{{"type":"text","text":"Neutral text","metadata":null}},{{"type":"thinking","thinking":"Neutral thought","signature":"neutral","metadata":{{"count":9007199254740993}}}}],"stop_reason":"end_turn"}}}}
"#
    );
    let path = world.input("metadata.jsonl", raw.as_bytes());
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000009601");
    wait_selected(&world, receipt["retained"]["id"].as_str().unwrap());
    let reference = reference();
    let out = world.cli(&[&reference, "read", "--format", "json"]);
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let doc = RetainedJson::parse(&out.stdout).unwrap();
    let blocks = &doc["messages"][0]["blocks"];
    assert_eq!(
        blocks[0]["data"]["caller"],
        RetainedJson::parse(br#"{"type":"direct","n":1.2300,"s":"\ud800","nullable":null}"#)
            .unwrap()
    );
    assert!(blocks[0].get("attrs").is_none());
    assert!(blocks[1]["attrs"]["metadata"].is_null());
    assert_eq!(
        blocks[2]["attrs"]["metadata"],
        RetainedJson::parse(br#"{"count":9007199254740993}"#).unwrap()
    );
    let export = RetainedJson::parse(&exported(&world, &reference)).unwrap();
    assert_eq!(doc["messages"], export["messages"]);
    let page = model_call(&world, &known_catalog_call(&reference), Some(16000));
    assert_ne!(page["outcome"], "error", "{page}");
    assert_eq!(page["primary_content_complete"], true);
    for (index, container) in [(0, "data"), (1, "attrs"), (2, "attrs")] {
        let shown = &page["shown"][index];
        let bag = RetainedJson::parse(
            shown["block_fields"][container]["json"]
                .as_str()
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(bag, blocks[index][container]);
        let origins = shown["field_origins"].as_array().unwrap();
        assert!(
            origins.iter().any(|o| o
                .to_string()
                .contains(&format!("blocks[{index}].{container}."))),
            "{shown}"
        );
    }
}
#[test]
fn codex_thinking_payload_and_extra_named_data_coexist_and_unknown_path_hint_is_scoped() {
    let world = World::new();
    let bytes=[json!({"type":"session_meta","payload":{"id":"block-metadata"}}),json!({"type":"response_item","payload":{"type":"message","id":"text","role":"assistant","content":[{"type":"output_text","text":"Neutral text"}],"internal_chat_message_metadata_passthrough":null}}),json!({"type":"response_item","payload":{"type":"reasoning","id":"reason","summary":[{"type":"summary_text","text":"Neutral thought"}],"encrypted_content":"neutral ciphertext","data":{"opaque":"extra data"},"internal_chat_message_metadata_passthrough":{"known":false}}}),json!({"type":"event_msg","payload":{"type":"future_topology_control"}})].iter().map(|v|v.to_string()+"\n").collect::<String>();
    let path = world.input("codex.jsonl", bytes.as_bytes());
    let out = world.cli(&[
        "import",
        path.to_str().unwrap(),
        "--format",
        "codex-rollout",
        "--machine",
        "00112233445566778899aabbccddeeff",
        "--host",
        "synthetic-collector",
        "--wait",
        "--wait-timeout",
        "90",
    ]);
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let reference = "codex:block-metadata";
    let out = world.cli(&[reference, "read", "--format", "json"]);
    assert_eq!(out.status.code(), Some(5), "{out:?}");
    let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
    let blocks = doc["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["blocks"].as_array().unwrap())
        .collect::<Vec<_>>();
    let thinking = blocks.iter().find(|b| b["kind"] == "thinking").unwrap();
    assert_eq!(thinking["data"], "neutral ciphertext");
    assert_eq!(thinking["attrs"]["data"], json!({"opaque":"extra data"}));
    assert!(blocks.iter().any(|b| b["kind"] == "text"
        && b["attrs"]["internal_chat_message_metadata_passthrough"].is_null()));
    let export: Value = serde_json::from_slice(&exported(&world, reference)).unwrap();
    assert_eq!(doc["messages"], export["messages"]);
    let default = model_call(&world, &transcript_call(reference), Some(16000));
    assert_eq!(default["outcome"], "error");
    assert!(
        default["error"]["hint"]
            .as_str()
            .unwrap()
            .contains("--branches")
    );
    assert!(default.get("shown").is_none());
    let page = model_call(&world, &known_catalog_call(reference), Some(16000));
    assert_ne!(page["outcome"], "error", "{page}");
    assert_eq!(page["outcome"], "degraded");
    let shown = page["shown"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["kind"] == "thinking")
        .unwrap();
    assert_eq!(
        opaque_page_value(&shown["block_fields"]["data"], false),
        "neutral ciphertext"
    );
    assert_eq!(
        opaque_page_value(&shown["block_fields"]["attrs"], false)["data"],
        json!({"opaque":"extra data"})
    );
    // An established Session with a pending input must not recommend branches.
    let pending_world = World::new();
    let complete = json!({"type":"user","uuid":"ready","sessionId":SID,"message":{"role":"user","content":"Known prefix"}}).to_string()+"\n";
    let complete_path = pending_world.input("complete.jsonl", complete.as_bytes());
    let ready = pending_world.import(&complete_path, "01900000-0000-7000-8000-000000009602");
    wait_selected(&pending_world, ready["retained"]["id"].as_str().unwrap());
    let mut partial_bytes = complete.into_bytes();
    partial_bytes.extend_from_slice(b"{\"type\":\"user\",\"uuid\":\"incomplete\"");
    let partial = pending_world.input("partial.jsonl", &partial_bytes);
    let receipt = pending_world.import(&partial, "01900000-0000-7000-8000-000000009603");
    let until = Instant::now() + WAIT;
    loop {
        let progress = pending_world.progress(receipt["retained"]["id"].as_str().unwrap());
        if progress["parser"]["status"] == "waiting_tail" {
            break;
        }
        assert!(Instant::now() < until, "{progress}");
        thread::sleep(Duration::from_millis(10));
    }
    let refusal = model_call(
        &pending_world,
        &transcript_call(&super::reference()),
        Some(16000),
    );
    assert_eq!(refusal["outcome"], "error");
    assert_eq!(
        refusal["error"]["reason"], "current_selected_basis_unavailable",
        "{refusal}"
    );
    assert!(refusal["error"]["hint"].as_str().is_none(), "{refusal}");
}

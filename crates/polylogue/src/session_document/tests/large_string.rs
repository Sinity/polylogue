use super::*;

#[test]
fn large_malformed_codex_arguments_stay_one_string_with_origin() {
    let world = World::new();
    let argument = format!("{}\n\\\"🦀 unfinished {{", "neutral ".repeat(17000));
    assert!(argument.len() > 131072);
    let lines = [
        json!({"type":"session_meta","payload":{"id":SID,"source":"cli"}}),
        json!({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Neutral request"}]}}),
        json!({"type":"response_item","payload":{"type":"function_call","name":"synthetic_tool","call_id":"synthetic-call","arguments":argument}}),
    ];
    let raw = lines.iter().map(|v| format!("{v}\n")).collect::<String>();
    fs::write(&world.path, &raw).unwrap();
    let input = world
        .owner
        .as_ref()
        .unwrap()
        .handle()
        .try_import(ImportRequest {
            format: InputFormat::CodexRollout,
            context: vec![],
            path: world.path.clone(),
            observation: Observation {
                id: "large-codex-input".into(),
                machine: "synthetic".into(),
                host: "synthetic".into(),
                path: world.path.to_str().unwrap().into(),
                seen_at: 0,
            },
            metadata_allowance: 65536,
        })
        .unwrap()
        .recv_timeout(WAIT)
        .unwrap()
        .unwrap();
    let deadline = Instant::now() + SOURCE_WAIT;
    loop {
        let p = world.progress(&input);
        assert!(p.failure.is_none(), "{p:?}");
        if !p.active
            && !matches!(
                p.source_rows.state,
                polylogue_model::source_rows::SourceRowState::Pending { .. }
            )
        {
            break;
        }
        assert!(Instant::now() < deadline, "{p:?}");
        thread::sleep(POLL);
    }
    fs::remove_file(&world.path).unwrap();
    let session = SessionId::new(&Origin::new("codex").unwrap(), SID).unwrap();
    let view = world
        .reader
        .selected_session(&session, 65536)
        .unwrap()
        .unwrap();
    let mut bytes = Vec::new();
    write(&view, &mut bytes, Instant::now() + WAIT).unwrap();
    let doc: Value = serde_json::from_slice(&bytes).unwrap();
    let message = doc["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| {
            m["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|b| b["kind"] == "tool_call")
        })
        .unwrap();
    let block = &message["blocks"][0];
    assert_eq!(block["input"], argument);
    let origin = message["field_origins"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["field"] == "blocks[0].input")
        .unwrap();
    assert_eq!(
        origin["origin"]["selection"]["transform"],
        json!({"ToolFieldV1":{"field":"input"}})
    );
    let source = &origin["origin"]["selection"]["inputs"][0];
    assert_eq!(source["path"], json!([{"Key":"payload"}]));
    let start = source["record"]["start"].as_u64().unwrap() as usize;
    let end = source["record"]["end"].as_u64().unwrap() as usize;
    assert_eq!(
        serde_json::from_slice::<Value>(&raw.as_bytes()[start..end]).unwrap()["payload"]["arguments"],
        argument
    );
    // Model metadata still has the 64 KiB field cap; only documents stream
    // large strings.
    let m = world
        .reader
        .selected_message(
            &MessageId::from_stored(message["id"].as_str().unwrap(), session).unwrap(),
        )
        .unwrap()
        .unwrap();
    let b = m.block_at(0).unwrap().unwrap();
    assert!(
        block_metadata(&b, Instant::now() + WAIT)
            .unwrap_err()
            .to_string()
            .contains("exceeds 64 KiB")
    );
}

use super::*;

#[test]
fn bounded_selected_message_uses_current_proof_while_observed_suffix_is_pending() {
    let mut world = World::new();
    world.kill();
    let initial = observed_user("earlier", "Independent earlier human text.", None);
    let main = world.input(&format!("{SID}.jsonl"), &initial);
    let config = world.input(
        "configured.json",
        json!({"claude_main":{"path":main,"machine":"synthetic-machine","host":"synthetic-host","acquisition_quantum_bytes":65536}}).to_string().as_bytes(),
    );
    world.child = Some(OwnedChild(
        world
            .command()
            .args(["daemon", "--config", config.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::from(File::create(&world.log).unwrap()))
            .spawn()
            .unwrap(),
    ));
    let deadline = Instant::now() + WAIT;
    let source = loop {
        let out = world.cli(&[&reference(), "read", "--format", "json"]);
        if matches!(out.status.code(), Some(0 | 5)) {
            let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
            if doc["messages"]
                .as_array()
                .is_some_and(|rows| rows.len() == 1)
            {
                break doc["messages"][0]["sources"][0]["input"]
                    .as_str()
                    .unwrap()
                    .to_owned();
            }
        }
        assert!(
            Instant::now() < deadline,
            "initial read: {out:?}; {}",
            world.log()
        );
        thread::sleep(Duration::from_millis(10));
    };
    let later = observed_user("later", "Later complete human text.", Some("earlier"));
    let split = later.len() / 2;
    let mut raw = initial;
    raw.extend_from_slice(&later[..split]);
    fs::write(&main, &raw).unwrap();
    await_pending(&world, &source, raw.len());
    let selected = format!("message:{}:earlier", reference());
    let page = model_call(&world, &json!({"ref":selected}), Some(20000));
    assert_eq!(
        page["shown"][0]["text"], "Independent earlier human text.",
        "{page}"
    );
    let block = model_call(&world, &json!({"ref":model_ref("earlier")}), Some(20000));
    assert_eq!(
        block["shown"][0]["text"], "Independent earlier human text.",
        "{block}"
    );
    let whole = model_call(&world, &transcript_call(&reference()), Some(20000));
    assert_eq!(whole["error"]["kind"], "not_yet_available", "{whole}");
    raw.extend_from_slice(&later[split..]);
    fs::write(&main, &raw).unwrap();
    let deadline = Instant::now() + WAIT;
    loop {
        let out = world.cli(&[&reference(), "read", "--format", "json"]);
        if matches!(out.status.code(), Some(0 | 5)) {
            let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
            if doc["messages"]
                .as_array()
                .is_some_and(|rows| rows.len() == 2)
            {
                break;
            }
        }
        assert!(Instant::now() < deadline, "finished suffix: {out:?}");
        thread::sleep(Duration::from_millis(10));
    }
    let body = world.work.path().join(SID).join("tool-results/call.txt");
    fs::create_dir_all(body.parent().unwrap()).unwrap();
    let call = json!({"type":"assistant","uuid":"call","sessionId":SID,"parentUuid":"later","requestId":"request","timestamp":"2026-10-01T08:01:00Z","message":{"id":"response","model":"synthetic-model","role":"assistant","content":[{"type":"tool_use","id":"call","name":"exec_command","input":{"cmd":"synthetic fixture"}}]}});
    let result = json!({"type":"user","uuid":"result","sessionId":SID,"parentUuid":"call","timestamp":"2026-10-01T08:02:00Z","toolUseResult":{"persistedOutputPath":body},"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"call","content":"Preview; complete output is stored externally."}]}});
    for record in [call, result] {
        raw.extend_from_slice((record.to_string() + "\n").as_bytes());
    }
    fs::write(&main, &raw).unwrap();
    let deadline = Instant::now() + WAIT;
    loop {
        let progress = world.raw_progress(&source);
        if progress["parser"]["acquired_upto"] == raw.len() as u64 && progress["active"] == false {
            break;
        }
        assert!(Instant::now() < deadline, "missing body: {progress}");
        thread::sleep(Duration::from_millis(10));
    }
    let independent = model_call(&world, &json!({"ref":selected}), Some(20000));
    assert_eq!(
        independent["shown"][0]["text"], "Independent earlier human text.",
        "{independent}"
    );
    let missing = model_call(
        &world,
        &transcript_call(&format!("message:{}:result", reference())),
        Some(20000),
    );
    assert_eq!(missing["error"]["kind"], "not_found", "{missing}");
    assert!(missing.get("shown").is_none(), "{missing}");
    fs::write(&body, b"Late synthetic result.").unwrap();
    let deadline = Instant::now() + WAIT;
    loop {
        let out = world.cli(&[&reference(), "read", "--format", "json"]);
        if matches!(out.status.code(), Some(0 | 5)) {
            let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
            if doc["messages"]
                .as_array()
                .is_some_and(|rows| rows.len() == 4)
            {
                break;
            }
        }
        assert!(Instant::now() < deadline, "late body: {out:?}");
        thread::sleep(Duration::from_millis(10));
    }
    // An unrecognized complete record has no source audit authority. A later
    // partial record cannot turn the earlier selected body into that authority.
    raw.extend_from_slice(b"{\"type\":\"unsupported-synthetic-record\"}\n{\"type\":");
    fs::write(&main, &raw).unwrap();
    await_pending(&world, &source, raw.len());
    let refused = model_call(&world, &json!({"ref":selected}), Some(20000));
    assert_eq!(refused["error"]["kind"], "not_yet_available", "{refused}");
}

fn await_pending(world: &World, source: &str, length: usize) {
    let deadline = Instant::now() + WAIT;
    loop {
        let progress = world.raw_progress(source);
        if progress["parser"]["acquired_upto"] == length as u64
            && progress["parser"]["status"] == "waiting_tail"
            && progress["active"] == false
        {
            assert_ne!(progress["source_rows"]["state"], "settled");
            return;
        }
        assert!(
            Instant::now() < deadline,
            "pending prefix: {progress}; {}",
            world.log()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn observed_user(native: &str, text: &str, parent: Option<&str>) -> Vec<u8> {
    (json!({"type":"user","uuid":native,"sessionId":SID,"parentUuid":parent,"isSidechain":false,"promptSource":"typed","timestamp":"2026-10-01T08:00:00.000Z","message":{"role":"user","content":text}}).to_string()+"\n").into_bytes()
}

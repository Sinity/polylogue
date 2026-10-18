//! Modern flat-path membership needs source-defined ordinary events, not EOF.
use super::*;
fn row(kind: &str, payload: Value) -> Value {
    json!({"type":kind,"payload":payload})
}
fn ordinary(sid: &str, family: &str) -> Vec<Value> {
    let mut metadata = json!({"id":sid,"source":"cli"});
    if family == "base_instructions" || family == "all" {
        metadata["base_instructions"] = json!({"text":"Neutral instructions"});
    }
    let mut records = vec![row("session_meta", metadata)];
    if matches!(family, "turn_start" | "turn_end" | "all") {
        records.push(row(
            "event_msg",
            json!({"type":"task_started","turn_id":"turn"}),
        ));
    }
    records.push(row("response_item",json!({"type":"message","id":"question","role":"user","content":[{"type":"input_text","text":"Neutral question"}]})));
    if family == "world_state" || family == "all" {
        records.push(row(
            "world_state",
            json!({"full":true,"state":{"instructions":"Neutral"}}),
        ));
    }
    if family == "turn_context" || family == "all" {
        records.push(row(
            "turn_context",
            json!({"turn_id":"turn","model":"synthetic"}),
        ));
    }
    records.push(row("response_item",json!({"type":"message","id":"answer","role":"assistant","content":[{"type":"output_text","text":"Neutral answer"}]})));
    let mut usage = json!({"type":"token_count","info":null});
    if family == "rate_limit" || family == "all" {
        usage["rate_limits"] = json!({"primary":{"used_percent":1},"secondary":null});
    }
    records.push(row("event_msg", usage));
    if family == "turn_end" || family == "all" {
        records.push(row(
            "event_msg",
            json!({"type":"task_complete","turn_id":"turn"}),
        ));
    }
    records
}
fn bytes(records: &[Value]) -> Vec<u8> {
    records
        .iter()
        .flat_map(|r| (r.to_string() + "\n").into_bytes())
        .collect()
}
fn import(world: &World, name: &str, raw: &[u8]) -> Value {
    let path = world.input(name, raw);
    let out = world.cli(&[
        "import",
        path.to_str().unwrap(),
        "--format",
        "codex-rollout",
        "--machine",
        "00112233445566778899aabbccddeeff",
        "--host",
        "synthetic",
    ]);
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    serde_json::from_slice(&out.stdout).unwrap()
}
fn await_idle(world: &World, input: &str) -> Value {
    let until = Instant::now() + WAIT;
    loop {
        let p = world.raw_progress(input);
        if p["parser"].is_object()
            && p["active"] == false
            && (p["failure"].is_string()
                || p["parser"]["status"] == "waiting_tail"
                || (p["parser"]["parsed_upto"] == p["length"]
                    && p["source_rows"]["state"] != "pending"))
        {
            return p;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(10));
    }
}
fn page(world: &World, sid: &str, branches: bool) -> Value {
    let mut call = transcript_call(&format!("codex:{sid}"));
    call["branches"] = json!(branches);
    model_call(world, &call, Some(65000))
}
#[test]
fn each_finite_event_family_and_empty_legacy_effects_preserve_current_default_values() {
    let mut world = World::new();
    for family in [
        "none",
        "base_instructions",
        "world_state",
        "turn_start",
        "turn_context",
        "rate_limit",
        "turn_end",
        "all",
    ] {
        let sid = format!("flat-{family}");
        let raw = bytes(&ordinary(&sid, family));
        let receipt = import(&world, &format!("{sid}.jsonl"), &raw);
        let input = receipt["retained"]["id"].as_str().unwrap();
        world.read(&format!("codex:{sid}"), input, 2);
        let selected = page(&world, &sid, false);
        assert_ne!(selected["outcome"], "error", "{family}: {selected}");
        let shown = selected["shown"].as_array().unwrap();
        assert_eq!(shown.len(), 2);
        assert_eq!(shown[0]["text"], "Neutral question");
        assert_eq!(shown[1]["text"], "Neutral answer");
        assert!(
            shown
                .iter()
                .all(|b| b["sources"].as_array().is_some_and(|s| !s.is_empty())
                    && b["field_origins"].as_array().is_some_and(|o| !o.is_empty()))
        );
        assert_eq!(selected["shown"], page(&world, &sid, true)["shown"]);
        if family == "all" {
            assert_eq!(selected["outcome"], "degraded"); // Event products still unsupported.
            fs::remove_file(world.work.path().join(format!("{sid}.jsonl"))).unwrap();
            world.restart();
            assert_eq!(selected["shown"], page(&world, &sid, false)["shown"]);
        }
    }
    let legacy=b"{\"id\":\"flat-legacy\"}\n{\"type\":\"message\",\"id\":\"question\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Neutral question\"}]}\n{\"type\":\"message\",\"id\":\"answer\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Neutral answer\"}]}\n";
    let receipt = import(&world, "flat-legacy.jsonl", legacy);
    world.read(
        "codex:flat-legacy",
        receipt["retained"]["id"].as_str().unwrap(),
        2,
    );
    assert_ne!(page(&world, "flat-legacy", false)["outcome"], "error");
}
#[test]
fn malformed_unknown_topology_and_lineage_controls_remain_unavailable() {
    let world = World::new();
    for mode in [
        "malformed",
        "wrong-turn",
        "unknown",
        "rollback",
        "compaction",
        "repeated",
        "fork",
        "conflict",
        "later-valid",
    ] {
        let sid = format!("refuse-{mode}");
        let mut records = ordinary(&sid, "all");
        match mode {
            "malformed" => records.push(row("world_state", json!({"full":true,"state":[]}))),
            "wrong-turn" => records.insert(3, row("turn_context", json!({"turn_id":"foreign"}))),
            "unknown" => records.push(row("event_msg", json!({"type":"future_control"}))),
            "rollback" => records.push(row(
                "event_msg",
                json!({"type":"thread_rolled_back","num_turns":1}),
            )),
            "compaction" => records.push(row("compacted", json!({"replacement_history":[]}))),
            "repeated" => records.push(row("session_meta", json!({"id":sid}))),
            "fork" => records[0]["payload"]["forked_from_id"] = json!("parent"),
            "conflict" => {
                records.push(row("response_item",json!({"type":"custom_tool_call","id":"same","call_id":"same","name":"exec_command","input":"first"})));
                records.push(row("response_item",json!({"type":"custom_tool_call","id":"same","call_id":"same","name":"different_tool","input":"changed"})));
            }
            "later-valid" => {
                records.push(row("world_state", json!({"full":true,"state":[]})));
                records.push(row("world_state", json!({"full":true,"state":{}})));
            }
            _ => unreachable!(),
        }
        let receipt = import(&world, &format!("{sid}.jsonl"), &bytes(&records));
        await_idle(&world, receipt["retained"]["id"].as_str().unwrap());
        let selected = page(&world, &sid, false);
        assert_eq!(selected["outcome"], "error", "{mode}: {selected}");
    }
}
#[test]
fn an_incomplete_tail_cannot_borrow_benign_event_authority() {
    let mut world = World::new();
    let mut raw = bytes(&ordinary("flat-partial", "all"));
    raw.extend_from_slice(b"{\"type\":\"event_msg\",\"payload\":");
    let receipt = import(&world, "flat-partial.jsonl", &raw);
    let input = receipt["retained"]["id"].as_str().unwrap();
    let p = await_idle(&world, input);
    assert_eq!(p["parser"]["status"], "waiting_tail");
    assert_eq!(page(&world, "flat-partial", false)["outcome"], "error");
    world.restart();
    assert_eq!(page(&world, "flat-partial", false)["outcome"], "error");
}

#[test]
fn a_lawful_input_correction_keeps_current_body_and_origin_without_inventing_a_result() {
    let mut world = World::new();
    let mut records = ordinary("flat-correction", "all");
    for input in ["first", "changed"] {
        records.push(row("response_item",json!({"type":"custom_tool_call","id":"call","call_id":"call","name":"exec_command","input":input})));
    }
    let raw = bytes(&records);
    let receipt = import(&world, "flat-correction.jsonl", &raw);
    let input = receipt["retained"]["id"].as_str().unwrap();
    let doc = world.read("codex:flat-correction", input, 3);
    assert_eq!(doc["messages"][2]["blocks"][0]["input"], "changed");
    let selected = page(&world, "flat-correction", false);
    assert_ne!(selected["outcome"], "error", "{selected}");
    let call = selected["shown"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["kind"] == "tool_call")
        .unwrap();
    assert_eq!(call["block_fields"]["input"]["json"], "\"changed\"");
    assert!(call["block_fields"].get("outcome").is_none());
    assert!(
        call["field_origins"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
    );
    fs::remove_file(world.work.path().join("flat-correction.jsonl")).unwrap();
    world.restart();
    assert_eq!(
        selected["shown"],
        page(&world, "flat-correction", false)["shown"]
    );
}

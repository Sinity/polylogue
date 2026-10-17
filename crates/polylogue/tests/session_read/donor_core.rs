//! Actual public default transcript reads require retained flat-path evidence.
use super::*;

fn codex_import(world: &World, name: &str, bytes: &[u8]) -> Value {
    let path = world.input(name, bytes);
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
    assert!(
        matches!(out.status.code(), Some(0 | 5)),
        "{out:?}; {}",
        world.log()
    );
    let receipt: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        receipt["source_rows"]["state"] == "settled"
            || receipt["source_rows"]["usable_result"] == true,
        "{receipt}"
    );
    receipt
}

#[test]
fn flat_codex_conversations_read_without_branches_but_rollback_and_ambiguous_controls_refuse() {
    for (name, reference, bytes) in [
        (
            "legacy.jsonl",
            "codex:legacy-response-session",
            include_bytes!("../../../sources/tests/fixtures/donor/legacy-response-records.jsonl")
                .as_slice(),
        ),
        (
            "headers.jsonl",
            "codex:tool-result-headers",
            include_bytes!("../../../sources/tests/fixtures/donor/tool-result-headers.jsonl")
                .as_slice(),
        ),
    ] {
        let world = World::new();
        let receipt = codex_import(&world, name, bytes);
        let page = model_call(&world, &transcript_call(reference), Some(65000));
        assert_ne!(page["outcome"], "error", "{page}; {receipt}");
        assert_eq!(page["primary_content_complete"], true);
        if reference == "codex:legacy-response-session" {
            let out = world.cli(&[reference, "read", "--format", "json"]);
            assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
            let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(doc["session"]["started_at"], "2025-08-01T10:00:00.000Z");
        }
        assert!(
            page["shown"]
                .as_array()
                .unwrap()
                .iter()
                .any(|shown| shown["kind"] == "tool_result")
        );
    }
    for kind in ["thread_rolled_back", "future_topology_control"] {
        let world = World::new();
        let mut bytes =
            include_bytes!("../../../sources/tests/fixtures/donor/tool-result-headers.jsonl")
                .to_vec();
        bytes.extend_from_slice(
            (json!({"type":"event_msg","payload":{"type":kind,"num_turns":1}}).to_string() + "\n")
                .as_bytes(),
        );
        let receipt = codex_import(&world, "uncertain.jsonl", &bytes);
        assert_eq!(receipt["source_rows"]["state"], "unavailable");
        let page = model_call(
            &world,
            &transcript_call("codex:tool-result-headers"),
            Some(65000),
        );
        assert_eq!(page["outcome"], "error", "{kind}: {page}");
        assert!(page.get("shown").is_none());
        let known = model_call(
            &world,
            &known_catalog_call("codex:tool-result-headers"),
            Some(65000),
        );
        assert_ne!(known["outcome"], "error", "{known}");
        assert_eq!(known["outcome"], "degraded");
    }
}

#[test]
fn mixed_prompt_history_cannot_certify_complete_codex_conversation_membership() {
    let mut world = World::new();
    codex_import(
        &world,
        "headers.jsonl",
        include_bytes!("../../../sources/tests/fixtures/donor/tool-result-headers.jsonl"),
    );
    world.kill();
    let home = world.work.path().join("codex-home");
    fs::create_dir_all(home.join("sessions")).unwrap();
    fs::write(home.join("history.jsonl"), b"{\"session_id\":\"tool-result-headers\",\"text\":\"Independent retained prompt reference.\",\"ts\":1792250000}\n").unwrap();
    let config = world.input(
        "codex-config.json",
        json!({"codex_home":{"home":home,"machine":"synthetic-machine","host":"synthetic-host"}})
            .to_string()
            .as_bytes(),
    );
    let out = world.cli(&[
        "daemon",
        "--config",
        config.to_str().unwrap(),
        "--until-idle",
        "--json",
    ]);
    assert_eq!(out.status.code(), Some(5), "{out:?}");
    world.child = Some(OwnedChild(
        world
            .command()
            .arg("daemon")
            .stdout(Stdio::null())
            .stderr(Stdio::from(File::create(&world.log).unwrap()))
            .spawn()
            .unwrap(),
    ));
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let status = world.cli(&["status"]);
        if status.status.code() == Some(5) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "mixed source startup: {status:?}; {}",
            world.log()
        );
        thread::sleep(Duration::from_millis(10));
    }
    for call in [
        transcript_call("codex:tool-result-headers"),
        known_catalog_call("codex:tool-result-headers"),
    ] {
        let page = model_call(&world, &call, Some(65000));
        assert_eq!(page["outcome"], "error", "{page}");
        assert!(page.get("shown").is_none());
    }
}

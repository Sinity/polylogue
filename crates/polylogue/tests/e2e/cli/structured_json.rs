use super::*;

const SECRET: &str = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AAAAAAAA";

#[test]
fn crossing_quoted_credentials_are_masked_by_public_read_and_export_without_changing_stored_text() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-synthetic");
    std::fs::create_dir_all(&project).unwrap();
    let texts = [
        "--secret=\"unfinished --token=\"synthetic-later-value\" tail-double",
        "--secret='unfinished --token='synthetic-later-value' tail-single",
    ];
    let source = project.join("crossing.jsonl");
    let raw = format!(
        "{}{}",
        line("u1", "crossing", texts[0]),
        line("u2", "crossing", texts[1])
    );
    std::fs::write(&source, &raw).unwrap();
    let archive = dir.path().join("archive");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(2));
    for args in [
        vec!["read", "claude-code:crossing"],
        vec!["--json", "read", "claude-code:crossing"],
        vec!["export", "claude-code:crossing", "--format", "json"],
        vec!["export", "claude-code:crossing", "--format", "html"],
    ] {
        let out = cli(&env, &args);
        assert!(out.status.success(), "{args:?}: {out:?}");
        let output = String::from_utf8(out.stdout).unwrap();
        assert!(
            !output.contains("synthetic-later-value"),
            "{args:?}: {output}"
        );
        assert!(output.contains("tail-double") && output.contains("tail-single"));
    }
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    let reader = pool.reader().unwrap();
    let session = polylogue_model::SessionId::parse("claude-code:crossing").unwrap();
    let messages = reader.messages(&session, false).unwrap();
    assert_eq!(messages.len(), 2);
    for (message, text) in messages.iter().zip(texts) {
        assert_eq!(
            message.blocks[0],
            polylogue_model::Block::Text { text: text.into() }
        );
    }
    assert_eq!(std::fs::read_to_string(source).unwrap(), raw);
}

fn structured_command(args: &[&str]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-synthetic");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("masked.jsonl"),
        line("u1", "masked", &format!("synthetic credential {SECRET}")),
    )
    .unwrap();
    let archive = dir.path().join("archive");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(1));
    let out = cli(&env, args);
    assert!(out.status.success(), "{out:?}");
    out
}

#[test]
fn structured_json_read_masks_credentials_through_the_public_command() {
    let out = structured_command(&["--json", "read", "claude-code:masked"]);
    let value = json(&out);
    assert_eq!(value["session"]["id"], "claude-code:masked");
    assert!(
        value["messages"][0]["blocks"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("synthetic credential ")
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains(SECRET),
        "structured read exposed the synthetic credential"
    );
}

#[test]
fn structured_json_export_masks_credentials_and_preserves_the_import_shape() {
    let out = structured_command(&["export", "claude-code:masked", "--format", "json"]);
    let text = String::from_utf8(out.stdout).unwrap();
    let document = polylogue_model::export::Document::from_json(&text).unwrap();
    assert_eq!(
        document.sessions[0].session.id.to_string(),
        "claude-code:masked"
    );
    assert_eq!(document.sessions[0].messages.len(), 1);
    assert!(
        !text.contains(SECRET),
        "structured export exposed the synthetic credential"
    );
}

#[test]
fn json_export_import_round_trip_keeps_structured_fields_that_are_not_secret() {
    use polylogue_model::export::Document;
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("source");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
    ];
    let _source = start_daemon(&env);
    let seed: Document = serde_json::from_value(serde_json::json!({
        "format": "polylogue-export", "version": 1,
        "sessions": [{"session": {"id": "claude-code:fidelity", "kind": "main", "repo": "https://example.invalid/synthetic/project", "mode": "bg"},
        "messages": [{"id": "claude-code:fidelity/m1", "session": "claude-code:fidelity", "role": "assistant", "model": "synthetic-model", "duration_ms": 0,
            "production": {"effort": "high"}, "blocks": [
                {"type": "tool_call", "call_id": "c1", "tool": "Bash", "action": "shell", "command": "printf synthetic", "input": {"empty": "", "nil": null, "count": 0, "enabled": false, "nested": {"paths": ["a", "b"]}}},
                {"type": "tool_result", "call_id": "c1", "outcome": {"kind": "ok"}, "output": "", "data": {"values": [0, false, null, ""]}}
            ]}]}]
    })).unwrap();
    let input = dir.path().join("seed.json");
    std::fs::write(&input, seed.to_json().unwrap()).unwrap();
    let imported = cli(&env, &["import", input.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let first = cli(
        &env,
        &["export", "claude-code:fidelity", "--format", "json"],
    );
    assert!(first.status.success(), "{first:?}");
    let first = Document::from_json(std::str::from_utf8(&first.stdout).unwrap()).unwrap();
    assert_eq!(first, seed);
    let transfer = dir.path().join("transfer.json");
    std::fs::write(&transfer, first.to_json().unwrap()).unwrap();
    let restored = dir.path().join("restored");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", restored.as_path()),
    ];
    let _restored = start_daemon(&env);
    let imported = cli(&env, &["import", transfer.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let second = cli(
        &env,
        &["export", "claude-code:fidelity", "--format", "json"],
    );
    assert!(second.status.success(), "{second:?}");
    assert_eq!(
        Document::from_json(std::str::from_utf8(&second.stdout).unwrap()).unwrap(),
        first
    );
}

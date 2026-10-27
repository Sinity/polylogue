use super::settings::merge_hooks;
use super::*;

#[test]
fn hook_spool_files_get_one_line_per_event() {
    let dir = tempfile::tempdir().unwrap();
    append_to_spool(
        dir.path(),
        Harness::ClaudeCode,
        "Stop",
        &json!({"session_id": "a"}),
    )
    .unwrap();
    append_to_spool(
        dir.path(),
        Harness::Codex,
        "SessionEnd",
        &json!({"session_id": "b"}),
    )
    .unwrap();
    let file = dir
        .path()
        .join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let text = std::fs::read_to_string(file).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["provider"], "claude-code");
    assert_eq!(lines[0]["session_id"], "a");
    assert_eq!(lines[1]["event"], "SessionEnd");
    assert_eq!(lines[1]["provider"], "codex");
    assert_eq!(lines[1]["session_id"], "b");
}

#[test]
fn a_hermes_turn_keeps_its_event_without_the_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let payload = json!({
        "hook_event_name": "pre_llm_call",
        "session_id": "20260101_000000_abc",
        "cwd": "/tmp",
        "extra": {"is_first_turn": true, "user_message": "hi", "conversation_history": [{"role": "user"}]},
    });
    append_to_spool(dir.path(), Harness::Hermes, "pre_llm_call", &payload).unwrap();
    let file = dir
        .path()
        .join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let line: Value = serde_json::from_str(std::fs::read_to_string(file).unwrap().trim()).unwrap();
    assert_eq!(line["provider"], "hermes");
    assert!(
        line["payload"]["extra"]
            .get("conversation_history")
            .is_none()
    );
    assert_eq!(line["payload"]["extra"]["user_message"], "hi");
    let native = line["session_id"].as_str().unwrap();
    assert!(
        native.starts_with("20260101_000000_abc@profile-"),
        "{native}"
    );
}

#[test]
fn each_harness_gets_context_in_its_own_shape() {
    // Without a daemon the hooks fail open: no context, no output.
    let start = json!({"session_id": "a", "cwd": "/nonexistent", "source": "startup"});
    assert!(respond(Harness::Codex, "Stop", &start).is_none());
    assert!(respond(Harness::Hermes, "SessionStart", &start).is_none());
}

#[test]
fn pasted_ids_are_found_in_prose() {
    let ids = pasted_session_ids(
        "pick up from claude-code:7f3e9a1c-0000-4000-8000-000000000001, and see \
         `0199a1b2-3c4d-7e8f-9012-345678901234`.",
    );
    let texts: Vec<String> = ids.iter().map(ToString::to_string).collect();
    assert_eq!(
        texts,
        [
            "claude-code:7f3e9a1c-0000-4000-8000-000000000001",
            "claude-code:0199a1b2-3c4d-7e8f-9012-345678901234",
            "codex:0199a1b2-3c4d-7e8f-9012-345678901234",
        ]
    );
    assert!(pasted_session_ids("no ids here: http://x").is_empty());
}

#[test]
fn hook_install_merges_and_is_idempotent() {
    let other = json!({
        "model": "opus",
        "hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": "notify-send done"}]}]
        }
    });
    let once = merge_hooks(other.clone(), Harness::ClaudeCode, true).unwrap();
    let twice = merge_hooks(once.clone(), Harness::ClaudeCode, true).unwrap();
    assert_eq!(once, twice);
    assert_eq!(once["model"], "opus");
    let stop = once["hooks"]["Stop"].as_array().unwrap();
    assert_eq!(stop.len(), 2);
    assert_eq!(stop[0]["hooks"][0]["command"], "notify-send done");
    let removed = merge_hooks(once, Harness::ClaudeCode, false).unwrap();
    assert_eq!(removed["hooks"]["Stop"].as_array().unwrap().len(), 1);
    assert!(removed["hooks"].get("SessionStart").is_none());
}

#[test]
fn install_writes_a_file_and_refuses_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    install(&path, Harness::ClaudeCode, true).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("polylogue hook SessionStart"));
    let link = dir.path().join("linked.json");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert!(install(&link, Harness::ClaudeCode, true).is_err());
}

#[test]
fn codex_hooks_name_the_harness_and_replace_their_own() {
    let once = merge_hooks(json!({}), Harness::Codex, true).unwrap();
    let start = &once["hooks"]["SessionStart"][0]["hooks"][0]["command"];
    assert_eq!(start, "polylogue hook SessionStart --harness codex");
    assert!(
        once["hooks"].get("Notification").is_none(),
        "Codex has no such event"
    );
    let twice = merge_hooks(once.clone(), Harness::Codex, true).unwrap();
    assert_eq!(once, twice);
    let removed = merge_hooks(twice, Harness::Codex, false).unwrap();
    assert!(removed["hooks"].as_object().unwrap().is_empty());
}

#[test]
fn install_preserves_permissions_and_ignores_an_old_temporary_name() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let other = dir.path().join("other.json");
    std::fs::write(&path, r#"{"model":"example"}"#).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    std::fs::write(&other, b"unchanged").unwrap();
    let stale = dir
        .path()
        .join(format!(".settings.json.polylogue-{}", std::process::id()));
    std::os::unix::fs::symlink(&other, stale).unwrap();
    install(&path, Harness::ClaudeCode, true).unwrap();
    assert_eq!(std::fs::read(&other).unwrap(), b"unchanged");
    assert!(!path.is_symlink());
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o640
    );
    let settings: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(settings["model"], "example");
    assert!(settings["hooks"]["SessionStart"].is_array());
}

mod settings;

#[test]
fn web_links_resolve_to_the_same_sessions_as_captures() {
    let links = [
        ("https://chatgpt.com/c/demo?model=x", "chatgpt:demo"),
        ("https://chat.openai.com/c/demo", "chatgpt:demo"),
        (
            "https://chatgpt.com/g/g-project/c/demo#turn",
            "chatgpt:demo",
        ),
        ("https://gemini.google.com/u/0/app/demo", "gemini-app:demo"),
        (
            "https://aistudio.google.com/u/2/prompts/demo",
            "ai-studio:demo",
        ),
        ("https://claude.ai/chat/demo", "claude-ai:demo"),
        ("https://gemini.google.com/app/demo", "gemini-app:demo"),
        ("https://aistudio.google.com/prompts/demo", "ai-studio:demo"),
        ("https://grok.com/c/demo", "grok:demo"),
    ];
    for (url, expected) in links {
        assert_eq!(
            pasted_session_ids(&format!("continue [{url}]({url}).")),
            vec![SessionId::parse(expected).unwrap()]
        );
    }
    for url in [
        "https://chatgpt.com.evil.invalid/c/demo",
        "http://chatgpt.com/c/demo",
        "https://u@chatgpt.com/c/demo",
        "https://chatgpt.com/c/demo/extra",
        "https://chatgpt.com/share/demo",
        "https://claude.ai/g/project/chat/demo",
        "https://grok.com/u/0/c/demo",
        "https://gemini.google.com/u/account/app/demo",
        "https://aistudio.google.com/g/project/prompts/demo",
        "https://chatgpt.com/u/0/c/demo",
    ] {
        assert!(pasted_session_ids(url).is_empty(), "{url}");
    }
}

#[test]
fn gemini_hooks_use_the_documented_events_and_millisecond_timeouts() {
    let settings =
        json!({"hooks":{"BeforeAgent":[{"hooks":[{"type":"command", "command":"other"}]}]}});
    let once = merge_hooks(settings.clone(), Harness::GeminiCli, true).unwrap();
    assert_eq!(
        merge_hooks(once.clone(), Harness::GeminiCli, true).unwrap(),
        once
    );
    assert_eq!(
        once["hooks"]["SessionStart"][0]["hooks"][0]["timeout"],
        10000
    );
    assert_eq!(once["hooks"]["BeforeTool"][0]["hooks"][0]["timeout"], 2000);
    assert_eq!(
        once["hooks"]["BeforeAgent"][1]["hooks"][0]["command"],
        "polylogue hook BeforeAgent --harness gemini-cli"
    );
    assert_eq!(
        merge_hooks(once, Harness::GeminiCli, false).unwrap(),
        settings
    );
    assert_eq!(
        session_of(Harness::GeminiCli, &json!({"session_id":"test"}))
            .unwrap()
            .to_string(),
        "gemini-cli:test"
    );
}

#[test]
fn gemini_context_output_matches_the_native_protocol() {
    assert_eq!(
        output_for(Harness::GeminiCli, "SessionStart", "brief"),
        json!({"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"brief"}})
    );
    assert_eq!(
        output_for(Harness::GeminiCli, "BeforeAgent", "takeover"),
        json!({"hookSpecificOutput":{"hookEventName":"BeforeAgent","additionalContext":"takeover"}})
    );
}

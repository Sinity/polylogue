use super::*;

#[test]
fn carriers_get_one_line_per_event() {
    let dir = tempfile::tempdir().unwrap();
    append_carrier(dir.path(), "Stop", &json!({"session_id": "a"})).unwrap();
    append_carrier(dir.path(), "SessionEnd", &json!({"session_id": "a"})).unwrap();
    let file = dir
        .path()
        .join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let text = std::fs::read_to_string(file).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["event"], "SessionEnd");
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
    let once = merge_hooks(other.clone(), true).unwrap();
    let twice = merge_hooks(once.clone(), true).unwrap();
    assert_eq!(once, twice);
    assert_eq!(once["model"], "opus");
    let stop = once["hooks"]["Stop"].as_array().unwrap();
    assert_eq!(stop.len(), 2);
    assert_eq!(stop[0]["hooks"][0]["command"], "notify-send done");
    let removed = merge_hooks(once, false).unwrap();
    assert_eq!(removed["hooks"]["Stop"].as_array().unwrap().len(), 1);
    assert!(removed["hooks"].get("SessionStart").is_none());
}

#[test]
fn install_writes_a_file_and_refuses_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    install(&path, true).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("polylogue hook SessionStart"));
    let link = dir.path().join("linked.json");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert!(install(&link, true).is_err());
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
    install(&path, true).unwrap();
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

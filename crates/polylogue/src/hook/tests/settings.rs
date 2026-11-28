use super::*;
use crate::installation::backup_path;
use std::os::unix::fs::PermissionsExt;

#[test]
fn malformed_or_conflicting_settings_refuse_without_changing_bytes_or_permissions() {
    for (text, expected) in [
        ("not json", "expected ident at line 2 column 2"),
        ("null", "settings: expected an object"),
        ("[]", "settings: expected an object"),
        ("42", "settings: expected an object"),
        (r#"{"hooks":null}"#, "hooks: expected an object"),
        (r#"{"hooks":[]}"#, "hooks: expected an object"),
        (r#"{"hooks":{"Stop":{}}}"#, "hooks.Stop: expected an array"),
        (
            r#"{"hooks":{"Stop":[null]}}"#,
            "hooks.Stop[0]: expected an object",
        ),
        (
            r#"{"hooks":{"Stop":[{}]}}"#,
            "hooks.Stop[0].hooks: expected an array",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":{}}]}}"#,
            "hooks.Stop[0].hooks: expected an array",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":[null]}]}}"#,
            "hooks.Stop[0].hooks[0]: expected an object",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":42}]}]}}"#,
            "hooks.Stop[0].hooks[0].command: expected a string",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":[{"type":"command"}]}]}}"#,
            "hooks.Stop[0].hooks[0].command: expected a string",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":[{"type":1}]}]}}"#,
            "hooks.Stop[0].hooks[0].type: expected a string",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":[{"command":"echo example"}]}]}}"#,
            "hooks.Stop[0].hooks[0].type: expected a string",
        ),
        (
            r#"{"hooks":{},"hooks":{"Stop":[]}}"#,
            "duplicate settings key \"hooks\" at line 2 column 19",
        ),
        (
            r#"{"hooks":{"Stop":[],"Stop":[]}}"#,
            "duplicate settings key \"Stop\" at line 2 column 26",
        ),
        (
            r#"{"future":{"keep":1,"keep":2},"hooks":{}}"#,
            "duplicate settings key \"keep\" at line 2 column 26",
        ),
        (
            r#"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"echo keep","command":"polylogue internal hook Stop"}]}]}}"#,
            "duplicate settings key \"command\" at line 2 column 77",
        ),
        (
            r#"{"hooks":{},"\u0068ooks":{"Stop":[]}}"#,
            "duplicate settings key \"hooks\" at line 2 column 24",
        ),
    ] {
        for on in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("settings.json");
            let original = format!(" \n{text}\n ");
            std::fs::write(&path, &original).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
            let result = install(
                &path,
                Harness::ClaudeCode,
                || panic!("malformed settings must be refused before probing the harness"),
                on,
            );
            assert_eq!(
                result.unwrap_err().to_string(),
                expected,
                "on={on}, input={text}"
            );
            assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o640
            );
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }
}

#[test]
fn mixed_hook_entries_keep_other_commands_matchers_and_unknown_fields() {
    for on in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let other = json!({"type":"command","command":"notify-send done","timeout":7});
        let prompt = json!({"type":"prompt","prompt":"synthetic prompt","future":{"value":7}});
        let entry = json!({"matcher":"Bash|Read","future":{"keep":true},"hooks":[
            {"type":"command","command":"polylogue internal hook Stop","timeout":2},other.clone(),prompt.clone()
        ]});
        let original = json!({"model":"example","hooks":{"Stop":[entry],"UnknownFutureEvent":[]},"future":{"a":[{"key":1},{"key":2}]}});
        std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
        install(&path, Harness::ClaudeCode, || None, on).unwrap();
        let once: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            once["hooks"]["Stop"][0],
            json!({"matcher":"Bash|Read","future":{"keep":true},"hooks":[other,prompt]})
        );
        assert_eq!(once["hooks"]["UnknownFutureEvent"], json!([]));
        assert_eq!(once["model"], original["model"]);
        assert_eq!(once["future"], original["future"]);
        install(&path, Harness::ClaudeCode, || None, on).unwrap();
        let twice: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(once, twice);
    }
}

#[test]
fn unknown_hook_kinds_and_json_numbers_survive_setup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let original = r#"{"large":900719925474099312345,"real":1.23456789012345,"hooks":{"Stop":[{"hooks":[{"type":"future","command":"polylogue internal hook Stop","opaque":null}]}]}}"#;
    std::fs::write(&path, original).unwrap();
    let expected: Value = serde_json::from_str(original).unwrap();
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    let installed: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(installed["large"], expected["large"]);
    assert_eq!(installed["real"], expected["real"]);
    assert_eq!(installed["hooks"]["Stop"][0], expected["hooks"]["Stop"][0]);
    assert_eq!(
        installed["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        "polylogue internal hook SessionStart"
    );
    install(&path, Harness::ClaudeCode, || None, false).unwrap();
    let actual: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn uninstall_preserves_empty_containers_and_shell_programs_it_did_not_install() {
    for original in [
        json!({"model":"example"}),
        json!({"hooks":{"Stop":[]}}),
        json!({"hooks":{"Stop":[{"hooks":[]}]}}),
        json!({"hooks":{"Stop":[{"hooks":[{"type":"command","command":"polylogue internal hook Stop && notify-send done"}]}]}}),
    ] {
        assert_eq!(
            merge_hooks(original.clone(), Harness::ClaudeCode, None, false).unwrap(),
            original
        );
    }
}

#[test]
fn only_valid_installs_probe_the_version_and_keep_confirmed_events() {
    for on in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, "{}").unwrap();
        let probed = std::cell::Cell::new(false);
        install(
            &path,
            Harness::ClaudeCode,
            || {
                probed.set(true);
                HarnessVersion::parse("2.1.316")
            },
            on,
        )
        .unwrap();
        assert_eq!(probed.get(), on);
        let settings: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for (event, _) in CONFIRMED_FROM {
            assert_eq!(settings["hooks"].get(*event).is_some(), on);
        }
    }
}

/// Settings as a person wrote them: keys out of order, four-space indents,
/// another program's hook beside an event Polylogue also uses, and a number serde
/// would print differently.
const HAND_WRITTEN: &str = r#"{
    "zeta": 1.50,
    "hooks": {
        "Stop": [
            {
                "matcher": "",
                "hooks": [ { "type": "command", "command": "notify-send done" } ]
            }
        ],
        "Custom": []
    },
    "alpha": "\u00e9"
}
"#;

#[test]
fn install_keeps_other_entries_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    std::fs::write(&path, HAND_WRITTEN).unwrap();
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    let installed = std::fs::read_to_string(&path).unwrap();
    let other_hook = r#"            {
                "matcher": "",
                "hooks": [ { "type": "command", "command": "notify-send done" } ]
            }"#;
    for kept in [
        "{\n    \"zeta\": 1.50,\n    \"hooks\": {\n        \"Stop\": [\n",
        other_hook,
        "\"Custom\": []",
        "\n    \"alpha\": \"\\u00e9\"\n}\n",
    ] {
        assert!(installed.contains(kept), "{kept:?} lost from:\n{installed}");
    }
    assert!(installed.ends_with("    },\n    \"alpha\": \"\\u00e9\"\n}\n"));
    let value: Value = serde_json::from_str(&installed).unwrap();
    assert_eq!(
        value["hooks"]["Stop"][1]["hooks"][0]["command"],
        "polylogue internal hook Stop"
    );
    assert_eq!(
        value["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        "polylogue internal hook SessionStart"
    );
    // Installing again rewrites Polylogue's entries to the same bytes.
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), installed);
    // Removal without the backup edits the text back to the original.
    std::fs::remove_file(backup_path(&path)).unwrap();
    install(&path, Harness::ClaudeCode, || None, false).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), HAND_WRITTEN);
}

#[test]
fn install_writes_backup_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    std::fs::write(&path, HAND_WRITTEN).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    let backup = backup_path(&path);
    assert_eq!(backup, dir.path().join("settings.json.polylogue-backup"));
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), HAND_WRITTEN);
    assert_eq!(
        std::fs::metadata(&backup).unwrap().permissions().mode() & 0o777,
        0o640
    );
    let edited = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"zeta\": 1.50", "\"zeta\": 2");
    std::fs::write(&path, &edited).unwrap();
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), HAND_WRITTEN);
    // A file that did not exist has no bytes to keep.
    let fresh = dir.path().join("fresh.json");
    install(&fresh, Harness::ClaudeCode, || None, true).unwrap();
    assert!(!backup_path(&fresh).exists());
}

#[test]
fn uninstall_restores_original_bytes() {
    for version in [None, HarnessVersion::parse("2.1.316")] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let original = "{\"model\":\"example\",\"n\":1e2}";
        std::fs::write(&path, original).unwrap();
        install(&path, Harness::ClaudeCode, || version.clone(), true).unwrap();
        install(
            &path,
            Harness::ClaudeCode,
            || panic!("removal probes nothing"),
            false,
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        assert!(!backup_path(&path).exists());
    }
    // A change made after the install is kept; only Polylogue's entries go,
    // with the `hooks` object they alone filled.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    std::fs::write(&path, "{\"model\":\"example\"}").unwrap();
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    let changed = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"example\"", "\"other\"");
    std::fs::write(&path, changed).unwrap();
    install(&path, Harness::ClaudeCode, || None, false).unwrap();
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "{\"model\":\"other\"}"
    );
}

#[test]
fn appended_events_keep_the_declared_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    std::fs::write(&path, r#"{"hooks":{"Other":[]}}"#).unwrap();
    install(&path, Harness::Codex, || None, true).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let position = |event: &str| text.find(&format!("\"{event}\"")).unwrap();
    let order = [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
    ];
    for pair in order.windows(2) {
        assert!(position(pair[0]) < position(pair[1]), "{text}");
    }
}

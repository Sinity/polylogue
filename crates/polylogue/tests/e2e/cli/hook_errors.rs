use super::*;
use serde_json::json;

#[test]
fn cli_read_shows_native_hook_errors_without_an_injected_context_message() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-hooks");
    std::fs::create_dir_all(&project).unwrap();
    let source = project.join("hooks.jsonl");
    let summary = json!({"type":"system", "subtype":"stop_hook_summary", "uuid":"h1",
        "sessionId":"hooks", "hookErrors":["synthetic CLI hook failure"],
        "hookAdditionalContext":[], "preventedContinuation":true, "stopReason":"check blocked"});
    let original = format!("{summary}\n");
    std::fs::write(&source, &original).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let id = polylogue_model::SessionId::parse("claude-code:hooks").unwrap();
    loop {
        let ready = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive))
            .and_then(|pool| {
                let ready = pool
                    .reader()?
                    .events(&id)?
                    .iter()
                    .any(|event| event.data == summary);
                Ok(ready)
            })
            .unwrap_or(false);
        if ready {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // The pass that stored the summary has also derived from it.
    wait_status(&env, |_| true);
    for format in ["terminal", "markdown", "html", "json"] {
        let out = cli(
            &env,
            &["--json", "read", "claude-code:hooks", "--format", format],
        );
        assert!(out.status.success(), "{out:?}");
        let value = json(&out);
        assert!(
            value.to_string().contains("synthetic CLI hook failure"),
            "{format}: {value}"
        );
        assert!(
            value.to_string().contains("check blocked"),
            "{format}: {value}"
        );
    }
    let out = cli(
        &env,
        &["--json", "read", "claude-code:hooks", "--budget", "1000"],
    );
    assert!(out.status.success(), "{out:?}");
    assert!(
        json(&out)["text"]
            .as_str()
            .unwrap()
            .contains("synthetic CLI hook failure")
    );
    assert_eq!(std::fs::read_to_string(source).unwrap(), original);
}

#[test]
fn hook_setup_cli_refuses_malformed_settings_without_replacing_them() {
    for original in [
        "null",
        r#"{"hooks":[]}"#,
        r#"{"hooks":{},"hooks":{"Stop":[]}}"#,
    ] {
        for action in ["install", "uninstall"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("settings.json");
            std::fs::write(&path, original).unwrap();
            let out = crate::command(dir.path())
                .args([
                    "--json",
                    "hooks",
                    action,
                    "--settings",
                    path.to_str().unwrap(),
                ])
                .output()
                .unwrap();
            assert_eq!(out.status.code(), Some(1), "{out:?}");
            assert_eq!(json(&out)["error"]["kind"], "io", "{out:?}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        }
    }
}

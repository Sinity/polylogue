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
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let ready = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive))
            .and_then(|pool| {
                Ok(pool
                    .reader()?
                    .events(&id)?
                    .iter()
                    .any(|event| event.data == summary))
            })
            .unwrap_or(false);
        if ready {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "native hook summary was not ingested"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
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

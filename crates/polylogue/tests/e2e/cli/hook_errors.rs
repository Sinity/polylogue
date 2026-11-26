use super::*;
use serde_json::json;

#[test]
fn cli_read_shows_hook_error_events_in_every_format() {
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
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    // The daemon has caught up: the pass that stored the summary has also
    // derived from it.
    let id = polylogue_model::SessionId::parse("claude-code:hooks").unwrap();
    let stored = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive))
        .unwrap()
        .reader()
        .unwrap()
        .events(&id)
        .unwrap();
    assert!(
        stored.iter().any(|event| event.data == summary),
        "{stored:?}"
    );
    for format in ["terminal", "markdown", "html", "json"] {
        let items = hook_items(&env, format, "10000");
        assert!(
            items.iter().any(|item| match &item.kind {
                polylogue_model::read::ItemKind::Other { data, .. } =>
                    data["event"]["data"] == summary,
                _ => false,
            }),
            "{format}: hook event must keep its recorded fields: {items:?}"
        );
    }
    let items = hook_items(&env, "json", "1000");
    assert!(
        items.iter().any(|item| match &item.kind {
            polylogue_model::read::ItemKind::Other { data, .. } => data["event"]["data"] == summary,
            _ => false,
        }),
        "paged hook event lost source data: {items:?}"
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
        for remove in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("settings.json");
            std::fs::write(&path, original).unwrap();
            use std::os::unix::fs::PermissionsExt;
            let tools = dir.path().join("tools");
            std::fs::create_dir(&tools).unwrap();
            let probe = tools.join("claude");
            std::fs::write(
                &probe,
                "#!/bin/sh\n: > \"$HOOK_PROBE_MARKER\"\nprintf '2.1.316\\n'\n",
            )
            .unwrap();
            std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o700)).unwrap();
            let marker = dir.path().join("probed");
            let mut command = crate::command(dir.path());
            command.args([
                "--json",
                "install",
                "claude-code",
                "--settings",
                path.to_str().unwrap(),
            ]);
            if remove {
                command.arg("--remove");
            }
            let out = command
                .env("PATH", &tools)
                .env("HOOK_PROBE_MARKER", &marker)
                .output()
                .unwrap();
            assert!(!marker.exists(), "malformed settings invoked the harness");
            assert_eq!(out.status.code(), Some(4), "{out:?}");
            assert_eq!(json(&out)["error"]["kind"], "io", "{out:?}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        }
    }
}

fn hook_items(
    fixture: &crate::Fixture,
    format: &str,
    budget: &str,
) -> Vec<polylogue_model::read::ReadItem> {
    let arguments: Vec<String> = [
        "--json",
        "session:claude-code:hooks",
        "read",
        "--format",
        format,
        "--budget",
        budget,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let output = cli(
        fixture,
        &arguments.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    assert!(output.status.success(), "{output:?}");
    let mut page = read_page(json(&output));
    let mut items = super::read_current::Items::default();
    let mut seen = std::collections::BTreeSet::new();
    loop {
        assert!(
            !page.document.items.is_empty(),
            "hook read made no progress"
        );
        items.push(&page);
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(
            seen.insert(serde_json::to_string(&next.request).unwrap()),
            "repeated continuation"
        );
        page = followed_page(fixture, &next);
    }
    items.into_values()
}

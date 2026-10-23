use super::*;
use serde_json::{Value, json};

fn exported(name: &str, text: &str, parent: Option<(&str, Option<&str>)>) -> Value {
    let id = format!("claude-code:{name}");
    let links = parent.map(|(parent, anchor)| json!({
        "kind": "fork", "parent": format!("claude-code:{parent}"), "child": id,
        "at": anchor.map(|s| format!("claude-code:{s}")), "evidence": "synthetic fork marker"
    })).into_iter().collect::<Vec<_>>();
    json!({"session": {"id": id, "kind": "main"}, "messages": [{
        "id": format!("claude-code:{name}-message"), "session": id, "role": "user",
        "blocks": [{"type": "text", "text": text}]
    }], "links": links})
}

fn logical(env: &[(&str, &Path)], session: &str, format: &str) -> Value {
    let out = cli(
        env,
        &["--json", "read", session, "--logical", "--format", format],
    );
    assert!(out.status.success(), "{out:?}");
    json(&out)
}

#[test]
fn ordinary_cli_reads_deep_lineage_and_reports_incomplete_parent_prefixes() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let mut sessions = Vec::new();
    for i in 0usize..1030 {
        let parent = format!("s{}", i.saturating_sub(1));
        let anchor = format!("{parent}-message");
        sessions.push(exported(
            &format!("s{i}"),
            &format!("message {i} λ🙂"),
            (i > 0).then_some((parent.as_str(), Some(anchor.as_str()))),
        ));
    }
    sessions.push(exported(
        "prefix-parent",
        "unselected parent tail",
        Some(("s1", Some("s1-message"))),
    ));
    sessions.push(exported(
        "prefix-child",
        "selected child",
        Some(("prefix-parent", Some("s0-message"))),
    ));
    sessions.push(exported(
        "missing-anchor",
        "available λ🙂 --token=synthetic-lineage-secret",
        Some(("s0", Some("absent-message"))),
    ));
    sessions.push(exported(
        "absent-anchor",
        "available child",
        Some(("s0", None)),
    ));
    sessions.push(exported(
        "orphan",
        "available orphan",
        Some(("unavailable", Some("unavailable-message"))),
    ));
    sessions.push(exported(
        "cycle-a",
        "cycle a",
        Some(("cycle-b", Some("cycle-b-message"))),
    ));
    sessions.push(exported(
        "cycle-b",
        "cycle b",
        Some(("cycle-a", Some("cycle-a-message"))),
    ));
    let path = dir.path().join("synthetic-export.json");
    std::fs::write(
        &path,
        json!({"format": "polylogue-export", "version": 1, "sessions": sessions}).to_string(),
    )
    .unwrap();
    let imported = cli(&env, &["--json", "import", path.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let deep = logical(&env, "claude-code:s1029", "json");
    assert_eq!(deep["lineage"]["complete"], true);
    let messages = deep["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1030);
    for (i, message) in messages.iter().enumerate() {
        assert_eq!(message["blocks"][0]["text"], format!("message {i} λ🙂"));
    }
    let prefix = logical(&env, "claude-code:prefix-child", "json");
    assert_eq!(prefix["lineage"]["complete"], true);
    assert_eq!(prefix["messages"].as_array().unwrap().len(), 2);
    assert_eq!(prefix["messages"][0]["id"], "claude-code:s0-message");
    assert_eq!(
        prefix["messages"][1]["id"],
        "claude-code:prefix-child-message"
    );
    for (name, kind, count) in [
        ("missing-anchor", "missing_anchor", 1),
        ("absent-anchor", "missing_anchor", 1),
        ("orphan", "missing_parent", 1),
        ("cycle-a", "cycle", 2),
    ] {
        let sid = format!("claude-code:{name}");
        let read = logical(&env, &sid, "json");
        assert_eq!(read["lineage"]["complete"], false);
        assert_eq!(read["lineage"]["issues"][0]["kind"], kind);
        assert_eq!(read["messages"].as_array().unwrap().len(), count);
    }
    let own = cli(&env, &["--json", "read", "claude-code:missing-anchor"]);
    assert!(own.status.success());
    let own = json(&own);
    assert!(own["lineage"].is_null(), "a physical read does not inspect ancestry");
    assert_eq!(own["messages"].as_array().unwrap().len(), 1);
    for format in ["terminal", "markdown", "html"] {
        let rendered = logical(&env, "claude-code:missing-anchor", format);
        assert_eq!(rendered["lineage"]["complete"], false);
        let text = rendered["content"].as_str().unwrap();
        assert!(text.contains("Incomplete session ancestry"));
        assert!(!text.contains("message 0 λ"), "unproven parent included");
        assert!(!text.contains("synthetic-lineage-secret"));
    }
    let mut cursor = None;
    let mut pages = 0;
    loop {
        let mut args = vec![
            "--json",
            "read",
            "claude-code:missing-anchor",
            "--logical",
            "--budget",
            "2",
        ];
        if let Some(token) = cursor.as_deref() {
            args.extend(["--cursor", token]);
        }
        let out = cli(&env, &args);
        assert!(out.status.success(), "{out:?}");
        let page = json(&out);
        assert_eq!(page["lineage"]["complete"], false);
        assert_eq!(page["lineage"]["issues"][0]["kind"], "missing_anchor");
        assert!(!String::from_utf8_lossy(&out.stdout).contains("synthetic-lineage-secret"));
        cursor = page["next"].as_str().map(str::to_owned);
        pages += 1;
        if cursor.is_none() {
            break;
        }
        assert!(pages < 100);
    }
    assert!(pages > 1);
    let context = cli(&env, &["--json", "takeover", "claude-code:missing-anchor"]);
    assert!(context.status.success(), "{context:?}");
    let context = json(&context);
    assert!(
        context["prepared"]["text"]
            .as_str()
            .unwrap()
            .contains("Incomplete session ancestry")
    );
    assert!(
        context["prepared"]["left_out"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["what"] == "inherited session content")
    );
}

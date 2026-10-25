use super::*;

const THREAD: &str = "019a0000-0000-7000-8000-000000000001";

#[test]
fn native_codex_counts_usage_and_selected_items_keep_their_own_scope_and_masking_context() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let sessions = home.join(".codex/sessions/2026/10/23");
    std::fs::create_dir_all(&sessions).unwrap();
    let source = sessions.join(format!("rollout-synthetic-{THREAD}.jsonl"));
    let original = polylogue_testkit::codex::observations(THREAD);
    std::fs::write(&source, &original).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let deadline = Instant::now() + Duration::from_secs(15);
    let answer = loop {
        let out = cli(&env, &["--json", "measure", "usage"]);
        assert!(out.status.success(), "{out:?}");
        let answer = json(&out);
        if answer["data"]["recorded_usage"]["measured_completion_observations"] == 2 {
            break answer;
        }
        assert!(Instant::now() < deadline, "{answer}");
        std::thread::sleep(Duration::from_millis(20));
    };
    let data = &answer["data"];
    assert!(data["messages"].is_null());
    assert_eq!(data["unknown_boundary_items"], 3);
    assert_eq!(data["recorded_usage"]["known_usage"]["input"], 30);
    assert_eq!(data["recorded_usage"]["known_usage"]["output"], 5);
    assert_eq!(data["api_cost"]["unpriced_messages"], 0);
    let text = cli(&env, &["measure", "usage"]);
    assert!(text.status.success(), "{text:?}");
    let text = String::from_utf8(text.stdout).unwrap();
    assert!(text.contains("unknown responses"), "{text}");
    assert!(text.contains("2 recorded usage observations"), "{text}");
    let daily = cli(&env, &["--json", "measure", "daily", "tokens"]);
    assert!(daily.status.success(), "{daily:?}");
    let daily = json(&daily);
    let daily = &daily["data"];
    assert_eq!(daily["covers"], "recorded_known_values");
    assert_eq!(daily["timed_by"], "source_record_time");
    assert_eq!(daily["incomplete"], true);
    assert_eq!(daily["unknown_spend_time"], true);
    assert_eq!(
        daily["days"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d[1].as_f64().unwrap())
            .sum::<f64>(),
        35.0
    );
    assert!(
        daily["baseline_7d"]
            .as_array()
            .unwrap()
            .iter()
            .all(serde_json::Value::is_null)
    );
    assert!(daily["changes"].as_array().unwrap().is_empty());
    let quota = cli(&env, &["--json", "measure", "quota"]);
    assert!(quota.status.success(), "{quota:?}");
    let quota = json(&quota);
    let codex = quota["data"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["tool"] == "codex")
        .unwrap();
    assert_eq!(codex["unplaced"][0]["used_percent"], 12.0);
    assert!(codex["unplaced"][0]["observed_at"].is_null());
    wait_status(&env, |s| s["store"]["search_pending"] == 0);
    let query = cli(&env, &["--json", "messages where publicneedle"]);
    assert!(query.status.success(), "{query:?}");
    let query = json(&query);
    assert_eq!(query["count"], 1);
    assert_eq!(query["rows"][0]["id"], "codex:c1");
    assert!(
        !query.to_string().contains("synthetic-private-tail"),
        "{query}"
    );
    let id = format!("codex:{THREAD}");
    let read = cli(
        &env,
        &[
            "--json", "read", &id, "--view", "dialogue", "--format", "json",
        ],
    );
    assert!(read.status.success(), "{read:?}");
    assert!(!json(&read).to_string().contains("synthetic-private-tail"));
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    let reader = pool.reader().unwrap();
    let raw = reader
        .messages(&polylogue_model::SessionId::parse(&id).unwrap(), false)
        .unwrap();
    assert_eq!(raw.len(), 5);
    assert!(raw.iter().any(|m| {
        m.id.to_string() == "codex:c1"
            && serde_json::to_string(m)
                .unwrap()
                .contains("synthetic-private-tail")
    }));
    let child = raw
        .iter()
        .find(|m| m.id.to_string() == "codex:native-child")
        .unwrap();
    assert_eq!(child.model, None);
    assert_eq!(
        child.prompt_kind,
        Some(polylogue_model::PromptKind::Harness)
    );
    assert_eq!(std::fs::read_to_string(source).unwrap(), original);
}

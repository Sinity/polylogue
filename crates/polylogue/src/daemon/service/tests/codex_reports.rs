use super::*;
use serde_json::json;

#[test]
fn native_codex_independent_usage_and_rate_only_quota_reach_mcp() {
    let daemon = test_daemon::start("synthetic human prompt");
    let thread = "019a0000-0000-7000-8000-000000000001";
    let sessions = daemon.home.join(".codex/sessions/2026/10/23");
    std::fs::create_dir_all(&sessions).unwrap();
    let path = sessions.join(format!("rollout-synthetic-{thread}.jsonl"));
    let original = polylogue_testkit::codex::observations(thread);
    std::fs::write(&path, &original).unwrap();
    let id = polylogue_model::SessionId::parse(&format!("codex:{thread}")).unwrap();
    daemon.wait_for(|_| {
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .events(&id)
            .unwrap_or_default()
            .iter()
            .filter(|e| e.kind == "usage_report")
            .count()
            == 2
    });
    let archive = Archive::at(&daemon.daemon.archive.root);
    let usage: serde_json::Value = serde_json::from_str(
        &crate::mcp::call(
            &archive,
            "query",
            json!({"query":"sessions | measure usage"})
                .as_object()
                .unwrap()
                .clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(usage["data"]["messages"].is_null());
    assert_eq!(
        usage["data"]["recorded_usage"]["measured_completion_observations"],
        2
    );
    assert_eq!(usage["data"]["recorded_usage"]["known_usage"]["input"], 30);
    let quota: serde_json::Value = serde_json::from_str(
        &crate::mcp::call(
            &archive,
            "query",
            json!({"query":"sessions | measure quota"})
                .as_object()
                .unwrap()
                .clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let codex = quota["data"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["tool"] == "codex")
        .unwrap();
    assert_eq!(codex["unplaced"][0]["used_percent"], 12.0);
    assert!(codex["unplaced"][0]["measured_at"].is_null());
    assert_eq!(std::fs::read_to_string(path).unwrap(), original);
}

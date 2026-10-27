use super::*;

#[test]
fn scan_command_finds_secrets_and_status_never_scans() {
    let secret = format!("sk-ant-api03-{}", "a".repeat(60));
    let w = test_daemon::start(&format!("ANTHROPIC_API_KEY={secret}"));
    let archive = Archive::at(&w.daemon.archive.root);
    let scan = |args_value| {
        let text = call(&archive, "scan-secrets", args(args_value)).unwrap();
        assert!(!text.contains(&secret));
        serde_json::from_str::<serde_json::Value>(&text).unwrap()
    };
    assert!(tools().iter().any(|t| t.name == "scan-secrets"));
    assert_eq!(scan(json!({"status":true}))["remaining_pending"], 1);
    assert_eq!(scan(json!({"status":true}))["remaining_pending"], 1);
    assert_eq!(
        scan(json!({"session":"claude-code:missing"}))["status"],
        "not_found"
    );
    let result = scan(json!({"all":true,"limit":1}));
    assert_eq!(result["sessions_scanned"], 1);
    assert_eq!(result["blocks_scanned"], 1);
    assert_eq!(result["candidates_found"], 1);
    assert_eq!(result["findings"][0]["start"], 18);
    assert_eq!(scan(json!({"status":true}))["remaining_pending"], 0);
    assert_eq!(scan(json!({"all":true}))["sessions_scanned"], 0);
    let explicit = scan(json!({"session":"claude-code:s1"}));
    assert_eq!(explicit["findings"], result["findings"]);
    for bad in [
        json!({}),
        json!({"all":true,"status":true}),
        json!({"session":"x","all":true}),
        json!({"limit":0,"all":true}),
    ] {
        assert!(call(&archive, "scan-secrets", args(bad)).is_err());
    }
}

use super::*;

#[test]
fn secret_scan_cli_drains_pages_and_reports_counts_without_literals() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    let home = dir.path().join("home");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    let secret = "AKIAABCDEFGHIJKLMNOP";
    for i in 0..4 {
        let session = format!("s{i}");
        std::fs::write(
            project.join(format!("{session}.jsonl")),
            line(&format!("u{i}"), &session, secret),
        )
        .unwrap();
    }
    let env = [
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("POLYLOGUE_HOME", home.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |data| data["store"]["messages"] == 4);
    let status = cli(&env, &["--json", "scan-secrets", "--status"]);
    assert!(status.status.success(), "{status:?}");
    assert_eq!(json(&status)["remaining_pending"], 4);
    let page = cli(&env, &["--json", "scan-secrets", "--all", "--limit", "2"]);
    assert!(page.status.success(), "{page:?}");
    assert_eq!(json(&page)["sessions_scanned"], 2);
    assert_eq!(json(&page)["candidates_found"], 2);
    assert_eq!(json(&page)["remaining_pending"], 2);
    assert!(!String::from_utf8_lossy(&page.stdout).contains(secret));
    let rest = cli(&env, &["scan-secrets", "--all"]);
    assert!(rest.status.success(), "{rest:?}");
    let text = String::from_utf8_lossy(&rest.stdout);
    assert!(text.contains("blocks scanned: 2"), "{text}");
    assert!(text.contains("secret candidates found: 2"), "{text}");
    assert!(!text.contains(secret));
    for flags in [
        vec!["scan-secrets"],
        vec!["scan-secrets", "--all", "--status"],
        vec!["scan-secrets", "--session", "x", "--all"],
    ] {
        assert!(!cli(&env, &flags).status.success());
    }
}

use super::*;

#[test]
fn cli_read_pages_cover_utf8_once_and_refuse_malformed_cursors() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-pages");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("pages.jsonl"),
        line("u1", "pages", "λ🙂 synthetic prompt with several pages"),
    )
    .unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1
            && status["store"]["search_pending"] == 0
            && status["last_pass"]["seconds"]
                .as_f64()
                .is_some_and(|seconds| seconds > 0.0)
    });
    let whole = cli(
        &env,
        &["--json", "read", "claude-code:pages", "--budget", "100000"],
    );
    assert!(whole.status.success(), "{whole:?}");
    let expected = json(&whole)["text"].as_str().unwrap().to_owned();
    let mut cursor = None;
    let mut joined = String::new();
    for _ in 0..100 {
        let mut args = vec!["--json", "read", "claude-code:pages", "--budget", "1"];
        if let Some(cursor) = cursor.as_deref() {
            args.extend(["--cursor", cursor]);
        }
        let out = cli(&env, &args);
        assert!(out.status.success(), "{out:?}");
        let page = json(&out);
        joined.push_str(page["text"].as_str().unwrap());
        cursor = page["next"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    assert!(cursor.is_none());
    assert_eq!(joined, expected);
    let invalid = cli(
        &env,
        &[
            "--json",
            "read",
            "claude-code:pages",
            "--budget",
            "1",
            "--cursor",
            "broken",
        ],
    );
    assert!(!invalid.status.success());
    assert_eq!(json(&invalid)["error"]["kind"], "bad_request");
    let first = cli(
        &env,
        &["--json", "read", "claude-code:pages", "--budget", "1"],
    );
    let first = json(&first);
    let token = first["next"].as_str().unwrap();
    drop(daemon);
    polylogue_testkit::delete_index(&archive);
    let _rebuilt = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1 && status["store"]["search_pending"] == 0
    });
    let stale = cli(
        &env,
        &[
            "--json",
            "read",
            "claude-code:pages",
            "--budget",
            "1",
            "--cursor",
            token,
        ],
    );
    assert!(!stale.status.success());
    assert_eq!(json(&stale)["error"]["kind"], "stale_cursor");
}

use super::*;

#[test]
fn promoted_chatgpt_import_resolves_assets_and_pages_after_provider_free_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let source = dir.path().join("export");
    std::fs::create_dir(&home).unwrap();
    std::fs::create_dir(&source).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let unknown = dir.path().join("renamed.json");
    let body = conversation();
    std::fs::write(&unknown, &body).unwrap();
    std::fs::write(source.join("conversations.json"), &body).unwrap();
    std::fs::write(source.join("file-Native.dat"), b"promoted asset bytes").unwrap();
    std::fs::write(
        source.join("conversation_asset_file_names.json"),
        br#"{"file-Native.dat":"promoted.png"}"#,
    )
    .unwrap();
    let daemon = start_daemon(&env);
    let kept = cli(&env, &["--json", "import", unknown.to_str().unwrap()]);
    assert!(kept.status.success(), "{kept:?}");
    assert_eq!(json(&kept)["unrecognized"].as_array().unwrap().len(), 1);
    let promoted = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    assert!(promoted.status.success(), "{promoted:?}");
    assert!(
        json(&promoted)["pass"]["errors"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for rebuild in [false, true] {
        if rebuild {
            drop(daemon);
            std::fs::remove_file(&unknown).unwrap();
            std::fs::remove_dir_all(&source).unwrap();
            polylogue_testkit::delete_index(&archive);
            let _rebuilt = start_daemon(&env);
            let started = Instant::now();
            wait_status(&env, |status| {
                assert!(started.elapsed() < Duration::from_secs(10), "{status}");
                status["store"]["messages"] == 1
            });
            check(&env);
            return;
        }
        check(&env);
    }
}

fn check(env: &[(&str, &Path)]) {
    let descriptor = cli(env, &["--json", "attachment", MESSAGE, "0"]);
    assert!(descriptor.status.success(), "{descriptor:?}");
    assert_eq!(json(&descriptor)["state"], "available");
    assert_eq!(
        json(&descriptor)["candidates"][0]["names"],
        json!(["promoted.png"])
    );
    let page = cli(env, &["--json", "read", SESSION, "--budget", "100000"]);
    assert!(page.status.success(), "{page:?}");
    assert_eq!(json(&page)["attachments"][0]["state"], "available");
    assert!(
        json(&page)["text"]
            .as_str()
            .unwrap()
            .contains("promoted.png")
    );
    let refused = cli(
        env,
        &["--json", "excise", "text:synthetic-text-removal", "--yes"],
    );
    assert!(
        !refused.status.success(),
        "opaque input text removal must refuse: {refused:?}"
    );
    let after = cli(env, &["--json", "attachment", MESSAGE, "0"]);
    assert_eq!(json(&after), json(&descriptor));
}

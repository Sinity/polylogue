use super::*;

#[test]
fn export_folder_import_resolves_assets_after_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let source = dir.path().join("export");
    std::fs::create_dir(&home).unwrap();
    std::fs::create_dir(&source).unwrap();
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let unknown = dir.path().join("renamed.json");
    let body = conversation();
    std::fs::write(&unknown, &body).unwrap();
    std::fs::write(source.join("conversations.json"), &body).unwrap();
    std::fs::write(source.join("file-Native.dat"), b"imported asset bytes").unwrap();
    std::fs::write(
        source.join("conversation_asset_file_names.json"),
        br#"{"file-Native.dat":"imported.png"}"#,
    )
    .unwrap();
    let daemon = start_daemon(&env);
    let kept = cli(&env, &["--json", "import", unknown.to_str().unwrap()]);
    assert!(kept.status.success(), "{kept:?}");
    let kept: polylogue_engine::ImportReport = serde_json::from_value(json(&kept)).unwrap();
    assert_eq!(kept.refused.len(), 1);
    assert_eq!(kept.refused[0].path, unknown);
    assert!(!kept.refused[0].reason.is_empty());
    assert!(!kept.refused[0].structures_tried.is_empty());
    assert_eq!(std::fs::read(&unknown).unwrap(), body);
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let imported: polylogue_engine::ImportReport = serde_json::from_value(json(&imported)).unwrap();
    assert!(
        imported.pass.errors.is_empty(),
        "{:?}",
        imported.pass.errors
    );
    assert!(
        imported
            .members
            .iter()
            .any(|member| member.path.ends_with("conversations.json")
                && member.class == "parsed"
                && member.origin.as_deref() == Some("chatgpt"))
    );
    assert_eq!(
        std::fs::read(source.join("file-Native.dat")).unwrap(),
        b"imported asset bytes"
    );
    for rebuild in [false, true] {
        if rebuild {
            drop(daemon);
            std::fs::remove_file(&unknown).unwrap();
            std::fs::remove_dir_all(&source).unwrap();
            polylogue_testkit::delete_index(&archive);
            let _rebuilt = start_daemon(&env);
            wait_status(&env, |status| status["store"]["messages"] == 1);
            check(&env);
            return;
        }
        check(&env);
    }
}

fn check(env: &crate::Fixture) {
    let descriptor = cli(env, &["--json", ATTACHMENT, "read"]);
    assert!(descriptor.status.success(), "{descriptor:?}");
    let descriptor = attachment_page(&json(&descriptor))["descriptor"].clone();
    assert_eq!(descriptor["state"], "available");
    assert_eq!(descriptor["matches"][0]["names"], json!(["imported.png"]));
    let page = cli(env, &["--json", SESSION, "read", "--budget", "100000"]);
    let page = typed_page(&page);
    let item = one_attachment(&page.document.items);
    assert_available(env, item, &descriptor);
    assert_eq!(
        descriptor["matches"][0]["size"],
        b"imported asset bytes".len()
    );
    assert_eq!(
        descriptor["matches"][0]["blob"],
        json!(polylogue_model::BlobHash::of(b"imported asset bytes"))
    );
    let bytes = cli(env, &["--json", ATTACHMENT, "read"]);
    assert!(bytes.status.success(), "{bytes:?}");
    use base64::Engine as _;
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(
                attachment_page(&json(&bytes))["content_base64"]
                    .as_str()
                    .unwrap()
            )
            .unwrap(),
        b"imported asset bytes"
    );
}

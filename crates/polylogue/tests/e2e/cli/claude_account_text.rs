use super::*;
use serde_json::json;

#[test]
fn account_and_project_text_import_without_turns_or_invented_document_authorship() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let source = dir.path().join("export");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(source.join("projects")).unwrap();
    let memories = json!([{"account_uuid":"account-a","conversations_memory":"Provider account summary.","project_memories":{"project-a":"Provider project summary."}}]);
    let project = json!({"uuid":"project-a","name":"Synthetic knowledge","description":"Supplied project context","is_starter_project":false,"prompt_template":"Use supplied records.","created_at":"2026-05-01T00:00:00Z","docs":[{"uuid":"doc-a","filename":"manual.md","content":"Vendor document λ🙂","created_at":"2026-05-02T00:00:00Z"}]});
    for (name, value) in [
        ("memories.json", memories),
        ("projects/project-a.json", project),
    ] {
        std::fs::write(source.join(name), serde_json::to_vec(&value).unwrap()).unwrap();
    }
    std::fs::write(source.join("conversations.json"),r#"[{"uuid":"old-conversation","name":"Existing conversation","chat_messages":[{"uuid":"old-message","sender":"human","text":"Existing conversation text."}]}]"#).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let old = cli(&env, &["--json", "read", "claude-ai:old-conversation"]);
    assert!(old.status.success(), "{old:?}");
    assert_eq!(
        json(&old)["messages"][0]["blocks"][0]["text"],
        "Existing conversation text."
    );
    let documents = cli(
        &env,
        &[
            "--json",
            "messages where origin:claude-ai \"Vendor document\"",
        ],
    );
    assert!(documents.status.success(), "{documents:?}");
    assert_eq!(
        json(&documents)["count"],
        1,
        "retained project text must be queryable without pretending it is a human prompt"
    );
    let authored = cli(
        &env,
        &[
            "--json",
            "messages where origin:claude-ai written_by:human \"Vendor document\"",
        ],
    );
    assert!(authored.status.success(), "{authored:?}");
    assert_eq!(json(&authored)["count"], 0);
}

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
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let daemon = start_daemon(&env);
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
            "documents where origin:claude-ai \"Vendor document\"",
        ],
    );
    assert!(documents.status.success(), "{documents:?}");
    assert_eq!(
        json(&documents)["count"],
        1,
        "project text must be queryable without pretending it is a human prompt"
    );
    let authored = cli(
        &env,
        &[
            "--json",
            "documents where origin:claude-ai authorship:human \"Vendor document\"",
        ],
    );
    assert!(authored.status.success(), "{authored:?}");
    assert_eq!(json(&authored)["count"], 0);
    let reference = "document:claude-ai:project:project-a:doc:doc-a";
    let read = cli(&env, &["--json", "read", reference]);
    assert!(read.status.success(), "{read:?}");
    let before = json(&read);
    assert_eq!(
        before["document"]["blocks"][0]["text"],
        "Vendor document λ🙂"
    );
    assert_eq!(before["document"]["authorship"], "unknown");
    assert_eq!(before["document"]["source_time"], "2026-05-02T00:00:00Z");
    assert!(before.get("messages").is_none());
    assert_eq!(before["sources"].as_array().unwrap().len(), 1);
    let mut text = String::new();
    let mut next = None::<String>;
    loop {
        let mut args = vec!["--json", "read", reference, "--budget", "3"];
        if let Some(cursor) = &next {
            args.extend(["--cursor", cursor]);
        }
        let out = cli(&env, &args);
        assert!(out.status.success(), "{out:?}");
        let page = json(&out);
        text.push_str(page["text"].as_str().unwrap());
        next = page["next"].as_str().map(str::to_owned);
        if next.is_none() {
            break;
        }
    }
    assert!(text.ends_with("Vendor document λ🙂\n"), "{text}");
    for format in ["markdown", "html", "terminal"] {
        let out = cli(&env, &["--json", "read", reference, "--format", format]);
        assert!(out.status.success(), "{out:?}");
        assert!(
            json(&out)["content"]
                .as_str()
                .unwrap()
                .contains("Vendor document λ🙂")
        );
    }
    let exported = cli(&env, &["--json", "export", reference]);
    assert!(exported.status.success(), "{exported:?}");
    let exported = json(&exported);
    assert_eq!(exported["version"], 2);
    assert_eq!(exported["sessions"], json!([]));
    assert_eq!(exported["documents"][0], before["document"]);
    let copy = source.join("polylogue-docs.json");
    std::fs::write(&copy, serde_json::to_vec(&exported).unwrap()).unwrap();
    let imported = cli(&env, &["--json", "import", copy.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let read = cli(&env, &["--json", "read", reference]);
    assert_eq!(json(&read)["document"], before["document"]);
    let note = cli(
        &env,
        &[
            "--json",
            "note",
            "add",
            reference,
            "--body",
            "Synthetic document note",
        ],
    );
    assert!(note.status.success(), "{note:?}");
    drop(daemon);
    // Rebuild this test archive from its stored bytes after the source file is gone.
    std::fs::remove_dir_all(&source).unwrap();
    polylogue_testkit::delete_index(&archive);
    let daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["documents"].as_u64() == Some(4));
    let after = cli(&env, &["--json", "read", reference]);
    assert!(after.status.success(), "{after:?}");
    assert_eq!(json(&after)["document"], before["document"]);
    drop(daemon);
}

mod pages;

mod history;

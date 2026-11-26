use serde_json::json;

#[test]
fn embed_without_yes_previews() {
    let daemon = crate::test_daemon::start("Embedding preview specimen");
    let setting = crate::test_daemon::call(
        &daemon.daemon,
        "set",
        json!({"key":"embed.provider","value":"local"}),
    );
    assert!(setting.value().is_some(), "{setting:?}");
    let answer = crate::test_daemon::call(&daemon.daemon, "embed", json!({}));
    let value = answer
        .value()
        .cloned()
        .unwrap_or_else(|| panic!("{answer:?}"));
    let answer: crate::embeddings::Answer = serde_json::from_value(value).unwrap();
    let crate::embeddings::Answer::Preview(preview) = answer else {
        panic!("{answer:?}")
    };
    assert_eq!(preview.texts, 1);
    assert_eq!(
        preview.characters,
        "Embedding preview specimen".len() as u64
    );
}

#[test]
fn help_topic_pages_units() {
    let daemon = crate::test_daemon::start("Help unit pages");
    let first = crate::test_daemon::call(
        &daemon.daemon,
        "help",
        json!({"topic":"units","page_size":1}),
    );
    let first = first.value().unwrap();
    assert_eq!(first["entries"].as_array().unwrap().len(), 1);
    assert_eq!(first["entries"][0]["name"], "sessions");
    let next = first["next"].as_str().unwrap();
    let second = crate::test_daemon::Call::new("help", json!({"topic":"units","page_size":1}))
        .cursor(Some(next))
        .answer(&daemon.daemon);
    let second = second.value().unwrap();
    assert_eq!(second["entries"].as_array().unwrap().len(), 1);
    assert_eq!(second["entries"][0]["name"], "documents");
    assert_eq!(second["total"], first["total"]);
}

#[test]
fn historical_drive_setting_reads_with_hint_and_public_writes_name_install_remove() {
    let daemon = crate::test_daemon::start("Historical setting specimen");
    let answer = crate::test_daemon::call(&daemon.daemon, "set", json!({"key":"drive.ai-studio"}));
    assert!(answer.value().is_some(), "{answer:?}");
    assert!(
        answer
            .envelope
            .notes
            .iter()
            .any(|note| note.contains("polylogue install drive --remove"))
    );
    for value in ["on", "off"] {
        let answer = crate::test_daemon::call(
            &daemon.daemon,
            "set",
            json!({"key":"drive.ai-studio","value":value}),
        );
        let text = serde_json::to_string(&answer).unwrap();
        assert!(answer.value().is_none(), "{answer:?}");
        assert!(text.contains("polylogue install drive --remove"), "{text}");
    }
}

#[test]
fn absent_drive_authorization_is_a_status_note_without_a_pass_fault() {
    let daemon = crate::test_daemon::start("Unsigned Drive status specimen");
    let answer = crate::test_daemon::call(&daemon.daemon, "status", json!({}));
    assert!(answer.value().is_some(), "{answer:?}");
    assert!(
        answer
            .envelope
            .notes
            .iter()
            .any(|note| note.contains("polylogue install drive")),
        "{answer:?}"
    );
    assert!(daemon.daemon.status.get().last_pass.errors.is_empty());
}

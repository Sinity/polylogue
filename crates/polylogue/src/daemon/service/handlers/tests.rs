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

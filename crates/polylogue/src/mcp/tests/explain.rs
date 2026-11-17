use super::*;

#[test]
fn help_pages_declarations_and_lines_explain_queries() {
    let w = test_daemon::start("grammar discovery");
    let archive = Archive::at(&w.daemon.archive.root);
    assert!(operations(Tool::Read).any(|command| command.name == "help"));
    let page: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "help",
            args(json!({"topic": "fields", "page_size": 1})),
        )
        .unwrap(),
    )
    .unwrap();
    let next = page["next"].as_str().unwrap();
    let second: serde_json::Value = serde_json::from_str(
        &call_page(
            &archive,
            "help",
            json!({"topic":"fields","page_size":1}),
            Some(next),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(second["subject"], "fields");
    assert_ne!(page["entries"], second["entries"]);
    let query: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "rows",
            args(json!({"selection": "tool_calls where outcome:error", "explain":true})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        query["plan"]["explanation"]["program"]["unit"],
        "tool_calls"
    );
    assert!(
        call(
            &archive,
            "rows",
            args(json!({"selection":"messages where invalid:x", "explain":true}))
        )
        .is_err()
    );
}

#[test]
fn explain_preserves_query_error_kinds_and_correction_hints() {
    let w = test_daemon::start("grammar correction");
    let archive = Archive::at(&w.daemon.archive.root);
    let error = call(
        &archive,
        "rows",
        args(json!({"selection": "\"unclosed phrase", "explain":true})),
    )
    .unwrap_err();
    assert!(error.starts_with("error (invalid_query):"), "{error}");
    assert!(error.contains("try: \"unclosed phrase\""), "{error}");
}

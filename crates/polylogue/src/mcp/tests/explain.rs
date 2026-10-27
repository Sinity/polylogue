use super::*;

#[test]
fn explain_is_a_table_tool_with_paged_declarations_and_query_errors() {
    let w = test_daemon::start("grammar discovery");
    let archive = Archive::at(&w.daemon.archive.root);
    assert!(tools().iter().any(|t| t.name == "explain"));
    let page: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "explain",
            args(json!({"subject": "fields", "page_size": 1})),
        )
        .unwrap(),
    )
    .unwrap();
    let next = page["next"].as_str().unwrap();
    let second: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "explain",
            args(json!({"subject": "fields", "page_size": 1, "cursor": next})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(second["subject"], "fields");
    assert_ne!(page["entries"], second["entries"]);
    let query: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "explain",
            args(json!({"subject": "query", "expression": "tool_calls where outcome:error"})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(query["explanation"]["program"]["unit"], "tool_calls");
    for arguments in [
        json!({"subject": "query"}),
        json!({"subject": "query", "expression": "messages where invalid:x"}),
    ] {
        assert!(call(&archive, "explain", args(arguments)).is_err());
    }
}

#[test]
fn explain_preserves_query_error_kinds_and_correction_hints() {
    let w = test_daemon::start("grammar correction");
    let archive = Archive::at(&w.daemon.archive.root);
    let error = call(
        &archive,
        "explain",
        args(json!({"subject": "query", "expression": "\"unclosed phrase"})),
    )
    .unwrap_err();
    assert!(error.starts_with("error (invalid_query):"), "{error}");
    assert!(error.contains("try: \"unclosed phrase\""), "{error}");
}

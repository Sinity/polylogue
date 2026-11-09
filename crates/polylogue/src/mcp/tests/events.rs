use super::*;
use polylogue_model::read::{ItemKind, ReadPage};

#[test]
fn mcp_reads_input_events_through_the_same_typed_reader_and_budget_cursor() {
    let daemon = test_daemon::start("independent ordinary conversation");
    let source = daemon.home.join("synthetic.atof.jsonl");
    let text = " exact MCP λ🙂 evidence ".repeat(500);
    let credential = "--secret=\"unfinished --token=\"synthetic-later-value\" tail";
    std::fs::write(&source,format!("{}\n",json!({"atof_version":"0.1","kind":"mark","uuid":"synthetic","name":"neutral","data":{"text":text,"credential":credential}}))).unwrap();
    let imported = crate::test_daemon::Call::new("import", serde_json::json!({"paths": [source]}))
        .id("mcp-event-fixture")
        .answer(&daemon.daemon);
    crate::test_answers::ok(imported);
    let archive = Archive::at(&daemon.daemon.archive.root);
    let listed: Value = serde_json::from_str(
        &call(
            &archive,
            "query",
            args(json!({"query":"events where source:hermes"})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(listed["count"], 1);
    let reference = format!("event:{}", listed["rows"][0]["id"].as_str().unwrap());
    let ordinary: ReadPage = serde_json::from_str(
        &super::read_pages::read_without_deadline(
            &archive,
            json!({"target":reference,"format":"json"}),
        )
        .unwrap(),
    )
    .unwrap();
    let (ordinary, _) = super::read_pages::collect(
        &daemon,
        ordinary,
        None,
        None,
        command_surface::Deadline::None,
    );
    let item = ordinary
        .values
        .iter()
        .find(|item| matches!(item.kind, ItemKind::Other { .. }))
        .expect("event layout item");
    let ItemKind::Other { data, .. } = &item.kind else {
        unreachable!()
    };
    assert_eq!(data["event"]["data"]["data"]["text"], text);
    assert_eq!(data["event"]["data"]["data"]["credential"], credential);
    let page: ReadPage = serde_json::from_str(
        &super::read_pages::read_without_deadline(
            &archive,
            json!({"target":reference,"page_tokens":128}),
        )
        .unwrap(),
    )
    .unwrap();
    let next = page
        .document
        .footer
        .next
        .clone()
        .expect("structured event is paged");
    let token = next.mcp["arguments"]["cursor"].as_str().unwrap();
    assert!(!token.contains("synthetic-later-value"));
    let mut wrong = next.clone();
    wrong.mcp["arguments"]["arguments"]["view"] = json!("dialogue");
    let error = super::read_pages::execute_ready(&archive, &wrong).unwrap_err();
    // docs/query.md: "a cursor from another question is `invalid_query`";
    // a read continuation with other arguments is refused the same way.
    assert!(error.contains("invalid_query"), "{error}");
    let (reconstructed, pages) =
        super::read_pages::collect(&daemon, page, None, None, command_surface::Deadline::None);
    let actual = reconstructed
        .values
        .iter()
        .find(|item| matches!(item.kind, ItemKind::Other { .. }))
        .expect("complete structured event survived paging");
    assert_eq!(actual, item);
    assert!(pages > 1);
}

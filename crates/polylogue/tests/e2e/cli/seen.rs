//! The actual output destination, rather than merely asking for a page, marks it.
use super::*;
use serde_json::{Value, json};

pub(super) fn count(env: &crate::Fixture) -> u64 {
    let barrier = cli(env, &["set", "browser.actions", "on"]);
    assert!(barrier.status.success(), "{barrier:?}");
    let output = cli(env, &["--json", "annotations where kind:seen"]);
    query_rows(&output)["count"].as_u64().unwrap()
}

#[test]
fn pipe_file_and_finite_stream_do_not_mark_but_explicit_pager_does() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = super::read_pages::three_prompts(dir.path());
    let target = "session:claude-code:pager";
    let pipe = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    assert!(pipe.status.success(), "{pipe:?}");
    assert_eq!(count(&env), 0);
    let path = dir.path().join("reading.md");
    let file = cli(&env, &[target, "read", "--to", path.to_str().unwrap()]);
    assert!(file.status.success(), "{file:?}");
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("third marker gamma")
    );
    assert_eq!(count(&env), 0);
    let bundle = format!("{}/bundle/", dir.path().display());
    let stream = cli(&env, &[target, "read", "--to", &bundle]);
    assert!(stream.status.success(), "{stream:?}");
    assert_eq!(count(&env), 0);

    let displayed = env
        .command()
        .env("PAGER", "cat")
        .args([
            target,
            "read",
            "--view",
            "context",
            "--at",
            "claude-code:u1",
            "--to",
            "pager",
        ])
        .output()
        .unwrap();
    assert!(displayed.status.success(), "{displayed:?}");
    assert_eq!(count(&env), 1);
    let after = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    let after = read_page(json(&after));
    let mut text = super::read_current::Items::default();
    text.push(&after);
    assert_eq!(text.text(), "second marker betathird marker gamma");
    let file_after = dir.path().join("after.json");
    let out = cli(
        &env,
        &[
            target,
            "read",
            "--from",
            "seen",
            "--format",
            "json",
            "--to",
            file_after.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{out:?}");
    let document: polylogue_model::read::ReadDocument =
        serde_json::from_slice(&std::fs::read(file_after).unwrap()).unwrap();
    let mut items = super::read_current::Items::default();
    items.push_document(&document);
    assert_eq!(items.text(), "second marker betathird marker gamma");
    assert_eq!(
        count(&env),
        1,
        "exporting the remainder cannot advance its display mark"
    );
    let again = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    let again = read_page(json(&again));
    let mut items = super::read_current::Items::default();
    items.push(&again);
    assert_eq!(
        items.text(),
        "second marker betathird marker gamma",
        "export must leave the accepted mark unchanged"
    );
}

#[test]
fn actual_mcp_returned_page_marks_and_from_seen_reads_the_remainder() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = super::read_pages::three_prompts(dir.path());
    let mut mcp = super::query_surfaces::Mcp::start(&env);
    let mut read = |step: Value| {
        let reply = mcp.request(
            "tools/call",
            json!({"name":"polylogue_read", "arguments":{
                "operation":"rows", "request_id":uuid::Uuid::now_v7().to_string(),
                "arguments":{"selection":"session:claude-code:pager", "steps":[step]},
            }}),
        );
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        let answer: command_surface::Answer<Value> =
            serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
        read_page(answer.value().unwrap().clone())
    };
    let first =
        read(json!({"verb":"read", "view":"context", "at":"claude-code:u1", "budget":100000}));
    assert!(
        first
            .document
            .items
            .iter()
            .any(|item| item.source.message.is_some())
    );
    assert_eq!(count(&env), 1);
    let after = read(json!({"verb":"read", "from":"seen", "budget":100000}));
    let mut text = super::read_current::Items::default();
    text.push(&after);
    assert_eq!(text.text(), "second marker betathird marker gamma");
    assert_eq!(count(&env), 1);
    let end = read(json!({"verb":"read", "from":"seen", "budget":100000}));
    assert!(
        !end.document
            .items
            .iter()
            .any(|item| item.source.message.is_some())
    );
}

//! MCP tool calls against a real daemon on a temporary archive.

use super::*;
use crate::test_daemon;
use serde_json::json;

fn args(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn tools_answer_with_the_daemon_and_refuse_bad_calls() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);

    let status: serde_json::Value =
        serde_json::from_str(&call(&archive, "status", args(json!({}))).unwrap()).unwrap();
    assert_eq!(status["store"]["messages"], 1);

    // Every option of `read` reaches the daemon.
    let text = call(
        &archive,
        "read",
        args(json!({"session": "claude-code:s1", "format": "markdown"})),
    )
    .unwrap();
    let read: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(
        read["content"].as_str().unwrap().contains("walrus"),
        "{text}"
    );

    let text = call(&archive, "query", args(json!({"query": "walrus"}))).unwrap();
    assert!(text.contains("claude-code:s1"), "{text}");

    let e = call(
        &archive,
        "read",
        args(json!({"session": "claude-code:s1", "pages": 1})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (bad_request)"), "{e}");
    let e = call(
        &archive,
        "excise",
        args(json!({"target": "session:claude-code:s1"})),
    )
    .unwrap_err();
    assert!(e.contains("no tool excise"), "{e}");
    let e = call(
        &archive,
        "read",
        args(json!({"session": "claude-code:nope"})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (not_found)"), "{e}");
}

mod read_pages;

mod events;

mod attachments;

fn initialize() -> Value {
    json!({"jsonrpc":"2.0", "id":"init", "method":"initialize", "params":{
        "protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"test", "version":"1"}
    }})
}

#[test]
fn stdio_answers_initialize_tools_calls_and_errors() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);
    let requests = [
        initialize().to_string(),
        json!({"jsonrpc":"2.0", "method":"notifications/initialized"}).to_string(),
        json!({"jsonrpc":"2.0", "id":2, "method":"tools/list"}).to_string(),
        json!({"jsonrpc":"2.0", "id":3, "method":"tools/call", "params":{"name":"status"}}).to_string(),
        json!({"jsonrpc":"2.0", "id":4, "method":"tools/call", "params":{"name":"unknown"}}).to_string(),
        "not JSON".to_owned(),
        json!({"jsonrpc":"2.0", "id":5, "method":"unknown"}).to_string(),
        json!({"jsonrpc":"2.0", "id":6, "method":"tools/call", "params":{"name":"status", "arguments":[]}}).to_string(),
        json!({"jsonrpc":"2.0", "id":7, "method":"ping"}).to_string(),
    ].join("\n");
    let mut output = Vec::new();
    serve(requests.as_bytes(), &mut output, &archive).unwrap();
    let replies: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies.len(), 8);
    let reply = |id: Value| replies.iter().find(|reply| reply["id"] == id).unwrap();
    assert_eq!(
        reply(json!("init"))["result"]["protocolVersion"],
        "2025-11-25"
    );
    assert_eq!(reply(json!(2))["result"]["tools"], json!(tools()));
    let result = &reply(json!(3))["result"];
    assert_eq!(result["isError"], false);
    let status: Value =
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(status["store"]["messages"], 1);
    assert_eq!(reply(json!(4))["result"]["isError"], true);
    assert_eq!(reply(Value::Null)["error"]["code"], -32700);
    assert_eq!(reply(json!(5))["error"]["code"], -32601);
    assert_eq!(reply(json!(6))["error"]["code"], -32602);
    assert_eq!(reply(json!(7))["result"], json!({}));
}

#[test]
fn discovery_and_inline_requests_keep_protocol_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientCapabilities":{}});
    let mut session = Session::default();
    let discovery = session
        .prepare(
            json!({"jsonrpc":"2.0", "id":1, "method":"server/discover", "params":{"_meta":meta}}),
        )
        .answer(&archive)
        .unwrap();
    assert_eq!(discovery["result"]["supportedVersions"], json!(VERSIONS));
    assert_eq!(discovery["result"]["resultType"], "complete");
    let list = session
        .prepare(json!({"jsonrpc":"2.0", "id":2, "method":"tools/list", "params":{"_meta":meta}}))
        .answer(&archive)
        .unwrap();
    assert_eq!(list["result"]["tools"], json!(tools()));
    assert_eq!(list["result"]["resultType"], "complete");
    let invalid = session
        .prepare(json!({"jsonrpc":"2.0", "id":3, "method":"tools/list"}))
        .answer(&archive)
        .unwrap();
    assert_eq!(invalid["error"]["code"], -32602);
    let mut init = initialize();
    init["params"]["protocolVersion"] = json!("future");
    let answer = Session::default().prepare(init).answer(&archive).unwrap();
    assert_eq!(answer["result"]["protocolVersion"], "2025-11-25");
}

#[test]
fn protocol_ids_null_arguments_and_ping_follow_the_request_version() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let mut session = Session::default();
    session.prepare(initialize()).answer(&archive).unwrap();
    let id = json!(u64::MAX);
    let ping = session
        .prepare(json!({"jsonrpc":"2.0", "id":id, "method":"ping"}))
        .answer(&archive)
        .unwrap();
    assert_eq!(ping["id"], id);
    assert_eq!(ping["result"], json!({}));
    let call = session.prepare(json!({"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":{"name":"status", "arguments":null}}));
    match call {
        Reply::Tool { arguments, .. } => assert!(arguments.is_empty()),
        _ => panic!("null arguments must be an empty argument map"),
    }
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientCapabilities":{}});
    let ping = session
        .prepare(json!({"jsonrpc":"2.0", "id":3, "method":"ping", "params":{"_meta":meta}}))
        .answer(&archive)
        .unwrap();
    assert_eq!(ping["error"]["code"], -32601);
}

#[test]
fn malformed_request_ids_are_not_echoed_in_error_responses() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    for id in [json!({"wrong":1}), json!([1]), json!(true)] {
        let reply = Session::default()
            .prepare(json!({"jsonrpc":"2.0", "id":id, "method":"ping"}))
            .answer(&archive)
            .unwrap();
        assert_eq!(reply["id"], Value::Null);
        assert_eq!(reply["error"]["code"], -32600);
    }
}

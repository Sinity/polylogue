use super::*;

#[test]
fn completion_route_uses_full_query_cursor_and_preserves_the_suffix() {
    let daemon = crate::test_daemon::start("MCP query completion");
    let archive = &daemon.daemon.archive;
    let mut session = Session::default();
    session.initialized = true;
    let reply = session
        .prepare(json!({
            "jsonrpc":"2.0","id":1,"method":"completion/complete",
            "params":{"argument":{"name":"query","value":"live: | measure count"},
                "_meta":{"polylogue/position":5}}
        }))
        .answer(archive)
        .unwrap();
    assert!(reply.get("error").is_none(), "{reply}");
    let rich: crate::protocol::complete::Answer =
        serde_json::from_value(reply["result"]["_meta"]["polylogue/completions"].clone()).unwrap();
    assert!(!rich.is_empty());
    let values = reply["result"]["completion"]["values"].as_array().unwrap();
    assert_eq!(values.len(), rich.len());
    assert!(
        values
            .iter()
            .all(|value| value.as_str().unwrap().ends_with(" | measure count"))
    );
    let (_, direct, failed) = call_with_limit(
        archive,
        "complete",
        json!({"text":"live: | measure count","position":5})
            .as_object()
            .unwrap()
            .clone(),
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )
    .unwrap();
    assert!(!failed, "{direct}");
    assert_eq!(
        reply["result"]["_meta"]["polylogue/completions"],
        direct["value"]
    );
}

#[test]
fn completion_route_rejects_a_cursor_inside_a_utf8_character() {
    let daemon = crate::test_daemon::start("MCP completion boundary");
    let mut session = Session::default();
    session.initialized = true;
    let reply = session
        .prepare(json!({
            "jsonrpc":"2.0","id":2,"method":"completion/complete",
            "params":{"argument":{"name":"query","value":"é"},"_meta":{"polylogue/position":1}}
        }))
        .answer(&daemon.daemon.archive)
        .unwrap();
    assert_eq!(reply["error"]["data"]["outcome"], "error");
    assert_eq!(reply["error"]["data"]["error"]["kind"], "bad_request");
}

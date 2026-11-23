use super::*;

#[test]
fn completion_route_uses_full_query_cursor_and_preserves_the_suffix() {
    let daemon = crate::test_daemon::start("MCP query completion");
    let archive = &daemon.daemon.archive;
    let mut session = Session {
        initialized: true,
        ..Default::default()
    };
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
    let (_, direct, failed) = run(
        archive,
        command_surface::RunnableRequest::new(
            "internal complete",
            json!({"text":"live: | measure count","position":5}),
            "complete-parity",
        ),
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )
    .unwrap();
    assert!(!failed, "{direct}");
    // The daemon may rewrite its completion catalog between the two calls;
    // only the time the catalog was written, at the end of each value's help,
    // may differ between the routes.
    let without_catalog_time = |completions: &Value| -> Value {
        let mut completions = completions.clone();
        for completion in completions.as_array_mut().unwrap() {
            if let Some(help) = completion["help"].as_str()
                && let Some(at) = help.find("values as of ")
            {
                completion["help"] = json!(help[..at].trim_end_matches(" · "));
            }
        }
        completions
    };
    assert_eq!(
        without_catalog_time(&reply["result"]["_meta"]["polylogue/completions"]),
        without_catalog_time(&direct["value"])
    );
}

#[test]
fn completion_route_rejects_a_cursor_inside_a_utf8_character() {
    let daemon = crate::test_daemon::start("MCP completion boundary");
    let mut session = Session {
        initialized: true,
        ..Default::default()
    };
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

#[test]
fn completion_without_a_position_completes_at_the_end_of_the_text() {
    let daemon = crate::test_daemon::start("MCP default completion position");
    let archive = &daemon.daemon.archive;
    let mut session = Session {
        initialized: true,
        ..Default::default()
    };
    let text = "live: | measure count";
    let mut complete = |meta: Value| {
        let reply = session
            .prepare(json!({
                "jsonrpc":"2.0","id":1,"method":"completion/complete",
                "params":{"argument":{"name":"query","value":text}, "_meta":meta}
            }))
            .answer(archive)
            .unwrap();
        assert!(reply.get("error").is_none(), "{reply}");
        reply["result"]["completion"]["values"].clone()
    };
    let default = complete(json!({}));
    let values = default.as_array().expect("completion values are an array");
    assert!(!values.is_empty());
    assert!(
        values
            .iter()
            .all(|value| value.as_str().unwrap().starts_with("live:")),
        "{values:?}"
    );
    let explicit = complete(json!({"polylogue/position": text.len()}));
    assert_eq!(default, explicit);
}

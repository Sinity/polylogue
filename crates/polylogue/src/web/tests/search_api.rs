//! Search operations use the same answers on the socket, HTTP and MCP.
//! Blocking surfaces run outside the runtime used for HTTP exchanges.
use super::*;
use polylogue_render::html::encode_path_segment as enc;
use serde_json::{Value, json};

/// The rows of a query that matches something.
fn data(r: &Reader, text: &str, cursor: Option<String>) -> Value {
    crate::test_answers::ok(
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": text, "page_size": Some(50), "why": true}),
        )
        .cursor(cursor.as_deref())
        .answer(&r.web.daemon),
    )["reply"]
        .clone()
}

#[test]
fn facet_counts_cover_the_population_beyond_the_visible_page() {
    let r = reader();
    for n in 0..120 {
        std::fs::write(
            r.w.home
                .join(format!(".claude/projects/-p/extra-{n}.jsonl")),
            polylogue_testkit::claude_user(
                &format!("u-{n}"),
                &format!("extra-{n}"),
                "synthetic parser prompt",
            ),
        )
        .unwrap();
    }
    r.w.wait_for(|s| s["store"]["sessions"] == 121 && s["store"]["search_pending"] == 0);
    let text = "sessions where (parser OR missing) origin:claude-code";
    let counts = data(&r, &format!("{text} | group by origin | count"), None);
    assert_eq!(counts["rows"][0]["count"], 121, "{counts}");
    let mut cursor = None;
    let mut ids = std::collections::HashSet::new();
    loop {
        let answer = data(&r, text, cursor);
        assert!(answer["rows"].as_array().unwrap().len() <= 50);
        for row in answer["rows"].as_array().unwrap() {
            assert!(ids.insert(row["id"].as_str().unwrap().to_owned()));
        }
        cursor = answer["next"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(ids.len(), 121);
}

#[test]
fn search_writes_are_available_through_the_command_table() {
    let r = reader();
    rt().block_on(async {
        for (id, name) in [("saved-search", Some("parser")), ("kept-search", None)] {
            let body = json!({"request_id":id,"selection":"sessions where parser","steps":[{"verb":"save","name":name}]});
            for repeat in [false, true] {
                let (status, _, raw) = send(&r, own_origin(authed(post("/api/v1/rows"))).body(Body::from(body.to_string())).unwrap()).await;
                assert_eq!(status, StatusCode::OK, "{raw}");
                let value: Value = serde_json::from_str(&raw).unwrap();
                assert_eq!(value["repeat"], repeat);
                if name.is_some() { assert_eq!(value["saved"]["new_revision"], true); }
            }
        }
    });
    for selection in ["query:parser", "result:kept-search-1"] {
        let rows = data(&r, selection, None)["rows"].clone();
        assert_eq!(rows.as_array().unwrap().len(), 1, "{selection}");
        assert_eq!(rows[0]["id"], "claude-code:s1", "{selection}");
    }
    let kept = crate::mcp::tests::structured_call_as(
        &polylogue_store::Archive::at(&r.web.daemon.archive.root),
        "rows",
        json!({"selection":"sessions where parser","steps":[{"verb":"save"}]})
            .as_object()
            .unwrap()
            .clone(),
        "mcp-kept",
    );
    let crate::protocol::line::StepAnswer::Saved(kept) = crate::test_answers::step(kept) else {
        panic!("expected saved selection")
    };
    assert_eq!(kept.saved_rows.result.as_str(), "result:mcp-kept-1");
}

#[test]
fn filtered_semantic_and_mixed_results_keep_lower_ranked_matches() {
    let r = reader();
    std::fs::write(
        r.w.home.join(".claude/projects/-p/other.jsonl"),
        polylogue_testkit::claude_user("other-u", "other", "synthetic parser match"),
    )
    .unwrap();
    r.w.wait_for(|s| s["store"]["messages"] == 2 && s["store"]["search_pending"] == 0);
    let provider = r.w.local_embeddings();
    let http = rt();
    for source in [
        "sessions where -id:claude-code:s1 similar:\"parser\"",
        "sessions where -id:claude-code:s1 parser similar:\"parser\"",
    ] {
        let text = format!("{source} origin:claude-code");
        let answer = data(&r, &text, None);
        assert_eq!(answer["rows"].as_array().unwrap().len(), 1, "{answer}");
        assert_eq!(answer["rows"][0]["id"], "claude-code:other");
        let (_, _, raw) = http.block_on(send(
            &r,
            authed(get(&format!(
                "/api/v1/rows?selection={}&page_size=50&why=true",
                enc(&text)
            )))
            .body(Body::empty())
            .unwrap(),
        ));
        let api: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(api["rows"], answer["rows"]);
        let args = json!({"selection":text,"page_size":50,"why":true})
            .as_object()
            .unwrap()
            .clone();
        let mcp = crate::mcp::tests::structured_call_as(
            &polylogue_store::Archive::at(&r.web.daemon.archive.root),
            "rows",
            args,
            "web-semantic-parity",
        );
        assert_eq!(api["rows"], mcp.value().unwrap()["reply"]["rows"]);
        let (status, _, raw) = http.block_on(send(
            &r,
            own_origin(authed(post("/api/v1/rows")))
                .body(Body::from(
                    json!({"request_id":format!("save-{source}"),"selection":text,"steps":[{"verb":"save"}]}).to_string(),
                ))
                .unwrap(),
        ));
        assert_eq!(status, StatusCode::OK, "{raw}");
        let outcome: Value = serde_json::from_str(&raw).unwrap();
        let kept = data(
            &r,
            &format!(
                "result:{}",
                serde_json::to_string(
                    outcome["kept"]["result"]
                        .as_str()
                        .unwrap()
                        .strip_prefix("result:")
                        .unwrap()
                )
                .unwrap()
            ),
            None,
        );
        assert_eq!(kept["rows"].as_array().unwrap().len(), 1, "{kept}");
        assert_eq!(kept["rows"][0]["id"], "claude-code:other");
    }
    provider.join().unwrap();
}

#[test]
fn filtered_search_has_the_same_rows_on_every_surface() {
    let r = reader();
    let http = rt();
    for text in [
        "messages where (parser OR absent) -role:assistant",
        "sessions where origin:claude-code parser",
        "messages where origin:claude-code similar:\"parser\"",
        "messages where origin:claude-code parser similar:\"parser\"",
    ] {
        let request = crate::client::request(
            "rows",
            serde_json::json!({"selection": text, "page_size": 50, "why": true}),
            "socket-query".into(),
        );
        let socket = crate::client::call(&r.web.daemon.archive.socket(), &request).unwrap();
        let (status, _, raw) = http.block_on(send(
            &r,
            authed(get(&format!(
                "/api/v1/rows?selection={}&page_size=50&why=true",
                enc(text)
            )))
            .body(Body::empty())
            .unwrap(),
        ));
        let api: Value = serde_json::from_str(&raw).unwrap();
        match crate::protocol::into_value(socket) {
            Ok(answer) => {
                assert_eq!(status, StatusCode::OK, "{raw}");
                assert_eq!(api["rows"], answer["reply"]["rows"], "{text}");
                let args = json!({"selection":text,"page_size":50,"why":true})
                    .as_object()
                    .unwrap()
                    .clone();
                let mcp = crate::mcp::tests::structured_call_as(
                    &polylogue_store::Archive::at(&r.web.daemon.archive.root),
                    "rows",
                    args,
                    "web-search-parity",
                );
                assert_eq!(api["rows"], mcp.value().unwrap()["reply"]["rows"]);
            }
            Err(failure) => {
                assert_ne!(status, StatusCode::OK, "{raw}");
                assert_eq!(api["error"]["code"], failure.code.unwrap());
            }
        }
    }
}

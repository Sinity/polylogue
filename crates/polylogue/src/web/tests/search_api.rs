//! Search operations use the same answers on the socket, HTTP and MCP.
use super::*;
use polylogue_render::html::encode_path_segment as enc;
use serde_json::{Value, json};

fn data(r: &Reader, text: &str, cursor: Option<String>) -> Value {
    match handle(Request::Query {
        text: text.into(), cursor, page_size: Some(50), why: true,
        session: None, fields: vec![],
    }, &r.web.daemon) {
        Response::Ok { data } => data,
        other => panic!("{text}: {other:?}"),
    }
}

#[test]
fn facet_counts_cover_the_population_beyond_the_visible_page() {
    let r = reader();
    for n in 0..120 {
        std::fs::write(
            r.w.home.join(format!(".claude/projects/-p/extra-{n}.jsonl")),
            polylogue_testkit::claude_user(&format!("u-{n}"), &format!("extra-{n}"), "synthetic parser prompt"),
        ).unwrap();
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
        if cursor.is_none() { break; }
    }
    assert_eq!(ids.len(), 121);
}

#[test]
fn search_writes_are_available_through_the_command_table() {
    let r = reader();
    rt().block_on(async {
        for (operation, body) in [
            ("save", json!({"request_id":"saved-search","name":"parser","query":"sessions where parser"})),
            ("keep", json!({"request_id":"kept-search","query":"sessions where parser"})),
        ] {
            for repeat in [false, true] {
                let (status, _, raw) = send(&r, own_origin(authed(post(&format!("/api/v1/{operation}")))).body(Body::from(body.to_string())).unwrap()).await;
                assert_eq!(status, StatusCode::OK, "{raw}");
                let value: Value = serde_json::from_str(&raw).unwrap();
                assert_eq!(value["repeat"], repeat);
            }
        }
        assert_eq!(data(&r, "query:parser", None)["rows"].as_array().unwrap().len(), 1);
        assert_eq!(data(&r, "result:kept-search", None)["rows"].as_array().unwrap().len(), 1);
        let kept = crate::mcp::call(
            &polylogue_store::Archive::at(&r.web.daemon.archive.root), "keep",
            json!({"request_id":"mcp-kept","query":"sessions where parser"}).as_object().unwrap().clone(),
        ).unwrap();
        let receipt: Value = serde_json::from_str(&kept).unwrap();
        assert_eq!(receipt["id"], "mcp-kept");
    });
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
    rt().block_on(async {
        for source in [
            "sessions where -id:claude-code:s1 near:\"parser\"",
            "sessions where -id:claude-code:s1 parser near:\"parser\"",
        ] {
            let text = format!("{source} origin:claude-code");
            let answer = data(&r, &text, None);
            assert_eq!(answer["rows"].as_array().unwrap().len(), 1, "{answer}");
            assert_eq!(answer["rows"][0]["id"], "claude-code:other");
            let (_, _, raw) = send(
                &r,
                authed(get(&format!(
                    "/api/v1/query?query={}&limit=50&why=true",
                    enc(&text)
                )))
                .body(Body::empty())
                .unwrap(),
            )
            .await;
            let api: Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(api["rows"], answer["rows"]);
            let args = json!({"query":text,"limit":50,"why":true})
                .as_object()
                .unwrap()
                .clone();
            let mcp: Value = serde_json::from_str(
                &crate::mcp::call(
                    &polylogue_store::Archive::at(&r.web.daemon.archive.root),
                    "query",
                    args,
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(api["rows"], mcp["rows"]);
            let (status, _, raw) = send(
                &r,
                own_origin(authed(post("/api/v1/keep")))
                    .body(Body::from(
                        json!({"request_id":format!("keep-{source}"),"query":text}).to_string(),
                    ))
                    .unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{raw}");
            let receipt: Value = serde_json::from_str(&raw).unwrap();
            let kept = data(
                &r,
                &format!(
                    "result:{}",
                    serde_json::to_string(receipt["id"].as_str().unwrap()).unwrap()
                ),
                None,
            );
            assert_eq!(kept["rows"].as_array().unwrap().len(), 1, "{kept}");
            assert_eq!(kept["rows"][0]["id"], "claude-code:other");
        }
    });
    provider.join().unwrap();
}

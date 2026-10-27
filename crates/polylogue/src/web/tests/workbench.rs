use super::*;
use polylogue_render::html::encode_path_segment as enc;
use serde_json::{Value, json};

fn data(r: &Reader, text: &str, cursor: Option<String>) -> Value {
    match handle(
        super::super::search::query(text.into(), cursor),
        &r.web.daemon,
    ) {
        Response::Ok { data } => scrubbed(&r.web, data),
        other => panic!("{text}: {other:?}"),
    }
}

#[test]
fn filtered_search_has_the_same_rows_on_every_surface() {
    let r = reader();
    rt().block_on(async {
        for text in [
            "messages where (parser OR absent) -role:assistant",
            "sessions where origin:claude-code parser",
            "messages where origin:claude-code near:\"parser\"",
            "messages where origin:claude-code parser near:\"parser\"",
        ] {
            let request = super::super::search::query(text.into(), None);
            let socket = crate::cli::send(&r.web.daemon.archive.socket(), &request).unwrap();
            let (status, _, raw) = send(
                &r,
                authed(get(&format!(
                    "/api/v1/query?query={}&limit=50&why=true",
                    enc(text)
                )))
                .body(Body::empty())
                .unwrap(),
            )
            .await;
            let api: Value = serde_json::from_str(&raw).unwrap();
            let (_, _, html) = send(
                &r,
                authed(get(&format!("/search?q={}", enc(text))))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            match socket {
                Response::Ok { data: answer } => {
                    assert_eq!(status, StatusCode::OK, "{raw}");
                    assert_eq!(api["rows"], answer["rows"], "{text}");
                    for row in answer["rows"].as_array().unwrap() {
                        let id = row["id"].as_str().unwrap();
                        assert!(html.contains(&enc(id)), "{html}");
                    }
                    let args = json!({"query":text,"limit":50,"why":true})
                        .as_object()
                        .unwrap()
                        .clone();
                    let mcp = crate::mcp::call(
                        &polylogue_store::Archive::at(&r.web.daemon.archive.root),
                        "query",
                        args,
                    )
                    .unwrap();
                    let mcp: Value = serde_json::from_str(&mcp).unwrap();
                    assert_eq!(api["rows"], mcp["rows"]);
                }
                Response::Error { kind, .. } => {
                    assert_ne!(status, StatusCode::OK, "{raw}");
                    assert_eq!(api["error"], kind);
                    assert!(html.contains(&kind), "{html}");
                }
            }
        }
    });
}

#[test]
fn facet_counts_cover_more_than_the_visible_page_and_paging_reaches_every_row() {
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
    let text =
        super::super::search::filtered("sessions where parser OR missing", "origin", "claude-code")
            .unwrap();
    let counts = data(
        &r,
        &super::super::search::grouped(&text, "origin").unwrap(),
        None,
    );
    assert_eq!(counts["rows"][0]["count"], 121, "{counts}");
    let mut cursor = None;
    let mut ids = std::collections::HashSet::new();
    rt().block_on(async {
        loop {
            let answer = data(&r, &text, cursor.clone());
            let path = format!(
                "/search?q={}{}",
                enc(&text),
                cursor
                    .as_deref()
                    .map(|c| format!("&cursor={}", enc(c)))
                    .unwrap_or_default()
            );
            let (status, _, html) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{html}");
            for row in answer["rows"].as_array().unwrap() {
                let id = row["id"].as_str().unwrap();
                assert!(ids.insert(id.to_owned()), "duplicate {id}");
                assert!(html.contains(&enc(id)), "{html}");
            }
            cursor = answer["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
            assert!(html.contains("rel=\"next\""));
        }
        let (_, _, html) = send(
            &r,
            authed(get(&format!(
                "/search/facets?q={}&field=origin",
                enc(&text)
            )))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert!(html.contains("121"), "{html}");
    });
    assert_eq!(ids.len(), 121);
}

#[test]
fn facet_changes_reset_cursors_and_escape_query_text_in_every_template() {
    let r = reader();
    rt().block_on(async {
        let text = r#"sessions where title:"<script> & λ | x" OR origin:claude-code"#;
        let (status, headers, _) = send(
            &r,
            authed(get(&format!(
                "/search?q={}&field=origin&value=claude-code&cursor=obsolete",
                enc(text)
            )))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let location = headers[header::LOCATION].to_str().unwrap();
        assert!(!location.contains("cursor="));
        let (_, _, html) = send(&r, authed(get(location)).body(Body::empty()).unwrap()).await;
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("OR origin:claude-code"));
        let (_, _, home) = send(
            &r,
            authed(get("/?recent=true")).body(Body::empty()).unwrap(),
        )
        .await;
        assert!(home.contains("Recent sessions"));
        assert!(home.contains("?latest=true"));
        assert!(home.contains("ended"));
        assert!(home.contains("Notices have not been requested"));
    });
}

#[test]
fn saved_queries_and_kept_results_use_existing_idempotent_writes() {
    let r = reader();
    rt().block_on(async {
        for (operation, body) in [
            ("save",json!({"request_id":"saved-search","name":"parser","query":"sessions where parser"})),
            ("keep",json!({"request_id":"kept-search","query":"sessions where parser"})),
        ] {
            let mut first: Option<Value> = None;
            for _ in 0..2 {
                let (status,_,raw) = send(&r,own_origin(authed(post(&format!("/api/v1/{operation}")))).body(Body::from(body.to_string())).unwrap()).await;
                assert_eq!(status,StatusCode::OK,"{raw}");
                let value:Value = serde_json::from_str(&raw).unwrap();
                if let Some(first) = &first { assert_eq!(value["id"],first["id"]); assert_eq!(value["repeat"],true); } else { first=Some(value); }
            }
        }
        assert_eq!(data(&r,"query:parser",None)["rows"].as_array().unwrap().len(),1);
        assert_eq!(data(&r,"notes where kind:saved_query",None)["rows"].as_array().unwrap().len(),1);
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
            let text = super::super::search::filtered(source, "origin", "claude-code").unwrap();
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
            let (_, _, html) = send(
                &r,
                authed(get(&format!("/search?q={}", enc(&text))))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert!(html.contains("/s/claude-code%3Aother"), "{html}");
            assert!(!html.contains("/s/claude-code%3As1\""), "{html}");
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
                &format!("result:{}", receipt["id"].as_str().unwrap()),
                None,
            );
            assert_eq!(kept["rows"].as_array().unwrap().len(), 1, "{kept}");
            assert_eq!(kept["rows"][0]["id"], "claude-code:other");
        }
    });
    provider.join().unwrap();
}

#[test]
fn live_lists_waiting_sessions_first_with_recorded_evidence_and_recent_keeps_ended() {
    let r = reader();
    crate::hook::append_to_spool(
        &r.w.config.spool_dir("hook-spool"),
        crate::hook::Harness::ClaudeCode,
        "Notification",
        &json!({"session_id":"waiting","notification_type":"permission_prompt"}),
    )
    .unwrap();
    crate::hook::append_to_spool(
        &r.w.config.spool_dir("hook-spool"),
        crate::hook::Harness::ClaudeCode,
        "UserPromptSubmit",
        &json!({"session_id":"working"}),
    )
    .unwrap();
    r.w.wait_for(|s| s["store"]["sessions"] == 3);
    let answer = data(&r, super::super::live::LIVE, None);
    assert_eq!(answer["rows"][0]["id"], "claude-code:waiting", "{answer}");
    rt().block_on(async {
        let (_, _, html) = send(&r, authed(get("/")).body(Body::empty()).unwrap()).await;
        assert!(html.contains("permission"), "{html}");
        assert!(html.contains("hook_notification"), "{html}");
        assert!(
            html.find("claude-code%3Awaiting").unwrap()
                < html.find("claude-code%3Aworking").unwrap()
        );
        assert!(!html.contains("/s/claude-code%3As1?"));
        let (_, _, recent) = send(
            &r,
            authed(get("/?recent=true")).body(Body::empty()).unwrap(),
        )
        .await;
        assert!(recent.contains("/s/claude-code%3As1?"));
    });
}

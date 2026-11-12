fn feed_frames(text: &str) -> Vec<serde_json::Value> {
    command_surface::SseDecoder::new()
        .push(text.as_bytes())
        .unwrap()
        .into_iter()
        .map(|event| serde_json::from_str(&event.data).unwrap())
        .collect()
}

use super::*;

#[test]
fn page_navigation_and_message_routes_use_the_shared_selection() {
    let r = reader();
    rt().block_on(async {
        let first = read_page(&r, "session:claude-code:s1", &[("page_tokens", "2")]).await;
        let next = first
            .document
            .footer
            .next
            .as_ref()
            .expect("small budget continues");
        assert_eq!(next.request.operation, "read");
        let (_, headers, _) = send(
            &r,
            authed(get("/m/claude-code:u1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(
            headers[header::LOCATION],
            "/api/v1/read?target=message%3Aclaude-code%3Au1"
        );
        let (status, _, body) = send_html(&r, "/m/claude-code:u1").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        let path = polylogue_render::web::http_url(crate::commands::declarations(), &next.request)
            .unwrap();
        let (status, _, body) = send_html(&r, &path).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(!body.contains("history.back"));
    });
}

#[test]
fn command_and_sign_in_body_types_are_checked_before_parsing_or_writing() {
    let r = reader();
    rt().block_on(async {
        for path in ["/auth", "/api/v1/setting"] {
            for media in [
                None,
                Some("text/plain"),
                Some("application/jsonp"),
                Some("text/html"),
            ] {
                let mut req = own_origin(authed(post(path)))
                    .body(Body::from("{}"))
                    .unwrap();
                req.headers_mut().remove(header::CONTENT_TYPE);
                if let Some(media) = media {
                    req.headers_mut()
                        .insert(header::CONTENT_TYPE, HeaderValue::from_static(media));
                }
                let (status, headers, _) = send(&r, req).await;
                assert_eq!(
                    status,
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "{path}: {media:?}"
                );
                assert_eq!(headers[header::CACHE_CONTROL], "no-store");
            }
        }
        let mut req = own_origin(authed(post("/api/v1/setting")))
            .body(Body::from(setting_envelope("media-setting")))
            .unwrap();
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        assert_eq!(send(&r, req).await.0, StatusCode::OK);
    });
}

#[test]
fn finite_http_changes_are_authenticated_and_resume_only_from_complete() {
    let r = reader();
    rt().block_on(async {
        let (status, headers, body) = send(
            &r,
            authed(get("/api/v1/changes")).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(headers[header::CONTENT_TYPE], "text/event-stream");
        let lines = feed_frames(&body);
        assert_eq!(lines.first().unwrap()["type"], "begin");
        assert_eq!(lines.last().unwrap()["type"], "complete");
        let position = lines.last().unwrap()["position"].as_str().unwrap();
        let path = format!(
            "/api/v1/changes?since={}",
            polylogue_render::html::encode_path_segment(position)
        );
        let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("\"count\":0"), "{body}");
        assert_eq!(
            send(&r, get(&path).body(Body::empty()).unwrap()).await.0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            send(
                &r,
                authed(get("/api/v1/changes?since=broken"))
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
    });
}

#[test]
fn changed_generation_refuses_content_continuation_and_replaces_change_population() {
    let r = reader();
    rt().block_on(async {
        let first = read_page(&r, "session:claude-code:s1", &[("page_tokens", "2")]).await;
        let next = polylogue_render::web::http_url(
            crate::commands::declarations(),
            &first.document.footer.next.unwrap().request,
        )
        .unwrap();
        let (_, _, body) = send(
            &r,
            authed(get("/api/v1/changes")).body(Body::empty()).unwrap(),
        )
        .await;
        let frames = feed_frames(&body);
        let position = frames.last().unwrap()["position"].as_str().unwrap();
        let conn = rusqlite::Connection::open(r.w.daemon.archive.index_db()).unwrap();
        conn.execute(
            "UPDATE meta SET value=?1 WHERE key='generation'",
            [uuid::Uuid::now_v7().to_string()],
        )
        .unwrap();
        let (status, _, body) = send(&r, authed(get(&next)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("stale_cursor"));
        let url = format!(
            "/api/v1/changes?since={}",
            polylogue_render::html::encode_path_segment(position)
        );
        let (status, _, body) = send(&r, authed(get(&url)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let frames = feed_frames(&body);
        assert_eq!(frames[0]["generation_changed"], true);
        assert_eq!(frames[0]["replacement"], true);
    });
}

#[test]
fn web_client_authentication_recovery_and_visibility_contract() {
    let output = std::process::Command::new("node")
        .arg("--test")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/web/assets/tests/live.test.mjs"
        ))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/web/assets/tests/auth.test.mjs"
        ))
        .output()
        .expect("Node from the development shell");
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn every_relationship_is_reachable_and_search_pages_keep_the_query() {
    let r = reader();
    for n in 0..125 {
        let child = format!("related-{n}");
        let mut record: serde_json::Value = serde_json::from_str(&polylogue_testkit::claude_user(
            &format!("u-{child}"),
            &child,
            "synthetic related prompt",
        ))
        .unwrap();
        record["forkedFrom"] = serde_json::json!({"sessionId":"s1","messageUuid":"u1"});
        std::fs::write(
            r.w.home.join(format!(".claude/projects/-p/{child}.jsonl")),
            format!("{record}\n"),
        )
        .unwrap();
    }
    r.w.wait_for(|data| data["store"]["messages"] == 126);
    fn link(html: &str, label: &str) -> Option<String> {
        let end = html.find(&format!("\">{label}</a>"))?;
        let (_, path) = html[..end].rsplit_once("href=\"")?;
        Some(path.replace("&amp;", "&"))
    }
    rt().block_on(async {
        let items = read_items(&r, "session:claude-code:s1", &[("view", "links")]).await;
        let mut seen = std::collections::HashSet::new();
        for item in items {
            if let polylogue_model::read::ItemKind::Card { card } = item.kind
                && let polylogue_model::read::Card::LinkEdge {
                    from,
                    to,
                    relation,
                    how,
                    ..
                } = card.as_ref()
            {
                assert_eq!(from, "session:claude-code:s1");
                assert_eq!(relation, "fork");
                assert_eq!(how.as_deref(), None);
                assert!(seen.insert(to.clone()), "relationship repeated");
            }
        }
        assert_eq!(seen.len(), 125);
        assert_eq!(
            seen,
            (0..125)
                .map(|n| format!("session:claude-code:related-{n}"))
                .collect()
        );
        let (_, _, body) = send(
            &r,
            authed(get("/search?q=messages"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        let next = link(&body, "Next").expect("search continuation");
        assert!(next.starts_with("/search?q=messages&cursor="));
        assert!(!body.contains("more: --cursor"));
        let (status, _, body) = send(&r, authed(get(&next)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("polylogue list messages --cursor"));
    });
}

#[test]
fn exact_reference_routes_do_not_substitute_prefixes_or_current_note_revisions() {
    use polylogue_model::Reference;
    use polylogue_model::assertion::{
        AssertionKind, AssertionPatch, Change, LinkRole, NewAssertion, NewAssertionLink,
    };
    let r = reader();
    let note = NewAssertion {
        kind: AssertionKind::Note,
        key: None,
        value: Some(serde_json::json!("older exact note <text>")),
        links: vec![NewAssertionLink {
            role: LinkRole::About,
            reference: Reference::Session(
                polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            ),
        }],
        valid_until: None,
    };
    let Ok(data) = crate::protocol::into_value(
        crate::test_daemon::Call::new("assert", serde_json::json!({"assertions": [note]}))
            .id("exact-note")
            .answer(&r.web.daemon),
    ) else {
        panic!("note failed")
    };
    let old = data["id"].as_str().unwrap().to_owned();
    crate::test_answers::ok(
        crate::test_daemon::Call::new(
            "revise",
            serde_json::json!({"bases": [format!("assertion:{old}@{old}")], "patch": AssertionPatch {
                value: Change::Set(serde_json::json!("newer exact note")),
                ..Default::default()
            }}),
        )
        .id("exact-note-new")
        .answer(&r.web.daemon),
    );
    let exact = Reference::Assertion(polylogue_model::RevisionReference::exact(
        old.clone(),
        old.clone(),
    ))
    .to_string();
    rt().block_on(async {
        let (status, _, body) = send_html(
            &r,
            &format!("/a/{}", polylogue_render::html::encode_path_segment(&exact)),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("older exact note &lt;text&gt;"));
        assert!(!body.contains("newer exact note"));
        let page = read_page(&r, &exact, &[]).await;
        assert_eq!(page.document.header.target, exact);
        // A message prefix and a note prefix remain absent exact references.
        for path in ["/m/claude-code:u", "/a/assertion:exact@exact"] {
            assert_eq!(send_html(&r, path).await.0, StatusCode::NOT_FOUND, "{path}");
        }
    });
}

#[test]
fn inheritance_is_an_explicit_shared_read_option() {
    let r = reader();
    let mut record: serde_json::Value = serde_json::from_str(&polylogue_testkit::claude_user(
        "child-prompt",
        "child",
        "synthetic child only",
    ))
    .unwrap();
    record["forkedFrom"] = serde_json::json!({"sessionId":"s1","messageUuid":"u1"});
    std::fs::write(
        r.w.home.join(".claude/projects/-p/child.jsonl"),
        format!("{record}\n"),
    )
    .unwrap();
    r.w.wait_for(|data| data["store"]["messages"] == 2);
    rt().block_on(async {
        let (_, _, own) = send_html(&r, "/s/claude-code:child").await;
        assert!(own.contains("synthetic child only"));
        assert!(!own.contains("&lt;script&gt;alert"));
        let (status, _, inherited) = send_html(&r, "/s/claude-code:child?show=inherited").await;
        assert_eq!(status, StatusCode::OK, "{inherited}");
        assert!(inherited.contains("&lt;script&gt;alert"));
        let page = read_page(&r, "session:claude-code:child", &[("show", "inherited")]).await;
        assert!(page.document.items.iter().any(|item| item.source.inherited));
    });
}

#[test]
fn previous_links_reopen_exact_unicode_fragments_without_browser_history() {
    let r = reader();
    let text = "first λ🙂 ".repeat(800);
    let rewritten = polylogue_testkit::claude_user("u1", "s1", &text);
    std::fs::write(r.w.home.join(".claude/projects/-p/s1.jsonl"), &rewritten).unwrap();
    // The archive before the rewrite is already idle with nothing pending;
    // wait for the rewritten bytes, or the reads below race its re-parse,
    // which changes the session's content under the sealed links.
    let length = rewritten.len() as u64;
    r.w.wait_for(|data| {
        data["store"]["stored_bytes"]
            .as_u64()
            .is_some_and(|stored| stored >= length)
            && data["store"]["search_pending"] == 0
    });
    rt().block_on(async {
        let first = read_page(&r, "message:claude-code:u1", &[("page_tokens", "20")]).await;
        let next = first
            .document
            .footer
            .next
            .as_ref()
            .expect("long body continues");
        let second = follow_read(&r, next).await;
        let previous = second
            .document
            .footer
            .previous
            .as_ref()
            .expect("previous is supplied by service");
        let restored = follow_read(&r, previous).await;
        assert_eq!(restored.document.items, first.document.items);
        let path =
            polylogue_render::web::http_url(crate::commands::declarations(), &previous.request)
                .unwrap();
        let (status, _, html) = send_html(&r, &path).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(!html.contains("history.back"));
    });
}

#[test]
fn message_redirects_quote_literal_ids_without_resolving_source_rows() {
    let r = reader();
    let id = "claude-code:slash\\tab\tline\nquote\"AKIAIOSFODNN7EXAMPLE";
    rt().block_on(async {
        let path = format!("/m/{}", polylogue_render::html::encode_path_segment(id));
        let (status, headers, _) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            headers[header::LOCATION],
            format!(
                "/api/v1/read?target={}",
                polylogue_render::html::encode_path_segment(&format!("message:{id}"))
            )
        );
        let (status, _, body) = send_html(&r, &path).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    });
}

#[test]
fn message_identifiers_are_validated_even_when_a_cursor_is_supplied() {
    let r = reader();
    rt().block_on(async {
        let (status, _, body) = send_html(&r, "/m/not-an-id?cursor=invalid").await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    });
}

#[test]
fn live_initial_queries_page_through_the_real_http_api() {
    let r = reader();
    std::fs::write(
        r.w.home.join(".claude/projects/-p/s2.jsonl"),
        polylogue_testkit::claude_user("second", "s2", "second synthetic prompt"),
    )
    .unwrap();
    r.w.wait_for(|s| s["store"]["sessions"].as_u64().is_some_and(|n| n >= 2) && s["busy"] == false);
    rt().block_on(async {
        let mut cursor = None;
        let mut ids = Vec::new();
        loop {
            let mut params = vec![
                ("query".into(), "changes where kind:session".into()),
                ("limit".into(), "1".into()),
            ];
            if let Some(cursor) = cursor {
                params.push(("cursor".into(), cursor));
            }
            let path = format!(
                "/api/v1/query?{}",
                params
                    .iter()
                    .map(|(k, v): &(String, String)| format!(
                        "{}={}",
                        polylogue_render::html::encode_path_segment(k),
                        polylogue_render::html::encode_path_segment(v)
                    ))
                    .collect::<Vec<_>>()
                    .join("&")
            );
            let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let page: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert!(
                page["read_at"]["cursor"]
                    .as_str()
                    .and_then(polylogue_model::ChangePosition::parse)
                    .is_some(),
                "{page}"
            );
            for row in page["rows"].as_array().unwrap() {
                assert_eq!(row["kind"], "session");
                ids.push(row["ref"].as_str().unwrap().to_owned());
            }
            cursor = page["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
        assert!(ids.iter().any(|id| id == "claude-code:s1"));
        assert!(ids.len() >= 2);
        let distinct: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(distinct.len(), ids.len());
    });
}

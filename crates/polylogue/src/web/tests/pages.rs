use super::*;

#[test]
fn page_navigation_and_message_routes_use_the_shared_selection() {
    let r = reader();
    rt().block_on(async {
        let (status, _, first) = send(
            &r,
            authed(get("/s/claude-code:s1?budget=2"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{first}");
        assert!(first.contains("Next</a>"));
        let (_, text) = first.split_once("<div class=\"text\">").unwrap();
        let (text, _) = text.split_once("</div>").unwrap();
        assert!(!text.contains("alert(1)"), "content must be paged: {text}");
        let (status, headers, _) = send(
            &r,
            authed(get("/m/claude-code:u1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(
            headers[header::LOCATION],
            "/s/claude-code%3As1?message=claude-code%3Au1"
        );
        let (status, _, body) = send(
            &r,
            authed(get("/s/claude-code:s1?message=claude-code:u1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        let (status, _, body) = send(
            &r,
            authed(get("/api/v1/outline?session=claude-code:s1&page_size=1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let data: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(data["entries"][0]["message"], "claude-code:u1");
        assert!(
            !body.contains("alert"),
            "outline must not load a transcript body"
        );
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
            .body(Body::from(r#"{"key":"a","value":"b"}"#))
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
        assert_eq!(headers[header::CONTENT_TYPE], "application/x-ndjson");
        let lines = body
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines.first().unwrap()["data"]["type"], "begin");
        assert_eq!(lines.last().unwrap()["data"]["type"], "complete");
        let position = lines.last().unwrap()["data"]["position"].as_str().unwrap();
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
        let (_, _, body) = send(
            &r,
            authed(get(
                "/api/v1/read?session=claude-code:s1&with_parents=true&budget=2",
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        let page: serde_json::Value = serde_json::from_str(&body).unwrap();
        let token = page["next"].as_str().unwrap();
        let (_, _, body) = send(
            &r,
            authed(get("/api/v1/changes")).body(Body::empty()).unwrap(),
        )
        .await;
        let last: serde_json::Value = serde_json::from_str(body.lines().last().unwrap()).unwrap();
        let position = last["data"]["position"].as_str().unwrap();
        let conn = rusqlite::Connection::open(r.w.daemon.archive.index_db()).unwrap();
        conn.execute(
            "UPDATE meta SET value=?1 WHERE key='generation'",
            [uuid::Uuid::now_v7().to_string()],
        )
        .unwrap();
        let secret = "AKIAIOSFODNN7EXAMPLE";
        let url = format!(
            "/s/claude-code:s1?message=claude-code:{secret}&budget=2&cursor={}",
            polylogue_render::html::encode_path_segment(token)
        );
        let (status, _, body) = send(&r, authed(get(&url)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert!(body.contains("Refresh from this message"));
        assert!(
            body.contains(secret),
            "valid identifiers remain intact in links"
        );
        let url = format!(
            "/api/v1/changes?since={}",
            polylogue_render::html::encode_path_segment(position)
        );
        let (status, _, body) = send(&r, authed(get(&url)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let begin: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(begin["data"]["generation_changed"], true);
        assert_eq!(begin["data"]["replacement"], true);
    });
}

#[test]
fn live_client_recovery_and_visibility_contract() {
    let output = std::process::Command::new("node")
        .arg("--test")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/web/assets/tests/live.test.mjs"
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
        let mut path = "/s/claude-code:s1".to_owned();
        let mut seen = std::collections::HashSet::new();
        let mut pages = 0;
        loop {
            let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            for n in 0..125 {
                if body.contains(&format!("href=\"/s/claude-code%3Arelated-{n}\"")) {
                    assert!(seen.insert(n), "relationship repeated");
                }
            }
            pages += 1;
            let Some(next) = link(&body, "Next relationships page") else {
                break;
            };
            path = next;
            assert!(pages < 5);
        }
        assert_eq!(seen.len(), 125);
        assert_eq!(pages, 3);
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
    use polylogue_model::{Author, NewNote, NoteKind, Target};
    let r = reader();
    let note = NewNote {
        kind: NoteKind::Note,
        key: None,
        value: Some("older exact note <text>".into()),
        author: Author::Operator,
        targets: vec![Target::Session(
            polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
        )],
        relations: vec![],
        revises: None,
        valid_until: None,
    };
    let Response::Ok { data, .. } = handle(
        Request::AddNote {
            request_id: "exact-note".into(),
            note: note.clone(),
        },
        &r.web.daemon,
    ) else {
        panic!("note failed")
    };
    let old = data["id"].as_str().unwrap().to_owned();
    let revised = NewNote {
        value: Some("newer exact note".into()),
        revises: Some(old.clone()),
        ..note
    };
    assert!(matches!(
        handle(
            Request::AddNote {
                request_id: "exact-note-new".into(),
                note: revised
            },
            &r.web.daemon
        ),
        Response::Ok { .. }
    ));
    rt().block_on(async {
        let (status, _, body) = send(
            &r,
            authed(get(&format!(
                "/n/{}",
                polylogue_render::html::encode_path_segment(&old)
            )))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("older exact note &lt;text&gt;"));
        assert!(!body.contains("newer exact note"));
        let (status, _, body) = send(
            &r,
            authed(get(&format!(
                "/api/v1/note/read?id={}",
                polylogue_render::html::encode_path_segment(&old)
            )))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"],
            old
        );
        // A message prefix and a note prefix remain absent exact references.
        for path in ["/m/claude-code:u", "/n/exact"] {
            assert_eq!(
                send(&r, authed(get(path)).body(Body::empty()).unwrap())
                    .await
                    .0,
                StatusCode::NOT_FOUND,
                "{path}"
            );
        }
    });
}

fn page_link(html: &str, label: &str) -> String {
    let end = html.find(&format!("\">{label}</a>")).unwrap();
    html[..end]
        .rsplit_once("href=\"")
        .unwrap()
        .1
        .replace("&amp;", "&")
}

#[test]
fn fork_parent_toggle_opens_the_selected_content_and_outline_is_optional() {
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
        let (status, _, html) = send(
            &r,
            authed(get("/s/claude-code:child"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(html.contains("Open outline"));
        let path = page_link(&html, "Exclude parents");
        assert!(!path.contains("message="));
        let (status, _, html) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("synthetic child only"));
        assert!(!html.contains("&lt;script&gt;alert"));
        let path = page_link(&html, "Open outline");
        let (status, _, outline) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(outline.contains("<summary>Outline</summary>"));
        let path = page_link(&html, "Include parents");
        let (status, _, html) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(html.contains("&lt;script&gt;alert"));
    });
}

#[test]
fn previous_links_work_for_direct_anchors_and_within_large_blocks() {
    let r = reader();
    let records = polylogue_testkit::claude_user("u1", "s1", &"first ".repeat(500))
        + &polylogue_testkit::claude_user("u2", "s1", &"second ".repeat(500));
    std::fs::write(r.w.home.join(".claude/projects/-p/s1.jsonl"), records).unwrap();
    r.w.wait_for(|data| data["store"]["messages"] == 2);
    rt().block_on(async {
        let (_, _, html) = send(
            &r,
            authed(get("/s/claude-code:s1?message=claude-code:u2&budget=100"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        let previous = page_link(&html, "Previous");
        let (status, _, html) = send(&r, authed(get(&previous)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("first first"));
        assert!(!html.contains("second second"));
        let previous = page_link(&html, "Previous");
        let (status, _, html) = send(&r, authed(get(&previous)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("first first"));
        assert!(!html.contains("second second"));
        assert!(!html.contains("history.back"));
    });
}

#[test]
fn message_redirects_quote_literal_ids_and_preserve_them_in_links() {
    let r = reader();
    let id = "claude-code:slash\\tab\tline\nquote\"AKIAIOSFODNN7EXAMPLE";
    let conn = rusqlite::Connection::open(r.w.daemon.archive.index_db()).unwrap();
    conn.execute("UPDATE message SET id=?1", [id]).unwrap();
    rt().block_on(async {
        let path = format!("/m/{}", polylogue_render::html::encode_path_segment(id));
        let (status, headers, body) =
            send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::SEE_OTHER, "{body}");
        assert_eq!(
            headers[header::LOCATION],
            format!(
                "/s/claude-code%3As1?message={}",
                polylogue_render::html::encode_path_segment(id)
            )
        );
        let path = headers[header::LOCATION].to_str().unwrap();
        let (status, _, body) = send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("AKIAIOSFODNN7EXAMPLE"));
    });
}

#[test]
fn message_identifiers_are_validated_even_when_a_cursor_is_supplied() {
    let r = reader();
    rt().block_on(async {
        let (status, _, body) = send(
            &r,
            authed(get("/s/claude-code:s1?cursor=invalid&message=not-an-id"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("not a message ID"), "{body}");
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

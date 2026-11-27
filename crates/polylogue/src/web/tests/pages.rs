/// The feed's lines, without the final answer that ends the reply.
fn feed_frames(text: &str) -> Vec<serde_json::Value> {
    command_surface::SseDecoder::new()
        .push(text.as_bytes())
        .unwrap()
        .into_iter()
        .filter(|event| event.event != "answer")
        .map(|event| serde_json::from_str(&event.data).unwrap())
        .collect()
}

use super::*;

#[test]
fn page_navigation_and_message_routes_use_the_shared_selection() {
    let r = reader();
    rt().block_on(async {
        let first = read_page(&r, "session:claude-code:s1", &[("budget", "2")]).await;
        let next = first
            .document
            .footer
            .next
            .as_ref()
            .expect("small budget continues");
        assert_eq!(next.request.operation, "rows");
        let (status, _, body) = send_html(&r, "/m/claude-code:u1").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body.matches("role=\"alert\"").count(), 0, "{body}");
        assert_eq!(html_elements(&body, "<div class=\"archived-text\">", "</div>"),
            ["<div class=\"archived-text\"><p>look at &lt;script&gt;alert(1)&lt;/script&gt; &amp; the parser</p>\n</div>"]);
        let path =
            polylogue_render::ready::http_url(crate::commands::declarations(), &next.request)
                .unwrap();
        let (status, _, body) = send_html(&r, &path).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let api = follow_read(&r, next).await;
        assert_reader_items(&body, &api);
        assert_reader_navigation(&body, &api);
    });
}

#[test]
fn command_and_sign_in_body_types_are_checked_before_parsing_or_writing() {
    let r = reader();
    rt().block_on(async {
        for path in ["/auth", "/api/v1/set"] {
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
        let mut req = own_origin(authed(post("/api/v1/set")))
            .body(Body::from(setting_request("media-setting")))
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
            authed(get("/api/v1/dev/feed")).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(headers[header::CONTENT_TYPE], "text/event-stream");
        let lines = feed_frames(&body);
        assert_eq!(lines.first().unwrap()["type"], "begin");
        assert_eq!(lines.last().unwrap()["type"], "complete");
        let position = lines.last().unwrap()["position"].as_str().unwrap();
        let path = format!(
            "/api/v1/dev/feed?after={}",
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
                authed(get("/api/v1/dev/feed?after=broken"))
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
        let first = read_page(&r, "session:claude-code:s1", &[("budget", "2")]).await;
        let next = polylogue_render::ready::http_url(
            crate::commands::declarations(),
            &first.document.footer.next.unwrap().request,
        )
        .unwrap();
        let (_, _, body) = send(
            &r,
            authed(get("/api/v1/dev/feed")).body(Body::empty()).unwrap(),
        )
        .await;
        let frames = feed_frames(&body);
        let position = frames.last().unwrap()["position"].as_str().unwrap();
        let conn = rusqlite::Connection::open(r.w.daemon.archive.index_db()).unwrap();
        conn.execute(
            "UPDATE meta SET value=?1 WHERE key='index_instance'",
            [uuid::Uuid::now_v7().to_string()],
        )
        .unwrap();
        let (status, _, body) = send(&r, authed(get(&next)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("stale_cursor"));
        let url = format!(
            "/api/v1/dev/feed?after={}",
            polylogue_render::html::encode_path_segment(position)
        );
        let (status, _, body) = send(&r, authed(get(&url)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let frames = feed_frames(&body);
        assert_eq!(frames[0]["index_instance_changed"], true);
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
        let items = read_items(&r, "session:claude-code:s1", &[("view", "tree")]).await;
        let mut seen = std::collections::HashSet::new();
        for item in items {
            if let polylogue_model::read::ItemKind::Card { card } = item.kind
                && let polylogue_model::read::Card::LinkEdge {
                    from,
                    to,
                    relation,
                    reason,
                    ..
                } = card.as_ref()
            {
                assert_eq!(from, "session:claude-code:s1");
                assert_eq!(relation, "fork");
                assert_eq!(reason.as_deref(), None);
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
        assert!(body.contains("polylogue messages --cursor"));
    });
}

#[test]
fn exact_reference_routes_do_not_substitute_prefixes_or_current_note_revisions() {
    use polylogue_model::Reference;
    use polylogue_model::annotation::{
        AnnotationKind, AnnotationPatch, Change, LinkRole, NewAnnotation, NewAnnotationLink,
    };
    let r = reader();
    let note = NewAnnotation {
        kind: AnnotationKind::Note,
        key: None,
        value: Some(serde_json::json!("older exact note <text>")),
        links: vec![NewAnnotationLink {
            role: LinkRole::About,
            reference: Reference::Session(
                polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            ),
        }],
        valid_until: None,
    };
    let invocation = r
        .web
        .daemon
        .invocation(&polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        });
    let old = crate::test_answers::ok(crate::daemon::service::write(
        &r.web.daemon,
        &invocation,
        "exact-note".into(),
        polylogue_model::user::UserWrite::Assert {
            annotations: vec![note],
        },
    ))
    .id;
    crate::test_answers::ok(crate::daemon::service::write(
        &r.web.daemon,
        &invocation,
        "exact-note-new".into(),
        polylogue_model::user::UserWrite::ReviseAnnotations {
            bases: vec![polylogue_model::RevisionReference::exact(
                old.clone(),
                old.clone(),
            )],
            patch: AnnotationPatch {
                value: Change::Set(serde_json::json!("newer exact note")),
                ..Default::default()
            },
        },
    ));
    let exact = Reference::Annotation(polylogue_model::RevisionReference::exact(
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
        for path in ["/m/claude-code:u", "/a/annotation:exact@exact"] {
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
        assert_eq!(
            page.document
                .items
                .iter()
                .filter(|item| item.source.inherited)
                .count(),
            1,
            "only the parent's u1 is inherited"
        );
    });
}

#[test]
fn previous_links_reopen_unicode_fragments_without_browser_history() {
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
        let first = read_page(&r, "message:claude-code:u1", &[("budget", "20")]).await;
        let next = first
            .document
            .footer
            .next
            .as_ref()
            .expect("long body continues");
        let second = follow_read(&r, next).await;
        let next_path =
            polylogue_render::ready::http_url(crate::commands::declarations(), &next.request)
                .unwrap();
        let (status, _, current_html) = send_html(&r, &next_path).await;
        assert_eq!(status, StatusCode::OK, "{current_html}");
        assert_reader_navigation(&current_html, &second);

        let previous = second
            .document
            .footer
            .previous
            .as_ref()
            .expect("previous is supplied by service");
        let restored = follow_read(&r, previous).await;
        assert_eq!(restored.document.items, first.document.items);
        let path =
            polylogue_render::ready::http_url(crate::commands::declarations(), &previous.request)
                .unwrap();
        let (status, _, html) = send_html(&r, &path).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert_reader_items(&html, &restored);
        assert_reader_navigation(&html, &restored);
    });
}

#[test]
fn message_urls_read_literal_ids_without_resolving_source_rows() {
    let r = reader();
    let id = "claude-code:slash\\tab\tline\nquote\"AKIAIOSFODNN7EXAMPLE";
    rt().block_on(async {
        let path = format!("/m/{}", polylogue_render::html::encode_path_segment(id));
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
    r.w.wait_for(|s| {
        s["store"]["sessions_physical"]
            .as_u64()
            .is_some_and(|n| n >= 2)
            && s["busy"] == false
    });
    rt().block_on(async {
        let mut cursor = None;
        let mut ids = Vec::new();
        loop {
            let mut params = vec![
                ("selection".into(), "feed where kind:session".into()),
                ("page_size".into(), "1".into()),
            ];
            if let Some(cursor) = cursor {
                params.push(("cursor".into(), cursor));
            }
            let path = format!(
                "/api/v1/rows?{}",
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
                page["read_position"]["cursor"]
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
        ids.sort();
        assert_eq!(ids, ["claude-code:s1", "claude-code:s2"]);
    });
}

#[test]
fn finite_http_context_streams_share_paged_items_and_require_read_authority() {
    use polylogue_model::read::{ReadFooter, ReadPart, ReadRecord};
    let r = reader();
    let position =
        r.w.daemon
            .reads
            .reader()
            .unwrap()
            .change_position()
            .unwrap()
            .to_string();
    rt().block_on(async {
        // The selection names the context message; `--at` reads the
        // annotations as they stood.
        for (target, at) in [
            ("session:claude-code:s1", None),
            ("message:claude-code:u1", None),
            ("session:claude-code:s1", Some("2026-11-21T10:00:00Z")),
            ("session:claude-code:s1", Some(position.as_str())),
        ] {
            let mut options = vec![("view", "context"), ("format", "jsonl")];
            if let Some(at) = at {
                options.push(("at", at));
            }
            let mut paged = options.clone();
            paged.retain(|(key, _)| *key != "format");
            let expected = read_page(&r, target, &paged).await;
            options.push(("delivery", "stream"));
            let path = read_url(target, &options);
            assert_eq!(
                send(&r, get(&path).body(Body::empty()).unwrap()).await.0,
                StatusCode::UNAUTHORIZED
            );
            let (status, headers, body) =
                send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(headers[header::CONTENT_TYPE], "text/event-stream");
            let frames = command_surface::SseDecoder::new()
                .push(body.as_bytes())
                .unwrap();
            assert_eq!(frames.first().unwrap().event, "begin");
            assert_eq!(frames.last().unwrap().event, "answer");
            let completion = &frames[frames.len() - 3];
            assert_eq!(completion.event, "complete");
            let report = &frames[frames.len() - 2];
            assert_eq!(report.event, "steps");
            assert_eq!(
                frames.iter().filter(|frame| frame.event == "steps").count(),
                1
            );
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| frame.event == "answer")
                    .count(),
                1
            );
            assert_eq!(
                frames.iter().filter(|frame| frame.event == "begin").count(),
                1
            );
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| frame.event == "complete")
                    .count(),
                1
            );
            let mut items = Vec::new();
            let mut messages = 0;
            for frame in &frames[1..frames.len() - 3] {
                assert_eq!(frame.event, "record");
                let record: ReadRecord = serde_json::from_str(&frame.data).unwrap();
                if matches!(
                    record.record(),
                    Some(polylogue_model::bundle::Record::Message(_))
                ) {
                    messages += 1;
                }
                for part in record.parts() {
                    if let ReadPart::Item(item) = part {
                        items.push(item.clone());
                    }
                }
            }
            assert_eq!(messages, 1);
            assert_eq!(items, expected.document.items);
            let complete: ReadFooter = serde_json::from_str(&completion.data).unwrap();
            assert_eq!(complete.counts.messages, messages);
            assert_eq!(complete.shown, items.len() as u64);
            let answer: command_surface::Answer<crate::protocol::line::LineAnswer> =
                serde_json::from_str(&report.data).unwrap();
            let crate::protocol::line::LineAnswer::Steps(steps) = answer.value().unwrap() else {
                panic!("the line must report its streamed read: {answer:?}");
            };
            assert_eq!(steps.steps.len(), 1);
            assert_eq!(steps.steps[0].verb, "read");
            let crate::protocol::line::StepOutcome::Done {
                reply: crate::protocol::line::StepAnswer::Streamed(reported),
                undo,
            } = &steps.steps[0].outcome
            else {
                panic!("the read must finish successfully: {steps:?}");
            };
            assert_eq!(reported.as_ref(), &complete);
            assert!(undo.is_empty());
            let final_answer: command_surface::Answer<crate::protocol::line::LineAnswer> =
                serde_json::from_str(&frames.last().unwrap().data).unwrap();
            let crate::protocol::line::LineAnswer::Steps(final_steps) =
                final_answer.value().unwrap()
            else {
                panic!("the line must close successfully: {final_answer:?}");
            };
            assert!(
                final_steps.steps.is_empty(),
                "the completed read is reported once"
            );
            assert_eq!(final_steps.resolved, steps.resolved);
            let deadline =
                crate::daemon::execution::deadline(std::time::Duration::ZERO).to_string();
            let (status, _, body) = send(
                &r,
                authed(get(&path))
                    .header("x-polylogue-deadline-ms", deadline)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
            assert!(
                !body.contains("event: begin"),
                "expired requests must not begin a stream"
            );
        }
    });
}

#[test]
fn http_get_pages_and_streams_never_acknowledge_display() {
    let r = reader();
    rt().block_on(async {
        for target in ["session:claude-code:s1", "root:claude-code:s1"] {
            for delivery in ["page", "stream"] {
                let path = read_url(
                    target,
                    &[
                        ("delivery", delivery),
                        ("from", "seen"),
                        ("displayed", "true"),
                    ],
                );
                let (status, _, body) =
                    send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
                assert_eq!(status, StatusCode::OK, "{delivery}: {body}");
                if delivery == "page" {
                    let page: polylogue_model::read::ReadPage =
                        serde_json::from_str(&body).unwrap();
                    assert_eq!(page.document.header.target, target);
                    assert!(
                        !page.document.items.is_empty(),
                        "GET actually reads the stored session"
                    );
                } else {
                    let frames = command_surface::SseDecoder::new()
                        .push(body.as_bytes())
                        .unwrap();
                    let complete = frames
                        .iter()
                        .find(|frame| frame.event == "complete")
                        .unwrap();
                    let complete: polylogue_model::read::ReadFooter =
                        serde_json::from_str(&complete.data).unwrap();
                    assert_eq!(
                        complete.counts.messages, 1,
                        "the stream actually delivers the stored message"
                    );
                }
            }
        }
    });
    r.w.set("browser.actions", "on");
    let answer = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"annotations where kind:seen"}),
    )
    .answer(&r.w.daemon);
    let rows = crate::test_answers::empty(crate::test_answers::listing(answer));
    assert_eq!(
        rows["count"], 0,
        "GET cannot turn a caller flag into an authored mark"
    );
}

#[test]
fn http_read_budget_none_keeps_transfer_continuations_and_deadlines() {
    use polylogue_model::read::ItemKind;
    use std::io::Write;
    let r = reader();
    let text = "synthetic unlimited token transfer λ🙂 ".repeat(1500);
    std::fs::OpenOptions::new()
        .append(true)
        .open(r.w.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap()
        .write_all(polylogue_testkit::claude_user("u2", "s1", &text).as_bytes())
        .unwrap();
    r.w.wait_for(|s| s["store"]["messages"] == 2);
    rt().block_on(async {
        let first = read_page(&r, "session:claude-code:s1", &[("budget", "none")]).await;
        let next = first
            .document
            .footer
            .next
            .as_ref()
            .expect("none lifts tokens, not the byte transfer bound");
        assert!(
            next.request.arguments["steps"][0]
                .as_object()
                .unwrap()
                .contains_key("budget")
        );
        assert_eq!(
            next.request.arguments["steps"][0]["budget"],
            serde_json::Value::Null
        );
        let second = follow_read(&r, next).await;
        let previous = second
            .document
            .footer
            .previous
            .as_ref()
            .expect("a previous transfer");
        assert!(
            previous.request.arguments["steps"][0]
                .as_object()
                .unwrap()
                .contains_key("budget")
        );
        assert_eq!(
            previous.request.arguments["steps"][0]["budget"],
            serde_json::Value::Null
        );
        assert_eq!(
            follow_read(&r, previous).await.document.items,
            first.document.items
        );
        let all = read_items(&r, "session:claude-code:s1", &[("budget", "none")]).await;
        assert!(
            all.iter().any(
                |item| matches!(&item.kind, ItemKind::Text { text: actual } if actual == &text)
            )
        );
        let numeric = read_items(&r, "session:claude-code:s1", &[("budget", "1000000")]).await;
        assert_eq!(all, numeric);
        let zero = read_page(&r, "session:claude-code:s1", &[("budget", "0")]).await;
        assert!(!zero.document.items.is_empty());
        assert_eq!(
            zero.document
                .footer
                .next
                .as_ref()
                .unwrap()
                .request
                .arguments["steps"][0]["budget"],
            0
        );
        let default = read_page(&r, "session:claude-code:s1", &[]).await;
        assert_ne!(
            zero.document.items, default.document.items,
            "zero must not become the default allowance"
        );
        let path = read_url("session:claude-code:s1", &[("budget", "none")]);
        let deadline = crate::daemon::execution::deadline(std::time::Duration::ZERO).to_string();
        let (status, _, body) = send(
            &r,
            authed(get(&path))
                .header("x-polylogue-deadline-ms", deadline)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    });
}

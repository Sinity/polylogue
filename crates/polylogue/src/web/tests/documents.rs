use super::*;
use polylogue_model::Reference;
use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
use serde_json::json;

fn assert_document_import(response: command_surface::Answer<serde_json::Value>) {
    let Some(data) = &response.value().cloned() else {
        panic!("document import failed: {response:?}")
    };
    let report: polylogue_engine::ImportReport = serde_json::from_value(data.clone()).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(report.members.len(), 1, "{report:?}");
    assert_eq!(report.members[0].origin.as_deref(), Some("claude-ai"));
    assert_eq!(report.members[0].class, "parsed");
    assert_eq!(
        report
            .packages
            .iter()
            .map(|package| package.documents)
            .sum::<u64>(),
        1,
        "{report:?}"
    );
}

#[test]
fn native_documents_search_and_read_through_one_typed_http_operation() {
    use polylogue_model::read::{ReadDelivery, ReadOptions, ReadPage};
    let r = reader();
    let literal = "AKIAIOSFODNN7EXAMPLE";
    let source = r.w._dir.path().join("synthetic-project-documents");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    std::fs::write(source.join("projects/project-a.json"),json!({"uuid":"project-a","name":"Synthetic project","docs":[{"uuid":"doc-a","filename":format!("Guide <script> {literal}"),"content":format!("Document needle λ🙂 <script>{literal}</script>"),"created_at":"2026-05-02T00:00:00Z"}]}).to_string()).unwrap();
    let imported = handle(
        Request::Import {
            request_id: "web-doc-import".into(),
            paths: vec![source],
        },
        &r.web.daemon,
    );
    assert_document_import(imported);
    r.w.wait_for(|s| s["store"]["documents"].as_u64().is_some_and(|n| n > 0));
    let id = "claude-ai:project:project-a:doc:doc-a";
    let target = format!("document:{id}");
    let encoded = polylogue_render::html::encode_path_segment(id);
    rt().block_on(async {
        let (status, _, search) = send(
            &r,
            authed(get("/search?q=documents%20where%20needle"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{search}");
        assert!(search.contains(&format!("href=\"/d/{encoded}\"")));
        assert!(!search.contains(&format!("href=\"/s/{encoded}\"")));
        let (status, headers, page) = send_html(&r, &format!("/d/{encoded}")).await;
        assert_eq!(status, StatusCode::OK, "{page}");
        assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        assert!(
            page.contains(&format!(
                "Document needle λ🙂 &lt;script&gt;{literal}&lt;/script&gt;"
            )),
            "{page}"
        );
        assert!(!page.contains("<script>"));
        assert!(page.contains("polylogue read --request"));
        let api = read_page(&r, &target, &[]).await;
        let Ok(data) = crate::protocol::into_value(handle(
            Request::Read {
                target: target.clone(),
                options: ReadOptions::default(),
                delivery: ReadDelivery::Page,
                format: "json".into(),
                budget: None,
                with_blobs: false,
                cursor: None,
            },
            &r.web.daemon,
        )) else {
            panic!("document read failed")
        };
        let socket: ReadPage = serde_json::from_value(data).unwrap();
        assert_eq!(api.document.items, socket.document.items);
        assert_eq!(api.document.header, socket.document.header);
        for (path, expected) in [
            (
                format!("/d/{encoded}?view=dialogue"),
                StatusCode::BAD_REQUEST,
            ),
            ("/d/claude-ai%3Amissing".into(), StatusCode::NOT_FOUND),
        ] {
            let (status, _, body) = send_html(&r, &path).await;
            assert_eq!(status, expected, "{body}");
            if path.contains("dialogue") {
                assert!(body.contains("transcript") && body.contains("raw"));
            }
        }
        assert_eq!(
            send(
                &r,
                get(&format!("/d/{encoded}")).body(Body::empty()).unwrap()
            )
            .await
            .0,
            StatusCode::SEE_OTHER
        );
    });
}

#[test]
fn search_links_follow_reference_columns_in_native_and_grouped_rows() {
    let r = reader();
    let note = handle(
        Request::Assert {
            request_id: "claude-code:s1".into(),
            assertions: vec![NewAssertion {
                kind: AssertionKind::Tag,
                key: Some("claude-code:s1".into()),
                value: Some(json!({"value": "claude-code:s1"})),
                links: vec![NewAssertionLink {
                    role: LinkRole::About,
                    reference: Reference::Session(
                        polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                    ),
                }],
                valid_until: None,
            }],
        },
        &r.web.daemon,
    );
    crate::test_answers::ok(note);
    let kept = handle(
        Request::Keep {
            request_id: "web-kept-group".into(),
            text: "sessions | group by id".into(),
            name: None,
        },
        &r.web.daemon,
    );
    crate::test_answers::ok(kept);
    rt().block_on(async {
        let (status, _, messages) = send(
            &r,
            authed(get("/search?q=messages"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            messages.contains("href=\"/s/claude-code%3As1\""),
            "{messages}"
        );
        assert!(
            !messages.contains("href=\"/s/claude-code%3Au1\""),
            "{messages}"
        );
        for (query, linked) in [
            ("assertions", false),
            ("sessions | fields text(id) as id", false),
            ("sessions | group by id", true),
            ("sessions | agg min:id", true),
            ("result:web-kept-group", true),
        ] {
            let path = format!(
                "/search?q={}",
                polylogue_render::html::encode_path_segment(query)
            );
            let (status, _, page) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{page}");
            if linked {
                assert!(
                    page.contains("href=\"/s/claude-code%3As1\""),
                    "{query}: {page}"
                );
            } else {
                assert!(!page.contains("href=\"/s/"), "{query}: {page}");
            }
            assert!(
                !page.contains("href=\"/s/claude-code%3Au1\""),
                "{query}: {page}"
            );
        }
    });
}

#[test]
fn document_previous_pages_restore_exact_unicode_body_fragments() {
    let r = reader();
    let source = r.w._dir.path().join("paged-documents");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    std::fs::write(source.join("projects/paged-project.json"),json!({"uuid":"paged-project","name":"Synthetic project","docs":[{"uuid":"paged-doc","filename":"Synthetic document","content":"synthetic paragraph λ🙂 ".repeat(800)}]}).to_string()).unwrap();
    let imported = handle(
        Request::Import {
            request_id: "paged-document-import".into(),
            paths: vec![source],
        },
        &r.web.daemon,
    );
    assert_document_import(imported);
    r.w.wait_for(|data| data["store"]["documents"].as_u64().is_some_and(|n| n > 0));
    rt().block_on(async {
        let first = read_page(
            &r,
            "document:claude-ai:project:paged-project:doc:paged-doc",
            &[("page_tokens", "64")],
        )
        .await;
        let current = follow_read(
            &r,
            first
                .document
                .footer
                .next
                .as_ref()
                .expect("long document continues"),
        )
        .await;
        let previous = current
            .document
            .footer
            .previous
            .as_ref()
            .expect("previous issued by service");
        let restored = follow_read(&r, previous).await;
        assert_eq!(restored.document.items, first.document.items);
        let (status, _, html) = send_html(
            &r,
            &polylogue_render::web::http_url(&previous.http).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(!html.contains("history.back"));
    });
}

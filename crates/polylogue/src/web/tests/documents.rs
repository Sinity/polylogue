use super::*;
use polylogue_model::Reference;
use polylogue_model::annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink};
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
fn documents_search_and_read_in_one_http_operation() {
    let r = reader();
    let literal = "AKIAIOSFODNN7EXAMPLE";
    let source = r.w._dir.path().join("synthetic-project-documents");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    std::fs::write(source.join("projects/project-a.json"),json!({"uuid":"project-a","name":"Synthetic project","docs":[{"uuid":"doc-a","filename":format!("Guide <script> {literal}"),"content":format!("Document needle λ🙂 <script>{literal}</script>"),"created_at":"2026-05-02T00:00:00Z"}]}).to_string()).unwrap();
    let imported = crate::test_daemon::Call::new("import", serde_json::json!({"paths": [source]}))
        .id("web-doc-import")
        .answer(&r.web.daemon);
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
        assert_read_command(&page, &target);
        let api = read_page(&r, &target, &[]).await;
        let socket = crate::test_answers::read(
            crate::test_daemon::Call::new(
                "rows",
                serde_json::json!({"selection":target,"steps":[{"verb":"read","format":"json"}]}),
            )
            .answer(&r.web.daemon),
        );
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
fn links_follow_references_in_unit_and_grouped_rows() {
    let r = reader();
    let invocation = r
        .web
        .daemon
        .invocation(&polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        });
    crate::test_answers::ok(crate::daemon::service::write(
        &r.web.daemon,
        &invocation,
        "claude-code:s1".into(),
        polylogue_model::user::UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Tag,
                key: Some("claude-code:s1".into()),
                value: Some(json!({"value":"claude-code:s1"})),
                links: vec![NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::parse("session:claude-code:s1").unwrap(),
                }],
                valid_until: None,
            }],
        },
    ));
    let kept = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"sessions | group by id","steps":[{"verb":"save"}]}),
    )
    .id("web-kept-group")
    .answer(&r.web.daemon);
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
        assert_eq!(messages.matches("role=\"alert\"").count(), 0, "{messages}");
        assert_eq!(session_hrefs(&messages), ["/s/claude-code%3As1"]);
        assert_eq!(
            html_elements(&messages, "<td>", "</td>")[0],
            "<td><a href=\"/m/claude-code%3Au1\">claude-code:u1</a></td>"
        );
        for (query, linked) in [
            ("annotations | fields key", false),
            ("sessions | fields text(id) as id", false),
            ("sessions | group by id", true),
            ("sessions | agg min(id)", true),
            ("result:web-kept-group-1", true),
        ] {
            let path = format!(
                "/search?q={}",
                polylogue_render::html::encode_path_segment(query)
            );
            let (status, _, page) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{page}");
            assert_eq!(page.matches("role=\"alert\"").count(), 0, "{query}: {page}");
            let cells = html_elements(&page, "<td>", "</td>");
            if linked {
                assert_eq!(session_hrefs(&page), ["/s/claude-code%3As1"], "{query}");
                assert_eq!(
                    cells[0], "<td><a href=\"/s/claude-code%3As1\">claude-code:s1</a></td>",
                    "{query}"
                );
            } else {
                assert_eq!(session_hrefs(&page), Vec::<&str>::new(), "{query}");
                assert_eq!(cells, ["<td>claude-code:s1</td>"], "{query}");
            }
        }
    });
}

#[test]
fn document_previous_pages_restore_unicode_body_fragments() {
    let r = reader();
    let source = r.w._dir.path().join("paged-documents");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    std::fs::write(source.join("projects/paged-project.json"),json!({"uuid":"paged-project","name":"Synthetic project","docs":[{"uuid":"paged-doc","filename":"Synthetic document","content":"synthetic paragraph λ🙂 ".repeat(800)}]}).to_string()).unwrap();
    let imported = crate::test_daemon::Call::new("import", serde_json::json!({"paths": [source]}))
        .id("paged-document-import")
        .answer(&r.web.daemon);
    assert_document_import(imported);
    r.w.wait_for(|data| data["store"]["documents"].as_u64().is_some_and(|n| n > 0));
    rt().block_on(async {
        let first = read_page(
            &r,
            "document:claude-ai:project:paged-project:doc:paged-doc",
            &[("budget", "64")],
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
        let next_path = polylogue_render::ready::http_url(
            crate::commands::declarations(),
            &first.document.footer.next.as_ref().unwrap().request,
        )
        .unwrap();
        let (status, _, current_html) = send_html(&r, &next_path).await;
        assert_eq!(status, StatusCode::OK, "{current_html}");
        assert_reader_navigation(&current_html, &current);
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
            &polylogue_render::ready::http_url(crate::commands::declarations(), &previous.request)
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert_reader_items(&html, &restored);
        assert_reader_navigation(&html, &restored);
    });
}

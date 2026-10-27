use super::*;
use polylogue_model::{Author, NewNote, NoteKind, Target};
use serde_json::json;

#[test]
fn native_documents_search_and_read_through_http_with_masked_sources() {
    let r = reader();
    let secret = format!("sk-ant-api03-{}-AAAAAAAA", "A".repeat(94));
    let source = r.w._dir.path().join(format!("source-{secret}"));
    std::fs::create_dir_all(source.join("projects")).unwrap();
    std::fs::write(
        source.join("projects/project-a.json"),
        json!({
            "uuid":"project-a","name":"Synthetic project","docs":[{
                "uuid":"doc-a","filename":format!("Guide <script> {secret}"),
                "content":format!("Document needle λ🙂 <script>{secret}</script>"),
                "created_at":"2026-05-02T00:00:00Z"
            }]
        })
        .to_string(),
    )
    .unwrap();
    let imported = handle(
        Request::Import {
            request_id: "web-doc-import".into(),
            paths: vec![source],
        },
        &r.web.daemon,
    );
    assert!(matches!(imported, Response::Ok { .. }), "{imported:?}");
    r.w.wait_for(|s| s["store"]["documents"].as_u64().is_some_and(|n| n > 0));
    let kept = handle(
        Request::Keep {
            request_id: "web-kept-documents".into(),
            text: "documents where needle".into(),
            name: None,
        },
        &r.web.daemon,
    );
    assert!(matches!(kept, Response::Ok { .. }), "{kept:?}");
    let id = "claude-ai:project:project-a:doc:doc-a";
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
        assert!(
            search.contains(&format!("href=\"/d/{encoded}\"")),
            "{search}"
        );
        assert!(!search.contains(&format!("href=\"/s/{encoded}\"")));
        let (_, _, kept) = send(
            &r,
            authed(get("/search?q=result%3Aweb-kept-documents"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(kept.contains(&format!("href=\"/d/{encoded}\"")), "{kept}");
        let (status, headers, page) = send(
            &r,
            authed(get(&format!("/d/{encoded}")))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{page}");
        assert_eq!(headers[header::CACHE_CONTROL], "no-store");
        assert!(
            page.contains("Document needle λ🙂 &lt;script&gt;"),
            "{page}"
        );
        assert!(page.contains("authorship unknown"), "{page}");
        assert!(page.contains("Source records"));
        assert!(page.contains("shown"));
        assert!(page.contains("2026-05-02T00:00:00Z"));
        assert!(
            !page.contains(&secret),
            "document title, text or source leaked a credential"
        );
        assert!(!page.contains("<script>"));
        assert!(
            page.contains(
                "polylogue read document:claude-ai:project:project-a:doc:doc-a --view full"
            )
        );
        let (status, _, api) = send(
            &r,
            authed(get(&format!(
                "/api/v1/read?session=document%3A{encoded}&format=json"
            )))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{api}");
        let api: serde_json::Value = serde_json::from_str(&api).unwrap();
        let source_path = api["sources"][0]["locations"][0]["path"].as_str().unwrap();
        assert!(!source_path.contains(&secret));
        assert!(
            source_path.contains("masked"),
            "the supplied source path must be masked: {source_path}"
        );
        let Response::Ok { data } = handle(
            Request::Read {
                slices: true,
                start: None,
                target: crate::daemon::ReadTarget::Document(
                    polylogue_model::DocumentId::parse(id).unwrap(),
                ),
                with_parents: false,
                view: "full".into(),
                format: "json".into(),
                budget: None,
                cursor: None,
            },
            &r.web.daemon,
        ) else {
            panic!("document read failed")
        };
        assert_eq!(api, scrubbed(&r.web, data));
        for (path, expected) in [
            (
                format!("/d/{encoded}?view=actions"),
                StatusCode::BAD_REQUEST,
            ),
            ("/d/claude-ai%3Amissing".into(), StatusCode::NOT_FOUND),
        ] {
            assert_eq!(
                send(&r, authed(get(&path)).body(Body::empty()).unwrap())
                    .await
                    .0,
                expected
            );
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
fn message_note_group_and_string_columns_do_not_become_session_links() {
    let r = reader();
    let note = handle(
        Request::AddNote {
            request_id: "claude-code:s1".into(),
            note: NewNote {
                kind: NoteKind::Note,
                key: Some("claude-code:s1".into()),
                value: Some("claude-code:s1".into()),
                author: Author::Operator,
                targets: vec![Target::Session(
                    polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                )],
                relations: vec![],
                revises: None,
                valid_until: None,
            },
        },
        &r.web.daemon,
    );
    assert!(matches!(note, Response::Ok { .. }));
    let kept = handle(
        Request::Keep {
            request_id: "web-kept-group".into(),
            text: "sessions | group by id".into(),
            name: None,
        },
        &r.web.daemon,
    );
    assert!(matches!(kept, Response::Ok { .. }), "{kept:?}");
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
        for query in [
            "notes",
            "sessions | group by id",
            "sessions | agg min:id",
            "result:web-kept-group",
        ] {
            let path = format!(
                "/search?q={}",
                polylogue_render::html::encode_path_segment(query)
            );
            let (status, _, page) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{page}");
            assert!(!page.contains("href=\"/s/"), "{query}: {page}");
        }
    });
}

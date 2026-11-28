use super::*;
use tower::ServiceExt;

/// An object URL already names the selection and its read step; a query
/// value that would name them again is refused.
#[test]
fn a_path_target_cannot_be_overridden_by_query_parameters() {
    let daemon = crate::test_daemon::start("reader route refusal");
    let web = Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    });
    let response = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            super::super::router(web)
                .oneshot(
                    axum::http::Request::get("/s/test:session?target=another%20selection")
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
        });
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn html_reader_uses_the_request_deadline_and_identity() {
    let daemon = crate::test_daemon::start("HTML uses the same deadline");
    let web = Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    });
    let router = super::super::router(web);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let tail = "/rows?selection=sessions&deadline=0&request_id=web-deadline";
        for path in [tail.to_owned(), format!("/api/v1{tail}")] {
            let (status, body) = get(&router, &path).await;
            assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{path}: {body}");
            assert!(body.contains("deadline_exceeded"), "{path}: {body}");
            assert!(body.contains("web-deadline"), "{path}: {body}");
        }
        let fresh = "/rows?selection=sessions&fresh=true";
        let (status, _) = get(&router, &format!("/api/v1{fresh}")).await;
        assert_eq!(get(&router, fresh).await.0, status);
    });
}

#[test]
fn reader_view_link_starts_a_new_typed_read_from_a_continuation() {
    let daemon = crate::test_daemon::start(&"Synthetic cursor text λ. ".repeat(300));
    let web = Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    });
    let router = super::super::router(web);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let initial = "/s/claude-code:s1?deadline=null&budget_tokens=4000&budget=128&show=tools:line&show=inherited&request_id=original";
        let (status, html) = get(&router, initial).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        let next = href(&html, "rel=\"next\">Next");
        let (status, html) = get(&router, &next).await;
        assert_eq!(status, StatusCode::OK, "{html}");

        let link = href(&html, ">dialogue</a>");
        let (status, json) = get(&router, &format!("/api/v1{link}")).await;
        assert_eq!(status, StatusCode::OK, "{link}: {json}");
        let answer: command_surface::Answer<LineAnswer> = serde_json::from_str(&json).unwrap();
        let Some(LineAnswer::Steps(steps)) = answer.value() else {
            panic!("expected step replies: {answer:?}");
        };
        let [report] = steps.steps.as_slice() else {
            panic!("expected one step: {steps:?}");
        };
        let StepOutcome::Done { reply: StepAnswer::Read(page), .. } = &report.outcome else {
            panic!("expected a read reply: {:?}", report.outcome);
        };
        assert_eq!(page.document.header.target, "session:claude-code:s1");
        assert_eq!(page.document.header.view, View::Dialogue);
        assert_eq!(page.document.footer.position, Some(0));
        let request = request_of_link(&link);
        let previous = request_of_link(&next);
        assert_eq!(request.deadline, command_surface::Deadline::None);
        assert_eq!(request.budget_tokens, command_surface::TokenBudget::Tokens(4000));
        assert!(request.cursor.is_none());
        assert_ne!(request.request_id, previous.request_id);
        let rows = Rows::deserialize(&request.arguments).unwrap();
        let [Step::Read(read)] = rows.steps.as_slice() else {
            panic!("expected one read step: {rows:?}");
        };
        assert_eq!(read.options.show, ["inherited"]);
        assert_eq!(read.budget, polylogue_model::read::ReadBudget::Tokens(128));
        assert!(read.cursor.is_none());
        let (status, html) = get(&router, &link).await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("aria-current=\"page\">dialogue"), "{html}");
    });
}

async fn get(router: &Router, path: &str) -> (StatusCode, String) {
    let reply = router
        .clone()
        .oneshot(
            axum::http::Request::get(path)
                .header(header::HOST, "127.0.0.1:8765")
                .header(header::AUTHORIZATION, "Bearer synthetic")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = reply.status();
    let bytes = axum::body::to_bytes(reply.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn request_of_link(url: &str) -> RunnableRequest {
    let Query(pairs) = Query::<Vec<(String, String)>>::try_from_uri(&url.parse().unwrap()).unwrap();
    super::super::api::get_request(row(), &pairs).unwrap()
}

fn href(html: &str, label: &str) -> String {
    let (before, _) = html.split_once(label).expect("link label");
    let (_, start) = before.rsplit_once("href=\"").expect("link URL");
    let (link, _) = start.split_once('"').expect("quoted URL");
    link.replace("&amp;", "&")
}

#[test]
fn html_and_json_report_the_failed_step_and_keep_a_completed_read() {
    let seed = "Content retained from the completed read.";
    let daemon = crate::test_daemon::start(seed);
    let router = super::super::router(Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    }));
    let arguments = serde_json::json!({
        "selection": "session:claude-code:s1",
        "steps": [{"verb": "read"}, {"verb": "read", "show": ["unknown-kind"]}]
    });
    let tail = format!(
        "/rows?arguments={}",
        polylogue_render::html::encode_path_segment(&arguments.to_string())
    );
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for path in [tail.clone(), format!("/api/v1{tail}")] {
            let (status, body) = get(&router, &path).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
            assert!(body.contains("unknown-kind"), "{path}: {body}");
            assert!(body.contains(seed), "{path}: {body}");
        }
    });
}

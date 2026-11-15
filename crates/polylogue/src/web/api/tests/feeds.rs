use super::*;
use serde_json::{Value, json};
use tower::ServiceExt;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

async fn get(
    daemon: &crate::test_daemon::TestDaemon,
    uri: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, Vec<u8>) {
    let mut builder = axum::http::Request::get(uri)
        .header(header::HOST, "127.0.0.1:8765")
        .header(header::AUTHORIZATION, "Bearer synthetic");
    for (key, value) in headers {
        builder = builder.header(*key, *value);
    }
    let response = super::super::super::router(web(daemon))
        .oneshot(builder.body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    (
        status,
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
}

fn lines(bytes: &[u8]) -> Vec<Value> {
    let frames = command_surface::SseDecoder::new().push(bytes).unwrap();
    assert_eq!(frames.first().unwrap().event, "begin");
    assert_eq!(frames.last().unwrap().event, "complete");
    for event in &frames[1..frames.len() - 1] {
        assert_eq!(event.event, "record");
    }
    frames
        .iter()
        .map(|event| serde_json::from_str(&event.data).unwrap())
        .collect()
}

#[test]
fn http_get_feeds_execute_generated_canonical_calls_and_continue_the_same_position() {
    let daemon = crate::test_daemon::start("HTTP GET feed binding");
    let declarations = crate::commands::declarations();
    let mut completed = None;
    runtime().block_on(async {
        for deadline in [
            command_surface::Deadline::Default,
            command_surface::Deadline::None,
            command_surface::Deadline::Milliseconds(5000),
        ] {
            let request = command_surface::Request {
                operation: "changes".into(),
                arguments: json!({"since":null}),
                request_id: "HTTP feed exact identity".into(),
                deadline,
                budget_tokens: Default::default(),
                cursor: None,
                fresh: false,
                format: Default::default(),
            };
            let ready = daemon.daemon.ready_call(&request).unwrap();
            let url = polylogue_render::web::http_url(declarations, &ready.request).unwrap();
            let Query(pairs) =
                Query::<Vec<(String, String)>>::try_from_uri(&url.parse().unwrap()).unwrap();
            assert_eq!(
                declarations
                    .parse_http_query("changes", &pairs, "unused")
                    .unwrap(),
                request
            );
            let (status, bytes) = get(&daemon, &url, &[]).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
            let received = lines(&bytes);
            assert_eq!(received[0]["position"], Value::Null);
            assert_eq!(received[0]["replacement"], true);
            assert_eq!(
                received.last().unwrap()["count"].as_u64().unwrap(),
                (received.len() - 2) as u64
            );
            assert!(
                received
                    .iter()
                    .any(|line| line["type"] == "change" && line["change"]["kind"] == "session")
            );
            completed = Some(
                received.last().unwrap()["position"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            );
        }
        let position = completed.as_ref().unwrap();
        let request = crate::client::request(
            "changes",
            json!({"since":position}),
            "HTTP feed continuation".into(),
        );
        let url = polylogue_render::web::http_url(declarations, &request).unwrap();
        let (status, bytes) = get(&daemon, &url, &[]).await;
        assert_eq!(status, StatusCode::OK);
        let received = lines(&bytes);
        assert_eq!(received.len(), 2);
        assert_eq!(received[0]["position"], *position);
        assert_eq!(received[0]["replacement"], false);
        assert_eq!(received[1]["position"], *position);
        assert_eq!(received[1]["count"], 0);
    });
    let request = crate::client::request(
        "changes",
        json!({"since":completed.unwrap()}),
        "socket feed continuation".into(),
    );
    let mut received = Vec::new();
    crate::client::stream::run_call(
        &daemon.daemon.archive,
        &request,
        Some(std::time::Duration::from_secs(5)),
        |frame| {
            let done = frame.event == "complete";
            received.push(serde_json::from_str::<Value>(&frame.data).unwrap());
            Ok(done)
        },
    )
    .unwrap();
    assert_eq!(received.len(), 2);
    assert_eq!(received[1]["count"], 0);
}

#[test]
fn http_get_feeds_refuse_duplicates_wrong_routes_and_unsupported_options() {
    let daemon = crate::test_daemon::start("HTTP GET feed refusals");
    runtime().block_on(async {
        for (query, reason) in [
            ("since=one&since=two", "duplicate since"),
            ("request_id=one&request_id=two", "duplicate request_id"),
            (
                "operation=status&arguments=%7B%7D",
                "route does not match operation",
            ),
            ("request_id=", "request_id"),
            ("arguments=%5B%5D", "object"),
            ("arguments=%7B%22since%22%3A1%7D", "string"),
            ("cursor=ordinary-page", "cursor"),
            ("budget_tokens=null", "budget_tokens"),
            ("budget_tokens=10", "budget_tokens"),
            ("fresh=true", "fresh"),
            ("format=json", "format"),
            ("why=true", "unknown query field why"),
            ("fields=ref", "unknown query field fields"),
        ] {
            let (status, bytes) = get(&daemon, &format!("/api/v1/changes?{query}"), &[]).await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{query}: {}",
                String::from_utf8_lossy(&bytes)
            );
            let answer: command_surface::Answer<Value> = serde_json::from_slice(&bytes).unwrap();
            let command_surface::Outcome::Error(error) = answer.outcome else {
                panic!("{query} did not refuse")
            };
            assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
            assert!(error.message.contains(reason), "{query}: {error:?}");
        }
        let (status, _) = get(&daemon, "/api/v1/follow?target=sessions", &[]).await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "follow gains no HTTP exposure"
        );
    });
}

#[test]
fn http_get_feeds_keep_explicit_outer_deadlines_and_authentication() {
    let daemon = crate::test_daemon::start("HTTP GET feed lifetime");
    runtime().block_on(async {
        let (status, bytes) = get(&daemon, "/api/v1/changes?deadline=null", &[("x-polylogue-deadline-ms", "1")]).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{}", String::from_utf8_lossy(&bytes));
        let answer: command_surface::Answer<Value> = serde_json::from_slice(&bytes).unwrap();
        assert!(matches!(answer.outcome, command_surface::Outcome::Error(error) if error.kind==command_surface::FailureKind::Timeout));
        let response = super::super::super::router(web(&daemon)).oneshot(
            axum::http::Request::get("/api/v1/changes?deadline=null")
                .header(header::HOST, "127.0.0.1:8765")
                .body(axum::body::Body::empty()).unwrap()
        ).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    });
}

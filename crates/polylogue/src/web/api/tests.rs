use super::*;
use crate::daemon::{
    Request, Response,
    service::{Daemon, error},
};
use std::io::Write;

fn web(daemon: &crate::test_daemon::TestDaemon) -> Arc<Web> {
    Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        operations: Arc::clone(daemon.operations()),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    })
}

#[test]
fn stream_failure_before_first_event_uses_the_shared_http_status() {
    fn failed(
        _: &Request,
        _: &Daemon,
        _: &polylogue_model::authorship::Authorship,
        _: &mut dyn Write,
    ) -> Option<Result<(), Response>> {
        Some(Err(error("bad_request", "invalid selection")))
    }
    let daemon = crate::test_daemon::start("synthetic stream refusal");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = stream(
                &web(&daemon),
                Request::Changes { since: None },
                failed,
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let answer: command_surface::Answer<serde_json::Value> =
                serde_json::from_slice(&body).unwrap();
            assert_eq!(answer.exit_code(), 2);
        });
}

#[test]
fn stream_failure_after_begin_delivers_error_without_final_completion() {
    fn failed(
        _: &Request,
        _: &Daemon,
        authorship: &polylogue_model::authorship::Authorship,
        output: &mut dyn Write,
    ) -> Option<Result<(), Response>> {
        assert_eq!(authorship.author, polylogue_model::Author::Operator);
        assert_eq!(authorship.via, polylogue_model::authorship::Channel::Web);
        crate::protocol::stream::event(
            output,
            "begin",
            &serde_json::json!({"snapshot":"synthetic"}),
        )
        .unwrap();
        Some(Err(error("conflict", "source changed")))
    }
    let daemon = crate::test_daemon::start("synthetic stream late failure");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = stream(
                &web(&daemon),
                Request::Changes { since: None },
                failed,
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let events = command_surface::SseDecoder::new().push(&body).unwrap();
            assert_eq!(
                events
                    .iter()
                    .map(|event| event.event.as_str())
                    .collect::<Vec<_>>(),
                ["begin", "error"]
            );
            let answer: command_surface::Answer<serde_json::Value> =
                serde_json::from_str(&events[1].data).unwrap();
            assert_eq!(answer.http_status(), Some(409));
        });
}

#[test]
fn stream_headers_arrive_while_the_bounded_producer_is_waiting_for_the_reader() {
    fn large(
        _: &Request,
        _: &Daemon,
        _: &polylogue_model::authorship::Authorship,
        output: &mut dyn Write,
    ) -> Option<Result<(), Response>> {
        let result = (|| {
            crate::protocol::stream::event(output, "begin", &serde_json::json!({}))?;
            // Exceeds the transport buffer. Preparation cannot finish until the body is consumed.
            crate::protocol::stream::event(output, "record", &"x".repeat(128 * 1024))?;
            crate::protocol::stream::event(output, "complete", &serde_json::json!({}))
        })();
        Some(result.map_err(|error| crate::daemon::service::error("io", error)))
    }
    let daemon = crate::test_daemon::start("synthetic stream backpressure");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                stream(&web(&daemon), Request::Changes { since: None }, large, None),
            )
            .await
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            drop(response);
        });
}

#[test]
fn generated_http_ready_call_executes_without_translating_its_arguments() {
    use tower::ServiceExt;
    let daemon = crate::test_daemon::start("HTTP ready witness");
    let call = command_surface::Request {
        operation: "query".into(),
        arguments: serde_json::json!({"query":"sessions"}),
        request_id: "http-ready-id".into(),
        deadline: command_surface::Deadline::None,
    };
    let ready = daemon.daemon.ready_call(&call).unwrap();
    let encode = |text: &str| {
        text.as_bytes()
            .iter()
            .map(|byte| format!("%{byte:02X}"))
            .collect::<String>()
    };
    let pairs = ready
        .http
        .arguments
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| {
            let text = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string());
            format!("{}={}", encode(key), encode(&text))
        })
        .collect::<Vec<_>>()
        .join("&");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = super::super::router(web(&daemon))
                .oneshot(
                    axum::http::Request::builder()
                        .method(ready.http.method.as_str())
                        .uri(format!("{}?{pairs}", ready.http.path))
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let answer: command_surface::Answer<serde_json::Value> =
                serde_json::from_slice(&bytes).unwrap();
            assert_eq!(answer.value().unwrap()["count"], 1);
            let refusal = super::super::router(web(&daemon))
                .oneshot(
                    axum::http::Request::get(format!("/api/v1/status?{pairs}"))
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
        });
}

#[test]
fn read_html_negotiates_on_the_same_typed_operation_route() {
    use tower::ServiceExt;
    let daemon = crate::test_daemon::start("HTML same operation witness <script>");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = super::super::router(web(&daemon))
                .oneshot(
                    axum::http::Request::get(
                        "/api/v1/read?target=session%3Aclaude-code%3As1&format=html",
                    )
                    .header(header::HOST, "127.0.0.1:8765")
                    .header(header::AUTHORIZATION, "Bearer synthetic")
                    .header(header::ACCEPT, "text/html")
                    .body(axum::body::Body::empty())
                    .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                response.headers()[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html")
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(text.contains("HTML same operation witness"), "{text}");
            assert!(text.contains("&lt;script&gt;"), "{text}");
        });
}

#[test]
fn native_scope_reset_terminates_a_begun_http_stream_without_final_completion() {
    fn reset_after_begin(
        _: &Request,
        daemon: &Daemon,
        _: &polylogue_model::authorship::Authorship,
        output: &mut dyn Write,
    ) -> Option<Result<(), Response>> {
        let reader = match daemon.reads.reader() {
            Ok(reader) => reader,
            Err(error) => return Some(Err(crate::daemon::service::store_error(&error))),
        };
        let result = (|| {
            crate::protocol::stream::event(
                output,
                "begin",
                &serde_json::json!({"generation":reader.change_position()?.to_string()}),
            )?;
            daemon.archive.advance_generation_for_test()?;
            crate::protocol::stream::event(
                output,
                "record",
                &serde_json::json!({"stale":"must not escape"}),
            )?;
            crate::protocol::stream::event(output, "complete", &serde_json::json!({}))?;
            Ok::<(), Box<dyn std::error::Error>>(())
        })();
        Some(result.map_err(|error| crate::daemon::service::error("io", error)))
    }
    let daemon = crate::test_daemon::start("native generation stream witness");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = stream(
                &web(&daemon),
                Request::Changes { since: None },
                reset_after_begin,
                None,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let events = command_surface::SseDecoder::new().push(&bytes).unwrap();
            assert_eq!(
                events
                    .iter()
                    .map(|event| event.event.as_str())
                    .collect::<Vec<_>>(),
                ["begin", "error"]
            );
            let answer: command_surface::Answer<serde_json::Value> =
                serde_json::from_str(&events[1].data).unwrap();
            let command_surface::Answer::Error { error, .. } = answer else {
                panic!("stale stream was successful")
            };
            assert_eq!(error.code.as_deref(), Some("scope_changed"));
            assert_eq!(error.kind, command_surface::FailureKind::Conflict);
        });
}

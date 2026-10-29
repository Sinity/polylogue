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
    fn failed(_: &Request, _: &Daemon, _: &mut dyn Write) -> Option<Result<(), Response>> {
        Some(Err(error("bad_request", "invalid selection")))
    }
    let daemon = crate::test_daemon::start("synthetic stream refusal");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = stream(&web(&daemon), Request::Changes { since: None }, failed).await;
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
    fn failed(_: &Request, _: &Daemon, output: &mut dyn Write) -> Option<Result<(), Response>> {
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
            let response = stream(&web(&daemon), Request::Changes { since: None }, failed).await;
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
    fn large(_: &Request, _: &Daemon, output: &mut dyn Write) -> Option<Result<(), Response>> {
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
                stream(&web(&daemon), Request::Changes { since: None }, large),
            )
            .await
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            drop(response);
        });
}

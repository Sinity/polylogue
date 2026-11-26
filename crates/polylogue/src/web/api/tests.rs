mod feeds;
mod rows;
use super::*;
use axum::extract::Query;

fn web(daemon: &crate::test_daemon::TestDaemon) -> Arc<Web> {
    Arc::new(Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    })
}

/// The caller of one HTTP request whose connection is `stop` and whose
/// outer bound is `outer_deadline`.
fn caller(
    stop: polylogue_store::cancellation::Cancellation,
    outer_deadline: Option<std::time::Instant>,
) -> crate::operation::Caller {
    super::super::Lifetime {
        stop,
        outer_deadline,
    }
    .caller(&Default::default())
}

fn get(web: &Arc<Web>, uri: &str) -> HttpResponse {
    use tower::ServiceExt;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            super::super::router(Arc::clone(web))
                .oneshot(
                    axum::http::Request::get(uri)
                        .header(header::HOST, "127.0.0.1:8765")
                        .header(header::AUTHORIZATION, "Bearer synthetic")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
        })
}

fn body(response: HttpResponse) -> Vec<u8> {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
        .unwrap()
        .to_vec()
}

/// A reply refused before any event is one JSON body with the failure's
/// HTTP status; a reply whose first event was written is an event stream
/// ending with its final answer (docs/interfaces.md, operation table).
#[test]
fn a_reply_is_json_until_its_first_event_and_a_stream_after_it() {
    let daemon = crate::test_daemon::start("synthetic stream refusal");
    let web = web(&daemon);
    let refused = get(&web, "/api/v1/dev/feed?after=broken");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    assert!(
        refused.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json")
    );
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&body(refused)).unwrap();
    assert_eq!(answer.exit_code(), 2);
    let streamed = get(&web, "/api/v1/dev/feed");
    assert_eq!(streamed.status(), StatusCode::OK);
    assert_eq!(
        streamed.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let frames = command_surface::SseDecoder::new()
        .push(&body(streamed))
        .unwrap();
    let names: Vec<&str> = frames.iter().map(|frame| frame.event.as_str()).collect();
    assert_eq!(names.first(), Some(&"begin"));
    assert_eq!(&names[names.len() - 2..], ["complete", "answer"]);
    let answer: command_surface::Answer<String> =
        serde_json::from_str(&frames.last().unwrap().data).unwrap();
    assert_eq!(answer.exit_code(), 0);
}

/// A GET runs only what reads and changes nothing; anything else is
/// refused with its POST route.
#[test]
fn a_get_runs_only_a_request_that_reads() {
    let daemon = crate::test_daemon::start("synthetic GET rule");
    let web = web(&daemon);
    for uri in [
        "/api/v1/set?key=browser.actions&value=on",
        "/api/v1/rows?selection=sessions&steps=tag&name=x",
    ] {
        let refused = get(&web, uri);
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST, "{uri}");
        let answer: command_surface::Answer<serde_json::Value> =
            serde_json::from_slice(&body(refused)).unwrap();
        let message = &answer.failure().unwrap().message;
        assert!(message.contains("use POST /api/v1/"), "{uri}: {message}");
    }
    assert_eq!(
        get(&web, "/api/v1/set?key=browser.actions").status(),
        StatusCode::OK
    );
}

#[test]
fn generated_http_ready_call_executes_without_translating_its_arguments() {
    use tower::ServiceExt;
    let daemon = crate::test_daemon::start("HTTP ready example");
    let call = command_surface::Request {
        operation: "rows".into(),
        arguments: serde_json::json!({"selection":"sessions"}),
        request_id: "http-ready-id".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let ready = daemon.daemon.ready_call(&call).unwrap();
    let url =
        polylogue_render::ready::http_url(crate::commands::declarations(), &ready.request).unwrap();
    let pairs = url.split_once('?').unwrap().1.to_owned();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = super::super::router(web(&daemon))
                .oneshot(
                    axum::http::Request::builder()
                        .method("GET")
                        .uri(url)
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
            assert_eq!(answer.value().unwrap()["reply"]["count"], 1);
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

/// The API answers a read as JSON whatever the client accepts; the reader
/// route renders the same typed reply as HTML.
#[test]
fn a_read_is_json_in_the_api_and_html_in_the_reader_route() {
    let daemon = crate::test_daemon::start("HTML same operation example <script>");
    let web = web(&daemon);
    let api = get(
        &web,
        "/api/v1/rows?steps=read&selection=session%3Aclaude-code%3As1&format=html",
    );
    assert_eq!(api.status(), StatusCode::OK);
    assert!(
        api.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json")
    );
    let page = get(
        &web,
        "/rows?steps=read&selection=session%3Aclaude-code%3As1&format=html",
    );
    assert_eq!(page.status(), StatusCode::OK);
    assert!(
        page.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    let text = String::from_utf8(body(page)).unwrap();
    assert!(text.contains("HTML same operation example"), "{text}");
    assert!(text.contains("&lt;script&gt;"), "{text}");
}

#[test]
fn a_captured_parent_deadline_is_not_renewed_during_dispatch_wait() {
    let daemon = crate::test_daemon::start("parent queue bound");
    let web = web(&daemon);
    let parent = std::time::Instant::now() + std::time::Duration::from_millis(10);
    let request = command_surface::Request {
        operation: "status".into(),
        arguments: serde_json::json!({}),
        request_id: "queued-parent-bound".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let response = runtime.block_on(async {
        tokio::time::sleep_until(tokio::time::Instant::from_std(parent)).await;
        respond(&web, request, caller(Default::default(), Some(parent))).await
    });
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let bytes = runtime
        .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
        .unwrap();
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&bytes).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("original expired parent must reach the blocking handler");
    };
    assert_eq!(error.kind, command_surface::FailureKind::Timeout);
    assert_eq!(error.code.as_deref(), Some("deadline_exceeded"));
}

#[test]
fn no_outer_header_does_not_supply_a_default_parent_for_a_null_request() {
    assert!(caller(Default::default(), None).outer_deadline.is_none());
}

#[test]
fn a_cancelled_http_parent_without_a_header_cannot_submit_a_pure_write() {
    let daemon = crate::test_daemon::start("cancelled HTTP write acceptance");
    let web = web(&daemon);
    let cancellation = polylogue_store::cancellation::Cancellation::default();
    cancellation.cancel();
    let id = "cancelled-parent-no-new-write";
    let request = command_surface::Request {
        operation: "set".into(),
        arguments: serde_json::json!({"key":"reading.alternates","value":"[]"}),
        request_id: id.into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let response = runtime.block_on(respond(&web, request, caller(cancellation, None)));
    let bytes = runtime
        .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
        .unwrap();
    let answer: command_surface::Answer<crate::protocol::setting::Answer> =
        serde_json::from_slice(&bytes).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("cancelled acceptance cannot yield a write result")
    };
    assert_eq!(error.kind, command_surface::FailureKind::Cancelled);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .user_request(id)
            .unwrap()
            .is_none()
    );
}

#[test]
fn writer_dispatch_refuses_queued_expiry_but_finishes_an_already_owned_write() {
    use polylogue_store::writer::DispatchPhase;
    use std::time::{Duration, Instant};
    for phase in [DispatchPhase::BeforeParentCheck, DispatchPhase::Owned] {
        let daemon = crate::test_daemon::start("HTTP writer pause");
        let web = web(&daemon);
        let id = if phase == DispatchPhase::Owned {
            "owned-before-expiry"
        } else {
            "queued-before-expiry"
        };
        let writer = daemon.daemon.writes.as_ref().unwrap();
        let pause = writer.pause_user_dispatch_for_test(id, phase);
        let parent = Instant::now() + Duration::from_secs(2);
        let request = command_surface::Request {
            operation: "set".into(),
            arguments: serde_json::json!({"key":"reading.alternates","value":"[]"}),
            request_id: id.into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let task = runtime.spawn(async move {
            respond(&web, request, caller(Default::default(), Some(parent))).await
        });
        // This is the writer's own dispatch, not a client-response delay.
        pause.wait().unwrap();
        runtime.block_on(async {
            tokio::time::sleep_until(tokio::time::Instant::from_std(parent)).await;
        });
        pause.release();
        let response = runtime.block_on(task).unwrap();
        let status = response.status();
        let bytes = runtime
            .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
            .unwrap();
        let answer: command_surface::Answer<crate::protocol::setting::Answer> =
            serde_json::from_slice(&bytes).unwrap();
        let recorded = daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .user_request(id)
            .unwrap();
        if phase == DispatchPhase::Owned {
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
            let Some(crate::protocol::setting::Answer::Outcome(write_result)) = answer.value()
            else {
                panic!("owned write must keep the writer acknowledgment")
            };
            assert_eq!(write_result.id, id);
            assert!(!write_result.repeat);
            let recorded = recorded.expect("owned write is durably recorded");
            assert_eq!(recorded.id, write_result.id);
            assert_eq!(
                recorded.authorship.via,
                polylogue_model::authorship::Channel::Web
            );
        } else {
            assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
            let command_surface::Answer {
                outcome: command_surface::Outcome::Error(error),
                ..
            } = answer
            else {
                panic!("unaccepted write cannot return a write result")
            };
            assert_eq!(error.kind, command_surface::FailureKind::Timeout);
            assert!(recorded.is_none());
        }
    }
}

#[test]
fn ready_http_calls_preserve_absent_null_and_finite_deadlines() {
    let daemon = crate::test_daemon::start("HTTP deadline envelope roundtrip");
    for deadline in [
        command_surface::Deadline::Default,
        command_surface::Deadline::None,
        command_surface::Deadline::Milliseconds(37),
    ] {
        let request = command_surface::Request {
            operation: "rows".into(),
            arguments: serde_json::json!({"selection":"sessions","page_size":null,"fields":[]}),
            request_id: "ready-http-deadline".into(),
            deadline,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let ready = daemon.daemon.ready_call(&request).unwrap();
        assert_eq!(ready.request, request);
        let url =
            polylogue_render::ready::http_url(crate::commands::declarations(), &ready.request)
                .unwrap();
        let uri: axum::http::Uri = url.parse().unwrap();
        let Query(pairs) = Query::<Vec<(String, String)>>::try_from_uri(&uri).unwrap();
        let has_deadline = pairs.iter().any(|(key, _)| key == "deadline");
        assert_eq!(
            has_deadline,
            !matches!(deadline, command_surface::Deadline::Default)
        );
        let parsed = crate::commands::declarations()
            .parse_http_query("rows", &pairs, "unused")
            .unwrap();
        assert_eq!(
            parsed, request,
            "generated call changed its accepted envelope: {url}"
        );
        assert!(
            parsed
                .arguments
                .get("page_size")
                .is_some_and(serde_json::Value::is_null)
        );
    }
}

use super::*;
use crate::daemon::{Failure, Request, service::Daemon};
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
        _: &std::sync::Arc<crate::operation::Context>,
        _: &mut dyn Write,
    ) -> Option<Result<(), crate::daemon::Failure>> {
        Some(Err(Failure::new("bad_request", "invalid selection")))
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
        authorship: &std::sync::Arc<crate::operation::Context>,
        output: &mut dyn Write,
    ) -> Option<Result<(), crate::daemon::Failure>> {
        assert_eq!(
            authorship.authorship.author,
            polylogue_model::Author::Operator
        );
        assert_eq!(
            authorship.authorship.via,
            polylogue_model::authorship::Channel::Web
        );
        let accepted = authorship
            .accepted_request()
            .expect("canonical accepted request");
        assert_eq!(accepted.request_id, "http-accepted-stream");
        assert_eq!(accepted.deadline, command_surface::Deadline::None);
        assert_eq!(accepted.arguments["uncapped"], true);
        crate::protocol::stream::event(
            output,
            "begin",
            &serde_json::json!({"snapshot":"synthetic"}),
        )
        .unwrap();
        Some(Err(Failure::new("conflict", "source changed")))
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
                Some(command_surface::Request {
                    operation: "changes".into(),
                    arguments: serde_json::json!({"uncapped":true}),
                    request_id: "http-accepted-stream".into(),
                    deadline: command_surface::Deadline::None,
                    budget_tokens: Default::default(),
                    cursor: None,
                    fresh: false,
                    format: Default::default(),
                }),
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
        _: &std::sync::Arc<crate::operation::Context>,
        output: &mut dyn Write,
    ) -> Option<Result<(), crate::daemon::Failure>> {
        let result = (|| {
            crate::protocol::stream::event(output, "begin", &serde_json::json!({}))?;
            // Exceeds the transport buffer. Preparation cannot finish until the body is consumed.
            crate::protocol::stream::event(output, "record", &"x".repeat(128 * 1024))?;
            crate::protocol::stream::event(output, "complete", &serde_json::json!({}))
        })();
        Some(result.map_err(|error| Failure::new("io", error)))
    }
    let daemon = crate::test_daemon::start("synthetic stream backpressure");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                stream(
                    &web(&daemon),
                    Request::Changes { since: None },
                    large,
                    None,
                    None,
                ),
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
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
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
        _: &std::sync::Arc<crate::operation::Context>,
        output: &mut dyn Write,
    ) -> Option<Result<(), crate::daemon::Failure>> {
        let reader = match daemon.reads.reader() {
            Ok(reader) => reader,
            Err(error) => return Some(Err(Failure::from(error))),
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
        Some(result.map_err(|error| Failure::new("io", error)))
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
            let command_surface::Answer {
                outcome: command_surface::Outcome::Error(error),
                ..
            } = answer
            else {
                panic!("stale stream was successful")
            };
            assert_eq!(error.code.as_deref(), Some("scope_changed"));
            assert_eq!(error.kind, command_surface::FailureKind::Conflict);
        });
}

#[test]
fn scope_reset_after_http_encoding_cannot_activate_the_old_body() {
    let daemon = crate::test_daemon::start("native HTTP activation fence");
    let web = web(&daemon);
    let scope = daemon.daemon.reads.scope_token().unwrap();
    let (status, bytes) = super::super::prepare_json_answer(
        &web,
        command_surface::Answer::ok(serde_json::json!({"stale":"synthetic-old-scope-value"})),
    );
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    let response = scoped_body_response(
        &web,
        status,
        "application/json",
        bytes,
        &crate::protocol::ReadScopes::single(Some(scope)),
    );
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
        .unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("synthetic-old-scope-value"));
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&bytes).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("old scope must not activate");
    };
    assert_eq!(error.code.as_deref(), Some("scope_changed"));
}

#[test]
fn scope_reset_before_polling_an_encoded_http_body_aborts_its_frame() {
    let daemon = crate::test_daemon::start("native HTTP frame fence");
    let web = web(&daemon);
    let scope = daemon.daemon.reads.scope_token().unwrap();
    let (status, bytes) = super::super::prepare_json_answer(
        &web,
        command_surface::Answer::ok(serde_json::json!({"stale":"synthetic-old-scope-value"})),
    );
    let response = scoped_body_response(
        &web,
        status,
        "application/json",
        bytes,
        &crate::protocol::ReadScopes::single(Some(scope)),
    );
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(axum::body::to_bytes(response.into_body(), usize::MAX));
    assert!(
        result.is_err(),
        "a revoked HTTP body must not publish a complete value"
    );
}

#[test]
fn a_captured_parent_deadline_is_not_renewed_during_dispatch_wait() {
    let daemon = crate::test_daemon::start("parent queue bound");
    let web = web(&daemon);
    let observed = daemon.watch_reads();
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
    let response = runtime.block_on(crate::web::READ_LIMIT.scope(
        (Some(parent), Default::default()),
        async {
            tokio::time::sleep_until(tokio::time::Instant::from_std(parent)).await;
            answer(
                &web,
                crate::commands::command("status").unwrap(),
                Ok(Input::Ready(request)),
                &HeaderMap::new(),
            )
            .await
        },
    ));
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
    let observation = observed
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert_eq!(observation.sqlite_steps(), 0);
}

#[test]
fn no_outer_header_does_not_supply_a_default_parent_for_a_null_request() {
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(
            crate::web::READ_LIMIT.scope((None, Default::default()), async {
                assert!(crate::web::parent_deadline().is_none());
            }),
        );
}

#[test]
fn a_cancelled_http_parent_without_a_header_cannot_submit_a_pure_write() {
    let daemon = crate::test_daemon::start("cancelled HTTP write admission");
    let web = web(&daemon);
    let cancellation = polylogue_store::read::limit::ReadCancellation::default();
    cancellation.cancel();
    let id = "cancelled-parent-no-new-write";
    let request = command_surface::Request {
        operation: "setting".into(),
        arguments: serde_json::json!({"key":"reading.variants","value":"[]"}),
        request_id: id.into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let response = runtime.block_on(crate::web::READ_LIMIT.scope((None, cancellation), async {
        answer(
            &web,
            crate::commands::command("setting").unwrap(),
            Ok(Input::Ready(request)),
            &HeaderMap::new(),
        )
        .await
    }));
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
        panic!("cancelled admission cannot yield a receipt")
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
    for phase in [DispatchPhase::BeforeAdmission, DispatchPhase::Owned] {
        let daemon = crate::test_daemon::start("HTTP writer admission barrier");
        let web = web(&daemon);
        let id = if phase == DispatchPhase::Owned {
            "owned-before-expiry"
        } else {
            "queued-before-expiry"
        };
        let writer = daemon.daemon.writes.as_ref().unwrap();
        let barrier = writer.pause_user_dispatch_for_test(id, phase);
        let parent = Instant::now() + Duration::from_secs(2);
        let request = command_surface::Request {
            operation: "setting".into(),
            arguments: serde_json::json!({"key":"reading.variants","value":"[]"}),
            request_id: id.into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let task = runtime.spawn(crate::web::READ_LIMIT.scope(
            (Some(parent), Default::default()),
            async move {
                answer(
                    &web,
                    crate::commands::command("setting").unwrap(),
                    Ok(Input::Ready(request)),
                    &HeaderMap::new(),
                )
                .await
            },
        ));
        // This is the native writer's dispatch, not a client-response delay.
        barrier.wait().unwrap();
        runtime.block_on(async {
            tokio::time::sleep_until(tokio::time::Instant::from_std(parent)).await;
        });
        barrier.release();
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
            let Some(crate::protocol::setting::Answer::Receipt(receipt)) = answer.value() else {
                panic!("owned write must retain the actual writer ACK")
            };
            assert_eq!(receipt.id, id);
            assert!(!receipt.repeat);
            let recorded = recorded.expect("owned write is durably recorded");
            assert_eq!(recorded.id, receipt.id);
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
                panic!("unadmitted write cannot return a receipt")
            };
            assert_eq!(error.kind, command_surface::FailureKind::Timeout);
            assert!(recorded.is_none());
        }
    }
}

#[test]
fn actual_ready_http_calls_preserve_absent_null_and_finite_deadlines() {
    let daemon = crate::test_daemon::start("HTTP deadline envelope roundtrip");
    for deadline in [
        command_surface::Deadline::Default,
        command_surface::Deadline::None,
        command_surface::Deadline::Milliseconds(37),
    ] {
        let request = command_surface::Request {
            operation: "query".into(),
            arguments: serde_json::json!({"query":"sessions","session":null,"limit":null,"fields":[]}),
            request_id: "actual-ready-http-deadline".into(),
            deadline,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let ready = daemon.daemon.ready_call(&request).unwrap();
        assert_eq!(
            ready.http.arguments,
            serde_json::to_value(&request).unwrap()
        );
        let url = polylogue_render::web::http_url(&ready.http).unwrap();
        let uri: axum::http::Uri = url.parse().unwrap();
        let Query(pairs) = Query::<Vec<(String, String)>>::try_from_uri(&uri).unwrap();
        let has_deadline = pairs.iter().any(|(key, _)| key == "deadline");
        assert_eq!(
            has_deadline,
            !matches!(deadline, command_surface::Deadline::Default)
        );
        let parsed = ready_pairs(pairs).unwrap();
        assert_eq!(
            parsed, request,
            "actual generated call changed its accepted envelope: {url}"
        );
        assert!(
            parsed
                .arguments
                .get("session")
                .is_some_and(serde_json::Value::is_null)
        );
    }
}

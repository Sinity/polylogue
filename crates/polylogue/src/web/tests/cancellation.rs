use super::*;
use std::io::Write;
use std::time::Duration;

/// Protocol tests inspect the canonical envelope before any domain projection.
async fn send_wire(r: &Reader, request: Http<Body>) -> (StatusCode, HeaderMap, String) {
    let response = router(Arc::clone(&r.web)).oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, headers, String::from_utf8(bytes.to_vec()).unwrap())
}

#[test]
fn http_disconnect_interrupts_sql_and_releases_the_reader_slot() {
    let daemon = test_daemon::start("HTTP cancellation");
    let reads = daemon.watch_reads();
    let mut server = start_on(
        Arc::clone(&daemon.daemon),
        Arc::clone(daemon.operations()),
        0,
    )
    .unwrap();
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let sql = "WITH%20RECURSIVE%20n(x)%20AS%20(SELECT%201%20UNION%20ALL%20SELECT%20x%2B1%20FROM%20n%20WHERE%20x%3C2000000000)%20SELECT%20count(*)%20FROM%20n";
    write!(stream, "GET /api/v1/sql?statement={sql} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\n\r\n", server.port, server.token).unwrap();
    let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    test_daemon::TestDaemon::wait_for_sql(&limit);
    drop(stream);
    let elapsed = daemon.assert_read_released(&limit);
    assert!(limit.cancellation.is_cancelled());
    eprintln!("HTTP disconnect released reader in {elapsed:?}");
    server.shutdown().unwrap();
}

#[test]
fn aborting_an_http_handler_cancels_its_blocking_read() {
    let r = reader();
    let reads = r.w.watch_reads();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let web = Arc::clone(&r.web);
    let task = runtime.spawn(async move {
        ask(&web, "sql", serde_json::json!({"statement": "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}), None).await
    });
    let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    test_daemon::TestDaemon::wait_for_sql(&limit);
    task.abort();
    runtime.block_on(async {
        assert!(task.await.unwrap_err().is_cancelled());
    });
    r.w.assert_read_released(&limit);
    assert!(limit.cancellation.is_cancelled());
}

#[test]
fn http_deadline_returns_a_typed_refusal_with_an_unknown_count() {
    let r = reader();
    let reads = r.w.watch_reads();
    let deadline = crate::daemon::execution::deadline(Duration::ZERO).to_string();
    let request = authed(get("/api/v1/status"))
        .header("x-polylogue-deadline-ms", deadline)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(send_wire(&r, request));
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_str(&body).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("expired parent must fail")
    };
    assert_eq!(error.kind, command_surface::FailureKind::Timeout);
    assert_eq!(error.code.as_deref(), Some("deadline_exceeded"));
    assert!(error.retryable);
    let observed = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(observed.sqlite_steps(), 0, "expired parent reached SQL");
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        body["error"]["count"].is_null(),
        "unmeasured rows must not become zero"
    );
}

fn typed_url(
    daemon: &TestDaemon,
    operation: &str,
    arguments: serde_json::Value,
    request_id: &str,
    deadline: command_surface::Deadline,
) -> String {
    let request = command_surface::Request {
        operation: operation.into(),
        arguments,
        request_id: request_id.into(),
        deadline,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let ready = daemon.daemon.ready_call(&request).unwrap();
    polylogue_render::web::http_url(&ready.http).unwrap()
}

#[test]
fn no_header_status_keeps_default_null_and_finite_envelope_deadlines() {
    let r = reader();
    rt().block_on(async {
        for (id, deadline) in [
            ("no-parent-default", command_surface::Deadline::Default),
            ("no-parent-null", command_surface::Deadline::None),
            (
                "no-parent-finite",
                command_surface::Deadline::Milliseconds(1000),
            ),
        ] {
            let path = typed_url(&r.w, "status", serde_json::json!({}), id, deadline);
            let (status, _, body) =
                send_wire(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{id}: {body}");
            let answer: command_surface::Answer<serde_json::Value> =
                serde_json::from_str(&body).unwrap();
            assert!(answer.value().is_some());
        }
    });
}

#[test]
fn outer_parent_expires_during_actual_sql_under_a_null_envelope() {
    let r = reader();
    let reads = r.w.watch_reads();
    let path = typed_url(
        &r.w,
        "sql",
        serde_json::json!({"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
        "parent-during-sql",
        command_surface::Deadline::None,
    );
    let request = authed(get(&path))
        .header(
            "x-polylogue-deadline-ms",
            crate::daemon::execution::deadline(Duration::from_secs(1)).to_string(),
        )
        .body(Body::empty())
        .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let web = Arc::clone(&r.web);
    let task = runtime.spawn(async move { router(web).oneshot(request).await.unwrap() });
    let observed = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    TestDaemon::wait_for_sql(&observed);
    let response = runtime.block_on(task).unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    let bytes = runtime
        .block_on(to_bytes(response.into_body(), usize::MAX))
        .unwrap();
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&bytes).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("SQL parent expiry must fail")
    };
    assert_eq!(error.kind, command_surface::FailureKind::Timeout);
    assert_eq!(error.code.as_deref(), Some("deadline_exceeded"));
    assert!(observed.sqlite_steps() >= 10_000, "no SQL-phase evidence");
    r.w.assert_read_released(&observed);
}

#[test]
fn a_null_stream_envelope_is_still_capped_by_the_outer_parent() {
    let r = reader();
    rt().block_on(async {
        let path = typed_url(&r.w,"read",serde_json::json!({"target":"session:claude-code:s1","delivery":"stream","format":"jsonl"}),"parent-before-stream",command_surface::Deadline::None);
        let request = authed(get(&path))
            .header("x-polylogue-deadline-ms",crate::daemon::execution::deadline(Duration::ZERO).to_string())
            .body(Body::empty()).unwrap();
        let (status,_,body) = send_wire(&r,request).await;
        assert_eq!(status,StatusCode::GATEWAY_TIMEOUT,"{body}");
        let answer: command_surface::Answer<serde_json::Value> = serde_json::from_str(&body).unwrap();
        let command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } = answer else { panic!("expired stream parent must fail before begin") };
        assert_eq!(error.kind,command_surface::FailureKind::Timeout);
        assert!(!body.contains("event: complete"));
    });
}

#[test]
fn a_finite_stream_envelope_keeps_its_shorter_bound_than_the_outer_parent() {
    let r = reader();
    rt().block_on(async {
        let path = typed_url(&r.w,"read",serde_json::json!({"target":"session:claude-code:s1","delivery":"stream","format":"jsonl"}),"logical-before-parent",command_surface::Deadline::Milliseconds(0));
        let request = authed(get(&path))
            .header("x-polylogue-deadline-ms",crate::daemon::execution::deadline(Duration::from_secs(1)).to_string())
            .body(Body::empty()).unwrap();
        let (status,_,body) = send_wire(&r,request).await;
        assert_eq!(status,StatusCode::GATEWAY_TIMEOUT,"{body}");
        let answer: command_surface::Answer<serde_json::Value> = serde_json::from_str(&body).unwrap();
        let command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } = answer else { panic!("expired logical stream cap must fail") };
        assert_eq!(error.kind,command_surface::FailureKind::Timeout);
    });
}

#[test]
fn an_expired_http_parent_never_submits_a_new_pure_write() {
    let r = reader();
    let id = "expired-parent-no-write";
    rt().block_on(async {
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
        let request = own_origin(authed(post("/api/v1/setting")))
            .header(
                "x-polylogue-deadline-ms",
                crate::daemon::execution::deadline(Duration::ZERO).to_string(),
            )
            .body(Body::from(serde_json::to_vec(&request).unwrap()))
            .unwrap();
        let (status, _, body) = send_wire(&r, request).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        let answer: command_surface::Answer<crate::protocol::setting::Answer> =
            serde_json::from_str(&body).unwrap();
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = answer
        else {
            panic!("expired admission cannot yield a receipt: {body}")
        };
        assert_eq!(error.kind, command_surface::FailureKind::Timeout);
    });
    assert!(
        r.w.daemon
            .reads
            .reader()
            .unwrap()
            .user_request(id)
            .unwrap()
            .is_none()
    );
}

#[test]
fn no_header_null_sql_has_no_implicit_read_cap_and_disconnect_still_cleans_up() {
    let daemon = test_daemon::start("null HTTP SQL cancellation");
    let reads = daemon.watch_reads();
    let mut server = start_on(
        Arc::clone(&daemon.daemon),
        Arc::clone(daemon.operations()),
        0,
    )
    .unwrap();
    let path = typed_url(
        &daemon,
        "sql",
        serde_json::json!({"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
        "no-parent-null-sql",
        command_surface::Deadline::None,
    );
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\n\r\n",
        server.port, server.token
    )
    .unwrap();
    let observed = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    TestDaemon::wait_for_sql(&observed);
    // Exceed this operation's actual class default without adding it to null.
    let default = crate::commands::deadlines()
        .get(crate::commands::command("sql").unwrap().latency)
        .expect("SQL has a finite class default");
    std::thread::sleep(default + Duration::from_millis(20));
    observed.check().unwrap();
    drop(stream);
    daemon.assert_read_released(&observed);
    assert!(observed.cancellation.is_cancelled());
    server.shutdown().unwrap();
}

#[test]
fn an_expired_http_read_parent_creates_no_query_object_begin_or_outcome() {
    let r = reader();
    let custody = || {
        r.w.daemon.reads.reader().unwrap().query(
        "SELECT (SELECT count(*) FROM source_query) AS objects, (SELECT count(*) FROM source_query_run WHERE run IS NOT NULL) AS begins, (SELECT count(*) FROM source_query_run WHERE completion IS NOT NULL) AS outcomes", &[]).unwrap()
    };
    let before = custody();
    let reads = r.w.watch_reads();
    rt().block_on(async {
        let path = typed_url(
            &r.w,
            "read",
            serde_json::json!({"target":"session:claude-code:s1"}),
            "expired-before-read-begin",
            command_surface::Deadline::None,
        );
        let request = authed(get(&path))
            .header(
                "x-polylogue-deadline-ms",
                crate::daemon::execution::deadline(Duration::ZERO).to_string(),
            )
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send_wire(&r, request).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        let answer: command_surface::Answer<serde_json::Value> =
            serde_json::from_str(&body).unwrap();
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = answer
        else {
            panic!("expired read cannot begin: {body}")
        };
        assert_eq!(error.kind, command_surface::FailureKind::Timeout);
    });
    assert_eq!(custody(), before);
    // Admission observes the limit but does not start a statement or read body.
    let observed = reads.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(observed.sqlite_steps(), 0);
}

/// A read the table stops at its deadline names the reason in its code, so
/// every client prints it instead of a fallback.
#[test]
fn a_page_read_past_its_deadline_answers_deadline_exceeded() {
    let r = reader();
    rt().block_on(async {
        let path = typed_url(
            &r.w,
            "read",
            serde_json::json!({"target": "session:claude-code:s1"}),
            "page-past-deadline",
            command_surface::Deadline::Milliseconds(0),
        );
        let (status, _, body) =
            send_wire(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        let answer: command_surface::Answer<serde_json::Value> =
            serde_json::from_str(&body).unwrap();
        let command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } = answer
        else {
            panic!("a read past its deadline must fail: {body}")
        };
        assert_eq!(error.kind, command_surface::FailureKind::Timeout);
        assert_eq!(
            error.code.as_deref(),
            Some(command_surface::DEADLINE_EXCEEDED)
        );
    });
}

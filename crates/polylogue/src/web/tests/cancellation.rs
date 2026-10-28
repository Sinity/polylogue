use super::*;
use std::io::Write;
use std::time::Duration;

#[test]
fn http_disconnect_interrupts_sql_and_releases_the_reader_slot() {
    let daemon = test_daemon::start("HTTP cancellation");
    let _held = daemon.daemon.reads.reader().unwrap();
    let reads = daemon.watch_reads();
    let mut server = start_on(Arc::clone(&daemon.daemon), 0).unwrap();
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let sql = "WITH%20RECURSIVE%20n(x)%20AS%20(SELECT%201%20UNION%20ALL%20SELECT%20x%2B1%20FROM%20n%20WHERE%20x%3C2000000000)%20SELECT%20count(*)%20FROM%20n";
    write!(stream, "GET /api/v1/sql?statement={sql} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\n\r\n", server.port, server.token).unwrap();
    let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    test_daemon::TestDaemon::wait_for_sql(&limit);
    drop(stream);
    let elapsed = daemon.assert_slot_freed();
    assert!(limit.cancellation.is_cancelled());
    eprintln!("HTTP disconnect released reader in {elapsed:?}");
    server.shutdown().unwrap();
}

#[test]
fn aborting_an_http_handler_cancels_its_blocking_read() {
    let r = reader();
    let _held = r.w.daemon.reads.reader().unwrap();
    let reads = r.w.watch_reads();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let web = Arc::clone(&r.web);
    let task = runtime.spawn(async move {
        ask(&web, Request::Sql { statement: "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n".into() }).await
    });
    let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    test_daemon::TestDaemon::wait_for_sql(&limit);
    task.abort();
    runtime.block_on(async {
        assert!(task.await.unwrap_err().is_cancelled());
    });
    r.w.assert_slot_freed();
    assert!(limit.cancellation.is_cancelled());
}

#[test]
fn http_deadline_returns_a_typed_refusal_with_an_unknown_count() {
    let r = reader();
    let deadline = crate::daemon::execution::deadline(Duration::ZERO).to_string();
    let request = authed(get("/api/v1/status"))
        .header("x-polylogue-deadline-ms", deadline)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(send(&r, request));
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["error"], "deadline_exceeded");
    assert!(body["detail"]["count"].is_null());
}

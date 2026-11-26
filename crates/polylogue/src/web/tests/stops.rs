//! A deadline and a client's hang-up each stop a read in progress, on the
//! socket and on HTTP, through the requests a client sends; a stream the
//! daemon's stop cuts ends with `shutting_down` (docs/latency.md,
//! docs/operations.md).
use super::*;
use std::io::{Read, Write};
use std::time::Duration;

/// A statement that reads until something stops it.
const ENDLESS: &str = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n";

fn sql(id: &str, deadline: command_surface::Deadline) -> command_surface::RunnableRequest {
    command_surface::Request {
        deadline,
        ..crate::client::request(
            "dev sql",
            serde_json::json!({ "statement": ENDLESS }),
            id.into(),
        )
    }
}

/// The deadline refusal of a read, offering the same request without one.
fn assert_deadline_refusal(answer: &command_surface::Answer<serde_json::Value>) {
    let failure = answer.failure().unwrap_or_else(|| panic!("{answer:?}"));
    assert_eq!(
        failure.code.as_deref(),
        Some("deadline_exceeded"),
        "{answer:?}"
    );
    let lifted = answer
        .envelope
        .next
        .iter()
        .find(|next| next.label == "complete without a deadline")
        .unwrap_or_else(|| panic!("no lifted call: {answer:?}"));
    assert_eq!(lifted.request.deadline, command_surface::Deadline::None);
    assert_eq!(lifted.request.arguments["statement"], ENDLESS);
}

#[test]
fn a_deadline_stops_a_read_on_the_socket() {
    let daemon = test_daemon::start("socket deadline stop");
    let answer = crate::client::call(
        &daemon.daemon.archive.socket(),
        &sql(
            "socket-deadline",
            command_surface::Deadline::Milliseconds(200),
        ),
    )
    .unwrap();
    assert_deadline_refusal(&answer);
    daemon.assert_reads_released();
}

#[test]
fn a_hang_up_stops_a_read_on_the_socket() {
    let daemon = test_daemon::start("socket hang-up stop");
    let mut socket =
        std::os::unix::net::UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    writeln!(
        socket,
        "{}",
        serde_json::to_string(&sql("socket-hang-up", command_surface::Deadline::None)).unwrap()
    )
    .unwrap();
    daemon.wait_for_a_read();
    drop(socket);
    let elapsed = daemon.assert_reads_released();
    eprintln!("the socket hang-up released the reader in {elapsed:?}");
}

/// The daemon's HTTP port for `daemon` and a raw GET on it.
fn http_get(server: &crate::web::Server, path: &str) -> std::net::TcpStream {
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    write!(
        peer,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
        server.port, server.token
    )
    .unwrap();
    peer
}

fn sql_path(deadline: &str) -> String {
    format!(
        "/api/v1/dev/sql?statement={}&deadline={deadline}",
        polylogue_render::html::encode_path_segment(ENDLESS)
    )
}

#[test]
fn a_deadline_stops_a_read_on_http() {
    let daemon = test_daemon::start("HTTP deadline stop");
    let mut server = start_on(Arc::clone(&daemon.daemon), 0).unwrap();
    let mut peer = http_get(&server, &sql_path("200"));
    peer.set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut response = String::new();
    peer.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 504"), "{head}");
    assert!(head.contains("content-type: application/json"), "{head}");
    assert_deadline_refusal(&serde_json::from_str(body).unwrap());
    daemon.assert_reads_released();
    server.shutdown().unwrap();
}

#[test]
fn a_hang_up_stops_a_read_on_http() {
    let daemon = test_daemon::start("HTTP hang-up stop");
    let mut server = start_on(Arc::clone(&daemon.daemon), 0).unwrap();
    let peer = http_get(&server, &sql_path("null"));
    daemon.wait_for_a_read();
    drop(peer);
    let elapsed = daemon.assert_reads_released();
    eprintln!("the HTTP hang-up released the reader in {elapsed:?}");
    server.shutdown().unwrap();
}

/// A follow the daemon's stop cuts ends with the stop as its final answer,
/// not with a stream that merely closes.
#[test]
fn a_stream_cut_by_the_daemon_stopping_ends_with_shutting_down() {
    let mut daemon = test_daemon::start("stream cut by a stop");
    let request = crate::client::request(
        "rows",
        serde_json::json!({"selection":"session:claude-code:s1","steps":[{"verb":"read","follow":true}]}),
        "cut-by-stop".into(),
    );
    let archive = daemon.daemon.archive.clone();
    let (first, began) = std::sync::mpsc::channel();
    let reply = std::thread::spawn(move || {
        let mut first = Some(first);
        crate::client::call_streaming(&archive, &request, None, |event| {
            if let Some(first) = first.take() {
                first.send(event.event).unwrap();
            }
            Ok(false)
        })
    });
    assert_eq!(
        began.recv_timeout(Duration::from_secs(10)).unwrap(),
        "record"
    );
    daemon.stop().unwrap();
    let answer = reply.join().unwrap().unwrap().expect("the stream answers");
    assert_eq!(
        answer.failure().and_then(|failure| failure.code.as_deref()),
        Some("shutting_down"),
        "{answer:?}"
    );
}

use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::time::Duration;

const CONTEXT_TOOLS: &[&str] = &["brief", "recall", "takeover"];

fn tool_call(name: &str) -> Value {
    let arguments = match name {
        "brief" => json!({"session":"claude-code:s1", "cwd":"/synthetic/project"}),
        "recall" => json!({"query":"MCP context", "cwd":"/synthetic/project"}),
        "takeover" => json!({"session":"claude-code:s1"}),
        "query" => json!({"query":"sessions"}),
        _ => panic!("no fixture for {name}"),
    };
    super::tool_call(name, arguments, &format!("cancel-{name}"))
}

struct PeerCleanup<'a> {
    socket: UnixStream,
    lock: Option<&'a std::fs::File>,
}
impl Drop for PeerCleanup<'_> {
    fn drop(&mut self) {
        if let Some(lock) = self.lock {
            let _ = lock.unlock();
        }
        let _ = self.socket.shutdown(Shutdown::Both);
    }
}

fn cancelled_read(disconnect: bool, name: &str) {
    let mut daemon = test_daemon::start_with_workers("MCP cancellation", 4);
    // As in the deadline fixture, join the unrelated signal poller before
    // blocking storage, so its background work cannot retain a reader here.
    daemon.pause_signals();
    let lock = std::fs::File::open(daemon.daemon.archive.archive_db()).unwrap();
    lock.lock().unwrap();
    let reads = daemon.watch_reads();
    let (mut client, server) = UnixStream::pair().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut input = BufReader::new(client.try_clone().unwrap());
    let archive = Archive::at(&daemon.daemon.archive.root);
    std::thread::scope(|scope| {
        let _cleanup = PeerCleanup {
            socket: client.try_clone().unwrap(),
            lock: Some(&lock),
        };
        let output = server.try_clone().unwrap();
        let service = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        writeln!(client, "{}", super::initialize()).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        assert!(serde_json::from_str::<Value>(&line).unwrap()["result"].is_object());
        let mut params = tool_call(name);
        // The caller's cancellation is the stop condition under test.
        params["arguments"]["deadline"] = if name == "query" {
            Value::Null
        } else {
            json!(60_000)
        };
        writeln!(
            client,
            "{}",
            json!({"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":params})
        )
        .unwrap();
        let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while !limit.reader_acquired() && std::time::Instant::now() < until {
            std::thread::yield_now();
        }
        let started = limit.reader_acquired();
        let acquisition_check = format!("{:?}", limit.check());
        let acquisition_slots = daemon.daemon.reads.checked_out();
        let mut acquisition_reply = Vec::new();
        if !started {
            client
                .set_read_timeout(Some(Duration::from_millis(1)))
                .unwrap();
            loop {
                line.clear();
                match input.read_line(&mut line) {
                    Ok(0) => {
                        acquisition_reply.push("MCP closed without an answer".into());
                        break;
                    }
                    Ok(_) => {
                        let answered = serde_json::from_str::<Value>(&line)
                            .is_ok_and(|reply| reply["id"] == 2);
                        acquisition_reply.push(line.clone());
                        if answered {
                            break;
                        }
                    }
                    Err(error) => {
                        acquisition_reply.push(format!("MCP read: {error:?}"));
                        break;
                    }
                }
            }
            client
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
        }
        let stop_error = if disconnect {
            client.shutdown(Shutdown::Both)
        } else {
            writeln!(client, "{}", json!({"jsonrpc":"2.0", "method":"notifications/cancelled", "params":{"requestId":2, "reason":"caller stopped"}}))
        }.err().map(|error| error.to_string());
        let until = std::time::Instant::now() + Duration::from_secs(2);
        while (daemon.daemon.reads.checked_out_by(limit.owner) != 0
            || !limit.cancellation.is_cancelled())
            && std::time::Instant::now() < until
        {
            std::thread::yield_now();
        }
        let owner_slots = daemon.daemon.reads.checked_out_by(limit.owner);
        let released = owner_slots == 0;
        let cancelled = limit.cancellation.is_cancelled();
        let slots = daemon.daemon.reads.checked_out();
        // Teardown must also finish when the regression retains its reader.
        lock.unlock().unwrap();
        let _ = client.shutdown(Shutdown::Both);
        let _ = service.join().unwrap();
        assert!(
            stop_error.is_none(),
            "MCP {name} stop failed: {stop_error:?}"
        );
        assert!(
            released,
            "MCP {name} retained its reader after cancellation: cancelled={cancelled}, owner_slots={owner_slots}, pool_slots={slots}, limit={:?}",
            limit.limit
        );
        assert!(cancelled, "MCP {name} did not cancel its daemon read");
        assert!(
            started,
            "MCP {name} never acquired its reader: request={params}, reply={acquisition_reply:?}, acquisition_check={acquisition_check}, pool_slots={acquisition_slots}, queued_cancellation={cancelled}, released_slots={owner_slots}, stop_error={stop_error:?}, limit={:?}",
            limit.limit
        );
    });
    assert_no_prepared(&daemon);
}

#[test]
fn mcp_cancellation_interrupts_the_daemon_read() {
    cancelled_read(false, "query");
}

#[test]
fn mcp_disconnect_interrupts_the_daemon_read() {
    cancelled_read(true, "query");
}

#[test]
fn stdio_tool_body_preserves_the_explicit_shared_deadline() {
    use std::os::unix::net::UnixListener;
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let listener = UnixListener::bind(archive.socket()).unwrap();
    let deadline = 60_000u64;
    listener.set_nonblocking(true).unwrap();
    let (mut client, server) = UnixStream::pair().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut input = BufReader::new(client.try_clone().unwrap());
    std::thread::scope(|scope| {
        let _cleanup = PeerCleanup {
            socket: client.try_clone().unwrap(),
            lock: None,
        };
        let output = server.try_clone().unwrap();
        let service = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        let daemon = scope.spawn(|| {
            let start = std::time::Instant::now();
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && start.elapsed() < Duration::from_secs(5) =>
                    {
                        std::thread::yield_now()
                    }
                    Err(error) => panic!("MCP did not forward the request: {error}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut line = String::new();
            BufReader::new(&socket).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["operation"], "status");
            assert_eq!(request["request_id"], "bounded-forward");
            assert_eq!(request["deadline"], deadline);
            writeln!(
                socket,
                "{}",
                serde_json::to_value(command_surface::Answer::ok(json!({"forwarded":true})))
                    .unwrap()
            )
            .unwrap();
        });
        writeln!(client, "{}", super::initialize()).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        writeln!(client, "{}", json!({"jsonrpc":"2.0", "id":"bounded", "method":"tools/call", "params":{"name":"polylogue_read","arguments":{"operation":"status","arguments":{},"request_id":"bounded-forward","deadline":deadline}}})).unwrap();
        line.clear();
        input.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(reply["id"], "bounded");
        assert_eq!(reply["result"]["isError"], false);
        let result: Value =
            serde_json::from_str(reply["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(result["forwarded"], true);
        client.shutdown(Shutdown::Write).unwrap();
        daemon.join().unwrap();
        service.join().unwrap().unwrap();
    });
}

fn context_deadline(name: &str) {
    let mut daemon = test_daemon::start_with_workers("MCP context deadline", 4);
    daemon.pause_signals();
    let lock = std::fs::File::open(daemon.daemon.archive.archive_db()).unwrap();
    lock.lock().unwrap();
    let reads = daemon.watch_reads();
    let (mut client, server) = UnixStream::pair().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut input = BufReader::new(client.try_clone().unwrap());
    let archive = Archive::at(&daemon.daemon.archive.root);
    let (answer, started, acquisition, owner_slots, pool_slots) = std::thread::scope(|scope| {
        let _cleanup = PeerCleanup {
            socket: client.try_clone().unwrap(),
            lock: Some(&lock),
        };
        let output = server.try_clone().unwrap();
        let service = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        writeln!(client, "{}", super::initialize()).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        let deadline = crate::daemon::execution::deadline(Duration::from_millis(500));
        let mut params = tool_call(name);
        params["_meta"] = json!({"deadline_ms":deadline});
        writeln!(
            client,
            "{}",
            json!({"jsonrpc":"2.0", "id":"context", "method":"tools/call", "params":params})
        )
        .unwrap();
        let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
        let until = std::time::Instant::now() + Duration::from_secs(2);
        while !limit.reader_acquired() && limit.check().is_ok() && std::time::Instant::now() < until
        {
            std::thread::yield_now();
        }
        let started = limit.reader_acquired();
        let acquisition = format!(
            "request={params}, check={:?}, remaining={:?}, owner_slots={}, pool_slots={}, limit={:?}",
            limit.check(),
            limit.remaining(),
            daemon.daemon.reads.checked_out_by(limit.owner),
            daemon.daemon.reads.checked_out(),
            limit.limit,
        );
        line.clear();
        let answer = input.read_line(&mut line).map(|_| line);
        // The transport timeout can win the race with daemon cleanup. Keep the
        // archive blocked while allowing the daemon to release its reader.
        let until = std::time::Instant::now() + Duration::from_secs(2);
        while daemon.daemon.reads.checked_out_by(limit.owner) != 0
            && std::time::Instant::now() < until
        {
            std::thread::yield_now();
        }
        let owner_slots = daemon.daemon.reads.checked_out_by(limit.owner);
        let pool_slots = daemon.daemon.reads.checked_out();
        // Release fixture holds before assertions so a failing regression also joins.
        lock.unlock().unwrap();
        let _ = client.shutdown(Shutdown::Both);
        let _ = service.join().unwrap();
        (answer, started, acquisition, owner_slots, pool_slots)
    });
    assert!(
        started,
        "MCP {name} never acquired its reader: {acquisition}, reply={answer:?}"
    );
    assert!(
        owner_slots == 0,
        "MCP {name} retained its daemon reader after the deadline: owner_slots={owner_slots}, pool_slots={pool_slots}"
    );
    let answer = answer.expect("MCP context did not answer its explicit deadline");
    let reply: Value = serde_json::from_str(&answer).unwrap();
    assert_eq!(reply["result"]["isError"], true);
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("deadline_exceeded"),
        "{reply}"
    );
    assert_no_prepared(&daemon);
}

fn assert_no_prepared(daemon: &test_daemon::TestDaemon) {
    let response = crate::daemon::service::handle(
        crate::daemon::Request::Query {
            text: "deliveries".into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
        &daemon.daemon,
    );
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}");
    };
    assert_eq!(
        data["count"], 0,
        "a refused context must not record prepared delivery"
    );
}

#[test]
fn mcp_context_deadline_releases_blocked_read_without_a_prepared_delivery() {
    for name in CONTEXT_TOOLS {
        context_deadline(name);
    }
}

#[test]
fn mcp_context_cancellation_interrupts_preparation_before_its_deadline() {
    for name in CONTEXT_TOOLS {
        cancelled_read(false, name);
    }
}

#[test]
fn mcp_context_disconnect_interrupts_preparation_before_its_deadline() {
    for name in CONTEXT_TOOLS {
        cancelled_read(true, name);
    }
}

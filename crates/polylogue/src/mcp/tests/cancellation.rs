use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::time::Duration;

fn cancelled_read(disconnect: bool, context: bool) {
    let daemon = test_daemon::start("MCP cancellation");
    let _held = daemon.daemon.reads.reader().unwrap();
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
        let output = server.try_clone().unwrap();
        let service = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        writeln!(client, "{}", super::initialize()).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        assert!(serde_json::from_str::<Value>(&line).unwrap()["result"].is_object());
        let params = if context {
            json!({"name":"takeover", "arguments":{"session":"claude-code:s1"}})
        } else {
            json!({"name":"query", "arguments":{"query":"sessions"}})
        };
        writeln!(
            client,
            "{}",
            json!({"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":params})
        )
        .unwrap();
        let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while daemon.daemon.reads.checked_out() < 2 {
            assert!(
                std::time::Instant::now() < until,
                "MCP never acquired its reader"
            );
            std::thread::yield_now();
        }
        if disconnect {
            client.shutdown(Shutdown::Both).unwrap();
        } else {
            writeln!(client, "{}", json!({"jsonrpc":"2.0", "method":"notifications/cancelled", "params":{"requestId":2, "reason":"caller stopped"}})).unwrap();
            let elapsed = daemon.assert_slot_freed();
            eprintln!("MCP cancellation released reader in {elapsed:?}");
            client.shutdown(Shutdown::Both).unwrap();
        }
        let elapsed = daemon.assert_slot_freed();
        assert!(limit.cancellation.is_cancelled());
        eprintln!("MCP disconnect released reader in {elapsed:?}");
        drop(lock);
        let _ = service.join().unwrap();
    });
}

#[test]
fn mcp_cancellation_interrupts_the_daemon_read() {
    cancelled_read(false, false);
}

#[test]
fn mcp_disconnect_interrupts_the_daemon_read() {
    cancelled_read(true, false);
}

#[test]
fn stdio_tool_metadata_preserves_the_explicit_daemon_deadline() {
    use std::os::unix::net::UnixListener;
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let listener = UnixListener::bind(archive.socket()).unwrap();
    let deadline = crate::daemon::execution::deadline(Duration::from_secs(60));
    let (mut client, server) = UnixStream::pair().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut input = BufReader::new(client.try_clone().unwrap());
    std::thread::scope(|scope| {
        let output = server.try_clone().unwrap();
        let service = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        let daemon = scope.spawn(|| {
            let (mut socket, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&socket).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["op"], "status");
            assert_eq!(request["deadline_ms"], deadline);
            writeln!(
                socket,
                "{}",
                json!({"result":"ok", "data":{"forwarded":true}})
            )
            .unwrap();
        });
        writeln!(client, "{}", super::initialize()).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        writeln!(client, "{}", json!({"jsonrpc":"2.0", "id":"bounded", "method":"tools/call", "params":{"name":"status", "_meta":{"deadline_ms":deadline}}})).unwrap();
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

#[test]
fn mcp_takeover_deadline_releases_blocked_read_without_a_prepared_delivery() {
    let daemon = test_daemon::start("MCP context deadline");
    let held = daemon.daemon.reads.reader().unwrap();
    let lock = std::fs::File::open(daemon.daemon.archive.archive_db()).unwrap();
    lock.lock().unwrap();
    let reads = daemon.watch_reads();
    let (mut client, server) = UnixStream::pair().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut input = BufReader::new(client.try_clone().unwrap());
    let archive = Archive::at(&daemon.daemon.archive.root);
    let (answer, released) = std::thread::scope(|scope| {
        let output = server.try_clone().unwrap();
        let service = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        writeln!(client, "{}", super::initialize()).unwrap();
        let mut line = String::new();
        input.read_line(&mut line).unwrap();
        let deadline = crate::daemon::execution::deadline(Duration::from_millis(50));
        writeln!(client, "{}", json!({"jsonrpc":"2.0", "id":"context", "method":"tools/call", "params":{"name":"takeover", "arguments":{"session":"claude-code:s1"}, "_meta":{"deadline_ms":deadline}}})).unwrap();
        reads.recv_timeout(Duration::from_secs(5)).unwrap();
        line.clear();
        let answer = input.read_line(&mut line).map(|_| line);
        let released = daemon.daemon.reads.checked_out() == 1;
        // Release fixture holds before assertions so a failing regression also joins.
        drop(lock);
        drop(held);
        client.shutdown(Shutdown::Both).unwrap();
        let _ = service.join().unwrap();
        (answer, released)
    });
    assert!(
        released,
        "MCP context retained its daemon reader after the deadline"
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
    let crate::daemon::Response::Ok { data, .. } = response else {
        panic!("{response:?}");
    };
    assert_eq!(
        data["count"], 0,
        "a refused context must not record prepared delivery"
    );
}

#[test]
fn mcp_context_cancellation_interrupts_preparation_without_explicit_deadline() {
    cancelled_read(false, true);
}

#[test]
fn mcp_context_disconnect_interrupts_preparation_without_explicit_deadline() {
    cancelled_read(true, true);
}

use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::time::Duration;

fn cancelled_read(disconnect: bool) {
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
        writeln!(client, "{}", json!({"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":{"name":"query", "arguments":{"query":"sessions"}}})).unwrap();
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
    cancelled_read(false);
}

#[test]
fn mcp_disconnect_interrupts_the_daemon_read() {
    cancelled_read(true);
}

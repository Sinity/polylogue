use super::*;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn cancelled_read(disconnect: bool) {
    let daemon = test_daemon::start("MCP cancellation");
    let _held = daemon.daemon.reads.reader().unwrap();
    let reads = daemon.watch_reads();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let (client, server) = tokio::io::duplex(8192);
    let archive = daemon.daemon.archive.root.clone();
    let service = runtime.spawn(async move {
        let disconnected = tokio_util::sync::CancellationToken::new();
        let (input, output) = tokio::io::split(server);
        let input = Input {
            inner: input,
            disconnected: disconnected.clone(),
        };
        Polylogue {
            archive,
            disconnected,
        }
        .serve((input, output))
        .await
        .unwrap()
        .waiting()
        .await
        .unwrap();
    });
    let mut client = BufReader::new(client);
    runtime.block_on(async {
        client.get_mut().write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"synthetic\",\"version\":\"1\"}}}\n").await.unwrap();
        let mut line = String::new();
        tokio::time::timeout(Duration::from_secs(5), client.read_line(&mut line)).await.unwrap().unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&line).unwrap()["result"].is_object());
        client.get_mut().write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n").await.unwrap();
        let call = json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"sql","arguments":{"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}}});
        client.get_mut().write_all(format!("{call}\n").as_bytes()).await.unwrap();
    });
    let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    test_daemon::TestDaemon::wait_for_sql(&limit);
    if disconnect {
        drop(client);
    } else {
        runtime.block_on(async {
            client.get_mut().write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":2,\"reason\":\"caller stopped\"}}\n").await.unwrap();
        });
        let elapsed = daemon.assert_slot_freed();
        eprintln!("MCP cancellation released reader in {elapsed:?}");
        drop(client);
    }
    let elapsed = daemon.assert_slot_freed();
    assert!(limit.cancellation.is_cancelled());
    eprintln!("MCP disconnect released reader in {elapsed:?}");
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(5), service)
            .await
            .unwrap()
            .unwrap();
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

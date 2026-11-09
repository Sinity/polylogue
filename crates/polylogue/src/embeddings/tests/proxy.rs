use super::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

/// A server that answers one request with 200 and reports whether any
/// connection reached it within the wait.
fn recorder(wait: Duration) -> (String, std::thread::JoinHandle<bool>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let until = Instant::now() + wait;
        loop {
            match listener.accept() {
                Ok((mut peer, _)) => {
                    peer.set_nonblocking(false).unwrap();
                    peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                    let mut head = [0u8; 4096];
                    let _ = peer.read(&mut head);
                    let _ = peer.write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                    );
                    return true;
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < until =>
                {
                    std::thread::sleep(Duration::from_millis(2))
                }
                Err(_) => return false,
            }
        }
    });
    (address, server)
}

/// Sends one request to `target` through a client that starts with `proxy`
/// configured, after the provider's own client settings are applied.
fn send_through(kind: &ProviderKind, proxy: &str, target: &str) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // The client is built inside the runtime, as the provider's own request
    // does: building it outside one panics for want of a reactor.
    runtime.block_on(async {
        let builder = reqwest::Client::builder()
            .proxy(reqwest::Proxy::all(proxy).unwrap())
            .timeout(Duration::from_secs(5));
        let client = client_for(kind, builder).build().unwrap();
        let _ = client.post(format!("{target}/embeddings")).send().await;
    });
}

#[test]
fn the_local_provider_never_goes_through_a_proxy() {
    let (target, target_seen) = recorder(Duration::from_secs(5));
    let (proxy, proxy_seen) = recorder(Duration::from_millis(500));
    send_through(&ProviderKind::Local, &proxy, &target);
    assert!(
        target_seen.join().unwrap(),
        "the loopback server got no request"
    );
    assert!(
        !proxy_seen.join().unwrap(),
        "the local provider's request went through the proxy"
    );
}

#[test]
fn a_remote_provider_keeps_the_configured_proxy() {
    let (target, target_seen) = recorder(Duration::from_millis(500));
    let (proxy, proxy_seen) = recorder(Duration::from_secs(5));
    send_through(&ProviderKind::Voyage, &proxy, &target);
    assert!(
        proxy_seen.join().unwrap(),
        "the remote provider's request skipped the proxy"
    );
    assert!(!target_seen.join().unwrap());
}

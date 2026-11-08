use super::*;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::channel,
};
use std::time::{Duration, Instant};

struct Cleanup {
    stop_peer: Arc<AtomicBool>,
    cancellation: polylogue_store::read::limit::ReadCancellation,
}
impl Drop for Cleanup {
    fn drop(&mut self) {
        self.stop_peer.store(true, Ordering::Release);
        self.cancellation.cancel();
    }
}

fn stalled_provider(cancel: bool) {
    let mut daemon = crate::test_daemon::start("provider cancellation");
    daemon.pause_signals();
    let _held = daemon.daemon.reads.reader().unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let provider = Provider {
        kind: ProviderKind::Local,
        endpoint: format!("http://{}", listener.local_addr().unwrap()),
        model: "synthetic".into(),
        price_per_mtok: Some(0.0),
        credentials: Credentials::new(None),
    };
    let stop_peer = Arc::new(AtomicBool::new(false));
    let cancellation = polylogue_store::read::limit::ReadCancellation::default();
    let (ready, started) = channel();
    std::thread::scope(|scope| {
        // Drop before scoped joins even when setup/assertions fail. No peer
        // remains in accept and no provider worker keeps its reader alive.
        let _cleanup = Cleanup {
            stop_peer: Arc::clone(&stop_peer),
            cancellation: cancellation.clone(),
        };
        let server = scope.spawn(|| {
            let until = Instant::now() + Duration::from_secs(5);
            let stream = loop {
                if stop_peer.load(Ordering::Acquire) {
                    return;
                }
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < until =>
                    {
                        std::thread::sleep(Duration::from_millis(1))
                    }
                    Err(error) => panic!("stalled-provider fixture received no request: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut input = BufReader::new(stream);
            let mut length = 0;
            loop {
                let mut line = String::new();
                assert_ne!(input.read_line(&mut line).unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse::<usize>().unwrap();
                }
            }
            input.read_exact(&mut vec![0; length]).unwrap();
            ready.send(()).unwrap();
            // Only cancellation/deadline can close this stalled exchange:
            // the peer never sends response headers.
            assert_eq!(input.read(&mut [0]).unwrap(), 0);
        });
        let worker = scope.spawn(|| -> Result<_, EmbedError> {
            // Reader initialization is a bounded fixture precondition. It
            // must not consume the active provider request's 500 ms deadline.
            let setup = polylogue_store::read::limit::ReadLimit::new(
                Duration::from_secs(5),
                cancellation.child(),
            );
            let reader = setup.scope(|| -> Result<_, polylogue_store::StoreError> {
                let reader = daemon.daemon.reads.reader()?;
                reader.query("SELECT 1", &[])?;
                Ok(reader)
            })?;
            let limit = polylogue_store::read::limit::ReadLimit::new(
                if cancel {
                    Duration::from_secs(30)
                } else {
                    Duration::from_millis(500)
                },
                cancellation.child(),
            );
            let result = limit.scope(|| provider.embed(&["synthetic".into()], true));
            drop(reader);
            result
        });
        if let Err(error) = started.recv_timeout(Duration::from_secs(5)) {
            let slots = daemon.daemon.reads.checked_out();
            cancellation.cancel();
            let result = worker.join();
            panic!(
                "stalled provider never received the request: readiness={error:?}, worker={result:?}, pool_slots={slots}"
            );
        }
        let stopped = Instant::now();
        if cancel {
            cancellation.cancel();
        }
        let elapsed = daemon.assert_slot_freed();
        let result = worker.join().unwrap();
        if cancel {
            assert!(
                matches!(
                    &result,
                    Err(EmbedError::Store(
                        polylogue_store::StoreError::ReadCancelled
                    ))
                ),
                "{result:?}"
            );
        } else {
            assert!(
                matches!(
                    &result,
                    Err(EmbedError::Store(
                        polylogue_store::StoreError::ReadDeadlineExceeded
                    ))
                ),
                "{result:?}"
            );
        }
        assert!(stopped.elapsed() < Duration::from_secs(2));
        eprintln!("stalled provider released reader in {elapsed:?}");
        server.join().unwrap();
    });
}

#[test]
fn cancellation_closes_a_stalled_provider_and_releases_its_reader() {
    stalled_provider(true);
}

#[test]
fn deadline_closes_a_stalled_provider_and_releases_its_reader() {
    stalled_provider(false);
}

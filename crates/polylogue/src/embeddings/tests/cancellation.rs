use super::*;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

fn stalled_provider(cancel: bool) {
    let daemon = crate::test_daemon::start("provider cancellation");
    let _held = daemon.daemon.reads.reader().unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let provider = Provider {
        kind: ProviderKind::Local,
        endpoint: format!("http://{}", listener.local_addr().unwrap()),
        model: "synthetic".into(),
        price_per_mtok: Some(0.0),
    };
    let limit = polylogue_store::read::limit::ReadLimit::new(
        if cancel {
            Duration::from_secs(30)
        } else {
            Duration::from_millis(500)
        },
        Default::default(),
    );
    let (ready, started) = channel();
    std::thread::scope(|scope| {
        let server = scope.spawn(move || {
            let (stream, _) = listener.accept().unwrap();
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
            // A cancelled provider future must close its connection, even though
            // this peer never sends response headers.
            assert_eq!(input.read(&mut [0]).unwrap(), 0);
        });
        let worker = scope.spawn(|| {
            limit.scope(|| {
                let reader = daemon.daemon.reads.reader().unwrap();
                reader.query("SELECT 1", &[]).unwrap();
                let result = provider.embed(&["synthetic".into()], true);
                drop(reader);
                result
            })
        });
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        let stopped = Instant::now();
        if cancel {
            limit.cancellation.cancel();
        }
        let elapsed = daemon.assert_slot_freed();
        let result = worker.join().unwrap();
        if cancel {
            assert!(matches!(
                result,
                Err(EmbedError::Store(
                    polylogue_store::StoreError::ReadCancelled
                ))
            ));
        } else {
            assert!(matches!(
                result,
                Err(EmbedError::Store(
                    polylogue_store::StoreError::ReadDeadlineExceeded
                ))
            ));
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

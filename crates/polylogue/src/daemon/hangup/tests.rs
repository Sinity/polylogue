use super::*;
use std::time::{Duration, Instant};

fn cancelled_soon(cancellation: &ReadCancellation) -> bool {
    let until = Instant::now() + Duration::from_secs(10);
    while !cancellation.is_cancelled() {
        if Instant::now() > until {
            return false;
        }
        std::thread::yield_now();
    }
    true
}

#[test]
fn a_hang_up_cancels_only_that_connection_and_a_half_close_cancels_nothing() {
    let watcher = Watcher::start().unwrap();
    let (gone, gone_peer) = UnixStream::pair().unwrap();
    let (half, half_peer) = UnixStream::pair().unwrap();
    let (stays, _stays_peer) = UnixStream::pair().unwrap();
    let cancellations: Vec<ReadCancellation> = (0..3).map(|_| Default::default()).collect();
    let _watched: Vec<_> = [&gone, &half, &stays]
        .into_iter()
        .zip(&cancellations)
        .map(|(socket, cancellation)| watcher.watch(socket, cancellation.clone()).unwrap())
        .collect();
    half_peer.shutdown(std::net::Shutdown::Write).unwrap();
    drop(gone_peer);
    assert!(cancelled_soon(&cancellations[0]));
    // A half close is not a hang-up: that client still reads its answer.
    drop(watcher);
    assert!(!cancellations[1].is_cancelled());
    assert!(!cancellations[2].is_cancelled());
    drop(half);
}

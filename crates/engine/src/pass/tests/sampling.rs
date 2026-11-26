use super::*;
use polylogue_store::fault::{self, Action};
use std::sync::mpsc;

/// The thread that hands out parse work prepares each file read and decides
/// plans from the file's source state. With the writer stopped in the middle
/// of a write, both still come, and the state equals what a read answers
/// once the write is done.
#[test]
fn sampling_a_source_does_not_wait_for_a_busy_writer() {
    let home = Home::new();
    home.append(
        "sampled.jsonl",
        &claude_user("u1", "sampled", "stored first"),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let path = home.file("sampled.jsonl");

    fault::clear();
    fault::arm("write.coverage", "test-machine", Action::Hold, 0, 1);
    let sender = engine.writer.sender();
    let busy = std::thread::spawn(move || {
        sender.coverage(polylogue_model::capture::Coverage {
            version: 1,
            machine: "test-machine".into(),
            observed_at: polylogue_model::Timestamp::from_unix_ms(1_790_000_000_000).unwrap(),
            roots: Vec::new(),
        })
    });
    fault::wait_held();
    // Releases a writer this test would otherwise wait on forever.
    let (answered, answer) = mpsc::channel::<()>();
    let guard = std::thread::spawn(move || {
        let timed_out = answer.recv_timeout(Duration::from_secs(30)).is_err();
        if timed_out {
            fault::release();
        }
        timed_out
    });
    let context = engine.source_read_context(
        &path,
        "claude-code",
        polylogue_store::capture_read::ReadRoute::NativeCapture,
    );
    let state = engine.source_snapshot(context.source.clone()).unwrap();
    let still_held = fault::held();
    // The guard is gone once it has released the writer.
    let _ = answered.send(());
    let guard_released = guard.join().unwrap();
    fault::release();
    busy.join().unwrap().unwrap();
    fault::clear();
    assert!(
        still_held && !guard_released,
        "the source state was answered only after the writer finished"
    );

    let written = engine.source_snapshot(context.source.clone()).unwrap();
    assert_eq!(state, written);
    assert_eq!(
        state.status,
        polylogue_store::capture_read::SourceStatus::Ready
    );
    assert!(state.confirmed.is_some(), "{state:?}");
}

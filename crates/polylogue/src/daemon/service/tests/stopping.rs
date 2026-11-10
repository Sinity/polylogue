//! Stopping the daemon while its loop is in the middle of a pass.

use polylogue_model::SessionId;
use polylogue_store::fault::{self, Action};
use std::time::{Duration, Instant};

/// Six sessions written together reach one pass, whose one parse worker
/// takes them largest first. A stop while the second is being parsed lets
/// that parse commit whole and starts none of the rest (plnext-vn6t).
#[test]
fn a_stop_during_a_pass_commits_the_parse_in_hand_and_starts_no_more() {
    let mut daemon = crate::test_daemon::start_with_workers("before the pass", 1);
    let project = daemon.home.join(".claude/projects/-p");
    let names = ["stop-1", "stop-2", "stop-3", "stop-4", "stop-5", "stop-6"];
    fault::clear();
    fault::arm("parse", "stop-2", Action::Hold, 0, 1);
    for (i, name) in names.iter().enumerate() {
        // Each later session is shorter, so the worker takes them in order.
        let text = format!("{name} ").repeat(60 - 5 * i);
        let records: String = (0..3)
            .map(|n| polylogue_testkit::claude_user(&format!("{name}-{n}"), name, &text))
            .collect();
        std::fs::write(project.join(format!("{name}.jsonl")), records).unwrap();
    }
    while !fault::held() {
        std::thread::sleep(Duration::from_millis(1));
    }
    let stop = daemon.engine_stop();
    let releaser = std::thread::spawn(move || {
        while !stop.requested() {
            std::thread::sleep(Duration::from_millis(1));
        }
        fault::release();
    });
    let started = Instant::now();
    daemon.stop().unwrap();
    let took = started.elapsed();
    releaser.join().unwrap();
    fault::clear();
    // docs/operations.md: a stop within 5 s.
    assert!(took < Duration::from_secs(5), "stopping took {took:?}");

    let (archive, _lock) =
        polylogue_store::Archive::open_or_create(&daemon._dir.path().join("archive")).unwrap();
    let reads = polylogue_store::ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    for (i, name) in names.iter().enumerate() {
        let session = SessionId::parse(&format!("claude-code:{name}")).unwrap();
        let messages = reader.messages(&session, false).unwrap();
        let expected = if i < 2 { 3 } else { 0 };
        assert_eq!(
            messages.len(),
            expected,
            "{name}: the parse in hand and the one before it commit whole; none after starts"
        );
    }
}

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn selected_commits_queue_one_wake_until_the_subscriber_reads_them() {
    let events = CommitEvents::default();
    let wakes = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&wakes);
    let subscription = events.subscribe_with_wake(Committed::CONTENT, move || {
        observed.fetch_add(1, Ordering::SeqCst);
    });
    events.committed(Committed::AUTHORED);
    events.committed(Committed::QUERY_RECORDS);
    assert_eq!(wakes.load(Ordering::SeqCst), 0);
    assert!(matches!(
        subscription.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    for _ in 0..20 {
        events.committed(Committed::CONTENT);
    }
    assert_eq!(wakes.load(Ordering::SeqCst), 1);
    assert_eq!(subscription.try_recv().unwrap(), Committed::CONTENT);
    events.committed(Committed::CONTENT);
    assert_eq!(wakes.load(Ordering::SeqCst), 2);
    assert_eq!(subscription.try_recv().unwrap(), Committed::CONTENT);
    drop(subscription);
    events.committed(Committed::CONTENT);
    assert_eq!(wakes.load(Ordering::SeqCst), 2);
}

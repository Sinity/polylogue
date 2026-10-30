use super::*;
use shared_signals::notifications::{Notice, Route};
use signal_delivery::AttemptContext;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

fn fixture() -> crate::test_daemon::TestDaemon {
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(
            jiff::Timestamp::from_millisecond(1_792_332_000_000).unwrap(),
        ),
    );
    fixture.pause_signals();
    fixture
}

fn request(notice: Notice) -> PolicyRequest {
    PolicyRequest {
        delivery_id: notice.delivery_id.clone(),
        route: Route::Dbus,
        notices: vec![notice],
        attempt: AttemptContext::new(Duration::from_secs(60), None, CancellationToken::new()),
    }
}

fn seed(d: &Daemon) -> Notice {
    let outbox = crate::signals::tests::outbox("deadline-caller");
    let notice = outbox.metadata.notice.clone();
    let context = operation(d).unwrap();
    let reader = d.reads.reader().unwrap();
    let attention = local_attention(&reader, &context).unwrap();
    drop(reader);
    let previous = load(d).unwrap();
    let prior = previous
        .as_ref()
        .map(Checkpoint::identity)
        .transpose()
        .unwrap();
    let checkpoint = previous.unwrap_or_else(|| initial(&context, Some(&attention)).unwrap());
    // This fixture only records an outbox. It supplies no assertion closure,
    // inferred rule revision, actor privilege or native eligibility proof.
    assert!(
        persist(d, &context, prior, checkpoint, vec![outbox], vec![])
            .unwrap()
            .1
    );
    notice
}

#[tokio::test]
async fn unavailable_canonical_rule_proof_or_outbox_cannot_become_denial_or_permission() {
    let fixture = fixture();
    let notice = seed(&fixture.daemon);
    assert!(matches!(
        policy(fixture.daemon.clone())(request(notice)).await,
        Err(PolicyFailure::Unavailable)
    ));
    let absent = crate::signals::tests::outbox("absent").metadata.notice;
    assert!(matches!(
        policy(fixture.daemon.clone())(request(absent)).await,
        Err(PolicyFailure::Unavailable)
    ));
}

#[tokio::test]
async fn closed_store_is_operational_failure_and_interruption_keeps_its_typed_reason() {
    let fixture = fixture();
    let notice = seed(&fixture.daemon);
    fixture.daemon.reads.close();
    assert!(matches!(
        policy(fixture.daemon.clone())(request(notice.clone())).await,
        Err(PolicyFailure::Unavailable)
    ));
    let cancelled = request(notice.clone());
    cancelled.attempt.cancellation().cancel();
    assert!(matches!(
        policy(fixture.daemon.clone())(cancelled).await,
        Err(PolicyFailure::Interrupted(Interrupted::Cancelled))
    ));
    let mut expired = request(notice);
    expired.attempt = AttemptContext::new(
        Duration::from_secs(60),
        Some(tokio::time::Instant::now()),
        CancellationToken::new(),
    );
    assert!(matches!(
        policy(fixture.daemon.clone())(expired).await,
        Err(PolicyFailure::Interrupted(Interrupted::Deadline))
    ));
}

#[tokio::test]
async fn actual_native_stop_cancels_scoped_pool_wait_and_observes_blocking_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = polylogue_store::fixture::Fixture::new(directory.path()).unwrap();
    let pool =
        Arc::new(polylogue_store::ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap());
    let held = pool.reader().unwrap();
    let (stop, stopped) = std::sync::mpsc::channel();
    let mut bridge = StopBridge::new(stopped).unwrap();
    let attempt = AttemptContext::new(Duration::from_secs(60), None, bridge.token());
    let (entered, wait) = tokio::sync::oneshot::channel();
    let cleaned = Arc::new(AtomicBool::new(false));
    struct Cleanup(Arc<AtomicBool>);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let read_pool = pool.clone();
    let read_cleaned = cleaned.clone();
    let read = tokio::spawn(read_admission(attempt.clone(), move || {
        let _cleanup = Cleanup(read_cleaned);
        entered.send(()).unwrap();
        read_pool
            .reader()
            .map(drop)
            .map_err(|_| unavailable(&attempt))
    }));
    wait.await.unwrap();
    stop.send(()).unwrap();
    assert!(matches!(
        read.await.unwrap(),
        Err(PolicyFailure::Interrupted(Interrupted::Cancelled))
    ));
    assert!(
        cleaned.load(Ordering::Acquire),
        "owned blocking cleanup completed before the result"
    );
    drop(held);
    assert!(pool.reader().is_ok());
    bridge.finish().unwrap();
}

#[tokio::test]
async fn queued_or_disconnected_stop_prevents_signal_startup_work() {
    let fixture = fixture();
    let before = load(&fixture.daemon)
        .unwrap()
        .map(|checkpoint| checkpoint.identity().unwrap());
    for queued in [false, true] {
        let (stop, stopped) = std::sync::mpsc::channel();
        if queued {
            stop.send(()).unwrap();
        }
        drop(stop);
        let mut bridge = StopBridge::new(stopped).unwrap();
        assert!(bridge.token().is_cancelled());
        run(fixture.daemon.clone(), bridge.token()).await;
        bridge.finish().unwrap();
        assert_eq!(
            load(&fixture.daemon)
                .unwrap()
                .map(|checkpoint| checkpoint.identity().unwrap()),
            before
        );
    }
}

#[tokio::test]
async fn stopping_drains_submitted_unknown_ack_before_joining_the_worker() {
    let fixture = fixture();
    let notice = seed(&fixture.daemon);
    let id = notice.delivery_id.clone();
    let mut state = NotificationState::default();
    let receipts = state.begin(notice, Route::Dbus);
    let (updates, mut returned) = mpsc::unbounded_channel();
    let (stop, stopped) = std::sync::mpsc::channel();
    let mut bridge = StopBridge::new(stopped).unwrap();
    let (submitted, wait) = tokio::sync::oneshot::channel();
    let daemon = fixture.daemon.clone();
    let worker = tokio::spawn(async move {
        let (persisted, acknowledged) = tokio::sync::oneshot::channel();
        updates
            .send(NotificationUpdate {
                receipts,
                state,
                released: None,
                persisted,
            })
            .unwrap();
        submitted.send(()).unwrap();
        acknowledged.await.unwrap();
        assert_eq!(
            load(&daemon)
                .unwrap()
                .unwrap()
                .notifications
                .unwrap()
                .attempt(&id)
                .unwrap()
                .dbus,
            Some(shared_signals::DeliveryOutcome::Unknown)
        );
        Ok::<(), zbus::Error>(())
    });
    wait.await.unwrap();
    stop.send(()).unwrap();
    bridge.token().cancelled().await;
    drain_worker(&fixture.daemon, &mut returned, worker)
        .await
        .unwrap();
    bridge.finish().unwrap();
}

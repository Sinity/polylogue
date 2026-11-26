use super::*;
use crate::signals::tests::operation;
use shared_signals::notifications::Notice;

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

fn seed(d: &Daemon) -> Notice {
    let pending_delivery = crate::signals::tests::pending_delivery("deadline-caller");
    let notice = pending_delivery.metadata.notice.clone();
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
    let checkpoint = previous.unwrap_or_else(|| initial(d, &context, Some(&attention)).unwrap());
    // This fixture only records a pending delivery; no rule fired it.
    assert!(
        persist(
            d,
            &context,
            prior,
            checkpoint,
            vec![pending_delivery],
            vec![]
        )
        .unwrap()
    );
    notice
}

#[tokio::test]
async fn a_stop_made_before_startup_prevents_signal_startup_work() {
    let fixture = fixture();
    let before = load(&fixture.daemon)
        .unwrap()
        .map(|checkpoint| checkpoint.identity().unwrap());
    let stop = CancellationToken::new();
    stop.cancel();
    run(fixture.daemon.clone(), stop).await.unwrap();
    assert_eq!(
        load(&fixture.daemon)
            .unwrap()
            .map(|checkpoint| checkpoint.identity().unwrap()),
        before
    );
}

#[tokio::test]
async fn stopping_drains_submitted_unknown_ack_before_joining_the_worker() {
    let fixture = fixture();
    let notice = seed(&fixture.daemon);
    let id = notice.delivery_id.clone();
    let mut state = NotificationState::default();
    let receipts = state.begin(notice);
    let (updates, mut returned) = mpsc::unbounded_channel();
    let (submitted, wait) = tokio::sync::oneshot::channel();
    let daemon = fixture.daemon.clone();
    let worker = tokio::spawn(async move {
        let (persisted, acknowledged) = tokio::sync::oneshot::channel();
        updates
            .send(NotificationUpdate {
                receipts,
                state,
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
                .desktop,
            Some(shared_signals::DeliveryOutcome::Unknown)
        );
        Ok::<(), zbus::Error>(())
    });
    wait.await.unwrap();
    drain_worker(&fixture.daemon, &mut returned, worker)
        .await
        .unwrap();
}

#[tokio::test]
async fn closed_startup_reader_is_no_failure_when_work_stops() {
    let mut fixture = fixture();
    fixture.stop().unwrap();
    // The daemon's stop already ended reads, but this worker's own token
    // is not cancelled. No new restoration starts.
    run(fixture.daemon.clone(), CancellationToken::new())
        .await
        .unwrap();
}

#[tokio::test]
async fn notification_cleanup_preserves_the_delivery_workers_error() {
    let fixture = fixture();
    let (updates, mut returned) = mpsc::unbounded_channel();
    let worker = tokio::spawn(async move {
        drop(updates);
        Err::<(), _>(zbus::Error::Failure("synthetic delivery failure".into()))
    });
    let error = drain_worker(&fixture.daemon, &mut returned, worker)
        .await
        .unwrap_err();
    assert_eq!(error.reason, "desktop worker: synthetic delivery failure");
}

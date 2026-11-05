use super::*;
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
    let checkpoint = previous.unwrap_or_else(|| initial(d, &context, Some(&attention)).unwrap());
    // This fixture only records an outbox; no rule fired it.
    assert!(
        persist(d, &context, prior, checkpoint, vec![outbox], vec![])
            .unwrap()
            .1
    );
    notice
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
    let receipts = state.begin(notice);
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
    stop.send(()).unwrap();
    bridge.token().cancelled().await;
    drain_worker(&fixture.daemon, &mut returned, worker)
        .await
        .unwrap();
    bridge.finish().unwrap();
}

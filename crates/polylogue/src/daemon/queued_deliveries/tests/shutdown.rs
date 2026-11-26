use super::*;
use polylogue_model::user::{Delivery, UserWrite};
use polylogue_store::fault::{self, Action};

fn prepared_outcome(daemon: &crate::test_daemon::TestDaemon, prefix: &str) -> UserRequest {
    let invocation =
        daemon
            .daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "shutdown-outcome",
            ));
    let id = format!("{prefix}-delivery");
    daemon
        .daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .user(invocation.request(
            id.clone(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "session".into(),
                recipient_id: Some("claude-code:s1".into()),
                sender: polylogue_model::DeliverySender::Hook,
                event: polylogue_model::DeliveryEvent::SessionStart,
                cwd: None,
                text: "synthetic stop brief".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ))
        .unwrap();
    invocation.request(
        format!("{prefix}-outcome"),
        UserWrite::DeliveryEvent {
            delivery: id,
            what: polylogue_model::DeliveryOutcome::Delivered,
            detail: Some("accepted by the synthetic receiver".into()),
        },
    )
}

#[test]
fn a_queued_outcome_survives_loop_stop_and_writer_queue_pressure() {
    let mut daemon = crate::test_daemon::start("stop queued outcome fixture");
    let outcome = prepared_outcome(&daemon, "stop-queued");
    let invocation =
        daemon
            .daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "shutdown-outcome",
            ));
    let barrier = invocation.request(
        "stop-outcome-barrier".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    );
    let writes = daemon.daemon.writes.as_ref().unwrap();
    writes.archive().user(barrier.clone()).unwrap();
    fault::arm("write.user", "stop-outcome-barrier", Action::Hold, 0, 1);
    let _release = daemon.daemon.archive.stop().on_cancel(fault::release);
    assert!(matches!(
        writes.archive().try_submit(barrier.clone()).unwrap(),
        Queued::Committing(_)
    ));
    assert!(
        fault::wait_held_within(std::time::Duration::from_secs(30)),
        "no thread reached the held point"
    );
    // Fill the bounded public writer queue with repeats that change no user state.
    while matches!(
        writes.archive().try_submit(barrier.clone()).unwrap(),
        Queued::Committing(_)
    ) {}
    daemon
        .daemon
        .queued_deliveries
        .submit(writes, outcome, None, None)
        .unwrap();
    assert_eq!(daemon.daemon.queued_deliveries.pending_outcomes().0, 1);
    assert!(
        !daemon
            .daemon
            .queued_deliveries
            .0
            .locked()
            .waiting
            .is_empty()
    );
    let root = daemon.daemon.archive.root.clone();
    daemon.stop().unwrap();
    fault::clear();
    let (archive, _lock) = polylogue_store::Archive::open_or_create(&root).unwrap();
    let reads = polylogue_store::ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    let (delivery, text) = reader.delivery("stop-queued-delivery").unwrap().unwrap();
    assert_eq!(delivery.events, vec!["delivered"]);
    assert_eq!(text, "synthetic stop brief");
}

#[test]
fn an_unwritten_final_outcome_is_a_shutdown_error_with_its_request_kept() {
    let mut daemon = crate::test_daemon::start("failed final outcome fixture");
    let outcome = prepared_outcome(&daemon, "stop-failed");
    fault::arm("write.user", "stop-failed-outcome", Action::Fail, 0, 100);
    daemon
        .daemon
        .queued_deliveries
        .submit(
            daemon.daemon.writes.as_ref().unwrap(),
            outcome.clone(),
            None,
            None,
        )
        .unwrap();
    let error = daemon.stop().unwrap_err().to_string();
    assert!(error.contains("delivery outcome drain"), "{error}");
    assert!(error.contains("write.user"), "{error}");
    let (count, failure) = daemon.daemon.queued_deliveries.pending_outcomes();
    assert_eq!(count, 1);
    assert!(failure.unwrap().contains("write.user"));
    let state = daemon.daemon.queued_deliveries.0.locked();
    assert_eq!(state.waiting.front().unwrap().request, outcome);
    fault::clear();
}

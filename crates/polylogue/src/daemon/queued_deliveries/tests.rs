use super::*;
mod shutdown;

#[test]
fn a_failed_outcome_write_is_retried_and_reported() {
    use polylogue_model::user::{Delivery, UserWrite};
    use polylogue_store::fault::{self, Action};
    let daemon = crate::test_daemon::start("delivery retry fixture");
    let writes = daemon.daemon.writes.as_ref().unwrap();
    let invocation =
        daemon
            .daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "outcome-retry",
            ));
    let delivery = invocation.request(
        "retry-delivery".into(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some("claude-code:s1".into()),
            sender: "hook".into(),
            event: "session_start".into(),
            cwd: None,
            text: "synthetic brief".into(),
            refs: vec![],
            left_out: vec![],
        }),
    );
    writes.user(delivery).unwrap();
    let barrier = invocation.request(
        "outcome-status-barrier".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    );
    writes.user(barrier.clone()).unwrap();
    let outcome = invocation.request(
        "retry-outcome".into(),
        UserWrite::DeliveryEvent {
            delivery: "retry-delivery".into(),
            what: polylogue_model::DeliveryOutcome::Delivered,
            detail: Some("the synthetic receiver accepted it".into()),
        },
    );
    fault::arm("write.user", "retry-outcome", Action::Fail, 0, 1);
    fault::arm("write.user", "outcome-status-barrier", Action::Hold, 0, 1);
    // Keep the retry queue locked until the following writer request is queued.
    // Its hold makes the failed acknowledgment observable before any retry can commit.
    let held = {
        let mut state = daemon.daemon.queued_deliveries.0.locked();
        submit(&mut state, writes, outcome.clone(), None, None).unwrap();
        let Queued::Committing(held) = writes.try_enqueue_user(barrier.clone()).unwrap() else {
            panic!("the fixture writer queue has room")
        };
        held
    };
    let _release = daemon.daemon.archive.stop().on_cancel(fault::release);
    while !fault::held() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let status = crate::protocol::into_value(
        crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&daemon.daemon),
    )
    .unwrap();
    fault::release();
    assert_eq!(status["pending_writes"]["delivery_outcomes"], 1);
    assert!(
        status["pending_writes"]["last_error"]
            .as_str()
            .unwrap()
            .contains("write.user")
    );
    held.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    // A normal commit gives the scope worker its next bounded hand-over turn.
    writes
        .user(invocation.request(
            "outcome-retry-wake".into(),
            UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: "on".into(),
                machine: None,
            },
        ))
        .unwrap();
    daemon.wait_for(|status| status["pending_writes"]["delivery_outcomes"] == 0);
    assert_eq!(
        daemon.daemon.queued_deliveries.pending_outcomes(),
        (0, None)
    );
    daemon
        .daemon
        .queued_deliveries
        .submit(writes, outcome, None, None)
        .unwrap();
    writes.user(barrier).unwrap();
    assert_eq!(
        daemon.daemon.queued_deliveries.pending_outcomes(),
        (0, None)
    );
    let reader = daemon.daemon.reads.reader().unwrap();
    let (delivery, body) = reader.delivery("retry-delivery").unwrap().unwrap();
    assert_eq!(delivery.events, vec!["delivered"]);
    assert_eq!(body, "synthetic brief");
    fault::clear();
}

#[test]
fn captured_advice_remains_pending_until_current_history_confirms_application() {
    let queued = QueuedDeliveries::default();
    let (reply, committed) = std::sync::mpsc::sync_channel(1);
    queued.0.lock().unwrap().advice.insert(
        "advice-request".into(),
        PendingAdvice {
            recipient: "codex:recipient".into(),
            refs: vec![
                "assertion:root@revision".into(),
                "file:src/parser.rs@commit".into(),
            ],
            committed: Some(committed),
        },
    );
    reply
        .send(Ok(polylogue_store::user_writes::UserReceipt {
            outcome: polylogue_model::user::UserOutcome {
                id: "advice-request".into(),
                repeat: false,
                pending: true,
            },
            save: None,
            written: Vec::new(),
            already: Vec::new(),
        }))
        .unwrap();
    let first = queued.pending_advice("codex:recipient").unwrap();
    assert_eq!(
        first.len(),
        1,
        "a capture acknowledgment is not applied delivery history"
    );
    drop(reply);
    assert_eq!(
        queued.pending_advice("codex:recipient").unwrap(),
        first,
        "a consumed capture acknowledgment cannot end the pending hold"
    );
    queued.advice_applied(&["advice-request".into()]);
    assert!(queued.pending_advice("codex:recipient").unwrap().is_empty());
}

#[test]
fn advice_whose_hand_over_fails_is_not_left_pending() {
    use polylogue_model::user::{Delivery, UserWrite};
    let mut daemon = crate::test_daemon::start("advice hand-over failure fixture");
    let invocation =
        daemon
            .daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "advice-failure",
            ));
    let advice = invocation.request(
        "advice-request".into(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some("claude-code:s1".into()),
            sender: "hook".into(),
            event: polylogue_model::DeliveryEvent::Advice.as_str().into(),
            cwd: None,
            text: "synthetic advice".into(),
            refs: vec!["assertion:root@revision".into()],
            left_out: vec![],
        }),
    );
    // The writer has stopped, so the hand-over cannot succeed.
    daemon.stop().unwrap();
    let writes = daemon.daemon.writes.as_ref().unwrap();
    let error = daemon
        .daemon
        .queued_deliveries
        .submit(writes, advice, None, None)
        .unwrap_err();
    assert!(matches!(error, StoreError::WriterGone), "{error}");
    assert_eq!(
        daemon
            .daemon
            .queued_deliveries
            .pending_advice("claude-code:s1")
            .unwrap(),
        Vec::<(String, Vec<String>)>::new()
    );
}

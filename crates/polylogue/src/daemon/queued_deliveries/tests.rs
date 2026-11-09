use super::*;

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
        .send(Ok(UserOutcome {
            id: "advice-request".into(),
            repeat: false,
            pending: true,
        }))
        .unwrap();
    let first = queued.pending_advice("codex:recipient").unwrap();
    assert_eq!(
        first.len(),
        1,
        "capture ACK is not applied delivery history"
    );
    drop(reply);
    assert_eq!(
        queued.pending_advice("codex:recipient").unwrap(),
        first,
        "a consumed capture ACK cannot retire pending ownership"
    );
    queued.advice_applied(&["advice-request".into()]);
    assert!(queued.pending_advice("codex:recipient").unwrap().is_empty());
}

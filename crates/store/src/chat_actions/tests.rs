use super::*;
use crate::{Archive, ReadPool, Writer};
use polylogue_model::user::CHAT_CLAIM_MS;
use polylogue_model::user::UserWrite;

fn setup() -> (
    tempfile::TempDir,
    Archive,
    crate::ArchiveLock,
    Writer,
    ReadPool,
) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let reads = ReadPool::open(&archive).unwrap();
    (dir, archive, lock, writer, reads)
}

fn write(writer: &Writer, write: UserWrite) -> Result<polylogue_model::user::UserOutcome> {
    writer.user(crate::fixture::authored::operator_request(
        crate::user_writes::new_request_id(),
        write,
    ))
}

fn new(id: &str, key: &str, needs_approval: bool) -> NewChatAction {
    NewChatAction {
        id: id.into(),
        request_key: key.into(),
        request_hash: BlobHash::of(key.as_bytes()),
        request: serde_json::json!({"text": key}),
        stage_only: !needs_approval,
        needs_approval,
        attachments: vec![],
    }
}

#[test]
fn one_action_is_out_at_a_time_and_a_resent_request_is_the_same_action() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let first = write(&writer, UserWrite::ChatEnqueue(new("a1", "k1", false))).unwrap();
    assert_eq!((first.id.as_str(), first.repeat), ("a1", false));
    let again = write(&writer, UserWrite::ChatEnqueue(new("a9", "k1", false))).unwrap();
    assert_eq!((again.id.as_str(), again.repeat), ("a1", true));
    let mut other = new("a9", "k1", false);
    other.request_hash = BlobHash::of(b"other content");
    assert!(matches!(
        write(&writer, UserWrite::ChatEnqueue(other)),
        Err(StoreError::ChatAction("idempotency_key_reused"))
    ));
    write(&writer, UserWrite::ChatEnqueue(new("a2", "k2", false))).unwrap();

    let claimed = write(&writer, UserWrite::ChatClaim { by: "tab-1".into() }).unwrap();
    assert_eq!(claimed.id, "a1");
    assert!(matches!(
        write(&writer, UserWrite::ChatClaim { by: "tab-2".into() }),
        Err(StoreError::NotFound(_))
    ));
    let report = |by: &str, outcome: &str| {
        write(
            &writer,
            UserWrite::ChatReport {
                action: "a1".into(),
                by: by.into(),
                outcome: outcome.into(),
                phase: "composer".into(),
                detail: None,
                receipt: None,
            },
        )
    };
    assert!(matches!(
        report("tab-2", "progress"),
        Err(StoreError::ChatAction("not_the_claimant"))
    ));
    assert!(matches!(
        report("tab-1", "submitted"),
        Err(StoreError::ChatAction("stage_only_action_cannot_submit"))
    ));
    report("tab-1", "drafted").unwrap();
    assert_eq!(
        write(&writer, UserWrite::ChatClaim { by: "tab-2".into() })
            .unwrap()
            .id,
        "a2"
    );
    let r = reads.reader().unwrap();
    let a1 = r.chat_action("a1").unwrap().unwrap();
    assert_eq!(a1.status(i64::MAX / 2).0, ChatStatus::Drafted);
}

#[test]
fn an_expired_claim_returns_the_action_to_the_queue() {
    let action = ChatAction {
        id: "a".into(),
        request_key: "k".into(),
        request_hash: BlobHash::of(b"k"),
        request: serde_json::Value::Null,
        stage_only: true,
        needs_approval: false,
        created_at: 0,
        execution: polylogue_model::authored::ExecutionEligibility::Live,
        attachments: vec![],
        events: vec![ChatEvent {
            request: "event-request".into(),
            at: 1_000,
            kind: "claimed".into(),
            phase: "claimed".into(),
            detail: None,
            by: Some("tab".into()),
            receipt: None,
        }],
    };
    assert_eq!(
        action.status(1_000 + CHAT_CLAIM_MS),
        (ChatStatus::Claimed, Some("tab"))
    );
    assert_eq!(
        action.status(1_001 + CHAT_CLAIM_MS),
        (ChatStatus::Queued, None)
    );
}

#[test]
fn a_held_submit_waits_for_the_operator() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    write(&writer, UserWrite::ChatEnqueue(new("a1", "k1", true))).unwrap();
    assert!(matches!(
        write(&writer, UserWrite::ChatClaim { by: "tab".into() }),
        Err(StoreError::NotFound(_))
    ));
    let decide = |approve| {
        write(
            &writer,
            UserWrite::ChatDecide {
                action: "a1".into(),
                approve,
                by: "operator".into(),
                detail: None,
            },
        )
    };
    decide(true).unwrap();
    assert!(matches!(
        decide(false),
        Err(StoreError::ChatAction("not_awaiting_approval"))
    ));
    assert_eq!(
        write(&writer, UserWrite::ChatClaim { by: "tab".into() })
            .unwrap()
            .id,
        "a1"
    );
    let r = reads.reader().unwrap();
    let kinds: Vec<String> = r
        .chat_action("a1")
        .unwrap()
        .unwrap()
        .events
        .into_iter()
        .map(|e| e.kind)
        .collect();
    assert_eq!(kinds, ["awaiting_approval", "approved", "claimed"]);
}

fn write_at(
    writer: &Writer,
    at: i64,
    write: UserWrite,
) -> Result<polylogue_model::user::UserOutcome> {
    let mut request =
        crate::fixture::authored::operator_request(crate::user_writes::new_request_id(), write);
    request.at = polylogue_model::Timestamp::from_unix_ms(at).unwrap();
    writer.user(request)
}

#[test]
fn an_expired_sending_claim_is_unknown_and_never_handed_out_again() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let mut action = new("sending", "sending-key", false);
    action.stage_only = false;
    write_at(&writer, 0, UserWrite::ChatEnqueue(action)).unwrap();
    write_at(&writer, 1, UserWrite::ChatClaim { by: "tab-a".into() }).unwrap();
    for (at, phase) in [(2, "sending"), (3, "claimed"), (4, "capturing")] {
        write_at(
            &writer,
            at,
            UserWrite::ChatReport {
                action: "sending".into(),
                by: "tab-a".into(),
                outcome: "progress".into(),
                phase: phase.into(),
                detail: None,
                receipt: None,
            },
        )
        .unwrap();
    }
    let expired = 5 + CHAT_CLAIM_MS;
    let stored = reads
        .reader()
        .unwrap()
        .chat_action("sending")
        .unwrap()
        .unwrap();
    assert_eq!(stored.status(expired), (ChatStatus::OutcomeUnknown, None));
    for by in ["tab-a", "tab-b"] {
        assert!(matches!(
            write_at(&writer, expired, UserWrite::ChatClaim { by: by.into() }),
            Err(StoreError::NotFound(_))
        ));
        for outcome in ["progress", "outcome_unknown"] {
            assert!(matches!(
                write_at(
                    &writer,
                    expired,
                    UserWrite::ChatReport {
                        action: "sending".into(),
                        by: by.into(),
                        outcome: outcome.into(),
                        phase: "claimed".into(),
                        detail: None,
                        receipt: None,
                    }
                ),
                Err(StoreError::ChatAction("not_the_claimant"))
            ));
        }
    }
    write_at(
        &writer,
        expired,
        UserWrite::ChatEnqueue(new("next", "next-key", false)),
    )
    .unwrap();
    assert_eq!(
        write_at(
            &writer,
            expired,
            UserWrite::ChatClaim { by: "tab-b".into() }
        )
        .unwrap()
        .id,
        "next"
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .chat_action("sending")
            .unwrap()
            .unwrap()
            .events,
        stored.events,
    );
}

#[test]
fn an_expired_presend_chat_claim_can_be_reclaimed_without_accepting_stale_reports() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    write_at(
        &writer,
        0,
        UserWrite::ChatEnqueue(new("draft", "draft-key", false)),
    )
    .unwrap();
    write_at(&writer, 1, UserWrite::ChatClaim { by: "tab-a".into() }).unwrap();
    let at = 2 + CHAT_CLAIM_MS;
    assert_eq!(
        write_at(&writer, at, UserWrite::ChatClaim { by: "tab-b".into() })
            .unwrap()
            .id,
        "draft"
    );
    assert!(matches!(
        write_at(
            &writer,
            at,
            UserWrite::ChatReport {
                action: "draft".into(),
                by: "tab-a".into(),
                outcome: "progress".into(),
                phase: "sending".into(),
                detail: None,
                receipt: None,
            }
        ),
        Err(StoreError::ChatAction("not_the_claimant"))
    ));
    write_at(
        &writer,
        at,
        UserWrite::ChatReport {
            action: "draft".into(),
            by: "tab-b".into(),
            outcome: "drafted".into(),
            phase: "drafted".into(),
            detail: None,
            receipt: None,
        },
    )
    .unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .chat_action("draft")
            .unwrap()
            .unwrap()
            .status(at + CHAT_CLAIM_MS),
        (ChatStatus::Drafted, None),
    );
}

#[test]
fn only_the_original_sending_owner_can_late_confirm_the_captured_turn() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let mut action = new("late", "late-key", true);
    action.request["conversation"] = serde_json::json!("conversation-1");
    write_at(&writer, 0, UserWrite::ChatEnqueue(action)).unwrap();
    write_at(
        &writer,
        1,
        UserWrite::ChatDecide {
            action: "late".into(),
            approve: true,
            by: "operator".into(),
            detail: None,
        },
    )
    .unwrap();
    write_at(&writer, 2, UserWrite::ChatClaim { by: "tab-a".into() }).unwrap();
    for (at, phase) in [(3, "sending"), (4, "claimed")] {
        write_at(
            &writer,
            at,
            UserWrite::ChatReport {
                action: "late".into(),
                by: "tab-a".into(),
                outcome: "progress".into(),
                phase: phase.into(),
                detail: None,
                receipt: None,
            },
        )
        .unwrap();
    }
    let at = 5 + CHAT_CLAIM_MS;
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .chat_action("late")
            .unwrap()
            .unwrap()
            .status(at),
        (ChatStatus::OutcomeUnknown, None)
    );
    let report = |by: &str, outcome: &str, receipt| {
        write_at(
            &writer,
            at,
            UserWrite::ChatReport {
                action: "late".into(),
                by: by.into(),
                outcome: outcome.into(),
                phase: outcome.into(),
                detail: None,
                receipt,
            },
        )
    };
    for receipt in [
        None,
        Some(serde_json::json!({})),
        Some(serde_json::json!({"conversation":"", "turn":"captured-turn"})),
        Some(serde_json::json!({"conversation":"conversation-1", "turn":" "})),
    ] {
        assert!(matches!(
            report("tab-a", "submitted", receipt),
            Err(StoreError::ChatAction("submission_receipt_required"))
        ));
    }
    assert!(matches!(
        report(
            "tab-a",
            "submitted",
            Some(serde_json::json!({"conversation":"another", "turn":"captured-turn"}))
        ),
        Err(StoreError::ChatAction("submission_conversation_mismatch"))
    ));
    let receipt = serde_json::json!({"conversation":"conversation-1", "turn":"captured-turn"});
    assert!(matches!(
        report("tab-b", "submitted", Some(receipt.clone())),
        Err(StoreError::ChatAction("not_the_claimant"))
    ));
    assert!(matches!(
        report("tab-a", "progress", Some(receipt.clone())),
        Err(StoreError::ChatAction("not_the_claimant"))
    ));
    for by in ["tab-a", "tab-b"] {
        assert!(matches!(
            write_at(&writer, at, UserWrite::ChatClaim { by: by.into() }),
            Err(StoreError::NotFound(_))
        ));
    }
    report("tab-a", "submitted", Some(receipt.clone())).unwrap();
    let stored = reads
        .reader()
        .unwrap()
        .chat_action("late")
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.status(at + CHAT_CLAIM_MS),
        (ChatStatus::Submitted, None)
    );
    assert_eq!(stored.events.last().unwrap().receipt, Some(receipt));
    assert!(matches!(
        report("tab-a", "outcome_unknown", None),
        Err(StoreError::ChatAction("action_finished"))
    ));
}

#[test]
fn late_confirmation_never_grants_a_draft_only_action_permission_to_submit() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    write_at(
        &writer,
        0,
        UserWrite::ChatEnqueue(new("draft-only", "draft-only-key", false)),
    )
    .unwrap();
    write_at(&writer, 1, UserWrite::ChatClaim { by: "tab-a".into() }).unwrap();
    write_at(
        &writer,
        2,
        UserWrite::ChatReport {
            action: "draft-only".into(),
            by: "tab-a".into(),
            outcome: "progress".into(),
            phase: "sending".into(),
            detail: None,
            receipt: None,
        },
    )
    .unwrap();
    let at = 3 + CHAT_CLAIM_MS;
    assert!(matches!(
        write_at(
            &writer,
            at,
            UserWrite::ChatReport {
                action: "draft-only".into(),
                by: "tab-a".into(),
                outcome: "submitted".into(),
                phase: "submitted".into(),
                detail: None,
                receipt: Some(
                    serde_json::json!({"conversation":"conversation", "turn":"captured-turn"})
                ),
            }
        ),
        Err(StoreError::ChatAction("stage_only_action_cannot_submit"))
    ));
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .chat_action("draft-only")
            .unwrap()
            .unwrap()
            .status(at),
        (ChatStatus::OutcomeUnknown, None)
    );
}

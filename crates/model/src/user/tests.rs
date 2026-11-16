use super::*;
#[test]
fn request_uses_the_supplied_clock_and_machine() {
    let metadata = RequestMetadata::new(
        "synthetic-machine".into(),
        crate::WallClock::fixed("2026-03-08T12:34:56.123456789Z".parse().unwrap()),
    );
    let request = metadata.request(
        "synthetic-request".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
        Authorship {
            author: crate::Author::Operator,
            via: crate::authorship::Channel::Cli,
        },
    );
    assert_eq!(request.machine, "synthetic-machine");
    assert_eq!(
        jiff::Timestamp::from(request.at),
        "2026-03-08T12:34:56.123456789Z"
            .parse::<jiff::Timestamp>()
            .unwrap()
    );
}

#[test]
fn write_intent_refuses_caller_supplied_identity() {
    for write in [
        serde_json::json!({"write":"withdraw", "note":"root", "author":"operator"}),
        serde_json::json!({"write":"set_setting", "key":"browser.actions", "value":"on", "machine":null, "author":"operator"}),
        serde_json::json!({"write":"add_note", "note":{"kind":"note", "value":"text", "targets":["repo:synthetic"], "author":"operator"}}),
    ] {
        assert!(serde_json::from_value::<UserWrite>(write).is_err());
    }
}

#[test]
fn revision_bases_are_pinned_assertion_references() {
    let write = UserWrite::ReviseAssertions {
        bases: vec![crate::RevisionReference::exact(
            "family".into(),
            "revision".into(),
        )],
        patch: Default::default(),
    };
    let encoded = serde_json::to_value(&write).unwrap();
    assert_eq!(
        encoded["bases"],
        serde_json::json!(["assertion:family@revision"])
    );
    assert_eq!(
        serde_json::from_value::<UserWrite>(encoded.clone()).unwrap(),
        write
    );
    for invalid in [
        serde_json::json!(["assertion:family"]),
        serde_json::json!(["message:codex:message"]),
        serde_json::json!([{"root":"family","revision":"revision"}]),
    ] {
        let mut invalid_write = encoded.clone();
        invalid_write["bases"] = invalid;
        assert!(serde_json::from_value::<UserWrite>(invalid_write).is_err());
    }
}

fn chat_event(kind: &str, phase: &str, by: &str, at: i64) -> ChatEvent {
    ChatEvent {
        request: format!("{kind}-{phase}-{at}"),
        at,
        kind: kind.into(),
        phase: phase.into(),
        detail: None,
        by: Some(by.into()),
        receipt: None,
    }
}

fn chat_action(events: Vec<ChatEvent>) -> ChatAction {
    ChatAction {
        id: "synthetic-action".into(),
        request_key: "synthetic-key".into(),
        request_hash: BlobHash::of(b"synthetic-turn"),
        request: serde_json::json!({"text":"synthetic turn", "submit":true}),
        stage_only: false,
        needs_approval: false,
        created_at: 0,
        execution: crate::authored::ExecutionEligibility::Live,
        attachments: vec![],
        events,
    }
}

#[test]
fn sending_survives_progress_and_claim_expiry() {
    for phase in ["sending", "claimed", "waiting", "capturing"] {
        let action = chat_action(vec![
            chat_event("claimed", "claimed", "tab-a", 0),
            chat_event("progress", "sending", "tab-a", 1),
            chat_event("progress", phase, "tab-a", 2),
        ]);
        assert_eq!(
            action.status(2 + CHAT_CLAIM_MS),
            (ChatStatus::Claimed, Some("tab-a")),
        );
        assert_eq!(
            action.status(3 + CHAT_CLAIM_MS),
            (ChatStatus::OutcomeUnknown, None),
            "renewal phase {phase}",
        );
        assert_eq!(
            action.unconfirmed_submitter(3 + CHAT_CLAIM_MS),
            Some("tab-a")
        );
        assert_eq!(action.unconfirmed_submitter(2 + CHAT_CLAIM_MS), None);
    }
}

#[test]
fn only_the_live_claim_holder_can_start_a_send() {
    for stale in [
        chat_event("progress", "sending", "tab-b", 1),
        chat_event("progress", "sending", "tab-a", CHAT_CLAIM_MS + 1),
    ] {
        let action = chat_action(vec![chat_event("claimed", "claimed", "tab-a", 0), stale]);
        assert_eq!(action.status(1), (ChatStatus::Claimed, Some("tab-a")));
        assert_eq!(action.status(CHAT_CLAIM_MS + 1), (ChatStatus::Queued, None));
    }
    let action = chat_action(vec![
        chat_event("claimed", "claimed", "tab-a", 0),
        chat_event("progress", "composer", "tab-a", 1),
    ]);
    assert_eq!(action.status(CHAT_CLAIM_MS + 2), (ChatStatus::Queued, None));
    let mut action = action;
    action
        .events
        .push(chat_event("outcome_unknown", "outcome_unknown", "tab-a", 2));
    assert_eq!(
        action.status(CHAT_CLAIM_MS + 3),
        (ChatStatus::OutcomeUnknown, None)
    );
    assert_eq!(action.unconfirmed_submitter(CHAT_CLAIM_MS + 3), None);
}

#[test]
fn reported_outcome_stays_after_claim_expiry() {
    for (kind, expected) in [
        ("submitted", ChatStatus::Submitted),
        ("drafted", ChatStatus::Drafted),
        ("outcome_unknown", ChatStatus::OutcomeUnknown),
        ("failed", ChatStatus::Failed),
        ("rate_limited", ChatStatus::Blocked),
        ("declined", ChatStatus::Cancelled),
    ] {
        let action = chat_action(vec![
            chat_event("claimed", "claimed", "tab-a", 0),
            chat_event("progress", "sending", "tab-a", 1),
            chat_event(kind, kind, "tab-a", 2),
            chat_event("claimed", "claimed", "tab-b", 3),
            chat_event("progress", "claimed", "tab-a", 4),
        ]);
        assert_eq!(action.status(CHAT_CLAIM_MS + 5), (expected, None));
        assert_eq!(
            action.unconfirmed_submitter(CHAT_CLAIM_MS + 5),
            (expected == ChatStatus::OutcomeUnknown).then_some("tab-a"),
        );
    }
}

#[test]
fn later_claim_keeps_the_first_sender() {
    let action = chat_action(vec![
        chat_event("claimed", "claimed", "tab-a", 0),
        chat_event("progress", "sending", "tab-a", 1),
        chat_event("claimed", "claimed", "tab-b", CHAT_CLAIM_MS + 2),
        chat_event("progress", "preparing", "tab-b", CHAT_CLAIM_MS + 3),
    ]);
    assert_eq!(
        action.status(CHAT_CLAIM_MS + 3),
        (ChatStatus::OutcomeUnknown, None)
    );
    assert_eq!(
        action.unconfirmed_submitter(CHAT_CLAIM_MS + 3),
        Some("tab-a")
    );
}

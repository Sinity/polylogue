use super::*;
#[test]
fn stamps_are_only_a_skip_hint_and_due_or_failed_full_checks_cannot_certify_new_content() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("history.sqlite");
    std::fs::write(&path, b"first").unwrap();
    let config = polylogue_store::configured_codex::Config {
        home: temp.path().into(),
        machine: "synthetic".into(),
        host: "synthetic".into(),
        rollouts: "sessions".into(),
        history: "history.jsonl".into(),
        catalog: "state.sqlite".into(),
        paginated_history: Some("history.sqlite".into()),
        paginated_interval_seconds: 300,
    };
    let first = polylogue_store::configured_codex::history_stamp(&config).unwrap();
    let now = Instant::now();
    let mut p = HistoryPoll::new(Duration::from_secs(300), now);
    assert_eq!(p.decision(Some(&first), now), HistoryDecision::Capture);
    p.verified(Some(first.clone()), Some(first.clone()), now);
    assert_eq!(
        p.decision(Some(&first), now + Duration::from_secs(300)),
        HistoryDecision::UnchangedMetadata
    );
    assert_eq!(
        p.decision(Some(&first), now + Duration::from_secs(3600)),
        HistoryDecision::Capture
    );
    // A failed full attempt does not call verified and remains due.
    assert_eq!(
        p.decision(Some(&first), now + Duration::from_secs(3601)),
        HistoryDecision::Capture
    );
    std::fs::write(&path, b"different length and stamp").unwrap();
    let changed = polylogue_store::configured_codex::history_stamp(&config).unwrap();
    let now = now + Duration::from_secs(3602);
    p.verified(Some(first.clone()), Some(first), now);
    assert_eq!(
        p.decision(Some(&changed), now),
        HistoryDecision::WaitingForQuiet
    );
    assert_eq!(
        p.decision(Some(&changed), now + Duration::from_secs(300)),
        HistoryDecision::Capture
    );
    // A producer commit during capture means its post-capture stamp was not read.
    p.verified(Some(changed.clone()), None, now + Duration::from_secs(300));
    assert!(p.verified_stamp.is_none());
    assert_eq!(
        p.decision(Some(&changed), now + Duration::from_secs(301)),
        HistoryDecision::Capture
    );
    assert_eq!(p.decision(None, now), HistoryDecision::Capture);
}

use super::*;

fn record() -> HistoricalAcceptedRequest {
    let intent = b"original accepted semantic bytes".to_vec();
    HistoricalAcceptedRequest {
        hash_evidence: AcceptedHashEvidence::Observed,
        native: Default::default(),
        id: "historical-action".into(),
        source: SourceSchema::PolylogueArchive { version: 4 },
        content_hash: BlobHash::of(&intent),
        original: Some(OriginalBytes {
            cached: vec![],
            request: Some(
                br#"{"id":"historical-action","write":{"write":"withdraw","note":"family"}}"#
                    .to_vec(),
            ),
            intent: Some(intent),
            outcome: None,
            effect: None,
            records: Vec::new(),
            imports: None,
        }),
        intent: Some(Intent::Withdrawal {
            note: "family".into(),
            reason: None,
        }),
        machine: None,
        at: None,
        authorship: None,
        outcome: None,
        conserved_effects: None,
        facts: vec![],
        imports: Default::default(),
    }
}

#[test]
fn historical_accepted_evidence_roundtrips_exact_bytes_and_unknown_provenance() {
    let record = record();
    assert_eq!(record.validate().unwrap(), Integrity::HashMatches);
    let restored: HistoricalAcceptedRequest =
        serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
    assert_eq!(restored, record);
    assert_eq!(restored.authorship, None);
    assert_eq!(restored.at, None);
    assert_eq!(restored.outcome, None);
}

#[test]
fn historical_hash_checks_cannot_turn_missing_or_changed_bytes_into_verified_identity() {
    let mut record = record();
    record
        .original
        .as_mut()
        .unwrap()
        .intent
        .as_mut()
        .unwrap()
        .push(0);
    assert!(matches!(record.validate(), Err(InvalidHistory::Hash)));
    record.original = None;
    assert!(matches!(record.validate(), Err(InvalidHistory::Intent)));
    record.intent = None;
    assert_eq!(record.validate().unwrap(), Integrity::Unrecorded);
}

#[test]
fn historical_identity_and_outcome_are_checked_against_original_bytes() {
    let mut history = record();
    history.id = "different-request".into();
    assert!(matches!(history.validate(), Err(InvalidHistory::Request)));
    history.id = "historical-action".into();
    let outcome = UserOutcome {
        id: "original-fact".into(),
        repeat: false,
        pending: false,
    };
    history.outcome = Some(outcome.clone());
    assert!(matches!(
        history.validate(),
        Err(InvalidHistory::OutcomeBytes)
    ));
    history.original.as_mut().unwrap().outcome = Some(serde_json::to_vec(&outcome).unwrap());
    assert_eq!(history.validate().unwrap(), Integrity::HashMatches);
    history.outcome.as_mut().unwrap().id = "changed-fact".into();
    assert!(matches!(
        history.validate(),
        Err(InvalidHistory::OutcomeBytes)
    ));
}

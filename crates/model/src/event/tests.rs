use super::*;

#[test]
fn unknown_event_time_has_unknown_evidence_and_unbound_facts_have_no_session() {
    let old = r#"{"session":"hermes:native","kind":"observation","data":{"n":7}}"#;
    let event: Event = serde_json::from_str(old).unwrap();
    assert_eq!(event.time_evidence(), crate::TimeEvidence::UNKNOWN);
    assert!(event.time_evidence().validate(event.at).is_ok());
    let unbound = Event {
        session: None,
        range: Some((10, 20)),
        ..event
    };
    let wire = serde_json::to_value(&unbound).unwrap();
    assert!(wire.get("session").is_none());
    assert_eq!(serde_json::from_value::<Event>(wire).unwrap(), unbound);
}

#[test]
fn link_statement_identity_preserves_recorded_coordinates() {
    let s = |n: &str| SessionId::parse(n).unwrap();
    let from_child = Link {
        kind: LinkKind::Spawn,
        copies_prefix: false,
        how: None,
        parent: s("claude-code:p"),
        child: s("claude-code:agent-1"),
        at: None,
        call_id: None,
        evidence: "agentId".into(),
    };
    let from_parent = Link {
        call_id: Some("toolu_1".into()),
        evidence: "agent_progress".into(),
        ..from_child.clone()
    };
    assert_ne!(from_child.selection_tag(), from_parent.selection_tag());
    let mut same_statement = from_parent.clone();
    same_statement.evidence = "another field".into();
    same_statement.copies_prefix = true;
    assert_eq!(same_statement.selection_tag(), from_parent.selection_tag());
    let mut different = from_parent.clone();
    different.kind = LinkKind::Continuation;
    assert_ne!(from_child.selection_tag(), different.selection_tag());
    different = from_parent;
    different.child = s("claude-code:agent-2");
    assert_ne!(from_child.selection_tag(), different.selection_tag());
}

use super::*;

#[test]
fn unknown_event_time_has_unknown_accuracy() {
    let old = r#"{"session":"hermes:native","kind":"observation","data":{"n":7}}"#;
    let event: Event = serde_json::from_str(old).unwrap();
    assert_eq!(event.time_accuracy(), crate::TimeAccuracy::UNKNOWN);
    assert!(event.time_accuracy().validate(event.at).is_ok());
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
        reason: None,
        parent: s("claude-code:p"),
        child: s("claude-code:agent-1"),
        branch_point: None,
        call_id: None,
        evidence: "agentId".into(),
    };
    let from_parent = Link {
        call_id: Some("toolu_1".into()),
        evidence: "agent_progress".into(),
        ..from_child.clone()
    };
    // blake3 of the bytes
    // ["polylogue.link.selection.v2",["spawn","claude-code:p","claude-code:agent-1",null,null,null]]
    // computed outside this code; the tag is stored and used for dedupe.
    assert_eq!(
        from_child.selection_tag(),
        "9d1e2e7b2fe0d02451d15f633d26c833c9c7c80a1df11fee21003f78973a1078"
    );
    assert_ne!(from_child.selection_tag(), from_parent.selection_tag());
    let mut same_statement = from_parent.clone();
    same_statement.evidence = "another field".into();
    same_statement.copies_prefix = true;
    assert_eq!(same_statement.selection_tag(), from_parent.selection_tag());
    let baseline = from_parent.selection_tag();
    let mut different = from_parent.clone();
    different.kind = LinkKind::Continuation;
    assert_ne!(baseline, different.selection_tag());
    let mut different = from_parent.clone();
    different.child = s("claude-code:agent-2");
    assert_ne!(baseline, different.selection_tag());
    let mut different = from_parent.clone();
    different.branch_point = Some(MessageId::parse("claude-code:m1").unwrap());
    assert_ne!(baseline, different.selection_tag());
    let mut different = from_parent;
    different.kind = LinkKind::Reference;
    let reference = different.clone();
    different.reason = Some(ReferenceReason::Send);
    assert_ne!(reference.selection_tag(), different.selection_tag());
}

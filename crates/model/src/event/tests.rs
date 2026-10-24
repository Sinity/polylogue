use super::*;

#[test]
fn bound_event_wire_bytes_stay_unchanged_and_unbound_facts_have_no_session() {
    let old = r#"{"session":"hermes:native","kind":"observation","data":{"n":7}}"#;
    let event: Event = serde_json::from_str(old).unwrap();
    assert_eq!(serde_json::to_string(&event).unwrap(), old);
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
fn link_parts_merge_from_either_side() {
    let s = |n: &str| SessionId::parse(n).unwrap();
    let mut from_child = Link {
        kind: LinkKind::Spawn,
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
    from_child.merge(&from_parent);
    assert_eq!(from_child.call_id.as_deref(), Some("toolu_1"));
    assert_eq!(from_child.evidence, "agentId");
}

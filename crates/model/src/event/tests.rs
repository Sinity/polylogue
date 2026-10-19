use super::*;

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

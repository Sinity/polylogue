use super::*;

#[test]
fn computed_authorship_preserves_channel_and_observed_identity() {
    let blank = CallerEvidence::default();
    assert_eq!(
        compute_authorship(Channel::Cli, &blank, None).author,
        Author::Unattributed
    );
    assert_eq!(
        compute_authorship(Channel::Mcp, &blank, None).author,
        Author::Agent(None)
    );
    let terminal = CallerEvidence {
        controlling_terminal: true,
        ..blank.clone()
    };
    assert_eq!(
        compute_authorship(Channel::Cli, &terminal, None).author,
        Author::Operator
    );
    assert!(
        compute_authorship(Channel::Mcp, &terminal, None)
            .author
            .is_agent()
    );
    let script = CallerEvidence {
        unit: Some("worker.service".into()),
        ..blank.clone()
    };
    assert_eq!(
        compute_authorship(Channel::Cli, &script, None).author,
        Author::Script("worker.service".into())
    );
    let harness = CallerEvidence {
        harness: Some(Origin::new("gemini-cli").unwrap()),
        controlling_terminal: true,
        ..blank
    };
    assert_eq!(
        compute_authorship(Channel::Cli, &harness, None)
            .author
            .to_string(),
        "agent:gemini-cli"
    );
    assert_eq!(
        compute_authorship(Channel::Web, &harness, None).author,
        Author::Operator
    );
}
#[test]
fn hook_refinement_requires_the_recorded_parent_and_tool_call() {
    let caller = CallerEvidence {
        harness: Some(Origin::new("claude-code").unwrap()),
        session: Some(SessionId::parse("claude-code:main").unwrap()),
        harness_pid: Some(42),
        tool_call_id: Some("call1".into()),
        ..CallerEvidence::default()
    };
    let mut hook = HookIdentity {
        harness_pid: 42,
        tool_call_id: Some("call1".into()),
        session: SessionId::parse("claude-code:resumed").unwrap(),
        agent_id: Some("worker".into()),
    };
    assert_eq!(
        compute_authorship(Channel::Mcp, &caller, Some(&hook))
            .author
            .to_string(),
        "agent:claude-code:agent-worker"
    );
    hook.tool_call_id = Some("other".into());
    assert_eq!(
        compute_authorship(Channel::Mcp, &caller, Some(&hook))
            .author
            .to_string(),
        "agent:claude-code:main"
    );
    hook.tool_call_id = None;
    hook.agent_id = None;
    assert_eq!(
        compute_authorship(Channel::Mcp, &caller, Some(&hook))
            .author
            .to_string(),
        "agent:claude-code:resumed"
    );
    hook.harness_pid = 43;
    assert_eq!(
        compute_authorship(Channel::Mcp, &caller, Some(&hook))
            .author
            .to_string(),
        "agent:claude-code:main"
    );
}
#[test]
fn honest_author_forms_round_trip_without_turning_unknowns_into_operator() {
    for text in [
        "operator",
        "agent",
        "agent:gemini-cli",
        "agent:codex:thread",
        "script:worker.service",
        "unattributed",
    ] {
        let author = Author::parse(text).unwrap();
        assert_eq!(author.to_string(), text);
        assert_eq!(
            serde_json::from_str::<Author>(&serde_json::to_string(&author).unwrap()).unwrap(),
            author
        );
    }
}

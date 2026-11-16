use super::*;

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

#[test]
fn malformed_authorship_is_an_error_without_fallback_attribution() {
    for text in [
        "",
        "agent:",
        "script:",
        "person:synthetic",
        "operator:extra",
    ] {
        assert!(Author::parse(text).is_err(), "{text}");
    }
    assert!(matches!(
        Author::parse("agent:"),
        Err(AuthorError::Id(_, _))
    ));
    assert!(matches!(
        Author::parse("person:synthetic"),
        Err(AuthorError::Unknown(_))
    ));
}

#[test]
fn each_computed_author_keeps_its_stored_text() {
    use command_surface::{AgentSession, Author as Computed, Harness, Via};
    let harness = |name: &str| Harness::new(name).unwrap();
    for (computed, stored) in [
        (Computed::Operator, "operator"),
        (
            Computed::Agent(AgentSession::new(harness("codex"), "thread").unwrap()),
            "agent:codex:thread",
        ),
        (
            Computed::AgentOfHarness(harness("gemini-cli")),
            "agent:gemini-cli",
        ),
        (Computed::UnidentifiedAgent, "agent"),
        (
            Computed::Script("worker.service".into()),
            "script:worker.service",
        ),
        (Computed::Unattributed, "unattributed"),
    ] {
        let kept = Authorship::from(command_surface::Authorship {
            author: computed.clone(),
            via: Via::Mcp,
        });
        assert_eq!(kept.author.to_string(), stored);
        assert_eq!(kept.author.to_string(), computed.to_string());
        assert_eq!(kept.via, Channel::Mcp);
    }
}

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
    for (text, expected) in [
        ("", AuthorError::Unknown(String::new())),
        (
            "agent:",
            AuthorError::Id("agent:".into(), IdError::BadOrigin(String::new())),
        ),
        ("script:", AuthorError::Unknown("script:".into())),
        (
            "person:synthetic",
            AuthorError::Unknown("person:synthetic".into()),
        ),
        (
            "operator:extra",
            AuthorError::Unknown("operator:extra".into()),
        ),
    ] {
        assert_eq!(Author::parse(text), Err(expected), "{text}");
    }
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

#[test]
fn each_computed_via_keeps_its_channel() {
    use command_surface::{Author as Computed, Via};
    for (via, channel) in [
        (Via::Cli, Channel::Cli),
        (Via::Mcp, Channel::Mcp),
        (Via::Web, Channel::Web),
        (Via::Extension, Channel::Extension),
        (Via::Desktop, Channel::Desktop),
        (Via::Hook, Channel::Hook),
        (Via::Markup, Channel::Markup),
        (Via::Import, Channel::Import),
        (Via::Daemon, Channel::Daemon),
    ] {
        let kept = Authorship::from(command_surface::Authorship {
            author: Computed::Operator,
            via,
        });
        assert_eq!(kept.via, channel);
    }
}

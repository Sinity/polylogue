use super::*;

#[test]
fn targets_round_trip_through_their_reference_form() {
    for text in [
        "session:claude-code:abc",
        "message:codex:m:1",
        "tool:claude-code:abc#toolu_01",
        "work:7",
        "result:018f",
        "note:018f.1",
        "repo:polylogue",
        "cwd:/realm/project/polylogue/repo",
    ] {
        let t = Target::parse(text).unwrap();
        assert_eq!(t.to_string(), text);
    }
    assert!(matches!(
        Target::parse("session"),
        Err(TargetError::NoKind(_))
    ));
    assert!(matches!(
        Target::parse("tool:claude-code:abc"),
        Err(TargetError::Incomplete(_))
    ));
}

#[test]
fn authors_round_trip_and_agents_are_marked() {
    for text in ["operator", "agent", "agent:codex:x", "script:importer"] {
        assert_eq!(Author::parse(text).unwrap().to_string(), text);
    }
    assert!(Author::parse("agent:codex:x").unwrap().is_agent());
    assert!(!Author::Operator.is_agent());
}

#[test]
fn kinds_use_their_stored_names() {
    assert_eq!(NoteKind::parse("saved_query"), Some(NoteKind::SavedQuery));
    assert_eq!(NoteKind::SavedQuery.as_str(), "saved_query");
    assert!(NoteKind::Handoff.briefed());
    assert!(!NoteKind::Tag.briefed());
}

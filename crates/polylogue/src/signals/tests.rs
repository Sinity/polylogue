use super::*;

#[test]
fn rule_duration_uses_shared_parser_without_overflow() {
    assert_eq!(duration("20m"), Some(1_200_000));
    assert_eq!(duration("-1m"), None);
    assert_eq!(duration("999999999999999999999d"), None);
}

#[test]
fn signal_schema_uses_common_closed_rule_and_delivery_contract() {
    let rule: shared_signals::Rule = serde_json::from_value(
        json!({"name":"test","query":"sessions","fires":{"on_change":{"what":"membership"}}}),
    )
    .unwrap();
    assert!(
        rule.validate(&shared_signals::Configuration {
            value_targets: Default::default(),
            action_commands: Default::default()
        })
        .is_ok()
    );
    assert!(
        serde_json::from_value::<shared_signals::Rule>(
            json!({"name":"forged","query":"sessions","operator":true})
        )
        .is_err()
    );
}

fn registry() -> Checkpoint {
    let day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::midnight(),
    )
    .unwrap();
    Checkpoint {
        owner: "fixture".into(),
        evaluator: Evaluator::new(0, vec![], Policy::default(), &DayCalendar::new(day)).unwrap(),
        rules: vec![],
        rule_sources: BTreeMap::new(),
        rule_revisions: BTreeMap::new(),
        host: None,
        notifications: None,
        outlet_attempts: BTreeMap::new(),
    }
}
fn note(author: polylogue_model::Author) -> polylogue_model::Note {
    polylogue_model::Note {
        id: "revision".into(),
        first_note: "family".into(),
        kind: polylogue_model::NoteKind::Signal,
        key: None,
        value: Some(json!({"name":"waiting","query":"sessions","interrupt":true}).to_string()),
        author,
        created_at: polylogue_model::Timestamp::from_unix_ms(0).unwrap(),
        valid_until: None,
        targets: vec![],
        relations: vec![],
    }
}
#[test]
fn only_exact_operator_authorship_has_shared_budget_exemption() {
    use polylogue_model::{Author, Origin};
    let config = crate::config::SignalDeployment::default().rule_configuration();
    for author in [
        Author::Operator,
        Author::Script("unit".into()),
        Author::Unattributed,
        Author::Agent(None),
        Author::AgentOrigin(Origin::new("codex").unwrap()),
    ] {
        let expected = author == Author::Operator;
        let active = rules::active(&note(author), &registry(), &config).unwrap();
        assert_eq!(active.author.budget_exempt(), expected);
    }
}
#[test]
fn rule_withdrawal_and_restore_cannot_reuse_an_adopted_revision() {
    let config = crate::config::SignalDeployment::default().rule_configuration();
    let mut state = registry();
    let note = note(polylogue_model::Author::Operator);
    let first = rules::active(&note, &state, &config).unwrap();
    state
        .rule_sources
        .insert(first.version.id.clone(), note.id.clone());
    state
        .rule_revisions
        .insert(first.version.id.clone(), first.version.revision);
    state.rules.push(first.clone());
    let restored: Checkpoint =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    assert_eq!(
        rules::active(&note, &restored, &config).unwrap().version,
        first.version
    );
    state.rules.clear();
    let again = rules::active(&note, &state, &config).unwrap();
    assert_eq!(again.version.id, first.version.id);
    assert_eq!(again.version.revision, first.version.revision + 1);
}

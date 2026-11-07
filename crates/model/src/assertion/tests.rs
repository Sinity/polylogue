use super::*;

#[test]
fn assertion_input_refuses_caller_authorship_and_seen_hash() {
    let value = serde_json::json!({"kind":"note","value":"A statement","links":[{"role":"about","reference":"message:codex:message"}]});
    let parsed: NewAssertion = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(parsed.links[0].role, LinkRole::About);
    for field in ["author", "via", "machine", "created_at"] {
        let mut with_actor = value.clone();
        with_actor[field] = "operator".into();
        assert!(
            serde_json::from_value::<NewAssertion>(with_actor).is_err(),
            "{field}"
        );
    }
    let mut with_hash = value;
    with_hash["links"][0]["seen_hash"] = "00".repeat(32).into();
    assert!(serde_json::from_value::<NewAssertion>(with_hash).is_err());
}
#[test]
fn assertion_revision_patch_distinguishes_inherit_clear_and_replace() {
    let inherit: AssertionPatch = serde_json::from_str("{}").unwrap();
    assert_eq!(inherit.key, Change::Keep);
    let change:AssertionPatch=serde_json::from_value(serde_json::json!({"key":{"change":"clear"},"value":{"change":"set","value":"Replacement"}})).unwrap();
    assert_eq!(change.key, Change::Clear);
    assert_eq!(change.value, Change::Set("Replacement".into()));
    assert_eq!(change.valid_until, Change::Keep);
}

#[test]
fn canonical_setting_target_is_not_inferred_from_predecessor_fields() {
    let mut fact: AssertionFact=serde_json::from_value(serde_json::json!({
        "id":"setting","root":"setting","kind":"setting","key":"signal.quiet_hours",
        "value":{"machine":null,"value":{"type":"quiet_hours","value":null},"predecessor":{"machine":"declared-host"}},
        "author":"unattributed","via":"import","machine":"importing-host","created_at":"2026-01-01T00:00:00Z","valid_until":null,"links":[]
    })).unwrap();
    assert!(matches!(
        fact.validate(),
        Err(AssertionError::Setting(
            crate::settings::Error::Invalid { .. }
        ))
    ));
    fact.value.as_mut().unwrap()["machine"] = "actual-target".into();
    fact.validate().unwrap();
    assert_eq!(fact.machine, "importing-host");
    fact.key = Some("unknown.setting".into());
    assert!(matches!(
        fact.validate(),
        Err(AssertionError::Setting(crate::settings::Error::UnknownKey(
            _
        )))
    ));
}

#[test]
fn withdrawal_identity_has_no_text_binding() {
    let mut fact: AssertionFact = serde_json::from_value(serde_json::json!({
        "id":"withdrawal","root":"withdrawal","kind":"withdrawal","key":null,
        "value":null,"author":"unattributed","via":"import","machine":"importing-host",
        "created_at":"2026-01-01T00:00:00Z","valid_until":null,
        "links":[{"role":"withdraws","reference":"assertion:family","seen_hash":null}]
    }))
    .unwrap();
    fact.validate().unwrap();
    fact.links[0].seen_hash = Some(BlobHash([7; 32]));
    assert!(matches!(
        fact.validate(),
        Err(AssertionError::IdentityTextHash)
    ));
}

#[test]
fn assertion_text_uses_declared_prose_fields_and_preserves_recorded_empty_strings() {
    let cases = [
        (
            AssertionKind::Role,
            Some("reviewer"),
            serde_json::json!({"instructions":"read 💡 exactly","harness":"codex"}),
            Some("read 💡 exactly"),
        ),
        (
            AssertionKind::Rubric,
            Some("quality"),
            serde_json::json!({"instructions":"check evidence","unit_query":"messages","fields":[],"abstain":true,"confidence":false,"rationale":false}),
            Some("check evidence"),
        ),
        (
            AssertionKind::Judgment,
            Some("accepted"),
            serde_json::json!({"reason":"","blind":null}),
            Some(""),
        ),
        (
            AssertionKind::Judgment,
            Some("accepted"),
            serde_json::json!({"reason":null,"blind":null}),
            None,
        ),
        (
            AssertionKind::Setting,
            Some("browser.actions"),
            serde_json::json!({"machine":null,"value":{"type":"boolean","value":true},"predecessor":{"text":"declared metadata is not the body"}}),
            None,
        ),
    ];
    for (kind, key, value, expected) in cases {
        let fact = AssertionFact {
            id: "exact".into(),
            root: "exact".into(),
            kind,
            key: key.map(str::to_owned),
            value: Some(value),
            author: crate::Author::Unattributed,
            via: crate::authorship::Channel::Import,
            machine: "importing-host".into(),
            created_at: crate::Timestamp::from_unix_ms(1).unwrap(),
            valid_until: None,
            links: if kind == AssertionKind::Judgment {
                vec![AssertionLink {
                    role: LinkRole::About,
                    reference: crate::Reference::parse("assertion:subject@revision").unwrap(),
                    seen_hash: None,
                    quote: None,
                }]
            } else {
                Vec::new()
            },
        };
        fact.validate().unwrap();
        assert_eq!(fact.text(), expected, "{kind:?}");
    }
}

/// law: record.authorship-non-null
#[test]
fn a_stored_fact_without_author_channel_machine_or_time_does_not_decode() {
    let complete = serde_json::json!({
        "id":"exact","root":"exact","kind":"note","key":null,"value":"text",
        "author":"operator","via":"cli","machine":"host","created_at":"2026-01-01T00:00:00Z",
        "valid_until":null,"links":[{"role":"about","reference":"session:codex:s","seen_hash":null}]
    });
    serde_json::from_value::<AssertionFact>(complete.clone()).unwrap();
    for field in ["author", "via", "machine", "created_at"] {
        let mut missing = complete.clone();
        missing[field] = serde_json::Value::Null;
        assert!(
            serde_json::from_value::<AssertionFact>(missing).is_err(),
            "{field}"
        );
    }
}

#[test]
fn a_quote_finds_its_text_once_or_not_at_all() {
    let text = "alpha beta gamma beta delta";
    let quote = Quote::of(text, 17, 21).unwrap();
    assert_eq!(quote.exact, "beta");
    assert_eq!(quote.prefix, "alpha beta gamma ");
    assert_eq!(quote.suffix, " delta");
    assert_eq!(quote.find(text), Some((17, 21)));
    // Forty characters inserted before the span move it; the context still
    // picks the same occurrence.
    let moved = format!("{}{text}", "x".repeat(40));
    assert_eq!(quote.find(&moved), Some((57, 61)));
    // The quoted text gone: nothing is found.
    assert_eq!(quote.find("alpha gamma delta"), None);
    // Two occurrences and no unique context: never a guess.
    let bare = Quote {
        exact: "beta".into(),
        prefix: String::new(),
        suffix: "!".into(),
    };
    assert_eq!(bare.find(text), None);
    assert_eq!(Quote::of(text, 3, 99), None);
}

#[test]
fn only_a_span_link_carries_a_quote() {
    let mut fact: AssertionFact = serde_json::from_value(serde_json::json!({
        "id":"exact","root":"exact","kind":"note","key":null,"value":"text",
        "author":"operator","via":"cli","machine":"host","created_at":"2026-01-01T00:00:00Z",
        "valid_until":null,"links":[{"role":"about","reference":"session:codex:s","seen_hash":null}]
    }))
    .unwrap();
    fact.validate().unwrap();
    fact.links[0].quote = Quote::of("text", 0, 4);
    assert!(matches!(
        fact.validate(),
        Err(AssertionError::QuoteWithoutSpan)
    ));
}

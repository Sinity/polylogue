use super::*;

#[test]
fn annotation_input_refuses_caller_authorship_and_seen_hash() {
    let value = serde_json::json!({"kind":"note","value":"A statement","links":[{"role":"about","reference":"message:codex:message"}]});
    let parsed: NewAnnotation = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(parsed.links[0].role, LinkRole::About);
    for field in ["author", "via", "machine"] {
        let mut with_actor = value.clone();
        with_actor[field] = "operator".into();
        assert!(
            serde_json::from_value::<NewAnnotation>(with_actor).is_err(),
            "{field}"
        );
    }
    let mut with_hash = value;
    with_hash["links"][0]["seen_hash"] = "00".repeat(32).into();
    assert!(serde_json::from_value::<NewAnnotation>(with_hash).is_err());
}
#[test]
fn annotation_revision_patch_distinguishes_inherit_clear_and_replace() {
    let inherit: AnnotationPatch = serde_json::from_str("{}").unwrap();
    assert_eq!(inherit.key, Change::Keep);
    let change:AnnotationPatch=serde_json::from_value(serde_json::json!({"key":{"change":"clear"},"value":{"change":"set","value":"Replacement"}})).unwrap();
    assert_eq!(change.key, Change::Clear);
    assert_eq!(change.value, Change::Set("Replacement".into()));
    assert_eq!(change.valid_until, Change::Keep);
}

#[test]
fn setting_target_ignores_python_polylogue_fields() {
    let mut fact: AnnotationFact=serde_json::from_value(serde_json::json!({
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
    let mut fact: AnnotationFact = serde_json::from_value(serde_json::json!({
        "id":"withdrawal","root":"withdrawal","kind":"withdrawal","key":null,
        "value":null,"author":"unattributed","via":"import","machine":"importing-host",
        "created_at":"2026-01-01T00:00:00Z","valid_until":null,
        "links":[{"role":"withdraws","reference":"annotation:family","seen_hash":null}]
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
fn annotation_text_keeps_declared_fields_and_empty_strings() {
    let cases = [
        (
            AnnotationKind::Role,
            Some("reviewer"),
            serde_json::json!({"instructions":"read 💡 exactly","harness":"codex"}),
            Some("read 💡 exactly"),
        ),
        (
            AnnotationKind::Rubric,
            Some("quality"),
            serde_json::json!({"instructions":"check evidence","unit_query":"messages","fields":[],"abstain":true,"confidence":false,"rationale":false}),
            Some("check evidence"),
        ),
        (
            AnnotationKind::Judgment,
            Some("accepted"),
            serde_json::json!({"reason":"","blind":null}),
            Some(""),
        ),
        (
            AnnotationKind::Judgment,
            Some("accepted"),
            serde_json::json!({"reason":null,"blind":null}),
            None,
        ),
        (
            AnnotationKind::Setting,
            Some("browser.actions"),
            serde_json::json!({"machine":null,"value":{"type":"boolean","value":true},"predecessor":{"text":"declared metadata is not the body"}}),
            None,
        ),
    ];
    for (kind, key, value, expected) in cases {
        let fact = AnnotationFact {
            id: "exact".into(),
            root: "exact".into(),
            kind,
            key: key.map(str::to_owned),
            value: Some(value),
            author: crate::Author::Unattributed,
            via: crate::authorship::Channel::Import,
            machine: "importing-host".into(),
            created_at: Some(crate::Timestamp::from_unix_ms(1).unwrap()),
            valid_until: None,
            links: if kind == AnnotationKind::Judgment {
                vec![AnnotationLink {
                    role: LinkRole::About,
                    reference: crate::Reference::parse("annotation:subject@revision").unwrap(),
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

#[test]
fn stored_annotation_needs_author_channel_and_machine() {
    let complete = serde_json::json!({
        "id":"exact","root":"exact","kind":"note","key":null,"value":"text",
        "author":"operator","via":"cli","machine":"host","created_at":"2026-01-01T00:00:00Z",
        "valid_until":null,"links":[{"role":"about","reference":"session:codex:s","seen_hash":null}]
    });
    serde_json::from_value::<AnnotationFact>(complete.clone()).unwrap();
    for field in ["author", "via", "machine"] {
        let mut missing = complete.clone();
        missing[field] = serde_json::Value::Null;
        assert!(
            serde_json::from_value::<AnnotationFact>(missing).is_err(),
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
    // Overlapping occurrences are two places: "aa" starts at 0 and 1 in "aaa".
    let overlapping = Quote {
        exact: "aa".into(),
        prefix: String::new(),
        suffix: String::new(),
    };
    assert_eq!(overlapping.find("aaa"), None);
    assert_eq!(overlapping.find("baab"), Some((1, 3)));
    assert_eq!(Quote::of(text, 3, 99), None);
}

#[test]
fn only_a_span_link_carries_a_quote() {
    let mut fact: AnnotationFact = serde_json::from_value(serde_json::json!({
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

#[test]
fn authored_unknown_time_and_known_wire_identity_round_trip() {
    let mut fact: AnnotationFact = serde_json::from_value(serde_json::json!({
        "id":"synthetic", "root":"synthetic", "kind":"rubric", "key":"synthetic",
        "value":{"instructions":"Classify", "unit_query":"sessions", "fields":[], "abstain":false,"confidence":false,"rationale":false},
        "author":"unattributed", "via":"import", "machine":"synthetic-host",
        "created_at":null, "valid_until":null, "links":[]
    })).unwrap();
    let wire = serde_json::to_value(&fact).unwrap();
    assert!(wire["created_at"].is_null());
    assert_eq!(
        serde_json::from_value::<AnnotationFact>(wire).unwrap(),
        fact
    );
    let time = crate::Timestamp::from_unix_ms(1234).unwrap();
    fact.created_at = Some(time);
    let wire = serde_json::to_value(&fact).unwrap();
    assert_eq!(wire["created_at"], serde_json::to_value(time).unwrap());
    assert_eq!(
        serde_json::from_value::<AnnotationFact>(wire).unwrap(),
        fact
    );
}

#[test]
fn canonical_links_are_ordered_by_reference_text() {
    let original = serde_json::json!({
        "id":"named-note","root":"named-note","kind":"note","key":null,
        "value":"exact note","author":"unattributed","via":"import","machine":"synthetic",
        "created_at":"2026-01-01T00:00:00Z","valid_until":null,
        "links":[
            {"role":"about","reference":"alternate:root@revision#2","seen_hash":null},
            {"role":"about","reference":"annotation:root@revision","seen_hash":null},
            {"role":"about","reference":"url:https://example.test/source","seen_hash":null}
        ]
    });
    let mut fact: AnnotationFact = serde_json::from_value(original.clone()).unwrap();
    fact.links.reverse();
    let canonical = fact.canonicalized().unwrap();
    assert_eq!(serde_json::to_value(&canonical).unwrap(), original);
    assert_eq!(
        canonical.links[0].reference.to_string(),
        "alternate:root@revision#2"
    );
}

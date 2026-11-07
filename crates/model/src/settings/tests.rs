use super::*;
#[test]
fn catalog_refuses_unknown_blank_and_primary_owned_values() {
    assert!(matches!(
        parse("day.timezone", "UTC", Owner::Standalone),
        Err(Error::UnknownKey(_))
    ));
    assert!(matches!(
        parse("browser.actions", " ", Owner::Standalone),
        Err(Error::Blank(_))
    ));
    assert!(matches!(
        current("signal.interrupts_per_hour", None, Owner::Primary),
        Err(Error::SinexOwned(_))
    ));
    assert!(matches!(
        parse("signal.quiet_hours", "null", Owner::Primary),
        Err(Error::SinexOwned(_))
    ));
}
#[test]
fn all_declared_defaults_validate_and_variants_keep_order() {
    for spec in CATALOG.iter().filter(|spec| !spec.key.ends_with('*')) {
        if let Some(value) = spec.default {
            parse(spec.key, value, Owner::Standalone).unwrap();
        }
    }
    assert_eq!(
        parse(
            "reading.variants",
            r#"["translation:pl","simplification"]"#,
            Owner::Standalone
        )
        .unwrap(),
        Value::Variants(vec!["translation:pl".into(), "simplification".into()])
    );
    assert!(parse("reading.variants", "[\"\"]", Owner::Standalone).is_err());
}
#[test]
fn typed_values_refuse_invalid_minutes_counts_and_plans() {
    for (key, value) in [
        ("signal.quiet_hours", r#"{"from":1440,"to":10}"#),
        ("signal.interrupts_per_hour", "-1"),
        ("embed.price_per_mtok", "NaN"),
        (
            "plan.example",
            r#"{"name":"named","origin":"example","cycle_anchor_day":32,"price_nusd":1,"budget":null}"#,
        ),
    ] {
        assert!(parse(key, value, Owner::Standalone).is_err(), "{key}");
    }
    assert_eq!(
        parse(
            "signal.quiet_hours",
            r#"{"from":1320,"to":420}"#,
            Owner::Standalone
        )
        .unwrap(),
        Value::QuietHours(Some([1320, 420]))
    );
}

#[test]
fn scope_resolution_uses_accepting_host_and_rejects_archive_overrides() {
    assert_eq!(
        target_machine("signal.quiet_hours", None, "host-a").unwrap(),
        Some("host-a".into())
    );
    assert_eq!(
        target_machine("signal.quiet_hours", Some("host-b"), "host-a").unwrap(),
        Some("host-b".into())
    );
    assert_eq!(
        target_machine("reading.variants", None, "host-a").unwrap(),
        None
    );
    assert!(target_machine("reading.variants", Some("host-b"), "host-a").is_err());
    assert!(target_machine("signal.quiet_hours", Some("*"), "host-a").is_err());
}

#[test]
fn captured_setting_conflict_does_not_select_a_default_or_disable_other_keys() {
    let reference = |text: &str| crate::Reference::parse(text).unwrap();
    let conflict = vec![
        reference("assertion:setting@left"),
        reference("assertion:setting@right"),
    ];
    let snapshot = Snapshot::new([
        Current {
            key: "browser.actions".into(),
            kind: Kind::Boolean,
            default: Some(Value::Boolean(false)),
            scope: Scope::Archive,
            owner: "polylogue".into(),
            value: Some(Value::Boolean(true)),
            authored: None,
            conflicts: conflict.clone(),
        },
        Current {
            key: "reading.variants".into(),
            kind: Kind::Variants,
            default: Some(Value::Variants(vec![])),
            scope: Scope::Archive,
            owner: "polylogue".into(),
            value: Some(Value::Variants(vec!["translation:pl".into()])),
            authored: None,
            conflicts: vec![],
        },
    ]);
    assert!(
        matches!(snapshot.browser_actions(),Err(Error::Conflict {key,revisions}) if key=="browser.actions" && revisions==conflict)
    );
    assert_eq!(snapshot.reading_variants().unwrap(), ["translation:pl"]);
    assert_eq!(snapshot.get("embed.endpoint").unwrap(), None);
}

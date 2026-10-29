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

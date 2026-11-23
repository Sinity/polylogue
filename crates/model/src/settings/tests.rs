use super::*;
#[test]
fn catalog_refuses_unknown_blank_and_primary_owned_values() {
    assert!(matches!(
        parse("day.timezone", "UTC", SettingMode::Standalone),
        Err(Error::UnknownKey(_))
    ));
    assert!(matches!(
        parse("browser.actions", " ", SettingMode::Standalone),
        Err(Error::Blank(_))
    ));
    assert!(matches!(
        current("signal.interrupts_per_hour", None, SettingMode::Primary),
        Err(Error::SinexOwned(_))
    ));
    assert!(matches!(
        parse("signal.quiet_hours", "null", SettingMode::Primary),
        Err(Error::SinexOwned(_))
    ));
}
#[test]
fn all_declared_defaults_validate_and_alternates_keep_order() {
    for spec in CATALOG.iter().filter(|spec| !spec.key.ends_with('*')) {
        if let Some(value) = spec.default {
            parse(spec.key, value, SettingMode::Standalone).unwrap();
        }
    }
    assert_eq!(
        parse(
            "reading.alternates",
            r#"["translation:pl","simplification"]"#,
            SettingMode::Standalone
        )
        .unwrap(),
        Value::Alternates(vec!["translation:pl".into(), "simplification".into()])
    );
    assert!(parse("reading.alternates", "[\"\"]", SettingMode::Standalone).is_err());
}
#[test]
fn typed_values_refuse_invalid_minutes_counts_and_plans() {
    for (key, value, expected) in [
        (
            "signal.quiet_hours",
            r#"{"from":1440,"to":10}"#,
            "civil minutes from 0 through 1439",
        ),
        (
            "signal.interrupts_per_hour",
            "-1",
            "a nonnegative whole interrupt count",
        ),
        (
            "embed.price_per_mtok",
            "NaN",
            "a nonnegative decimal with at most six decimals",
        ),
        (
            "plan.example",
            r#"{"name":"named","origin":"example","cycle_anchor_day":32,"price_nusd":1,"budget":null}"#,
            PLAN_EXPECTED,
        ),
    ] {
        assert_eq!(
            parse(key, value, SettingMode::Standalone),
            Err(Error::Invalid {
                key: key.into(),
                expected,
            }),
            "{key}"
        );
    }
    assert_eq!(
        parse(
            "signal.quiet_hours",
            r#"{"from":1320,"to":420}"#,
            SettingMode::Standalone
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
        target_machine("reading.alternates", None, "host-a").unwrap(),
        None
    );
    assert_eq!(
        target_machine("reading.alternates", Some("host-b"), "host-a"),
        Err(Error::Invalid {
            key: "reading.alternates".into(),
            expected: "archive scope without a machine override",
        })
    );
    assert_eq!(
        target_machine("signal.quiet_hours", Some("*"), "host-a"),
        Err(Error::Invalid {
            key: "signal.quiet_hours".into(),
            expected: "a nonempty host machine identity",
        })
    );
}

#[test]
fn setting_conflict_picks_no_default_and_spares_other_keys() {
    let reference = |text: &str| crate::Reference::parse(text).unwrap();
    let conflict = vec![
        reference("annotation:setting@left"),
        reference("annotation:setting@right"),
    ];
    let snapshot = Snapshot::new([
        Current {
            key: "browser.actions".into(),
            kind: Kind::Boolean,
            default: Some(Value::Boolean(false)),
            scope: Scope::Archive,
            supplier: "polylogue".into(),
            value: Some(Value::Boolean(true)),
            authored: None,
            conflicts: conflict.clone(),
        },
        Current {
            key: "reading.alternates".into(),
            kind: Kind::Alternates,
            default: Some(Value::Alternates(vec![])),
            scope: Scope::Archive,
            supplier: "polylogue".into(),
            value: Some(Value::Alternates(vec!["translation:pl".into()])),
            authored: None,
            conflicts: vec![],
        },
    ]);
    assert!(
        matches!(snapshot.browser_actions(),Err(Error::Conflict {key,revisions}) if key=="browser.actions" && revisions==conflict)
    );
    assert_eq!(snapshot.reading_alternates().unwrap(), ["translation:pl"]);
    assert_eq!(snapshot.get("embed.endpoint").unwrap(), None);
}

#[test]
fn old_drive_setting_replays_but_is_not_a_current_control() {
    assert_eq!(
        parse("drive.ai-studio", "on", SettingMode::Standalone).unwrap(),
        Value::Boolean(true)
    );
    assert_eq!(
        parse("drive.ai-studio", "off", SettingMode::Standalone).unwrap(),
        Value::Boolean(false)
    );
    assert!(
        CATALOG
            .iter()
            .all(|declaration| declaration.key != "drive.ai-studio")
    );
    let hint = historical_hint("drive.ai-studio").unwrap();
    assert_eq!(
        hint,
        "This setting is historical. Use `polylogue install drive` to authorize collection or `polylogue install drive --remove` to revoke it."
    );
    assert_eq!(
        parse(
            "drive.client_id",
            "synthetic-client",
            SettingMode::Standalone
        )
        .unwrap(),
        Value::Text("synthetic-client".into())
    );
}

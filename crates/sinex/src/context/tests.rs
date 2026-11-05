use super::*;

#[test]
fn private_context_token_is_not_in_the_public_schema() {
    let schema = serde_json::to_value(schemars::schema_for!(ResolvedContext)).unwrap();
    assert!(schema["properties"].get("context_token").is_none());
}

fn available() -> PublicContext {
    PublicContext::Available {
        settings_identity: "settings-1".into(),
        context_token: "synthetic-owner-context-proof".into(),
        owner_revision: 7,
        day: query_language::time::DaySettingWire {
            zone: "Europe/Warsaw".into(),
            start: "04:00".parse().unwrap(),
        },
        quiet_hours: Some(shared_signals::QuietHours {
            from: 1320,
            to: 420,
        }),
        interrupt_budget: sinex_protocol::PublicInterruptBudget {
            per_hour: 3,
            used: 1,
            remaining: 2,
            window_ms: 3600000,
        },
        observed_at: sinex_model::Timestamp::from_unix_secs(100).unwrap(),
        coverage: sinex_protocol::CoverageStatus::NotRequested,
    }
}

#[test]
fn unavailable_owner_never_promotes_a_known_context_to_current() {
    let mut cache = Cache::default();
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Missing
    );
    cache.record(Ok(available()));
    let first = cache
        .resolve(|name| jiff::tz::TimeZone::get(name).map_err(|e| e.to_string()))
        .unwrap();
    assert_eq!(first.owner_revision, 7);
    assert_eq!(first.interrupt_budget.remaining, 2);
    cache.record(Err("owner offline".into()));
    let failure = cache.resolve(|_| unreachable!()).unwrap_err();
    assert_eq!(failure.kind, ContextFailureKind::Unavailable);
    assert_eq!(failure.last_known_identity.as_deref(), Some("settings-1"));
    cache.record(Ok(PublicContext::Pending {
        reason: "new configuration has no checkpoint".into(),
    }));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Pending
    );
    assert_eq!(first.settings_identity, "settings-1");
}

#[test]
fn current_context_requires_its_exact_resolved_zone_and_observation_age() {
    let mut cache = Cache::default();
    cache.record(Ok(available()));
    let wrong = cache
        .resolve(|_| Ok(jiff::tz::TimeZone::get("Etc/UTC").unwrap()))
        .unwrap_err();
    assert_eq!(wrong.kind, ContextFailureKind::Invalid);
    cache.received = Some(Instant::now() - Duration::from_secs(10));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Stale
    );
    cache.record(Ok(PublicContext::Refused {
        reason: "public context grant denied".into(),
    }));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Refused
    );
}

#[test]
fn malformed_policy_never_reaches_query_or_signal_callers() {
    let mut reply = available();
    if let PublicContext::Available {
        interrupt_budget, ..
    } = &mut reply
    {
        interrupt_budget.remaining = 3;
    }
    let mut cache = Cache::default();
    cache.record(Ok(reply));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Invalid
    );
}

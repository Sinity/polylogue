use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn replayed_evaluation_time_never_rewrites_the_accepting_invocation() {
    let samples = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&samples);
    let metadata = polylogue_model::user::RequestMetadata::new(
        "synthetic-machine".into(),
        polylogue_model::WallClock::new(move || {
            observed.fetch_add(1, Ordering::Relaxed);
            "2026-10-29T08:00:00.123456789Z".parse().unwrap()
        }),
    );
    let authorship = Context::worker_authorship("polylogue-signals");
    let invocation = Invocation::capture(&metadata, &authorship);
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    let context = Context::standalone(invocation.clone(), day.clone());
    let replay = context
        .replay_clock(Clock::new("2026-10-28T08:00:00Z".parse().unwrap(), day))
        .unwrap();
    let request = replay.request(
        "same-step".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "off".into(),
            machine: None,
        },
    );
    assert_eq!(samples.load(Ordering::Relaxed), 1);
    assert_eq!(request.at, invocation.at);
    assert_eq!(request.authorship, authorship);
    assert_eq!(request.machine, "synthetic-machine");
    assert_ne!(jiff::Timestamp::from(request.at), replay.clock.now);
    assert_eq!(replay.ran_at, context.ran_at);
    assert_eq!(replay.cursor_versions(), context.cursor_versions());
    let changed = Clock::new(
        replay.clock.now,
        DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(6, 0, 0, 0),
        )
        .unwrap(),
    );
    assert!(context.replay_clock(changed).is_err());
}

#[cfg(feature = "sinex")]
#[test]
fn public_context_evidence_and_debug_never_expose_the_owner_bearer_token() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    let invocation = Invocation {
        at,
        machine: "synthetic-machine".into(),
        authorship: Context::worker_authorship("polylogue-signals"),
    };
    let policy = polylogue_sinex::ResolvedContext {
        context_token: "synthetic-private-owner-token".into(),
        settings_identity: "public-settings:17".into(),
        owner_revision: 17,
        day: DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
        quiet_hours: None,
        interrupt_budget: polylogue_sinex::PublicInterruptBudget {
            per_hour: 7,
            used: 2,
            remaining: 5,
            window_ms: 3_600_000,
        },
        observed_at: serde_json::from_value(serde_json::json!("2026-10-29T07:59:59Z")).unwrap(),
        received_seconds_ago: 1.0,
    };
    let context = Context::primary(invocation, policy).unwrap();
    assert_eq!(context.attention().unwrap().budget.remaining, 5);
    assert_eq!(context.owner_token(), Some("synthetic-private-owner-token"));
    let output = serde_json::to_string(&context.provenance()).unwrap();
    let debug = format!("{context:?}");
    let schema = schemars::schema_for!(Provenance).to_value().to_string();
    for public in [output, debug, schema] {
        assert!(!public.contains("context_token"), "{public}");
        assert!(
            !public.contains("synthetic-private-owner-token"),
            "{public}"
        );
    }
}

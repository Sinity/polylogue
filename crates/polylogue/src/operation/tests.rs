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
    let invocation = Invocation::new(
        at,
        "synthetic-machine".into(),
        Context::worker_authorship("polylogue-signals"),
    );
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
    assert_eq!(context.provenance().machine, "synthetic-machine");
    assert!(matches!(
        context.provenance().owner,
        Owner::Primary {
            owner_revision: 17,
            ..
        }
    ));
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

#[test]
fn accepted_envelope_survives_context_replay_without_becoming_public_evidence() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    for deadline in [
        command_surface::Deadline::Default,
        command_surface::Deadline::None,
        command_surface::Deadline::Milliseconds(37),
    ] {
        let envelope = Arc::new(command_surface::Request {
            operation: "read".into(),
            arguments: serde_json::json!({"target":"synthetic-private-input", "options":{"output_lines":0}}),
            request_id: "actual-accepted-id".into(),
            deadline,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        });
        let invocation =
            Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
                .with_accepted_request(Some(envelope.clone()));
        let day = DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap();
        let context = Context::standalone(invocation, day.clone());
        let replay = context
            .replay_clock(Clock::new("2026-10-28T08:00:00Z".parse().unwrap(), day))
            .unwrap();
        assert_eq!(replay.accepted_request(), Some(envelope.as_ref()));
        assert_eq!(
            replay.invocation().accepted_request(),
            Some(envelope.as_ref())
        );
        assert!(!format!("{replay:?}").contains("synthetic-private-input"));
        assert!(
            !serde_json::to_string(&replay.provenance())
                .unwrap()
                .contains("actual-accepted-id")
        );
        assert_eq!(replay.ran_at, at);
    }
    assert!(
        Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
            .accepted_request()
            .is_none()
    );
}

#[test]
fn server_run_ids_are_random_v4_and_independent_of_captured_time() {
    let at = Timestamp::from_unix_ms(-100).unwrap();
    let invocation = Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"));
    let first = new_run_id();
    let second = new_run_id();
    assert_ne!(first, second);
    for run in [first, second] {
        let id = uuid::Uuid::parse_str(run.as_str()).unwrap();
        assert_eq!(id.get_version_num(), 4);
        assert_eq!(id.get_variant(), uuid::Variant::RFC4122);
    }
    assert_eq!(invocation.at, at);
}

#[test]
fn parent_lifetime_survives_context_conversion_and_replay_without_becoming_public() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    let cancellation = polylogue_store::read::limit::ReadCancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let invocation = Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
        .with_parent(Some(parent.clone()));
    invocation.check_parent_admission().unwrap();
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    let context = Context::standalone(invocation, day.clone());
    let replay = context
        .replay_clock(Clock::new("2026-10-28T08:00:00Z".parse().unwrap(), day))
        .unwrap();
    assert!(Arc::ptr_eq(replay.parent.as_ref().unwrap(), &parent));
    assert!(Arc::ptr_eq(
        replay.invocation().parent.as_ref().unwrap(),
        &parent
    ));
    cancellation.cancel();
    assert!(replay.check_parent_admission().is_err());
    assert!(replay.invocation().check_parent_admission().is_err());
    assert!(!format!("{replay:?}").contains("parent"));
    assert!(
        !serde_json::to_string(&replay.provenance())
            .unwrap()
            .contains("parent")
    );
    Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
        .check_parent_admission()
        .unwrap();
    assert!(
        ParentLifetime::capture(Some(std::time::Instant::now()), Default::default())
            .check()
            .is_err()
    );
}

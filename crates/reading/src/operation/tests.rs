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
    let accepted_instant: jiff::Timestamp = "2026-10-29T08:00:00.123456789Z".parse().unwrap();
    assert_eq!(jiff::Timestamp::from(request.at), accepted_instant);
    assert_eq!(request.authorship, authorship);
    assert_eq!(request.machine, "synthetic-machine");
    assert_ne!(jiff::Timestamp::from(request.at), replay.clock.now);
    assert_eq!(replay.ran_at, context.ran_at);
    assert_eq!(jiff::Timestamp::from(replay.ran_at), accepted_instant);
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

#[test]
fn accepted_request_survives_context_replay() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    for deadline in [
        command_surface::Deadline::Default,
        command_surface::Deadline::None,
        command_surface::Deadline::Milliseconds(37),
    ] {
        let envelope = Arc::new(command_surface::Request {
            operation: "read".into(),
            arguments: serde_json::json!({"target":"synthetic-private-input", "options":{"output_lines":0}}),
            request_id: "accepted-id".into(),
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
            Context::standalone(replay.invocation(), replay.clock.day.clone()).accepted_request(),
            Some(envelope.as_ref())
        );
        assert!(!format!("{replay:?}").contains("synthetic-private-input"));
        // The run context is a closed record: none of the accepted request's
        // fields (its target, id or deadline) can appear in it.
        let wire = serde_json::to_value(replay.run_context()).unwrap();
        let mut fields: Vec<&str> = wire
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        fields.sort_unstable();
        assert_eq!(
            fields,
            ["authorship", "clock", "machine", "ran_at", "sinex"]
        );
        assert_eq!(wire["machine"], "fixture");
        assert_eq!(wire["ran_at"], "2026-10-29T08:00:00Z");
        assert_eq!(wire["sinex"], serde_json::json!({"mode":"standalone"}));
        assert_eq!(replay.ran_at, at);
    }
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    assert!(
        Context::standalone(
            Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture")),
            day
        )
        .accepted_request()
        .is_none()
    );
}

#[test]
fn server_run_ids_are_random_v4() {
    let first = new_run_id();
    let second = new_run_id();
    assert_ne!(first, second);
    for run in [first, second] {
        let id = uuid::Uuid::parse_str(run.as_str()).unwrap();
        assert_eq!(id.get_version_num(), 4);
        assert_eq!(id.get_variant(), uuid::Variant::RFC4122);
    }
}

#[test]
fn parent_lifetime_survives_context_conversion_and_replay_without_becoming_public() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    let cancellation = polylogue_store::cancellation::Cancellation::default();
    let parent = ParentLifetime::capture(None, cancellation.clone());
    let invocation = Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
        .with_parent(Some(parent.clone()));
    invocation.check_parent_lifetime().unwrap();
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
    assert!(replay.check_parent_lifetime().is_err());
    assert!(replay.invocation().check_parent_lifetime().is_err());
    assert!(!format!("{replay:?}").contains("parent"));
    assert!(
        !serde_json::to_string(&replay.run_context())
            .unwrap()
            .contains("parent")
    );
    Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
        .check_parent_lifetime()
        .unwrap();
    assert!(
        ParentLifetime::capture(Some(std::time::Instant::now()), Default::default())
            .check()
            .is_err()
    );
}

#[test]
fn sinex_context_records_its_settings_revision() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    let invocation = Invocation::new(
        at,
        "synthetic-machine".into(),
        Context::worker_authorship("polylogue-signals"),
    );
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    let sinex = SinexContext::Primary {
        settings_identity: "public-settings:17".into(),
        settings_revision: 17,
        observed_at: Timestamp::parse_rfc3339("2026-10-29T07:59:59Z").unwrap(),
        received_seconds_ago: 1.0,
    };
    let context = Context::primary(invocation, day, sinex);
    let wire = serde_json::to_value(context.run_context()).unwrap();
    assert_eq!(wire["machine"], "synthetic-machine");
    assert_eq!(
        wire["sinex"],
        serde_json::json!({
            "mode": "primary",
            "settings_identity": "public-settings:17",
            "settings_revision": 17,
            "observed_at": "2026-10-29T07:59:59Z",
            "received_seconds_ago": 1.0
        })
    );
}

#[test]
fn refreshed_observation_preserves_cursor_settings_and_its_replayed_cutoff() {
    let before = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    let after = Timestamp::parse_rfc3339("2026-10-29T08:00:10Z").unwrap();
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    let operation = |at, settings_revision| {
        Context::primary(
            Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture")),
            day.clone(),
            SinexContext::Primary {
                settings_identity: "public-settings".into(),
                settings_revision,
                observed_at: at,
                received_seconds_ago: 0.,
            },
        )
    };
    let first = operation(before, 17);
    let refreshed = operation(after, 17);
    assert_eq!(refreshed.cursor_versions(), first.cursor_versions());
    let continued = refreshed.replay_clock(first.clock.clone()).unwrap();
    assert_eq!(continued.clock.now, first.clock.now);
    assert_eq!(continued.ran_at, after);
    assert_eq!(continued.cursor_versions(), first.cursor_versions());
    assert_ne!(
        operation(after, 18).cursor_versions(),
        first.cursor_versions()
    );
}

#[test]
fn caller_working_directory_survives_replay_without_becoming_wire_context() {
    let at = Timestamp::parse_rfc3339("2026-10-29T08:00:00Z").unwrap();
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(4, 0, 0, 0),
    )
    .unwrap();
    let invocation = Invocation::new(at, "fixture".into(), Context::worker_authorship("fixture"))
        .with_working_directory(Some("/home/op/morrow".into()));
    let context = Context::standalone(invocation, day.clone());
    let replay = context
        .replay_clock(Clock::new(
            "2026-10-28T08:00:00Z".parse().unwrap(),
            day.clone(),
        ))
        .unwrap();
    assert_eq!(
        replay.working_directory(),
        Some(std::path::Path::new("/home/op/morrow"))
    );
    assert_eq!(
        Context::standalone(replay.invocation(), day).working_directory(),
        replay.working_directory()
    );
    assert!(
        !serde_json::to_string(&replay.run_context())
            .unwrap()
            .contains("morrow")
    );
}

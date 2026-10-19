use super::*;

const T0: i64 = 1_790_000_000_000;

#[test]
fn codex_slots_are_classified_by_length_not_name() {
    let q = serde_json::json!({"limit_id": "codex", "plan_type": "pro",
        "primary": {"used_percent": 42.0, "window_minutes": 10080, "resets_at": 1_790_600_000},
        "secondary": null});
    let o = observations("codex", T0, &q);
    assert_eq!(o.len(), 1);
    assert_eq!(o[0].window_minutes, WEEK);
    assert_eq!(o[0].resets_at, Some(1_790_600_000_000));
    assert_eq!(o[0].used_percent, Some(42.0));
    let q = serde_json::json!({"primary": {"used_percent": 1, "window_minutes": 300,
        "resets_in_seconds": 60}});
    assert_eq!(
        observations("codex", T0, &q)[0].resets_at,
        Some(T0 + 60_000)
    );
}

#[test]
fn a_refused_claude_turn_means_the_window_is_used_up() {
    let q = serde_json::json!({"status": "rejected", "resetsAt": 1_790_010_000,
        "rateLimitType": "five_hour"});
    let o = observations("claude-code", T0, &q);
    assert_eq!(o[0].used_percent, Some(100.0));
    let q = serde_json::json!({"status": "allowed_warning", "resetsAt": 1_790_010_000,
        "rateLimitType": "seven_day"});
    let o = observations("claude-code", T0, &q);
    assert_eq!((o[0].window_minutes, o[0].used_percent), (WEEK, None));
}

#[test]
fn a_past_reset_is_carried_forward_by_the_window_length() {
    let o = vec![Observation {
        at: T0,
        tool: "codex".into(),
        limit: "codex".into(),
        window_minutes: WEEK,
        used_percent: Some(10.0),
        resets_at: Some(T0 + HOUR),
        status: None,
    }];
    assert_eq!(
        current_window(&o, T0),
        Some((Basis::Reported, T0 + HOUR - WEEK * MINUTE, T0 + HOUR))
    );
    let later = T0 + HOUR + 3 * MINUTE;
    assert_eq!(
        current_window(&o, later),
        Some((Basis::Cadence, T0 + HOUR, T0 + HOUR + WEEK * MINUTE))
    );
}

/// Law 109: windows after a reported reset are unchanged when an earlier
/// message is inserted, and windows nothing reported are labelled
/// inferred.
#[test]
fn law_109_windows_anchor_to_reported_resets() {
    let five = 5 * HOUR;
    let reset = T0 + 10 * HOUR;
    let anchors = [(reset - five, reset)];
    let times = vec![
        T0 + 7 * HOUR,
        T0 + 11 * HOUR,
        T0 + 17 * HOUR,
        T0 + 23 * HOUR,
    ];
    let before = blocks(&times, &anchors, five);
    let mut more = times.clone();
    more.insert(0, T0 + 2 * HOUR + 30 * MINUTE);
    let after = blocks(&more, &anchors, five);
    let after_reset = |b: &[(Basis, i64, i64)]| -> Vec<(Basis, i64, i64)> {
        b.iter()
            .copied()
            .filter(|&(_, s, _)| s >= reset - five)
            .collect()
    };
    assert_eq!(after_reset(&before), after_reset(&after));
    for (basis, s, e) in &after {
        let reported = anchors.contains(&(*s, *e));
        assert_eq!(
            *basis == Basis::Reported,
            reported,
            "{s}..{e} is labelled {basis:?}"
        );
    }
    // The inserted message starts an inferred block cut short where the
    // reported window begins.
    let first = after[0];
    assert_eq!(first.0, Basis::Inferred);
    assert_eq!(first.2, reset - five);
}

#[test]
fn blocks_start_on_the_hour_and_last_five_hours() {
    let times = vec![T0 + 30 * MINUTE, T0 + 4 * HOUR, T0 + 6 * HOUR];
    let b = blocks(&times, &[], 5 * HOUR);
    let h = T0 - T0.rem_euclid(HOUR);
    assert_eq!(
        b,
        vec![
            (Basis::Inferred, h, h + 5 * HOUR),
            (Basis::Inferred, h + 6 * HOUR, h + 11 * HOUR)
        ]
    );
}

#[test]
fn burn_needs_two_reports_five_minutes_apart() {
    let o = |at, p| Observation {
        at,
        tool: "codex".into(),
        limit: "codex".into(),
        window_minutes: FIVE_HOURS,
        used_percent: Some(p),
        resets_at: None,
        status: None,
    };
    let (a, b) = (o(T0, 10.0), o(T0 + HOUR, 30.0));
    assert_eq!(percent_burn(&[&a, &b]), Some(20.0));
    let c = o(T0 + MINUTE, 11.0);
    assert_eq!(percent_burn(&[&a, &c]), None);
}

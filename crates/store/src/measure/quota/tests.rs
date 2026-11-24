use super::*;

const T0: i64 = 1_790_000_000_000;

#[test]
fn a_past_reset_is_carried_forward_by_the_window_length() {
    let o = vec![Observation {
        at: Some(T0),
        measured_at: Some(T0),
        tool: "codex".into(),
        limit: "codex".into(),
        window_minutes: WEEK,
        from_first_response: false,
        used_percent: Some(10.0),
        resets_at: Some(T0 + HOUR),
        status: None,
    }];
    assert_eq!(
        current_window(&o, T0),
        Some((WindowSource::Reported, T0 + HOUR - WEEK * MINUTE, T0 + HOUR))
    );
    let later = T0 + HOUR + 3 * MINUTE;
    assert_eq!(
        current_window(&o, later),
        Some((
            WindowSource::Projected,
            T0 + HOUR,
            T0 + HOUR + WEEK * MINUTE
        ))
    );
}

/// Windows after a reported reset are unchanged when an earlier
/// message is inserted, and windows nothing reported are labelled
/// inferred.
#[test]
fn windows_start_at_reported_resets() {
    let five = 5 * HOUR;
    let reset = T0 + 10 * HOUR;
    let reported_windows = [(reset - five, reset)];
    let times = vec![
        T0 + 7 * HOUR,
        T0 + 11 * HOUR,
        T0 + 17 * HOUR,
        T0 + 23 * HOUR,
    ];
    let before = blocks(&times, &reported_windows, five);
    let mut more = times.clone();
    more.insert(0, T0 + 2 * HOUR + 30 * MINUTE);
    let after = blocks(&more, &reported_windows, five);
    let after_reset = |b: &[(WindowSource, i64, i64)]| -> Vec<(WindowSource, i64, i64)> {
        b.iter()
            .copied()
            .filter(|&(_, s, _)| s >= reset - five)
            .collect()
    };
    assert_eq!(after_reset(&before), after_reset(&after));
    for (source, s, e) in &after {
        let reported = reported_windows.contains(&(*s, *e));
        assert_eq!(
            *source == WindowSource::Reported,
            reported,
            "{s}..{e} is labelled {source:?}"
        );
    }
    // The inserted message starts an inferred block cut short where the
    // reported window begins.
    let first = after[0];
    assert_eq!(first.0, WindowSource::Inferred);
    assert_eq!(first.2, reset - five);
}

#[test]
fn blocks_start_on_the_hour_and_last_five_hours() {
    let times = vec![T0 + 30 * MINUTE, T0 + 4 * HOUR, T0 + 6 * HOUR];
    let b = blocks(&times, &[], 5 * HOUR);
    // T0 floored to the hour, worked out by hand.
    let h = 1_789_999_200_000;
    assert_eq!(
        b,
        vec![
            (WindowSource::Inferred, h, h + 5 * HOUR),
            (WindowSource::Inferred, h + 6 * HOUR, h + 11 * HOUR)
        ]
    );
}

#[test]
fn burn_needs_two_reports_five_minutes_apart() {
    let o = |at, p| Observation {
        at: Some(at),
        measured_at: Some(at),
        tool: "codex".into(),
        limit: "codex".into(),
        window_minutes: FIVE_HOURS,
        from_first_response: false,
        used_percent: Some(p),
        resets_at: None,
        status: None,
    };
    let (a, b) = (o(T0, 10.0), o(T0 + HOUR, 30.0));
    assert_eq!(percent_burn(&[&a, &b]), Some(20.0));
    let c = o(T0 + MINUTE, 11.0);
    assert_eq!(percent_burn(&[&a, &c]), None);
    let just_short = o(T0 + 5 * MINUTE - 1, 12.0);
    assert_eq!(percent_burn(&[&a, &just_short]), None);
    let exactly = o(T0 + 5 * MINUTE, 12.0);
    assert!(percent_burn(&[&a, &exactly]).is_some());
}

#[test]
fn a_lower_reading_restarts_the_burn() {
    let o = |hours, p| Observation {
        at: Some(T0 + hours * HOUR),
        measured_at: Some(T0 + hours * HOUR),
        tool: "codex".into(),
        limit: "codex".into(),
        window_minutes: FIVE_HOURS,
        from_first_response: false,
        used_percent: Some(p),
        resets_at: None,
        status: None,
    };
    let readings = [o(0, 10.0), o(1, 50.0), o(2, 20.0), o(3, 40.0)];
    let inside: Vec<&Observation> = readings.iter().collect();
    // From the 20% reading: 20 points in one hour.
    assert_eq!(percent_burn(&inside[..]), Some(10.0 * 2.0));
}

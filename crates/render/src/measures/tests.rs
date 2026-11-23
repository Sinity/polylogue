use super::*;
use serde_json::json;

#[test]
fn outlook_labels_estimates_populations_and_unknown_values() {
    let text = outlook(
        &json!({"provider":"sample","projection_method":"linear","window":{"start":0,"end":86400000},"segments":[{"plan":{"name":"configured"},"from":0,"until":86400000,"responses":{"api_nusd":{"used":1_000_000_001_i64,"projected":null,"known_items":1,"unknown_items":2,"lower_bound":true,"observed_days":1,"missing_days":[]}},"recorded_reports":{"reported_nusd":{"used":0,"projected":0,"known_items":1,"unknown_items":0,"observed_days":1,"missing_days":[]}},"subscription_price_share_nusd":null,"unknown_boundary_items":3,"unconfirmed_reports":4}],"untimed_responses":5,"untimed_reports":6}),
    );
    for label in [
        "(estimate)",
        "API-equivalent USD",
        "provider-reported USD",
        "at least used $1.000000001",
        "projected unknown",
        "Subscription price share: unknown",
        "1970-01-01 00:00 UTC",
        "5; untimed reports: 6",
    ] {
        assert!(text.contains(label), "{label}: {text}");
    }
    assert!(!text.contains("null"));
}

#[test]
fn outlook_escapes_controls_in_plan_names_units_and_reasons() {
    let text = outlook(
        &json!({"provider":"sample\u{1b}[2J","projection_method":"linear","segments":[{"plan":{"name":"name\u{7}"},"responses":{"unit\u{85}":{}},"budget_pressure":{"unit":"budget\u{202e}","reason":"reason\u{7}"}}]}),
    );
    for part in [
        "sample\\x1b[2J",
        "name\\x07",
        "unit\\x85",
        "budget\\u{202e}",
        "reason\\x07",
    ] {
        assert!(text.contains(part), "{part}: {text}");
    }
    let missing = outlook(&json!({"reason":"missing\u{1b}[2J"}));
    assert!(missing.contains("missing\\x1b[2J"));
    assert!(missing.contains("--option provider:method"));
    assert!(outlook(&json!({"reason":"no_cycle_anchor"})).contains("cycle_anchor_day"));
}

#[test]
fn measure_times_print_in_answer_zone() {
    let mut answer = json!({
        "zone": "Europe/Warsaw",
        "tools": [{"tool": "sample", "windows": [{"resets_at": 0}]}]
    });
    assert!(quota(&answer).contains("1970-01-01 01:00 +01:00"));
    answer.as_object_mut().unwrap().remove("zone");
    assert!(quota(&answer).contains("1970-01-01 00:00 +00:00"));
    answer["zone"] = json!("not/a/timezone");
    assert!(quota(&answer).contains("1970-01-01 00:00 +00:00"));
    assert!(time(&Value::Null, &answer_zone(&answer)).contains("unknown"));
}

#[test]
fn failing_times_use_resolved_context() {
    let data = json!([{"repo":"sample","command":"cargo test","at":0,"last_ok_at":0}]);
    let warsaw = jiff::tz::TimeZone::get("Europe/Warsaw").unwrap();
    let output = failing(&data, &warsaw);
    assert_eq!(output.matches("1970-01-01 01:00 +01:00").count(), 2);
}

#[test]
fn token_counts_change_unit_at_the_rounded_figure() {
    for (count, expected) in [
        (999, "999"),
        (1_000, "1.0k"),
        (1_500, "1.5k"),
        (999_949, "999.9k"),
        (999_999, "1.0M"),
        (1_500_000, "1.5M"),
        (999_999_999, "1.00B"),
        (2_500_000_000, "2.50B"),
    ] {
        assert_eq!(tokens(count), expected, "{count}");
    }
}

#[test]
fn failing_groups_by_repository_identity_not_display_name() {
    let data = json!([
        {"repository":"root-a","repo":"same","command":"cargo test","at":0,"last_ok_at":null},
        {"repository":"root-b","repo":"same","command":"cargo test","at":0,"last_ok_at":null},
        {"repository":"root-a","repo":"same","command":"cargo clippy","at":0,"last_ok_at":null},
    ]);
    let output = failing(&data, &jiff::tz::TimeZone::UTC);
    assert_eq!(output.matches("same\n").count(), 2, "{output}");
    assert_eq!(output.matches("cargo").count(), 3, "{output}");
    let first = output.find("cargo test").unwrap();
    let second = output.rfind("same\n").unwrap();
    assert!(output.find("cargo clippy").unwrap() < second && first < second);
}

#[test]
fn partial_money_and_token_counts_state_what_is_missing() {
    assert_eq!(
        money(&json!({"known":null,"unpriced_messages":3})),
        "unknown (3 unpriced)"
    );
    assert_eq!(
        money(&json!({"known":null,"unpriced_messages":0})),
        "unknown"
    );
    assert_eq!(
        money(&json!({"known":null,"response_total_unknown":true})),
        "unknown (response grouping unknown)"
    );
    assert_eq!(
        token_count(&json!({"tokens":5000,"unknown_messages":2})),
        "5.0k (+2 unreported)"
    );
    assert_eq!(
        token_count(&json!({"tokens":0,"unknown_messages":2})),
        "unknown"
    );
    assert_eq!(
        token_count(&json!({"tokens":900,"response_total_unknown":true})),
        "900 known + unknown response usage"
    );
}

#[test]
fn drift_names_each_format_or_says_everything_was_recognized() {
    assert_eq!(drift(&json!([])), "every record and field was recognized\n");
    let item = json!({"format":"sample","by_design":true});
    assert!(drift_text(&json!([item])).contains(" (kept as a count by design)"));
    let item = json!({"format":"sample","by_design":false});
    assert!(!drift_text(&json!([item])).contains("by design"));
}

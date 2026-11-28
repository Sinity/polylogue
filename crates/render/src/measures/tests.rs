use super::*;
use serde_json::json;

#[test]
fn outlook_labels_estimates_populations_and_unknown_values() {
    let text = outlook(
        &json!({"provider":"sample","projection_method":"linear","window":{"start":0,"end":86400000},"segments":[{"plan":{"name":"configured"},"from":0,"until":86400000,"responses":{"api_nusd":{"used":1_000_000_001_i64,"projected":null,"known_items":1,"unknown_items":2,"lower_bound":true,"observed_days":1,"missing_days":[]}},"recorded_reports":{"reported_nusd":{"used":0,"projected":0,"known_items":1,"unknown_items":0,"observed_days":1,"missing_days":[]}},"subscription_price_share_nusd":null,"unknown_boundary_items":3,"unconfirmed_reports":4}],"untimed_responses":5,"untimed_reports":6}),
    );
    assert_eq!(
        text,
        "sample (estimate), method linear\n\
         UTC billing window 1970-01-01 00:00 UTC to 1970-01-02 00:00 UTC\n\
         configured: 1970-01-01 00:00 UTC to 1970-01-02 00:00 UTC\n\
         responses, API-equivalent USD: at least used $1.000000001, projected unknown (estimate); 1 known items, 2 unknown; 1 observed days, 0 missing days\n\
         recorded_reports, provider-reported USD: used $0, projected $0 (estimate); 1 known items, 0 unknown; 1 observed days, 0 missing days\n\
         Subscription price share: unknown. Unknown-boundary items: 3; unconfirmed reports: 4\n\
         Untimed responses: 5; untimed reports: 6; untimed unknown-boundary items: unknown. Billing before the first plan setting: unknown.\n"
    );
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
    assert_eq!(
        missing,
        "Cost outlook unavailable: missing\\x1b[2J. Configure plan.<provider> and choose --option provider:method.\n"
    );
    assert_eq!(
        outlook(&json!({"reason":"no_cycle_anchor"})),
        "Cost outlook unavailable: no_cycle_anchor. Set cycle_anchor_day in plan.<provider>.\n"
    );
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
    assert_eq!(time(&Value::Null, &answer_zone(&answer)), "unknown");
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
        {"repository":"root-a","repo":"same","command":"cargo test","at":0,"last_ok_at":null,"session":"s1","exit_code":1,"failures_since_ok":1},
        {"repository":"root-b","repo":"same","command":"cargo test","at":0,"last_ok_at":null,"session":"s2","exit_code":2,"failures_since_ok":3},
        {"repository":"root-a","repo":"same","command":"cargo clippy","at":0,"last_ok_at":null,"session":"s3","exit_code":101,"failures_since_ok":2},
    ]);
    assert_eq!(
        failing(&data, &jiff::tz::TimeZone::UTC),
        "same\n\
         \x20 cargo test  failed 1970-01-01 00:00 +00:00 (exit 1) in s1 · 1 failure(s) since ok · never succeeded\n\
         \x20 cargo clippy  failed 1970-01-01 00:00 +00:00 (exit 101) in s3 · 2 failure(s) since ok · never succeeded\n\
         same\n\
         \x20 cargo test  failed 1970-01-01 00:00 +00:00 (exit 2) in s2 · 3 failure(s) since ok · never succeeded\n"
    );
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

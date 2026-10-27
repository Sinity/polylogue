use super::*;

#[test]
fn new_measures_use_selected_sessions_and_profile_fields() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    assert!(run_q(&archive, ask("sessions | measure profiles:unsupported")).is_err());
    let all = q(&archive, "sessions | measure facets:deep");
    assert!(all["data"]["global"]["total_sessions"].as_u64().unwrap() > 1);
    let selected = q(
        &archive,
        "sessions where origin:claude-code | measure facets:deep-no-idf",
    );
    assert!(selected["data"]["idf"].as_object().unwrap().is_empty());
    assert!(
        selected["data"]["selected"]["total_sessions"]
            .as_u64()
            .unwrap()
            < all["data"]["global"]["total_sessions"].as_u64().unwrap()
    );
    let profiles = q(
        &archive,
        "sessions where workflow_shape:chat | measure profiles",
    );
    assert!(!profiles["data"].as_array().unwrap().is_empty());
    assert!(
        profiles["data"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["facts"]["workflow_shape"] == "chat")
    );
    let pace = q(
        &archive,
        "sessions where origin:claude-code | measure pace:60:0:1",
    );
    assert_eq!(pace["data"]["gaps"].as_array().unwrap().len(), 1);
    assert!(
        pace["data"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["session"].as_str().unwrap().starts_with("claude-code:"))
    );
    let distribution = q(
        &archive,
        "sessions | measure distribution:median_tool_call_ms",
    );
    assert!(distribution["data"]["total_sessions"].as_u64().is_some());
    let correlation = q(
        &archive,
        "sessions | measure correlation:message_count:word_count",
    );
    assert!(correlation["data"]["sample_count"].as_u64().unwrap() > 0);
    assert!(q(&archive, "sessions | measure compare")["data"]["profiles"].is_array());
    let similar = q(
        &archive,
        r#"sessions where id:"claude-code:7f3a2c10-0001-4000-8000-000000000001" | measure similar"#,
    );
    assert!(!similar["data"].as_array().unwrap().is_empty());
    assert!(run_q(&archive, ask("sessions | measure similar")).is_err());
    assert_eq!(
        q(&archive, "sessions | measure outlook")["data"]["reason"],
        "plans_not_configured"
    );
}

#[test]
fn outlook_settings_and_method_reach_the_public_query() {
    let mut archive = morrow();
    archive.fx.setting("cost.plans",r#"{"sample":{"origin":"claude-code","cycle_anchor_day":1,"quota_unit":"credits","quota":1000,"price_usd":20,"overage_rate_usd_per_unit":null}}"#).unwrap();
    let answer = q(
        &archive,
        "sessions | measure outlook:sample:trailing-7d-mean",
    );
    assert_eq!(answer["data"]["projection_method"], "trailing-7d-mean");
    assert_eq!(answer["data"]["estimate"], true);
    assert_eq!(
        answer["data"]["quota_pressure"]["reason"],
        "quota_unit_unobserved"
    );
    assert!(run_q(&archive, ask("sessions | measure outlook:missing")).is_err());
}

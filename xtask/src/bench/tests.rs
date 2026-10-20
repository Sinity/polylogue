use super::*;

#[test]
fn supplied_run_comparison_keeps_assumptions_and_unknown_identity_visible() {
    let estimate = bench::linear_estimate(Some(2.0), &BTreeMap::new(), Some(10.0));
    let report = validation_report(&estimate, &json!({"seconds": 25.0})).unwrap();
    assert_eq!(report["estimated_seconds"], 20.0);
    assert_eq!(report["calibrated"], false);
    assert_eq!(report["candidate_match"], "not_verified");
    assert_eq!(report["input_match"], "not_verified");
    assert!(report["full_run_load"].is_null());
    assert!(validation_report(&json!({}), &json!({"seconds": 25.0})).is_err());
    assert!(validation_report(&estimate, &json!({})).is_err());
    assert!(validation_report(&estimate, &json!({"seconds": -1.0})).is_err());
    assert!(
        validation_report(&estimate, &json!({"seconds": 0.0})).unwrap()["relative_error"].is_null()
    );
}

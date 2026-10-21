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

#[test]
fn first_build_refuses_ingestion_errors_or_empty_work() {
    let good = json!({"errors":0,"input_mb":8.0,"seconds":1.0,"messages_written":10,"unreadable_root_count":0});
    assert_eq!(
        first_build_result(&format!("progress\n{good}\n"), true).unwrap(),
        good
    );
    for (field, value) in [
        ("errors", json!(2)),
        ("unreadable_root_count", json!(1)),
        ("messages_written", json!(0)),
        ("input_mb", json!(0)),
        ("seconds", json!(0)),
    ] {
        let mut bad = good.clone();
        bad[field] = value;
        assert!(first_build_result(&bad.to_string(), true).is_err(), "{bad}");
    }
    let mut idle = good.clone();
    idle["messages_written"] = json!(0);
    assert_eq!(first_build_result(&idle.to_string(), false).unwrap(), idle);
    idle["errors"] = json!(1);
    assert!(first_build_result(&idle.to_string(), false).is_err());
    let mut unknown = good;
    unknown.as_object_mut().unwrap().remove("errors");
    assert!(first_build_result(&unknown.to_string(), true).is_err());
}

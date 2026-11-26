use super::*;

fn point(input_mb: f64, seconds: f64, stages: &[(&str, f64)]) -> Point {
    Point {
        input_mb,
        seconds,
        stages: stages.iter().map(|(n, s)| (n.to_string(), *s)).collect(),
    }
}

#[test]
fn each_stage_is_fitted_as_fixed_seconds_plus_seconds_per_mb() {
    let points = [
        point(1000.0, 110.0, &[("rows", 50.0), ("derive", 12.0)]),
        point(3000.0, 310.0, &[("rows", 150.0), ("derive", 12.0)]),
    ];
    let fitted = fit(&points, 10_000.0).unwrap();
    let close =
        |v: &Value, want: f64| assert!((v.as_f64().unwrap() - want).abs() < 1e-6, "{v} != {want}");
    close(&fitted["seconds"]["seconds"], 1010.0);
    close(&fitted["seconds"]["fixed_seconds"], 10.0);
    close(&fitted["stages"]["rows"]["seconds"], 500.0);
    close(&fitted["stages"]["derive"]["seconds"], 12.0);
    close(&fitted["stages"]["derive"]["seconds_per_gb"], 0.0);
}

#[test]
fn one_sample_size_is_not_a_fit() {
    let one = [point(1000.0, 1.0, &[]), point(1000.0, 2.0, &[])];
    assert_eq!(
        fit(&one, 5000.0).unwrap_err(),
        "a fit needs runs of one build at two or more sample sizes"
    );
    assert_eq!(
        fit(&[], 5000.0).unwrap_err(),
        "a fit needs runs of one build at two or more sample sizes"
    );
}

#[test]
fn a_full_run_is_compared_stage_by_stage() {
    let points = [
        point(1000.0, 100.0, &[("rows", 50.0)]),
        point(2000.0, 200.0, &[("rows", 100.0)]),
    ];
    let estimate = fit(&points, 4000.0).unwrap();
    let report = comparison(
        &estimate,
        &json!({"seconds": 500.0, "input_mb": 4000.0, "stages": {"rows": 160.0}}),
    )
    .unwrap();
    assert_eq!(report["seconds"]["estimated"], 400.0);
    assert_eq!(report["seconds"]["relative_error"], -0.2);
    assert_eq!(report["stages"]["rows"]["estimated"], 200.0);
    assert_eq!(report["stages"]["rows"]["actual"], 160.0);
    assert_eq!(
        comparison(&estimate, &json!({})).unwrap_err(),
        "the supplied run has no valid seconds value"
    );
}

#[test]
fn a_comparison_keeps_a_stage_only_the_full_run_recorded() {
    let points = [
        point(1000.0, 100.0, &[("rows", 50.0)]),
        point(2000.0, 200.0, &[("rows", 100.0)]),
    ];
    let estimate = fit(&points, 4000.0).unwrap();
    let report = comparison(
        &estimate,
        &json!({"seconds": 500.0, "input_mb": 4000.0, "stages": {"rows": 160.0, "derive": 30.0}}),
    )
    .unwrap();
    assert_eq!(report["stages"]["derive"]["actual"], 30.0);
    assert!(report["stages"]["derive"]["estimated"].is_null());
    assert!(report["stages"]["derive"]["relative_error"].is_null());
}

#[test]
fn a_pair_reports_each_change_from_the_first_build() {
    let diff = difference(
        &json!({"seconds": 100.0, "stages": {"rows": 40.0}}),
        &json!({"seconds": 80.0, "stages": {"rows": 10.0, "derive": 5.0}}),
    );
    assert_eq!(diff["seconds"]["change"], -0.2);
    assert_eq!(diff["stages"]["rows"]["change"], -0.75);
    assert!(diff["stages"]["derive"]["change"].is_null());
}

#[test]
fn a_negative_stage_prediction_refuses_an_otherwise_positive_wall_estimate() {
    let points = [
        point(1000.0, 110.0, &[("derive", 20.0)]),
        point(3000.0, 310.0, &[("derive", 10.0)]),
    ];
    let error = fit(&points, 10000.0).unwrap_err();
    assert_eq!(
        error,
        "the sampled derive times predict a negative or non-finite duration; these runs do not support a corpus estimate"
    );
}

#[test]
fn a_negative_wall_prediction_is_refused_without_clamping_it() {
    let points = [point(1000.0, 100.0, &[]), point(3000.0, 50.0, &[])];
    let error = fit(&points, 10000.0).unwrap_err();
    assert_eq!(
        error,
        "the sampled wall time times predict a negative or non-finite duration; these runs do not support a corpus estimate"
    );
}

#[test]
fn an_unknown_or_non_finite_corpus_size_is_not_estimated() {
    let points = [point(1000.0, 100.0, &[]), point(3000.0, 300.0, &[])];
    for input in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            fit(&points, input).unwrap_err(),
            "the corpus input size must be positive and finite",
            "input: {input}"
        );
    }
}

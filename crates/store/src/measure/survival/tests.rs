use super::*;

#[test]
fn kaplan_meier_matches_a_hand_computed_curve() {
    // Times 1,2,2(c),3,4(c): S(1)=4/5, S(2)=4/5*3/4, S(3)=...*1/2.
    let (curve, median) = kaplan_meier(vec![
        (1.0, true),
        (2.0, true),
        (2.0, false),
        (3.0, true),
        (4.0, false),
    ]);
    let s: Vec<f64> = curve.iter().map(|c| c.survival).collect();
    assert!((s[0] - 0.8).abs() < 1e-12);
    assert!((s[1] - 0.6).abs() < 1e-12);
    assert!((s[2] - 0.3).abs() < 1e-12);
    assert_eq!(median, Some(3.0));
    assert_eq!(curve[1].at_risk, 4);
}

#[test]
fn censoring_alone_never_lowers_survival() {
    let (curve, median) = kaplan_meier(vec![(1.0, false), (2.0, false)]);
    assert!(curve.is_empty());
    assert_eq!(median, None);
}

use super::*;

#[test]
fn a_level_shift_is_found_and_noise_is_not() {
    let mut xs: Vec<f64> = (0..20).map(|i| 10.0 + (i % 3) as f64).collect();
    xs.extend((0..20).map(|i| 50.0 + (i % 3) as f64));
    assert_eq!(changepoints(&xs, 7), vec![20]);
    let flat: Vec<f64> = (0..40).map(|i| 10.0 + (i % 3) as f64).collect();
    assert!(changepoints(&flat, 7).is_empty());
}

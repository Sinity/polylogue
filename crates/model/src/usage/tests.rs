use super::*;

#[test]
fn unknown_lanes_stay_unknown() {
    let u = Usage::default();
    assert!(u.is_empty());
    assert_eq!(u.total_input(), None);
    let u = Usage {
        input: Some(3),
        cached_input: Some(10),
        ..Usage::default()
    };
    assert_eq!(u.total_input(), Some(13));
}

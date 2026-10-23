use super::*;

#[test]
fn change_positions_preserve_both_counters_and_refuse_ambiguous_inputs() {
    let p = ChangePosition::parse("generation:12:3").unwrap();
    assert_eq!(p.index, 12);
    assert_eq!(p.archive, 3);
    assert_eq!(p.to_string(), "generation:12:3");
    for text in [
        "12.3",
        "generation:12",
        ":1:2",
        "g:-1:0",
        "g:0:-1",
        "g:1:2:3",
        "g g:1:2",
        "g:9223372036854775808:0",
    ] {
        assert!(ChangePosition::parse(text).is_none(), "{text}");
    }
}

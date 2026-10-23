use super::*;

#[test]
fn forward_chunks_fetch_each_normalized_interval_once() {
    let plan = Plan::new().unwrap();
    let ranges: Vec<_> = (0..10000).map(|n| (n * 2, n * 2 + 1)).collect();
    plan.matches("input", 1, &ranges).unwrap();
    plan.normalize("input", 1).unwrap();
    let key = ("input".into(), 1);
    let mut edits = plan.edits(&key, 0).unwrap();
    for at in 0..20000 {
        let mut bytes = *b"x";
        edits.blank(at, &mut bytes, b'*').unwrap();
        assert_eq!(bytes, if at % 2 == 0 { *b"*" } else { *b"x" });
    }
    assert!(edits.fetched <= 10001, "{} range lookups", edits.fetched);
    let mut earlier = *b"x";
    assert!(edits.blank(0, &mut earlier, b'*').is_err());
    assert_eq!(earlier, *b"x");
}

#[test]
fn one_normalized_overlap_carries_across_chunks_without_losing_hit_counts() {
    let plan = Plan::new().unwrap();
    plan.matches("input", 1, &[(1, 5), (3, 8), (3, 8)]).unwrap();
    plan.normalize("input", 1).unwrap();
    assert_eq!(plan.counts().unwrap()["input"], 2);
    let key = ("input".into(), 1);
    let mut edits = plan.edits(&key, 0).unwrap();
    let mut actual = b"ab\ncdefghi".to_vec();
    for (n, part) in actual.chunks_mut(2).enumerate() {
        edits.blank((n * 2) as u64, part, b'*').unwrap();
    }
    assert_eq!(actual, b"a*\n*****hi");
    assert_eq!(edits.fetched, 2);
}

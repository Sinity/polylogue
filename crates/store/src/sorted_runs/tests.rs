use super::*;

/// Records pushed in any order come out of the merge in key order, equal
/// keys in the order they were pushed, across spilled runs and a merge
/// pass over more runs than are merged at once.
#[test]
fn a_merge_of_spilled_runs_gives_every_record_in_key_order() {
    let dir = tempfile::tempdir().unwrap();
    // A budget of a few records spills often enough to need a merge pass.
    let mut sorter = Sorter::new(dir.path(), "t".into(), 64);
    let mut expected = Vec::new();
    for n in 0..2_000u32 {
        let key = ((n * 7919) % 500).to_be_bytes();
        let value = n.to_be_bytes();
        sorter.push(&key, &value).unwrap();
        expected.push((key.to_vec(), value.to_vec()));
    }
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    let runs = sorter.finish().unwrap();
    assert!(runs.len() > FAN_IN);
    let mut merged = Merger::open(runs, dir.path()).unwrap();
    let mut got = Vec::new();
    loop {
        // A peek shows the record the next call returns, and takes nothing.
        let peeked = merged
            .peek()
            .unwrap()
            .map(|(k, v)| (k.to_vec(), v.to_vec()));
        let Some((k, v)) = merged.next().unwrap() else {
            assert_eq!(peeked, None);
            break;
        };
        assert_eq!(peeked, Some((k.to_vec(), v.to_vec())));
        got.push((k.to_vec(), v.to_vec()));
    }
    assert_eq!(got, expected);
}

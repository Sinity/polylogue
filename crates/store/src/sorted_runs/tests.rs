use super::*;

/// Records pushed in any order come out of the merge in key order, equal
/// keys in the order they were pushed, across spilled runs and merge
/// passes over more runs than the merge memory reads at once.
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
    assert!(runs.len() > 4);
    let mut merged = Merger::open(runs, dir.path(), 4 * RUN_MEMORY).unwrap();
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

/// Runs that fit the merge memory are read as they were written; past it,
/// only the leading runs needed to bring the rest within it are written
/// again, as one run.
#[test]
fn a_merge_writes_again_only_the_runs_past_its_memory() {
    let dir = tempfile::tempdir().unwrap();
    let spill = |name: &str, runs: u32| {
        let mut sorter = Sorter::new(dir.path(), name.into(), 1);
        for n in 0..runs {
            sorter.push(&n.to_be_bytes(), name.as_bytes()).unwrap();
        }
        sorter.finish().unwrap()
    };
    let listing = || {
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    };
    let runs = spill("a", 3);
    let before = listing();
    let mut merged = Merger::open(runs, dir.path(), 3 * RUN_MEMORY).unwrap();
    assert_eq!(listing(), before);
    let mut keys = Vec::new();
    while let Some((key, _)) = merged.next().unwrap() {
        keys.push(key.to_vec());
    }
    assert_eq!(
        keys,
        (0..3u32)
            .map(|n| n.to_be_bytes().to_vec())
            .collect::<Vec<_>>()
    );

    let runs = spill("b", 5);
    let mut merged = Merger::open(runs, dir.path(), 3 * RUN_MEMORY).unwrap();
    // Five runs in the memory of three: the first three become one.
    let listed = listing();
    assert_eq!(
        listed[..5],
        [
            "a-00000.run",
            "a-00001.run",
            "a-00002.run",
            "b-00003.run",
            "b-00004.run"
        ]
    );
    assert_eq!(listed.len(), 6);
    assert!(listed[5].starts_with("merged-"));
    let mut keys = Vec::new();
    while let Some((key, _)) = merged.next().unwrap() {
        keys.push(key.to_vec());
    }
    assert_eq!(
        keys,
        (0..5u32)
            .map(|n| n.to_be_bytes().to_vec())
            .collect::<Vec<_>>()
    );
}

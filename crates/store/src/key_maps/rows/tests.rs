use super::*;

fn key(n: u64) -> [u8; 32] {
    *blake3::hash(&n.to_le_bytes()).as_bytes()
}

#[test]
fn spilled_updates_and_rewrites_keep_each_keys_latest_value() {
    let mut rows = KeyRows::new(64 * 1024).unwrap();
    for n in 0..1_000 {
        rows.set(key(n), n as i64 + 1).unwrap();
    }
    for n in 0..1_000 {
        assert_eq!(rows.get(&key(n)).unwrap(), Some(n as i64 + 1));
    }
    for n in 1_000..1_100 {
        assert_eq!(rows.get(&key(n)).unwrap(), None);
    }
    rows.transform_values(|value| -value).unwrap();
    for n in 0..1_000 {
        assert_eq!(rows.get(&key(n)).unwrap(), Some(-(n as i64 + 1)));
    }
    for n in (0..1_000).step_by(2) {
        rows.set(key(n), n as i64 + 1).unwrap();
    }
    for n in 0..1_000 {
        let expected = (n as i64 + 1) * if n % 2 == 0 { 1 } else { -1 };
        assert_eq!(rows.get(&key(n)).unwrap(), Some(expected));
    }
    rows.clear();
    for n in 0..1_000 {
        assert_eq!(rows.get(&key(n)).unwrap(), None);
    }
}

fn ordered(n: u64) -> [u8; 32] {
    let mut key = [0; 32];
    key[..8].copy_from_slice(&n.to_be_bytes());
    key
}

#[test]
fn spilled_lookup_narrows_disk_search_with_bounded_samples() {
    let mut writer = RunWriter::new(64 * 1024, 200_000).unwrap();
    for n in 0..200_000 {
        writer
            .push(Record {
                key: ordered(n * 2),
                value: n as i64,
                sequence: n,
            })
            .unwrap();
    }
    let run = writer.finish().unwrap();
    for n in [1, 401, 82_003, 199_999] {
        run.reads.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut found = Vec::new();
        let _ = run
            .visit(&ordered(n * 2), &mut |record| {
                found.push(record.value);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(found, vec![n as i64]);
        // A record binary search needs at least seventeen disk reads here.
        assert!(run.reads.load(std::sync::atomic::Ordering::Relaxed) <= 12);
        let mut missing = Vec::new();
        let _ = run
            .visit(&ordered(n * 2 - 1), &mut |record| {
                missing.push(record.value);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert!(missing.is_empty());
    }
}

#[test]
fn duplicate_keys_cross_sample_boundaries_without_losing_candidates() {
    let mut writer = RunWriter::new(1024, 1000).unwrap();
    for n in 0..1000 {
        writer
            .push(Record {
                key: ordered(if n < 83 {
                    0
                } else if n < 917 {
                    1
                } else {
                    2
                }),
                value: n,
                sequence: n as u64,
            })
            .unwrap();
    }
    let run = writer.finish().unwrap();
    let mut found = Vec::new();
    let _ = run
        .visit(&ordered(1), &mut |record| {
            found.push(record.value);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(found, (83..917).collect::<Vec<_>>());
    let mut calls = 0;
    let result = run
        .visit(&ordered(1), &mut |_| {
            calls += 1;
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert!(result.is_break());
    assert_eq!(calls, 1);
}

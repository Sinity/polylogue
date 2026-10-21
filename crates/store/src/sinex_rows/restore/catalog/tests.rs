use super::*;
use crate::sinex_rows::{ExcisionPublication, Replacement};

fn unit(sequence: i64, key: &str) -> RestoreUnit {
    RestoreUnit {
        unit: Unit {
            commit_seq: sequence,
            kind: "note".into(),
            key: key.into(),
        },
        batch: Some(format!("blake3:{}", "01".repeat(32))),
        bytes: None,
        rows: 1,
        blobs: vec![],
        excision: None,
    }
}

#[test]
fn reverse_confirmations_and_repeats_have_the_same_ordered_checkpoint() {
    let originals: Vec<_> = (1..=300)
        .map(|n| unit(n, &format!("note:{n:05}")))
        .collect();
    let mut catalog = Catalog::new().unwrap();
    for (index, unit) in originals.iter().enumerate().rev() {
        catalog.push(index as u64 + 1, unit).unwrap();
        catalog.push(index as u64 + 1, unit).unwrap();
    }
    let selection = catalog.finish().unwrap();
    assert_eq!(selection.len(), 300);
    assert_eq!(selection.commit_seq(), 300);
    assert_eq!(
        serde_json::to_vec(&selection).unwrap(),
        serde_json::to_vec(&originals).unwrap()
    );
    assert_eq!(
        selection.find(&originals[155].unit).unwrap(),
        Some(originals[155].clone())
    );
    assert!(selection.find(&unit(301, "absent").unit).unwrap().is_none());
}

#[test]
fn conflicts_gaps_and_bad_unit_order_refuse_before_install() {
    let mut conflicting = Catalog::new().unwrap();
    conflicting.push(1, &unit(1, "one")).unwrap();
    assert!(conflicting.push(1, &unit(1, "other")).is_err());
    assert!(conflicting.finish().is_err());
    for positions in [[1, 3], [0, 1], [2, 1]] {
        let mut catalog = Catalog::new().unwrap();
        let result = catalog
            .push(positions[0], &unit(1, "one"))
            .and_then(|_| catalog.push(positions[1], &unit(2, "two")));
        assert!(result.is_err() || catalog.finish().is_err());
    }
    let selection = Catalog::new().unwrap().finish().unwrap();
    assert!(selection.is_empty());
    assert_eq!(selection.commit_seq(), 0);
    assert_eq!(serde_json::to_vec(&selection).unwrap(), b"[]");
}

#[test]
fn later_replacements_preserve_exact_identity_and_authorize_only_their_tombstone() {
    let first = unit(1, "one");
    let mut excision = unit(2, "removal");
    excision.unit.kind = "excision".into();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![Replacement {
            unit: first.unit.clone(),
            batch: None,
            bytes: 0,
            rows: 0,
            blobs: vec![],
        }],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    excision.excision = Some(publication.clone());
    let mut catalog = Catalog::new().unwrap();
    catalog.push(2, &excision).unwrap();
    catalog.push(1, &first).unwrap();
    let selected = catalog.finish().unwrap();
    let changed = selected.find(&first.unit).unwrap().unwrap();
    assert!(changed.batch.is_none());
    assert_eq!(changed.rows, 0);
    assert_eq!(selected.find(&excision.unit).unwrap(), Some(excision));
    let mut illegal = first.clone();
    illegal.batch = None;
    illegal.rows = 0;
    let mut catalog = Catalog::new().unwrap();
    catalog.push(1, &illegal).unwrap();
    assert!(catalog.finish().is_err());
    publication.replacements[0].unit.key = "unknown".into();
    publication.digest = publication.computed_digest().unwrap();
    let mut invalid_excision = unit(2, "removal");
    invalid_excision.unit.kind = "excision".into();
    invalid_excision.excision = Some(publication);
    let mut catalog = Catalog::new().unwrap();
    catalog.push(1, &first).unwrap();
    catalog.push(2, &invalid_excision).unwrap();
    assert!(catalog.finish().is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn many_descriptors_stay_on_disk_instead_of_growing_the_process_heap() {
    fn peak_kib() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let before = peak_kib();
    let mut catalog = Catalog::new().unwrap();
    let suffix = "x".repeat(2048);
    for n in (1..=20_000).rev() {
        catalog
            .push(n as u64, &unit(n, &format!("note:{n:05}:{suffix}")))
            .unwrap();
    }
    let selection = catalog.finish().unwrap();
    assert_eq!(selection.len(), 20_000);
    assert_eq!(selection.commit_seq(), 20_000);
    let mut count = 0;
    selection
        .each(|_| {
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 20_000);
    let growth = peak_kib().saturating_sub(before);
    eprintln!("20,000 descriptors: process high-water growth {growth} KiB");
    assert!(
        growth < 32 * 1024,
        "metadata grew the process by {growth} KiB"
    );
}

#[test]
fn equal_sequence_keys_keep_rust_string_order_including_unicode_and_zero_bytes() {
    let mut originals: Vec<_> = ["a", "a\0", "a\0b", "a:b", "é", "𐀀"]
        .into_iter()
        .map(|key| unit(1, key))
        .collect();
    originals.sort_by(|a, b| a.unit.cmp(&b.unit));
    let mut catalog = Catalog::new().unwrap();
    for (index, unit) in originals.iter().enumerate().rev() {
        catalog.push(index as u64 + 1, unit).unwrap();
    }
    assert_eq!(
        serde_json::to_vec(&catalog.finish().unwrap()).unwrap(),
        serde_json::to_vec(&originals).unwrap()
    );
}

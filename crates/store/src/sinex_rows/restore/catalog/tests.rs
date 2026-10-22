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

#[test]
fn source_blobs_are_distinct_and_follow_only_the_final_replacements() {
    let hash = |byte: &str| format!("blake3:{}", byte.repeat(32));
    let old = hash("02");
    let shared = hash("03");
    let added = hash("04");
    let mut first = unit(1, "one");
    first.blobs = vec![old.clone(), shared.clone(), shared.clone()];
    let mut second = unit(2, "two");
    second.blobs = vec![shared.clone()];
    let mut excision = unit(3, "removal");
    excision.unit.kind = "excision".into();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![Replacement {
            unit: first.unit.clone(),
            batch: first.batch.clone(),
            bytes: 1,
            rows: 1,
            blobs: vec![shared.clone(), added.clone()],
        }],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    excision.excision = Some(publication);
    let mut catalog = Catalog::new().unwrap();
    for (index, row) in [first, second, excision].iter().enumerate() {
        catalog.push(index as u64 + 1, row).unwrap();
    }
    let selection = catalog.finish().unwrap();
    // Every verification pass repeats the complete selected source set.
    for _ in 0..2 {
        let mut seen = Vec::new();
        selection
            .verify_source_blobs(|name| {
                seen.push(name.to_owned());
                Ok(1)
            })
            .unwrap();
        assert_eq!(seen, [shared.clone(), added.clone()]);
        assert!(!seen.contains(&old));
    }
    let mut visited = 0;
    let error = selection
        .verify_source_blobs(|_| {
            visited += 1;
            Err(invalid("source bytes unavailable"))
        })
        .unwrap_err();
    assert_eq!(visited, 1);
    assert!(error.to_string().contains("source bytes unavailable"));
}

#[test]
fn a_tombstone_requires_none_of_its_old_source_blobs() {
    let mut first = unit(1, "one");
    first.blobs = vec![format!("blake3:{}", "03".repeat(32))];
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
    excision.excision = Some(publication);
    let mut catalog = Catalog::new().unwrap();
    catalog.push(1, &first).unwrap();
    catalog.push(2, &excision).unwrap();
    catalog
        .finish()
        .unwrap()
        .verify_source_blobs(|_| panic!("retired source was requested"))
        .unwrap();
    Catalog::new()
        .unwrap()
        .finish()
        .unwrap()
        .verify_source_blobs(|_| panic!("empty view requested bytes"))
        .unwrap();
}

#[test]
fn nested_manifest_checkpoint_keeps_exact_bytes_and_unit_identity() {
    let mut first = unit(1, "input:é\0\"/1");
    first.unit.kind = "revision".into();
    first.blobs = (0..512)
        .map(|n: u64| format!("blake3:{}", blake3::hash(&n.to_le_bytes())))
        .collect();
    let mut excision = unit(2, "remove:\"\n\0");
    excision.unit.kind = "excision".into();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![Replacement {
            unit: first.unit.clone(),
            batch: first.batch.clone(),
            bytes: 10,
            rows: 1,
            blobs: first.blobs.clone(),
        }],
        retired_batches: vec![format!("blake3:{}", blake3::hash(b"old"))],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    excision.excision = Some(publication);
    let mut catalog = Catalog::new().unwrap();
    catalog.push(1, &first).unwrap();
    catalog.push(2, &excision).unwrap();
    let selected = catalog.finish().unwrap();
    first.bytes = Some(10);
    let expected = vec![first, excision];
    assert_eq!(
        serde_json::to_vec(&(Some("archive:é\n"), &selected)).unwrap(),
        serde_json::to_vec(&(Some("archive:é\n"), &expected)).unwrap()
    );
    let mut identities = Vec::new();
    selected
        .each_unit(|unit| {
            identities.push(unit.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        identities,
        expected.iter().map(|u| u.unit.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn partial_source_verification_invalidates_an_earlier_complete_pass() {
    let mut row = unit(1, "note1");
    row.blobs = vec![
        format!("blake3:{}", "01".repeat(32)),
        format!("blake3:{}", "02".repeat(32)),
    ];
    let mut catalog = Catalog::new().unwrap();
    catalog.push(1, &row).unwrap();
    let selected = catalog.finish().unwrap();
    selected.verify_source_blobs(|_| Ok(3)).unwrap();
    let mut count = 0;
    assert!(
        selected
            .verify_source_blobs(|_| {
                count += 1;
                if count == 2 {
                    Err(invalid("late source failure"))
                } else {
                    Ok(3)
                }
            })
            .is_err()
    );
    assert_eq!(count, 2);
    assert!(selected.begin_install().is_err());
    let mut count = 0;
    selected
        .verify_source_blobs(|_| {
            count += 1;
            Ok(3)
        })
        .unwrap();
    assert_eq!(count, 2);
    selected.begin_install().unwrap();
    assert!(selected.begin_install().is_err());
    assert!(selected.verify_source_blobs(|_| Ok(u64::MAX)).is_err());
    assert!(selected.begin_install().is_err());
}

#[test]
fn source_verification_deduplicates_canonical_hashes() {
    let mut row = unit(1, "note1");
    let hash = blake3::hash(b"synthetic source").to_hex().to_string();
    row.blobs = vec![
        format!("blake3:{hash}"),
        format!("blake3:{}", hash.to_uppercase()),
    ];
    let mut catalog = Catalog::new().unwrap();
    catalog.push(1, &row).unwrap();
    let selected = catalog.finish().unwrap();
    let mut calls = 0;
    selected
        .verify_source_blobs(|_| {
            calls += 1;
            Ok(16)
        })
        .unwrap();
    assert_eq!(calls, 1);
    selected.begin_install().unwrap();
}

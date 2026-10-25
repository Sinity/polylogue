use super::*;

fn source() -> Connection {
    let source = Connection::open_in_memory().unwrap();
    source
        .execute_batch("CREATE TABLE extent(input TEXT,blob BLOB)")
        .unwrap();
    for hash in [vec![2; 32], vec![1; 32], vec![2; 32], vec![3; 32]] {
        source
            .execute("INSERT INTO extent VALUES ('input',?1)", [hash])
            .unwrap();
    }
    source
}

#[test]
fn captured_blob_order_survives_source_mutation_and_ranges_keep_multiplicity() {
    let source = source();
    let expected: Vec<Vec<u8>> = source
        .prepare("SELECT DISTINCT blob FROM extent WHERE input='input'")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let owner = Erasures::new().unwrap();
    owner.freeze_literals(&source, "input").unwrap();
    let mut hashes = vec![];
    owner
        .literal_blobs(|hash, ranges| {
            hashes.push(hash.to_vec());
            source.execute("DELETE FROM extent", []).unwrap();
            ranges.begin()?;
            for (start, end) in [(4, 9), (1, 4), (2, 3), (1, 4)] {
                ranges.insert(start, end)?;
            }
            assert!(ranges.ranges(|_, _| Ok(())).is_err());
            assert_eq!(ranges.finish()?, 4);
            let mut actual = vec![];
            ranges.ranges(|start, end| {
                actual.push((start, end));
                Ok(())
            })?;
            assert_eq!(actual, [(1, 4), (1, 4), (2, 3), (4, 9)]);
            assert!(
                ranges
                    .ranges(|_, _| Err(StoreError::NotFound("synthetic".into())))
                    .is_err()
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(hashes, expected);
}

#[test]
fn zero_matches_are_complete_and_a_new_scan_invalidates_prior_results() {
    let owner = Erasures::new().unwrap();
    owner.freeze_literals(&source(), "input").unwrap();
    owner
        .literal_blobs(|_, ranges| {
            assert!(ranges.ranges(|_, _| Ok(())).is_err());
            ranges.begin()?;
            assert!(ranges.ranges(|_, _| Ok(())).is_err());
            assert_eq!(ranges.finish()?, 0);
            ranges.ranges(|_, _| panic!("completed empty scan visited a range"))?;
            ranges.begin()?;
            ranges.insert(5, 8)?;
            assert_eq!(ranges.finish()?, 1);
            ranges.begin()?;
            assert!(ranges.ranges(|_, _| Ok(())).is_err());
            assert_eq!(ranges.finish()?, 0);
            ranges.ranges(|_, _| panic!("a new scan kept an earlier range"))?;
            Ok(())
        })
        .unwrap();
}

#[test]
fn many_literal_hits_use_the_existing_disk_owner_without_deduplication() {
    let owner = Erasures::new().unwrap();
    owner.freeze_literals(&source(), "input").unwrap();
    owner
        .literal_blobs(|_, ranges| {
            ranges.begin()?;
            for n in (0..50_000).rev() {
                ranges.insert(n * 4, n * 4 + 2)?;
                ranges.insert(n * 4, n * 4 + 2)?;
            }
            assert_eq!(ranges.finish()?, 100_000);
            let mut count = 0;
            ranges.ranges(|start, end| {
                assert_eq!(start, (count / 2) * 4);
                assert_eq!(end, start + 2);
                count += 1;
                Ok(())
            })?;
            assert_eq!(count, 100_000);
            Ok(())
        })
        .unwrap();
    let bytes: i64 = owner
        .conn
        .query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(bytes > 1 << 20);
    assert_eq!(owner.conn.path(), Some(""));
}

#[cfg(feature = "faults")]
#[test]
fn failed_scans_do_not_complete_and_retry_discards_their_partial_hits() {
    use crate::fault::{self, Action};
    for point in [
        "excision_literals.insert",
        "excision_literals.complete",
        "excision_literals.read",
    ] {
        fault::clear();
        let owner = Erasures::new().unwrap();
        owner.freeze_literals(&source(), "input").unwrap();
        owner
            .literal_blobs(|_, ranges| {
                ranges.begin()?;
                ranges.insert(1, 4)?;
                fault::arm(point, "", Action::Fail, 0, 1);
                if point.ends_with("insert") {
                    assert!(ranges.insert(5, 8).is_err());
                    assert!(ranges.ranges(|_, _| Ok(())).is_err());
                } else if point.ends_with("complete") {
                    assert!(ranges.finish().is_err());
                    assert!(ranges.ranges(|_, _| Ok(())).is_err());
                } else {
                    ranges.finish()?;
                    assert!(ranges.ranges(|_, _| Ok(())).is_err());
                }
                ranges.begin()?;
                ranges.insert(10, 13)?;
                assert_eq!(ranges.finish()?, 1);
                let mut actual = vec![];
                ranges.ranges(|start, end| {
                    actual.push((start, end));
                    Ok(())
                })?;
                assert_eq!(actual, [(10, 13)]);
                Ok(())
            })
            .unwrap();
    }
    fault::clear();
}

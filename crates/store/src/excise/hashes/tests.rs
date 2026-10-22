use super::*;
use std::collections::BTreeSet;

#[test]
fn hash_policy_bytes_and_counts_match_the_previous_sorted_set() {
    let mut db = Connection::open_in_memory().unwrap();
    for size in [0, 1, 2, 17, 4096] {
        let tx = db.transaction().unwrap();
        let hashes = RemovalHashes::new(&tx).unwrap();
        let mut expected = BTreeSet::new();
        for n in (0..size).rev() {
            let hash = polylogue_model::BlobHash::of(format!("synthetic-{n}").as_bytes()).0;
            hashes.text(&hash).unwrap();
            hashes.text(&hash).unwrap();
            expected.insert(hash.to_vec());
        }
        let old: Vec<_> = expected
            .iter()
            .map(|hash| crate::sinex_rows::hex(hash))
            .collect();
        let old_json = serde_json::to_string(&old).unwrap();
        assert_eq!(hashes.text_policy_json().unwrap(), old_json);
        assert_eq!(hashes.text_count().unwrap(), expected.len() as u64);
        let mut selected = Vec::new();
        hashes
            .visit_texts(|hash| {
                selected.push(hash.to_vec());
                Ok(())
            })
            .unwrap();
        assert_eq!(selected, expected.into_iter().collect::<Vec<_>>());
        hashes.finish().unwrap();
        assert_eq!(RemovalHashes::new(&tx).unwrap().text_count().unwrap(), 0);
        tx.commit().unwrap();
    }
}

#[test]
fn blob_candidates_are_distinct_from_text_exclusion_authority() {
    let mut db = Connection::open_in_memory().unwrap();
    let tx = db.transaction().unwrap();
    let hashes = RemovalHashes::new(&tx).unwrap();
    hashes.text(&[4; 32]).unwrap();
    for hash in [[9; 32], [2; 32], [9; 32]] {
        hashes.blob_candidate(&hash).unwrap();
    }
    let mut blobs = Vec::new();
    hashes
        .visit_blob_candidates(|hash| {
            blobs.push(hash.to_vec());
            Ok(())
        })
        .unwrap();
    assert_eq!(blobs, [vec![2; 32], vec![9; 32]]);
    assert_eq!(hashes.text_count().unwrap(), 1);
    assert_eq!(
        hashes.text_policy_json().unwrap(),
        serde_json::to_string(&vec![crate::sinex_rows::hex(&[4; 32])]).unwrap()
    );
    hashes.finish().unwrap();
    let hashes = RemovalHashes::new(&tx).unwrap();
    hashes
        .visit_blob_candidates(|_| panic!("stale candidate"))
        .unwrap();
    hashes.visit_texts(|_| panic!("stale text")).unwrap();
    hashes.finish().unwrap();
}

#[test]
fn hash_callback_and_database_failures_refuse_then_rollback_allows_reuse() {
    let mut db = Connection::open_in_memory().unwrap();
    let tx = db.transaction().unwrap();
    let hashes = RemovalHashes::new(&tx).unwrap();
    hashes.text(&[1; 32]).unwrap();
    hashes.blob_candidate(&[2; 32]).unwrap();
    let error = hashes
        .visit_texts(|_| Err(StoreError::Invalid("visit failed".into())))
        .unwrap_err();
    assert!(matches!(error,StoreError::Invalid(s) if s=="visit failed"));
    let error = hashes
        .visit_blob_candidates(|_| Err(StoreError::Invalid("blob visit failed".into())))
        .unwrap_err();
    assert!(matches!(error,StoreError::Invalid(s) if s=="blob visit failed"));
    tx.execute_batch(
        "DROP TABLE temp.excision_removed_text; DROP TABLE temp.excision_candidate_blob",
    )
    .unwrap();
    assert!(hashes.text(&[3; 32]).is_err());
    assert!(hashes.blob_candidate(&[3; 32]).is_err());
    assert!(hashes.text_count().is_err());
    assert!(hashes.text_policy_json().is_err());
    assert!(hashes.finish().is_err());
    tx.rollback().unwrap();
    let tx = db.transaction().unwrap();
    let hashes = RemovalHashes::new(&tx).unwrap();
    assert_eq!(hashes.text_count().unwrap(), 0);
    hashes
        .visit_blob_candidates(|_| panic!("failed operation leaked"))
        .unwrap();
    hashes.text(&[8; 32]).unwrap();
    hashes.finish().unwrap();
    tx.commit().unwrap();
    // Existing empty definitions also survive a later failed operation cleanly.
    let tx = db.transaction().unwrap();
    let hashes = RemovalHashes::new(&tx).unwrap();
    hashes.text(&[9; 32]).unwrap();
    hashes.blob_candidate(&[6; 32]).unwrap();
    assert!(
        hashes
            .visit_texts(|_| Err(StoreError::Invalid("later failure".into())))
            .is_err()
    );
    tx.rollback().unwrap();
    let tx = db.transaction().unwrap();
    let hashes = RemovalHashes::new(&tx).unwrap();
    assert_eq!(hashes.text_count().unwrap(), 0);
    hashes
        .visit_blob_candidates(|_| panic!("stale failed candidate"))
        .unwrap();
    hashes.finish().unwrap();
}

#[test]
fn accumulator_clears_rows_while_other_temp_cursors_remain_open() {
    let mut db = Connection::open_in_memory().unwrap();
    let tx = db.transaction().unwrap();
    tx.execute_batch("CREATE TEMP TABLE selection(id); INSERT INTO selection VALUES(1),(2)")
        .unwrap();
    let mut statement = tx.prepare("SELECT id FROM selection").unwrap();
    let mut rows = statement.query([]).unwrap();
    while let Some(row) = rows.next().unwrap() {
        let id: u8 = row.get(0).unwrap();
        let hashes = RemovalHashes::new(&tx).unwrap();
        assert_eq!(hashes.text_count().unwrap(), 0);
        hashes.text(&[id; 32]).unwrap();
        hashes.blob_candidate(&[id; 32]).unwrap();
        hashes.finish().unwrap();
    }
}

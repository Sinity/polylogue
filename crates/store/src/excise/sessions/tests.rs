use super::*;

#[test]
fn frozen_sessions_preserve_order_duplicates_and_opaque_ids_after_deletion() {
    let mut db = Connection::open_in_memory().unwrap();
    db.execute_batch("CREATE TABLE session(pk INTEGER PRIMARY KEY,id TEXT)")
        .unwrap();
    let tx = db.transaction().unwrap();
    let selected = SelectedSessions::new(&tx).unwrap();
    assert!(selected.is_empty().unwrap());
    let ids = ["z:一\0two", "a", "z:一\0two"];
    for (pk, id) in [9, 2, 9].into_iter().zip(ids) {
        tx.execute(
            "INSERT OR IGNORE INTO session VALUES (?1,?2)",
            params![pk, id],
        )
        .unwrap();
        selected.insert(pk, id).unwrap();
    }
    assert!(!selected.is_empty().unwrap());
    tx.execute_batch("DELETE FROM session").unwrap();
    assert_eq!(selected.ids().unwrap(), ids);
    let mut pks = Vec::new();
    selected
        .visit(|pk, _| {
            pks.push(pk);
            Ok(())
        })
        .unwrap();
    assert_eq!(pks, [9, 2, 9]);
    selected.finish().unwrap();
    tx.commit().unwrap();
}

#[test]
fn session_selection_is_reusable_after_success_or_transaction_rollback() {
    let mut db = Connection::open_in_memory().unwrap();
    // Rollback also removes a selection abandoned after an error or preview.
    for finish in [false, true, false, true] {
        let tx = db.transaction().unwrap();
        let selected = SelectedSessions::new(&tx).unwrap();
        assert!(selected.is_empty().unwrap());
        selected.insert(1, "synthetic-session").unwrap();
        if finish {
            selected.finish().unwrap();
            tx.commit().unwrap();
        } else {
            tx.rollback().unwrap();
        }
    }
}

#[test]
fn session_selection_propagates_setup_insert_read_and_callback_errors() {
    let mut db = Connection::open_in_memory().unwrap();
    let tx = db.transaction().unwrap();
    let selected = SelectedSessions::new(&tx).unwrap();
    assert!(SelectedSessions::new(&tx).is_err());
    selected.insert(1, "synthetic-session").unwrap();
    let error = selected
        .visit(|_, _| Err(crate::StoreError::Invalid("visit failed".into())))
        .unwrap_err();
    assert!(matches!(error, crate::StoreError::Invalid(s) if s == "visit failed"));
    tx.execute_batch("DROP TABLE temp.excision_sessions")
        .unwrap();
    assert!(selected.insert(2, "unwritten").is_err());
    assert!(selected.ids().is_err());
    assert!(selected.is_empty().is_err());
    assert!(selected.finish().is_err());
    tx.rollback().unwrap();
    let tx = db.transaction().unwrap();
    SelectedSessions::new(&tx).unwrap().finish().unwrap();
    tx.commit().unwrap();
}

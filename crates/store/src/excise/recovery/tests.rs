use super::*;

fn applied() -> (Connection, Connection) {
    let archive = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&archive).unwrap();
    archive.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('a','session:claude-code:gone',1,1)", []).unwrap();
    let index = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    let tx = index.unchecked_transaction().unwrap();
    acknowledge(&archive, &tx).unwrap();
    tx.commit().unwrap();
    (archive, index)
}

#[test]
fn newer_excision_metadata_preserves_policy_but_changed_rules_refuse() {
    for change in [
        "UPDATE excision SET reason='new explanation',commit_seq=7",
        "UPDATE excision SET scope='session:claude-code:other'",
        "UPDATE excision SET refused_json='[1]'",
        "UPDATE excision SET message_ids_json='[1]'",
        "DELETE FROM excision",
    ] {
        let (archive, index) = applied();
        assert!(!needs_rebuild(&archive, &index).unwrap());
        archive.execute_batch(change).unwrap();
        let result = needs_rebuild(&archive, &index);
        if change.contains("new explanation") {
            assert!(result.unwrap());
        } else {
            assert!(
                matches!(result, Err(StoreError::ExcisionConflict { .. })),
                "{change}: {result:?}"
            );
        }
    }
    let (archive, index) = applied();
    archive.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('b','session:claude-code:other',1,2)", []).unwrap();
    assert!(needs_rebuild(&archive, &index).unwrap());
}

#[test]
fn an_index_claiming_applied_removals_cannot_lose_its_receipts() {
    let (archive, index) = applied();
    index.execute_batch("DROP TABLE excision_applied").unwrap();
    assert!(matches!(
        needs_rebuild(&archive, &index),
        Err(StoreError::ExcisionConflict { .. })
    ));
}

#[test]
fn a_pre_revision_index_can_adopt_current_rules_during_rebuild() {
    let (archive, index) = applied();
    index.execute_batch("DROP TABLE excision_applied; DELETE FROM meta WHERE key IN ('excision_version','excision_revision');").unwrap();
    assert!(needs_rebuild(&archive, &index).unwrap());
}

use super::*;
use polylogue_model::BlobHash;
use serde_json::json;

fn proof(position: i64) -> crate::primary::RawInputProof {
    crate::primary::RawInputProof {
        stamp: "owner-write-proof".into(),
        position,
        policy_epoch: 1,
        scope: "synthetic-scope".into(),
    }
}

fn binding() -> Product {
    let version = BlobHash::of(b"original message");
    let hash = BlobHash::of(b"original receiving bytes");
    let record = serde_json::from_value(json!({
        "context":{"message":"codex:held-message","source_version":version,"input":"received-record","chain":1,"import_id":"received-namespace"},
        "claim":{"source_version":version,"namespace":{"input":"original-source","chain":2,"import_id":"original-namespace"},"exported_from":{"input":"export-source","chain":3,"import_id":"export-namespace"},"block":0,"attachment":0,"source_id":"held-attachment","candidates":[{"hash":hash,"size":24,"sources":[]}]},
        "members":[{"candidate":0,"witness":{"version":1,"member":{"import_id":"received-namespace","input":"received-member","member":"blobs/member","chain":1,"source_version":"original-material-pin","hash":hash,"size":24},"observation":"original-member-observation"}}]
    }))
    .unwrap();
    Product::AttachmentBinding {
        record: Box::new(record),
    }
}

#[test]
fn canonical_custody_survives_restart_without_proof_and_rolls_back_proof_attachment() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index.db");
    let mut conn = Connection::open(&path).unwrap();
    conn.execute_batch(SCHEMA).unwrap();
    let writes = vec![ProductWrite {
        product: binding(),
        location: None,
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    }];
    let tx = conn.transaction().unwrap();
    stage_products(&tx, "batch", &writes, None).unwrap();
    tx.commit().unwrap();
    drop(conn);
    let mut conn = Connection::open(&path).unwrap();
    let frozen: Vec<u8> = conn
        .query_row(
            "SELECT body FROM primary_derived WHERE identity='batch'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    {
        let tx = conn.transaction().unwrap();
        stage_products(&tx, "batch", &writes, Some(&proof(7))).unwrap();
        tx.rollback().unwrap();
    }
    let stored: Option<Vec<u8>> = conn
        .query_row(
            "SELECT proof FROM primary_derived WHERE identity='batch'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(stored.is_none());
    let tx = conn.transaction().unwrap();
    stage_products(&tx, "batch", &writes, Some(&proof(7))).unwrap();
    tx.commit().unwrap();
    let mut refreshed = proof(7);
    refreshed.stamp = "refreshed-owner-stamp".into();
    let tx = conn.transaction().unwrap();
    stage_products(&tx, "batch", &writes, Some(&refreshed)).unwrap();
    tx.commit().unwrap();
    let (body, stored): (Vec<u8>, Vec<u8>) = conn
        .query_row(
            "SELECT body,proof FROM primary_derived WHERE identity='batch'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(body, frozen);
    assert_eq!(
        crate::encoding::unpack_json::<crate::primary::RawInputProof>(&stored).unwrap(),
        proof(7)
    );
    let mut changed = writes.clone();
    changed[0].output_index = 1;
    let tx = conn.transaction().unwrap();
    assert!(matches!(
        stage_products(&tx, "batch", &changed, None),
        Err(StoreError::Conflict(_))
    ));
    tx.rollback().unwrap();
}

use super::*;
use polylogue_store::primary::{
    CanonicalCustody, Confirmed, ConfirmedChange, RawInputProof, ReplayControl, Sinex as StoreSinex,
};
use polylogue_store::{Archive, ReadPool, SqlValue, StoreError, Writer};
use std::sync::Arc;

pub(super) fn witness() -> polylogue_store::assets::bindings::AttachmentBindingRecord {
    let version = BlobHash::of(b"original message");
    let hash = BlobHash::of(b"original receiving bytes");
    // Imported serialized metadata is held intact and supplies no admission authority.
    serde_json::from_value(json!({
        "context":{"message":"codex:held-message","source_version":version,"input":"received-record","chain":1,"import_id":"received-namespace"},
        "claim":{"source_version":version,"namespace":{"input":"original-source","chain":2,"import_id":"original-namespace"},"exported_from":{"input":"export-source","chain":3,"import_id":"export-namespace"},"block":0,"attachment":0,"source_id":"held-attachment","candidates":[{"hash":hash,"size":24,"sources":[]}]},
        "members":[{"candidate":0,"witness":{"version":1,"member":{"import_id":"received-namespace","input":"received-member","member":"blobs/member","chain":1,"source_version":"original-material-pin","hash":hash,"size":24},"observation":"original-member-observation"}}]
    })).unwrap()
}
fn held(sinex: &Sinex) -> Vec<CanonicalCustody> {
    let mut values = Vec::new();
    sinex
        .visit_canonical_custody(&mut |value| {
            values.push(value.clone());
            Ok(())
        })
        .unwrap();
    values
}

#[test]
fn real_sdk_retains_revoked_witness_identity_and_history_after_index_move_aside() {
    let root = tempfile::tempdir().unwrap();
    let spool = root.path().join("producer");
    let cache = root.path().join("cache");
    let byte_root = root.path().join("bytes");
    let original;
    let original_sql_body;
    let history = RawInputProof {
        stamp: "historical-owner-receipt".into(),
        position: 3,
        policy_epoch: 7,
        scope: "old-scope".into(),
    };
    {
        let sinex = Arc::new(client(&spool));
        let (archive, lock) = Archive::open_primary(&cache, &byte_root).unwrap();
        let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
        let mut staged = polylogue_store::export_items::ExportItems::new().unwrap();
        staged
            .push(
                0..0,
                Vec::new(),
                None,
                &serde_json::to_vec(&polylogue_store::assets::AssetRow::Binding {
                    record: witness(),
                })
                .unwrap(),
            )
            .unwrap();
        assert!(matches!(
            writer.import_assets(staged),
            Err(StoreError::ReferenceUnavailable { .. })
        ));
        let prepared = held(&sinex);
        assert_eq!(prepared.len(), 1);
        original = prepared[0].clone();
        assert_eq!(original.products.len(), 1);
        assert!(original.historical_proof.is_none());
        assert!(!original.revoked);
        original_sql_body = ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .query(
                "SELECT body FROM primary_derived WHERE identity=?1",
                &[SqlValue::Text(original.identity.clone())],
            )
            .unwrap()[0]["body"]
            .clone();
        // A distinct original held intent retains its own historical receipt.
        StoreSinex::stage_canonical_products(
            sinex.as_ref(),
            "historical-held-intent",
            &original.products,
            Some(&history),
        )
        .unwrap();
        writer
            .apply_confirmed(vec![Confirmed {
                position: 0,
                output_index: 0,
                event: String::new(),
                native: None,
                change: ConfirmedChange::Control(Box::new(ReplayControl::Reset {
                    scope: "new-scope".into(),
                    policy_epoch: 8,
                })),
            }])
            .unwrap();
        assert!(held(&sinex).iter().all(|value| value.revoked));
        drop(writer);
        drop(lock);
        drop(archive);
        drop(sinex);
    }
    // The producer root is independent of the rebuildable index and remains intact.
    std::fs::rename(
        cache.join("index.db"),
        cache.join("index-before-rebuild.db"),
    )
    .unwrap();
    let sinex = Arc::new(client(&spool));
    let recovered = held(&sinex);
    let same = recovered
        .iter()
        .find(|value| value.identity == original.identity)
        .unwrap();
    assert_eq!(same.products, original.products);
    assert!(same.revoked);
    let historical = recovered
        .iter()
        .find(|value| value.identity == "historical-held-intent")
        .unwrap();
    assert_eq!(historical.historical_proof.as_ref(), Some(&history));
    assert!(historical.revoked);
    let (archive, _lock) = Archive::open_primary(&cache, &byte_root).unwrap();
    let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader.query("SELECT body,revoked,proof IS NULL AS no_publication_proof FROM primary_derived WHERE identity=?1", &[SqlValue::Text(original.identity.clone())]).unwrap();
    assert_eq!(rows[0]["body"], original_sql_body);
    assert_eq!(rows[0]["revoked"], 1);
    assert_eq!(rows[0]["no_publication_proof"], 1);
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM attachment_binding", &[])
            .unwrap()[0]["n"],
        0
    );
    assert_eq!(reader.confirmed_replay_state().unwrap().publication, None);
    let renewed = RawInputProof {
        stamp: "new-owner-proof".into(),
        position: 0,
        policy_epoch: 8,
        scope: "new-scope".into(),
    };
    assert!(matches!(
        sinex.prepare_canonical_products(&original.identity, &original.products, &renewed),
        Err(crate::Error::Refused { .. })
    ));
    assert!(held(&sinex).iter().all(|value| value.revoked));
    drop(reader);
    drop(pool);
    drop(writer);
}

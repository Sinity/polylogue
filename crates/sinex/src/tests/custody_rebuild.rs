use super::*;
use polylogue_store::{Archive, ReadPool, Writer};
use std::sync::Arc;

pub(super) fn witness() -> polylogue_store::assets::bindings::AttachmentBindingRecord {
    let version = BlobHash::of(b"original message");
    let hash = BlobHash::of(b"original receiving bytes");
    serde_json::from_value(json!({
        "context":{"message":"codex:held-message","source_version":version,"input":"received-record","chain":1,"import_id":"received-namespace"},
        "claim":{"source_version":version,"namespace":{"input":"original-source","chain":2,"import_id":"original-namespace"},"exported_from":{"input":"export-source","chain":3,"import_id":"export-namespace"},"block":0,"attachment":0,"source_id":"held-attachment","candidates":[{"hash":hash,"size":24,"sources":[]}]},
        "members":[{"candidate":0,"witness":{"version":1,"member":{"import_id":"received-namespace","input":"received-member","member":"blobs/member","chain":1,"source_version":"original-material-pin","hash":hash,"size":24},"observation":"original-member-observation"}}]
    })).unwrap()
}
#[test]
fn an_imported_binding_waits_in_the_spool_through_an_index_move_aside() {
    let root = tempfile::tempdir().unwrap();
    let spool = root.path().join("producer");
    let cache = root.path().join("cache");
    let byte_root = root.path().join("bytes");
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
        writer.import_assets(staged).unwrap();
        assert_eq!(sinex.status().unwrap().prepared_materials, 1);
        drop(writer);
        drop(lock);
        drop(archive);
    }
    // The spool is independent of the rebuildable index.
    std::fs::rename(
        cache.join("index.db"),
        cache.join("index-before-rebuild.db"),
    )
    .unwrap();
    let sinex = Arc::new(client(&spool));
    assert_eq!(sinex.status().unwrap().prepared_materials, 1);
    let (archive, _lock) = Archive::open_primary(&cache, &byte_root).unwrap();
    let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM attachment_binding", &[])
            .unwrap()[0]["n"],
        0,
        "only the confirmed stream writes the index"
    );
    drop(reader);
    drop(pool);
    drop(writer);
}

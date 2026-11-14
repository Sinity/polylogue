use super::*;
use polylogue_model::authored::{AuthoredEffect, PreparedRequest};
use polylogue_model::user::{UserOutcome, UserWrite};
use polylogue_store::{Archive, ReadPool};
use std::io::Read;

#[test]
fn rebuilding_the_source7_cache_preserves_actual_pending_authored_intent() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("native");
    let root = dir.path().join("archive");
    let predecessor = client(&dir.path().join("request-source"));
    let request = PreparedRequest {
        request: polylogue_store::fixture::authored::operator_request(
            "pending-through-primary-cache-transition".into(),
            UserWrite::Assert { assertions: vec![] },
        ),
        outcome: UserOutcome {
            id: "pending-through-primary-cache-transition".into(),
            repeat: false,
            pending: false,
        },
        effect: AuthoredEffect::default(),
    };
    predecessor.prepare_request(&request).unwrap();
    let identity = "request:pending-through-primary-cache-transition";
    let mut original = predecessor
        .spool
        .lock()
        .unwrap()
        .decode_frame(identity)
        .unwrap();
    let sinex_protocol::Ingress::Material { material, .. } = &mut original else {
        panic!("authored intent must retain its original material");
    };
    material.source.version = 7;
    let original_body = serde_json::to_vec(&request).unwrap();
    let queue = crate::spool::Spool::open(&store, "polylogue-test").unwrap();
    queue
        .prepare_material(original.clone(), original_body.as_slice())
        .unwrap();
    drop(queue);
    let sdk = client(&store);
    let queued = || {
        let queue = sdk.spool.lock().unwrap();
        let entry = queue.get(identity).unwrap().unwrap();
        let frame = queue.decode_frame(identity).unwrap();
        let mut body = Vec::new();
        queue
            .body(identity)
            .unwrap()
            .read_to_end(&mut body)
            .unwrap();
        (
            serde_json::to_value(&entry.record).unwrap(),
            frame,
            body,
            entry.phase.clone(),
        )
    };
    let before = queued();
    let (archive, lock) = Archive::open_primary(&root, &store).unwrap();
    drop(lock);
    std::fs::remove_file(archive.index_db()).unwrap();
    polylogue_store::fixture::primary_previous_index(&archive.index_db()).unwrap();
    let (archive, _lock) = Archive::open_primary(&root, &store).unwrap();
    assert_eq!(queued(), before);
    sdk.prepare_request(&request).unwrap();
    assert_eq!(
        queued(),
        before,
        "retry preserves the Source7 intent and immutable frame"
    );
    assert_eq!(before.1, original);
    assert_eq!(
        serde_json::from_slice::<PreparedRequest>(&before.2).unwrap(),
        request
    );
    assert!(
        ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .status()
            .unwrap()
            .index_rebuild
            .unwrap()
            .building
    );
    assert!(matches!(
        before.3,
        polylogue_store::primary::queue::Phase::Pending
    ));
}

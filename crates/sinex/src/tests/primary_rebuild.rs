use super::*;
use polylogue_model::LockExt;
use polylogue_model::authored::{AuthoredEffect, PreparedRequest};
use polylogue_model::user::{UserOutcome, UserWrite};
use polylogue_store::{Archive, ReadPool};
use std::io::Read;

#[test]
fn rebuilding_cache_preserves_source7_and_source8_pending_authored_intent() {
    for source_version in [7, 8] {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("sinex");
        let root = dir.path().join("archive");
        let previous = client(&dir.path().join("request-source"));
        let request = PreparedRequest {
            request: polylogue_store::fixture::authored::operator_request(
                "pending-through-primary-cache-rebuild".into(),
                UserWrite::Assert { assertions: vec![] },
            ),
            outcome: UserOutcome {
                id: "pending-through-primary-cache-rebuild".into(),
                repeat: false,
                pending: false,
            },
            effect: AuthoredEffect::default(),
        };
        previous.prepare_request(&request).unwrap();
        let identity = "request:pending-through-primary-cache-rebuild";
        let mut original = previous.spool.locked().decode_frame(identity).unwrap();
        let sinex_protocol::Ingress::Material { material, .. } = &mut original else {
            panic!("authored intent must keep its original material");
        };
        material.source.version = source_version;
        let original_body = serde_json::to_vec(&request).unwrap();
        let queue = crate::spool::Spool::open(&store, "polylogue-test").unwrap();
        queue
            .prepare_material(original.clone(), original_body.as_slice())
            .unwrap();
        drop(queue);
        let sdk = client(&store);
        let queued = || {
            let queue = sdk.spool.locked();
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
        let retry = sdk.prepare_request(&request).unwrap();
        assert!(retry.repeat && retry.pending);
        assert_eq!(
            queued(),
            before,
            "retry preserves the original source and immutable frame"
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
        drop(sdk);
        let mut extracted = Vec::new();
        assert_eq!(
            crate::line_population::extract(&store, "polylogue-test", &mut extracted).unwrap(),
            0
        );
        assert!(
            extracted.is_empty(),
            "unchanged authored intent needs no conversion"
        );
    }
}

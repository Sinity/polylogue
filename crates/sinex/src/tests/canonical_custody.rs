use super::custody_rebuild::witness;
use super::*;
use polylogue_store::primary::{CanonicalCustody, RawInputProof};

fn writes() -> Vec<ProductWrite> {
    vec![ProductWrite {
        product: Product::AttachmentBinding {
            record: Box::new(witness()),
        },
        location: None,
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    }]
}

#[test]
fn held_canonical_body_survives_restart_and_revocation_without_publication_authority() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let products = writes();
    let receipt = sinex
        .stage_canonical_products("original-intent", &products, None)
        .unwrap();
    let private = sinex
        .queue()
        .unwrap()
        .get(&receipt.identity)
        .unwrap()
        .unwrap();
    assert!(private.is_private() && private.canonical().is_none() && private.phase.is_pending());
    drop(sinex);
    let sinex = client(root.path());
    let mut restored = Vec::<CanonicalCustody>::new();
    sinex
        .visit_canonical_custody(&mut |body| {
            restored.push(body.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].products, products);
    assert_eq!(restored[0].historical_proof, None);
    assert!(!restored[0].revoked);
    sinex.revoke_canonical_custody().unwrap();
    drop(sinex);
    let sinex = client(root.path());
    let mut revoked = Vec::new();
    sinex
        .visit_canonical_custody(&mut |body| {
            revoked.push(body.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(revoked[0].products, products);
    assert!(revoked[0].revoked);
    let proof = RawInputProof {
        stamp: "new-owner-write-stamp".into(),
        position: 7,
        policy_epoch: 2,
        scope: "new-scope".into(),
    };
    assert!(matches!(
        sinex.prepare_canonical_products("original-intent", &products, &proof),
        Err(crate::Error::Refused { .. })
    ));
    let retry = sinex
        .stage_canonical_products("original-intent", &products, Some(&proof))
        .unwrap();
    assert_eq!(retry.identity, receipt.identity);
    sinex
        .visit_canonical_custody(&mut |body| {
            assert!(body.revoked);
            assert_eq!(body.historical_proof, None);
            Ok(())
        })
        .unwrap();
    sinex.queue().unwrap().retire(i64::MAX).unwrap();
    assert!(
        sinex
            .queue()
            .unwrap()
            .get(&receipt.identity)
            .unwrap()
            .unwrap()
            .revoked
    );
    assert!(
        sinex
            .queue()
            .unwrap()
            .get("original-intent")
            .unwrap()
            .is_none()
    );
}

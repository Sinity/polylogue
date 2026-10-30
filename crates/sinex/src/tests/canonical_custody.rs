use super::*;
use polylogue_model::work_products::{self, component::*};
use polylogue_store::primary::{CanonicalCustody, RawInputProof};

fn writes() -> Vec<ProductWrite> {
    vec![ProductWrite {
        product: Product::WorkComponentStatus {
            status: Box::new(ComponentStatus {
                version: VERSION,
                reducer_version: work_products::VERSION,
                position: 7,
                component_kind: ComponentKind::LogicalWork,
                physical: vec![PhysicalRef {
                    session: polylogue_model::SessionId::parse("codex:held").unwrap(),
                    evidence: vec![PhysicalEvidence {
                        event: "00000000-0000-7000-8000-000000000001".into(),
                        field: PhysicalField::EventSession,
                    }],
                }],
                links: Vec::new(),
                status: ComponentState::Unavailable {
                    reasons: vec!["logical_membership_unavailable".into()],
                },
            }),
        },
        location: None,
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    }]
}

#[test]
fn held_native_json_survives_restart_exact_retry_and_custody_visit() {
    use work_products::{NativeEvidence, SelectionVersion, SnapshotMembership, WorkSnapshot};

    let session = polylogue_model::SessionId::parse("codex:deep").unwrap();
    let work = work_products::membership::derive([session.clone()], [])
        .pop()
        .unwrap();
    let event = "00000000-0000-7000-8000-000000000001";
    let mut data = serde_json::json!({"retained": "native value"});
    for _ in 0..200 {
        data = serde_json::Value::Object(serde_json::Map::from_iter([("nested".into(), data)]));
    }
    let snapshot = WorkSnapshot {
        reducer_version: work_products::VERSION,
        position: 7,
        session_id: session.clone(),
        membership: SnapshotMembership::Established {
            basis: MembershipBasis {
                membership_version: work_products::membership::VERSION,
                selection_version: SelectionVersion::SUPPORTED,
                completed_position: 7,
                scope: "held-scope".into(),
                membership_hash: work.identity(),
            },
            work,
            physical_refs: vec![PhysicalRef {
                session,
                evidence: vec![PhysicalEvidence {
                    event: event.into(),
                    field: PhysicalField::EventSession,
                }],
            }],
        },
        session: None,
        retired: false,
        unavailable: vec!["native_acquisition_authority_unavailable".into()],
        messages: Vec::new(),
        calls: Vec::new(),
        turns: Vec::new(),
        sources: Vec::new(),
        native: vec![NativeEvidence {
            event: event.into(),
            material: "00000000-0000-7000-8000-000000000002".into(),
            anchor: 0,
            data,
        }],
    };
    snapshot.validate().unwrap();
    let products = vec![ProductWrite {
        product: Product::WorkSnapshot { snapshot },
        location: None,
        parents: Vec::new(),
        machine: None,
        output_index: 0,
    }];
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let original = sinex
        .stage_canonical_products("deep-native-intent", &products, None)
        .unwrap();
    drop(sinex);

    let sinex = client(root.path());
    let retry = sinex
        .stage_canonical_products("deep-native-intent", &products, None)
        .unwrap();
    assert_eq!(retry.identity, original.identity);
    let mut count = 0;
    sinex
        .visit_canonical_custody(&mut |body| {
            assert_eq!(body.identity, "deep-native-intent");
            assert_eq!(body.products, products);
            assert_eq!(body.historical_proof, None);
            assert!(!body.revoked);
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 1);
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
    assert!(private.private && !private.canonical && !private.acknowledged);
    assert_eq!(private.canonical_scope, None);
    assert_eq!(private.canonical_epoch, None);
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

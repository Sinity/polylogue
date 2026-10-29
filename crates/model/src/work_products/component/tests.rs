use super::*;

fn status() -> ComponentStatus {
    ComponentStatus {
        version: VERSION,
        reducer_version: REDUCER_VERSION,
        position: 42,
        component_kind: ComponentKind::LogicalWork,
        physical: vec![PhysicalRef {
            session: SessionId::parse("codex:physical").unwrap(),
            evidence: vec![PhysicalEvidence {
                event: "00000000-0000-7000-8000-000000000001".into(),
                field: PhysicalField::MessageSession,
            }],
        }],
        links: vec![],
        status: ComponentState::Unavailable {
            reasons: vec!["logical_membership_root_unavailable".into()],
        },
    }
}

#[test]
fn physical_routing_status_cannot_deserialize_a_logical_root_or_an_unregistered_evidence_path() {
    let original = status();
    original.validate().unwrap();
    let encoded = serde_json::to_value(&original).unwrap();
    assert!(encoded.get("root").is_none());
    assert!(encoded.get("session_id").is_none());
    let mut forged = encoded.clone();
    forged["root"] = "codex:invented-main".into();
    assert!(serde_json::from_value::<ComponentStatus>(forged).is_err());
    let mut forged = encoded;
    forged["physical"][0]["evidence"][0]["field"] = "/payload/anything".into();
    assert!(serde_json::from_value::<ComponentStatus>(forged).is_err());
}

#[test]
fn resolution_requires_a_same_position_membership_basis_and_supported_root_references() {
    let mut original = status();
    original.status = ComponentState::Resolved {
        superseding_root_refs: vec![RootRef {
            session: SessionId::parse("codex:established-root").unwrap(),
            reducer_version: REDUCER_VERSION,
        }],
        basis: MembershipBasis {
            membership_version: membership::VERSION,
            selection_version: "confirmed-selection-1".into(),
            completed_position: 42,
            scope: "owner-scope".into(),
            membership_hash: "confirmed-membership".into(),
        },
    };
    original.validate().unwrap();
    let ComponentState::Resolved { basis, .. } = &mut original.status else {
        unreachable!()
    };
    basis.completed_position = 41;
    assert!(original.validate().is_err());
}

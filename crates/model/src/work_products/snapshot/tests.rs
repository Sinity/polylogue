use super::super::{
    VERSION, WorkSnapshot,
    component::{PhysicalEvidence, PhysicalField},
};
use super::*;
use crate::{Link, LinkKind};

fn id(value: &str) -> SessionId {
    SessionId::parse(&format!("fixture:{value}")).unwrap()
}

fn snapshot(work: membership::LogicalWork, root: SessionId) -> WorkSnapshot {
    let mut physical_refs: Vec<_> = work
        .physical
        .iter()
        .map(|session| PhysicalRef {
            session: session.clone(),
            evidence: vec![PhysicalEvidence {
                event: format!("event-{session}"),
                field: PhysicalField::SessionId,
            }],
        })
        .collect();
    for reference in &mut physical_refs {
        for link in &work.links {
            if link.facts.parent == reference.session {
                reference.evidence.push(PhysicalEvidence {
                    event: link.event.clone(),
                    field: PhysicalField::LinkParent,
                });
            }
            if link.facts.child == reference.session {
                reference.evidence.push(PhysicalEvidence {
                    event: link.event.clone(),
                    field: PhysicalField::LinkChild,
                });
            }
        }
        reference.evidence.sort();
        reference.evidence.dedup();
    }
    let basis = MembershipBasis {
        membership_version: membership::VERSION,
        selection_version: SelectionVersion::Confirmed1,
        completed_position: 7,
        scope: "scope".into(),
        membership_hash: work.identity(),
    };
    let membership = if work.root.is_some() {
        SnapshotMembership::Established {
            work,
            physical_refs,
            basis,
        }
    } else {
        SnapshotMembership::Unavailable {
            work,
            physical_refs,
            basis,
        }
    };
    WorkSnapshot {
        reducer_version: VERSION,
        position: 7,
        session_id: root,
        membership,
        session: None,
        retired: false,
        unavailable: vec!["membership_unavailable".into()],
        messages: vec![],
        calls: vec![],
        turns: vec![],
        sources: vec![],
        native: vec![],
    }
}

fn link(event: &str, parent: &str, child: &str) -> membership::ConfirmedLink {
    membership::ConfirmedLink {
        event: event.into(),
        facts: Link {
            kind: LinkKind::Continuation,
            parent: id(parent),
            child: id(child),
            at: None,
            call_id: None,
            evidence: "explicit continuation".into(),
        },
    }
}

#[test]
fn a_snapshot_cannot_gain_membership_from_a_missing_field_or_an_unknown_selection_version() {
    let current = snapshot(membership::derive([id("root")], []).remove(0), id("root"));
    current.validate().unwrap();
    let mut encoded = serde_json::to_value(&current).unwrap();
    encoded.as_object_mut().unwrap().remove("membership");
    assert!(serde_json::from_value::<WorkSnapshot>(encoded).is_err());
    let mut encoded = serde_json::to_value(&current).unwrap();
    encoded["membership"]["basis"]["selection_version"] = "future-selection".into();
    assert!(serde_json::from_value::<WorkSnapshot>(encoded).is_err());
    let mut historical = current;
    historical.reducer_version = 2;
    assert_eq!(
        historical.validate(),
        Err("unsupported work reducer version")
    );
}

#[test]
fn established_membership_requires_the_explicit_root_and_every_physical_reference() {
    let work = membership::derive([], [link("link-1", "root", "child")]).remove(0);
    let mut current = snapshot(work, id("root"));
    current.validate().unwrap();
    current.session_id = id("child");
    assert_eq!(
        current.validate(),
        Err("established snapshot has no unambiguous matching root")
    );
    current.session_id = id("root");
    let SnapshotMembership::Established { physical_refs, .. } = &mut current.membership else {
        unreachable!()
    };
    physical_refs.pop();
    assert!(current.validate().is_err());
}

#[test]
fn cyclic_membership_cannot_claim_complete_or_emit_computed_entries() {
    let work = membership::derive(
        [],
        [
            link("link-1", "root", "child"),
            link("link-2", "child", "root"),
        ],
    )
    .remove(0);
    // The caller supplies a known preceding root key. The reducer supplies none.
    let mut current = snapshot(work, id("previously-established"));
    current.validate().unwrap();
    assert!(!current.complete());
    current.session = Some(crate::Session::new(
        current.session_id.clone(),
        crate::SessionKind::Main,
    ));
    assert_eq!(
        current.validate(),
        Err("unavailable membership contains computed facts or lacks a reason")
    );
}

#[test]
fn retirement_cannot_disagree_with_membership_or_retain_physical_evidence() {
    let mut current = snapshot(membership::derive([id("root")], []).remove(0), id("root"));
    current.retired = true;
    assert_eq!(
        current.validate(),
        Err("snapshot retirement differs from its membership state")
    );
    current.membership = SnapshotMembership::Retired {
        basis: current.membership.basis().clone(),
    };
    current.validate().unwrap();
    let encoded = serde_json::to_value(&current).unwrap();
    assert!(encoded["membership"].get("physical_refs").is_none());
    assert!(encoded["membership"].get("work").is_none());
}

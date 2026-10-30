//! Explicit confirmed source links determine logical work inside the apply transaction.
use super::*;
use crate::primary::RawInputProof;
use polylogue_model::work_products::component::{
    ComponentKind, ComponentState, ComponentStatus, MembershipBasis, PhysicalEvidence,
    PhysicalField, PhysicalRef, RootRef,
};
use polylogue_model::work_products::membership::{self, ConfirmedLink, LogicalWork};
use polylogue_model::work_products::{SelectionVersion, SnapshotMembership};
use std::collections::{BTreeMap, BTreeSet};

fn basis(work: &LogicalWork, proof: &RawInputProof) -> MembershipBasis {
    MembershipBasis {
        membership_version: membership::VERSION,
        selection_version: SelectionVersion::SUPPORTED,
        completed_position: proof.position,
        scope: proof.scope.clone(),
        membership_hash: work.identity(),
    }
}
fn empty(root: SessionId, membership: SnapshotMembership, proof: &RawInputProof) -> WorkSnapshot {
    let retired = matches!(membership, SnapshotMembership::Retired { .. });
    WorkSnapshot {
        reducer_version: work_products::VERSION,
        position: proof.position,
        session_id: root,
        membership,
        session: None,
        retired,
        unavailable: if retired {
            Vec::new()
        } else {
            vec!["logical_membership_unavailable".into()]
        },
        messages: Vec::new(),
        calls: Vec::new(),
        turns: Vec::new(),
        sources: Vec::new(),
        native: Vec::new(),
    }
}
fn write(product: Product, parents: &BTreeSet<String>, ordinal: usize) -> ProductWrite {
    ProductWrite {
        product,
        location: None,
        parents: parents.iter().cloned().collect(),
        machine: None,
        output_index: ordinal as u64,
    }
}

pub(super) fn stage(tx: &Transaction<'_>, proof: &RawInputProof) -> Result<()> {
    let dirty: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM primary_work_dirty)",
        [],
        |row| row.get(0),
    )?;
    if !dirty {
        return Ok(());
    }
    let mut physical = BTreeMap::<SessionId, BTreeSet<PhysicalEvidence>>::new();
    let mut links = Vec::new();
    let mut parents = BTreeSet::new();
    let mut query = tx.prepare("SELECT event,product,native_context FROM primary_next WHERE kind='parsed' AND state='live' AND product IS NOT NULL ORDER BY input,chain,source_order,event")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let event: String = row.get(0)?;
        let fact = product(&row.get::<_, Vec<u8>>(1)?)?;
        let fields = match &fact {
            Product::Session { facts, .. } => vec![(facts.id.clone(), PhysicalField::SessionId)],
            Product::Message { facts, .. } => {
                vec![(facts.message.session.clone(), PhysicalField::MessageSession)]
            }
            Product::Event { facts, .. } => facts
                .session
                .iter()
                .cloned()
                .map(|id| (id, PhysicalField::EventSession))
                .collect(),
            Product::Background { facts, .. } => {
                vec![(facts.session.clone(), PhysicalField::BackgroundSession)]
            }
            Product::Link { reading, facts } => {
                let mut selected_links=tx.prepare("SELECT l.kind,l.how FROM link l JOIN link_source s ON s.link_pk=l.pk JOIN input_ref i ON i.pk=s.input_pk WHERE i.id=?1 AND s.chain=?2 AND l.parent_id=?3 AND l.child_id=?4 AND EXISTS(SELECT 1 FROM primary_next f WHERE f.event=?5 AND f.input=i.id AND f.chain=s.chain AND f.reference=?6)")?;
                let mut choices = selected_links.query(params![
                    reading.input.id,
                    reading.chain,
                    facts.parent.to_string(),
                    facts.child.to_string(),
                    event,
                    facts.selection_tag()
                ])?;
                let mut selected = false;
                while let Some(choice) = choices.next()? {
                    let kind: String = choice.get(0)?;
                    let how: String = choice.get(1)?;
                    selected |=
                        polylogue_model::LinkKind::from_columns(&kind, &how) == Some(facts.kind);
                }
                // Missing selected source evidence cannot manufacture separate roots.
                if !selected {
                    return Ok(());
                }
                links.push(ConfirmedLink {
                    event: event.clone(),
                    facts: facts.clone(),
                });
                vec![
                    (facts.parent.clone(), PhysicalField::LinkParent),
                    (facts.child.clone(), PhysicalField::LinkChild),
                ]
            }
            _ => Vec::new(),
        };
        if fields.is_empty() {
            continue;
        }
        let Some(native) = row.get::<_, Option<Vec<u8>>>(2)? else {
            return Ok(());
        };
        let native: crate::primary::NativeFactContext = crate::encoding::unpack_json(&native)?;
        if native.projection.policy_epoch != proof.policy_epoch
            || native.kind_definition_hash.is_empty()
            || native.origin.producer_family.as_deref() != Some("polylogue")
            || !native
                .origin
                .source
                .as_ref()
                .is_some_and(|source| source.name == "polylogue" && source.version == 3)
            || native
                .origin
                .source_fingerprint
                .as_ref()
                .is_none_or(String::is_empty)
            || native
                .origin
                .producer_identity
                .as_ref()
                .is_none_or(String::is_empty)
        {
            return Ok(());
        }
        parents.insert(event.clone());
        for (session, field) in fields {
            physical
                .entry(session)
                .or_default()
                .insert(PhysicalEvidence {
                    event: event.clone(),
                    field,
                });
        }
    }
    drop(rows);
    drop(query);
    let works = membership::derive(physical.keys().cloned(), links);
    let mut prior_roots = BTreeMap::new();
    let mut prior_components = Vec::new();
    let mut query = tx.prepare("SELECT product FROM primary_next WHERE kind='derived' AND state='live' AND product IS NOT NULL")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        match product(&row.get::<_, Vec<u8>>(0)?)? {
            Product::WorkSnapshot { snapshot }
                if snapshot.reducer_version == work_products::VERSION && !snapshot.retired =>
            {
                prior_roots.insert(snapshot.session_id.clone(), snapshot);
            }
            Product::WorkComponentStatus { status }
                if matches!(status.status, ComponentState::Unavailable { .. }) =>
            {
                prior_components.push(*status)
            }
            _ => {}
        }
    }
    drop(rows);
    drop(query);
    let mut writes = Vec::new();
    let mut addressed = BTreeSet::new();
    for work in &works {
        let mut refs: Vec<_> = work
            .physical
            .iter()
            .map(|session| PhysicalRef {
                session: session.clone(),
                evidence: physical[session].iter().cloned().collect(),
            })
            .collect();
        refs.sort_by_key(|reference| reference.session.to_string());
        let membership_basis = basis(work, proof);
        if let Some(root) = &work.root {
            for physical in &work.physical {
                if let Some(pk) = tx
                    .query_row(
                        "SELECT pk FROM session WHERE id=?1",
                        [physical.to_string()],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()?
                {
                    crate::derive::select_confirmed(tx, pk)?;
                }
            }
            let membership = SnapshotMembership::Established {
                work: work.clone(),
                physical_refs: refs,
                basis: membership_basis,
            };
            let snapshot = snapshot(tx, work, membership, proof.position)?;
            snapshot
                .validate()
                .map_err(|error| StoreError::Invalid(error.into()))?;
            addressed.insert(root.clone());
            writes.push(write(
                Product::WorkSnapshot { snapshot },
                &parents,
                writes.len(),
            ));
        } else {
            for root in prior_roots
                .keys()
                .filter(|root| work.physical.contains(root))
            {
                let membership = SnapshotMembership::Unavailable {
                    work: work.clone(),
                    physical_refs: refs.clone(),
                    basis: membership_basis.clone(),
                };
                let snapshot = empty(root.clone(), membership, proof);
                snapshot
                    .validate()
                    .map_err(|error| StoreError::Invalid(error.into()))?;
                addressed.insert(root.clone());
                writes.push(write(
                    Product::WorkSnapshot { snapshot },
                    &parents,
                    writes.len(),
                ));
            }
            let status = ComponentStatus {
                version: work_products::component::VERSION,
                reducer_version: work_products::VERSION,
                position: proof.position,
                component_kind: ComponentKind::LogicalWork,
                physical: refs,
                links: work.links.clone(),
                status: ComponentState::Unavailable {
                    reasons: work.unavailable.clone(),
                },
            };
            status
                .validate()
                .map_err(|error| StoreError::Invalid(error.into()))?;
            writes.push(write(
                Product::WorkComponentStatus {
                    status: Box::new(status),
                },
                &parents,
                writes.len(),
            ));
        }
    }
    for (root, prior) in &prior_roots {
        if addressed.contains(root) {
            continue;
        }
        let mut membership_basis = prior.membership.basis().clone();
        membership_basis.completed_position = proof.position;
        membership_basis.scope = proof.scope.clone();
        let snapshot = empty(
            root.clone(),
            SnapshotMembership::Retired {
                basis: membership_basis,
            },
            proof,
        );
        snapshot
            .validate()
            .map_err(|error| StoreError::Invalid(error.into()))?;
        writes.push(write(
            Product::WorkSnapshot { snapshot },
            &parents,
            writes.len(),
        ));
    }
    for mut status in prior_components {
        let selected: Vec<_> = works
            .iter()
            .filter(|work| {
                work.physical
                    .iter()
                    .any(|id| status.sessions().any(|old| old == id))
            })
            .collect();
        if selected.is_empty()
            || selected.iter().any(|work| work.root.is_none())
            || status
                .sessions()
                .any(|old| !selected.iter().any(|work| work.physical.contains(old)))
        {
            continue;
        }
        let mut roots: Vec<_> = selected
            .iter()
            .map(|work| RootRef {
                session: work.root.clone().expect("checked root"),
                reducer_version: work_products::VERSION,
            })
            .collect();
        roots.sort_by_key(|root| root.session.to_string());
        roots.dedup_by(|left, right| left.session == right.session);
        status.position = proof.position;
        status.status = ComponentState::Resolved {
            superseding_root_refs: roots,
            basis: basis(selected[0], proof),
        };
        status
            .validate()
            .map_err(|error| StoreError::Invalid(error.into()))?;
        writes.push(write(
            Product::WorkComponentStatus {
                status: Box::new(status),
            },
            &parents,
            writes.len(),
        ));
    }
    if !writes.is_empty() {
        let generation: String =
            tx.query_row("SELECT value FROM meta WHERE key='generation'", [], |row| {
                row.get(0)
            })?;
        if generation.is_empty() {
            return Err(StoreError::Invalid(
                "canonical intent has no cache publication generation".into(),
            ));
        }
        let identity = format!(
            "work-batch:{}:{}:{}",
            work_products::VERSION,
            proof.position,
            blake3::hash(&serde_json::to_vec(&(
                "polylogue.work.intent.v1",
                &generation,
                proof.policy_epoch,
                &proof.scope,
                proof.position,
                &writes
            ))?)
            .to_hex()
        );
        stage_products(tx, &identity, &writes, Some(proof))?;
    }
    tx.execute("DELETE FROM primary_work_dirty", [])?;
    Ok(())
}

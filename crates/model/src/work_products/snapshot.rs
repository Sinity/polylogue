//! Required membership provenance for a complete logical work replacement.
use super::{
    component::{MembershipBasis, PhysicalField, PhysicalRef},
    membership,
};
use crate::SessionId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum SelectionVersion {
    #[serde(rename = "polylogue-confirmed-selection-1")]
    Confirmed1,
}

impl SelectionVersion {
    pub const SUPPORTED: Self = Self::Confirmed1;
}

/// An unavailable record addresses an already established canonical root.
/// First ambiguous components use ComponentStatus instead of inventing a root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum SnapshotMembership {
    Established {
        work: membership::LogicalWork,
        physical_refs: Vec<PhysicalRef>,
        basis: MembershipBasis,
    },
    Unavailable {
        work: membership::LogicalWork,
        physical_refs: Vec<PhysicalRef>,
        basis: MembershipBasis,
    },
    Retired {
        basis: MembershipBasis,
    },
}

impl MembershipBasis {
    pub fn validate(&self, position: i64) -> Result<(), &'static str> {
        if self.membership_version != membership::VERSION
            || self.completed_position != position
            || position < 0
            || self.scope.is_empty()
            || self.membership_hash.is_empty()
        {
            return Err("incomplete membership basis");
        }
        Ok(())
    }
}

pub(super) fn validate_physical(physical: &[PhysicalRef]) -> Result<(), &'static str> {
    if physical.is_empty() {
        return Err("membership needs physical evidence");
    }
    let mut previous = None;
    for reference in physical {
        let session = reference.session.to_string();
        if previous
            .as_ref()
            .is_some_and(|previous| previous >= &session)
        {
            return Err("physical sessions are repeated or not in canonical order");
        }
        previous = Some(session);
        if reference.evidence.is_empty()
            || reference
                .evidence
                .iter()
                .any(|evidence| evidence.event.is_empty())
            || reference.evidence.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("physical evidence is absent, repeated or not in canonical order");
        }
    }
    Ok(())
}

impl SnapshotMembership {
    pub fn basis(&self) -> &MembershipBasis {
        match self {
            Self::Established { basis, .. }
            | Self::Unavailable { basis, .. }
            | Self::Retired { basis } => basis,
        }
    }

    pub fn validate(&self, session: &SessionId, position: i64) -> Result<(), &'static str> {
        self.basis().validate(position)?;
        let (work, physical, established) = match self {
            Self::Retired { .. } => return Ok(()),
            Self::Established {
                work,
                physical_refs,
                ..
            } => (work, physical_refs, true),
            Self::Unavailable {
                work,
                physical_refs,
                ..
            } => (work, physical_refs, false),
        };
        validate_physical(physical)?;
        let selected: BTreeSet<_> = physical
            .iter()
            .map(|reference| &reference.session)
            .collect();
        if work.version != membership::VERSION
            || work.physical.len() != selected.len()
            || work.physical.iter().collect::<BTreeSet<_>>() != selected
            || work.identity() != self.basis().membership_hash
        {
            return Err(
                "snapshot membership differs from its physical evidence or derivation basis",
            );
        }
        for link in &work.links {
            for (session, field) in [
                (&link.facts.parent, PhysicalField::LinkParent),
                (&link.facts.child, PhysicalField::LinkChild),
            ] {
                if !physical.iter().any(|reference| {
                    &reference.session == session
                        && reference
                            .evidence
                            .iter()
                            .any(|evidence| evidence.event == link.event && evidence.field == field)
                }) {
                    return Err("membership link lacks its exact physical event evidence");
                }
            }
        }
        // This product-owned check does not grant receiver publication authority.
        let derived = membership::derive(work.physical.clone(), work.links.clone());
        if derived.as_slice() != [work.clone()] {
            return Err("snapshot membership differs from its explicit link reduction");
        }
        if established {
            if work.root.as_ref() != Some(session) || !work.unavailable.is_empty() {
                return Err("established snapshot has no unambiguous matching root");
            }
        } else if work.root.is_some() || work.unavailable.is_empty() {
            return Err("unavailable snapshot claims an established root");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

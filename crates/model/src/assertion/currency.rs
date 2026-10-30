//! Revision and withdrawal currency over indexed graph lookups. An adapter
//! supplies edges; this algorithm knows no SQL, clocks or local write order.
use crate::{Reference, RevisionReference};
use std::collections::{HashMap, HashSet};

/// Currency of a known assertion family in one pinned source snapshot. Every
/// head is retained when revisions conflict.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct FamilyState {
    pub root: Reference,
    pub withdrawn: bool,
    pub heads: Vec<Reference>,
}
impl FamilyState {
    pub fn conflict(&self) -> bool {
        self.heads.len() > 1
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct RevisionState {
    pub reference: Reference,
    pub current: bool,
    pub withdrawn: bool,
    pub conflict: bool,
}

/// Every returned revision is an exact assertion or variant address. Store
/// implementations use the root and incoming-link indexes for these lookups.
pub trait RevisionLinks {
    type Error;
    fn revisions(&mut self, root: &Reference) -> Result<Vec<Reference>, Self::Error>;
    fn revisers(&mut self, revision: &Reference) -> Result<Vec<Reference>, Self::Error>;
    fn withdrawers(&mut self, target: &Reference) -> Result<Vec<Reference>, Self::Error>;
}
#[derive(Debug, PartialEq, Eq)]
pub enum CurrencyError<E> {
    Source(E),
    NotRevision(Reference),
    Cycle(Reference),
}

pub struct Currency<'a, G: RevisionLinks> {
    graph: &'a mut G,
    withdrawn: HashMap<Reference, bool>,
}
fn root(reference: &Reference) -> Option<Reference> {
    match reference {
        Reference::Assertion(revision) => Some(Reference::Assertion(RevisionReference {
            root: revision.root.clone(),
            revision: None,
        })),
        Reference::Variant {
            revision,
            part: None,
        } => Some(Reference::Variant {
            revision: RevisionReference {
                root: revision.root.clone(),
                revision: None,
            },
            part: None,
        }),
        _ => None,
    }
}
fn exact(reference: &Reference) -> bool {
    matches!(
        reference,
        Reference::Assertion(RevisionReference {
            revision: Some(_),
            ..
        }) | Reference::Variant {
            revision: RevisionReference {
                revision: Some(_),
                ..
            },
            part: None
        }
    )
}
impl<'a, G: RevisionLinks> Currency<'a, G> {
    pub fn new(graph: &'a mut G) -> Self {
        Self {
            graph,
            withdrawn: HashMap::new(),
        }
    }
    pub fn withdrawn(&mut self, reference: &Reference) -> Result<bool, CurrencyError<G::Error>> {
        if let Some(value) = self.withdrawn.get(reference) {
            return Ok(*value);
        }
        enum Step {
            Enter(Reference),
            Finish(Reference, Vec<Reference>),
        }
        let mut visiting = HashSet::new();
        let mut steps = vec![Step::Enter(reference.clone())];
        while let Some(step) = steps.pop() {
            match step {
                Step::Enter(reference) => {
                    if self.withdrawn.contains_key(&reference) {
                        continue;
                    }
                    let root = root(&reference)
                        .ok_or_else(|| CurrencyError::NotRevision(reference.clone()))?;
                    if !visiting.insert(reference.clone()) {
                        return Err(CurrencyError::Cycle(reference));
                    }
                    let mut withdrawers = self
                        .graph
                        .withdrawers(&reference)
                        .map_err(CurrencyError::Source)?;
                    if root != reference {
                        withdrawers.extend(
                            self.graph
                                .withdrawers(&root)
                                .map_err(CurrencyError::Source)?,
                        );
                    }
                    withdrawers.sort();
                    withdrawers.dedup();
                    for withdrawal in &withdrawers {
                        if !exact(withdrawal) {
                            return Err(CurrencyError::NotRevision(withdrawal.clone()));
                        }
                    }
                    // Finish only after every edge has been checked, including
                    // inactive branches whose cycles must not depend on order.
                    steps.push(Step::Finish(reference, withdrawers.clone()));
                    steps.extend(withdrawers.into_iter().rev().map(Step::Enter));
                }
                Step::Finish(reference, withdrawers) => {
                    let any_active = withdrawers
                        .iter()
                        .any(|withdrawal| !self.withdrawn[withdrawal]);
                    visiting.remove(&reference);
                    self.withdrawn.insert(reference, any_active);
                }
            }
        }
        Ok(self.withdrawn[reference])
    }
    pub fn current(&mut self, reference: &Reference) -> Result<bool, CurrencyError<G::Error>> {
        if !exact(reference) {
            return Err(CurrencyError::NotRevision(reference.clone()));
        }
        if self.withdrawn(reference)? {
            return Ok(false);
        }
        let mut revisers = self
            .graph
            .revisers(reference)
            .map_err(CurrencyError::Source)?;
        revisers.sort();
        revisers.dedup();
        let mut any_active = false;
        for reviser in revisers {
            if !exact(&reviser) {
                return Err(CurrencyError::NotRevision(reviser));
            }
            any_active |= !self.withdrawn(&reviser)?;
        }
        Ok(!any_active)
    }
    /// All current heads remain visible; the caller reports a conflict when
    /// this contains more than one revision, and never selects by timestamp.
    pub fn heads(&mut self, root: &Reference) -> Result<Vec<Reference>, CurrencyError<G::Error>> {
        if root != &self::root(root).ok_or_else(|| CurrencyError::NotRevision(root.clone()))? {
            return Err(CurrencyError::NotRevision(root.clone()));
        }
        let mut heads = Vec::new();
        for revision in self.graph.revisions(root).map_err(CurrencyError::Source)? {
            if self.current(&revision)? {
                heads.push(revision);
            }
        }
        heads.sort();
        heads.dedup();
        Ok(heads)
    }
}
#[cfg(test)]
mod tests;

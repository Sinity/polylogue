//! Structural authored policy shared by commands, markup and the predecessor import.
//! Existence, currency, text evidence and configured rubric checks belong to
//! the accepting store snapshot; this module never resolves a reference.
use super::{About, AssertionKind, KeyShape, LinkRole, NewAssertion};
use crate::{Reference, RevisionReference};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidAssertion {
    #[error("{kind:?} key must be {expected}")]
    Key {
        kind: AssertionKind,
        expected: &'static str,
    },
    #[error("{kind:?} must be about {expected}")]
    About {
        kind: AssertionKind,
        expected: &'static str,
    },
    #[error("{role:?} cannot name {reference}")]
    Link {
        role: LinkRole,
        reference: Reference,
    },
    #[error("duplicate {role:?} link to {reference}")]
    DuplicateLink {
        role: LinkRole,
        reference: Reference,
    },
    #[error(
        "all revises links must name revisions of one assertion family, unless a single-valued kind resolves families holding one slot"
    )]
    RevisionFamilies,
    #[error("withdrawal requires at least one withdraws link")]
    NoWithdrawalTarget,
    #[error("only a withdrawal may carry withdraws links")]
    WithdrawalRole,
    #[error("a {role:?} link to {reference} also needs an about link to it")]
    RelationWithoutAbout {
        role: LinkRole,
        reference: Reference,
    },
}

type Result<T> = std::result::Result<T, InvalidAssertion>;

fn exact_assertion(reference: &Reference) -> bool {
    matches!(
        reference,
        Reference::Assertion(RevisionReference {
            revision: Some(_),
            ..
        })
    )
}
fn withdrawal_target(reference: &Reference) -> bool {
    matches!(
        reference,
        Reference::Assertion(_) | Reference::Variant { part: None, .. }
    )
}

/// A saved-query name that `query:NAME` can say: ASCII letters, digits, `_`,
/// `-` and `.`. A new save is refused under any other name; names are also
/// never entirely hexadecimal, which [`validate_links`] checks for every
/// `saved_query` key.
pub fn sayable_query_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Checks the kind declaration and relation address shapes without changing
/// authored content. Waiting targets are valid; existence is not inferred.
pub fn validate_links<'a>(
    kind: AssertionKind,
    key: Option<&str>,
    links: impl IntoIterator<Item = (LinkRole, &'a Reference)>,
) -> Result<()> {
    match (kind.spec().key, key) {
        (KeyShape::None, Some(_)) => {
            return Err(InvalidAssertion::Key {
                kind,
                expected: "absent",
            });
        }
        (KeyShape::Required, None) | (_, Some("")) => {
            return Err(InvalidAssertion::Key {
                kind,
                expected: "nonempty",
            });
        }
        _ => {}
    }
    if kind == AssertionKind::SavedQuery
        && key.is_some_and(|key| key.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(InvalidAssertion::Key {
            kind,
            expected: "a name that is not entirely hexadecimal",
        });
    }
    let mut seen = BTreeSet::new();
    let mut about = Vec::new();
    let mut relations = Vec::new();
    let mut revisions = BTreeSet::new();
    let mut withdraws = false;
    for (role, reference) in links {
        if !seen.insert((role, reference)) {
            return Err(InvalidAssertion::DuplicateLink {
                role,
                reference: reference.clone(),
            });
        }
        let valid = if role.requires_exact_assertion() {
            exact_assertion(reference)
        } else {
            match role {
                LinkRole::DerivedFrom => {
                    !matches!(reference, Reference::Assertion(_)) || exact_assertion(reference)
                }
                LinkRole::Withdraws => withdrawal_target(reference),
                LinkRole::Scope => matches!(
                    reference,
                    Reference::Repo(_) | Reference::Cwd(_) | Reference::Page(_)
                ),
                LinkRole::SlotOf | LinkRole::MemberOf => matches!(reference, Reference::Page(_)),
                LinkRole::RoleOf => matches!(reference, Reference::Role(_)),
                _ => true,
            }
        };
        if !valid {
            return Err(InvalidAssertion::Link {
                role,
                reference: reference.clone(),
            });
        }
        if role == LinkRole::About {
            about.push(reference);
        }
        if role.is_relation() {
            relations.push((role, reference));
        }
        if role == LinkRole::Withdraws {
            withdraws = true;
        }
        if role == LinkRole::Revises
            && let Reference::Assertion(revision) = reference
        {
            revisions.insert(&revision.root);
        }
    }
    // Only a single-valued kind can have several families in one slot, and
    // one revision naming all their heads resolves them; the writer checks
    // that the families do share a slot.
    let single_valued = kind
        .spec()
        .uniqueness
        .is_some_and(|uniqueness| uniqueness.when_held == authored::WhenHeld::Revise);
    if revisions.len() > 1 && !single_valued {
        return Err(InvalidAssertion::RevisionFamilies);
    }
    // A relation is a statement about the revision it relates to.
    if let Some((role, reference)) = relations
        .into_iter()
        .find(|(_, reference)| !about.contains(reference))
    {
        return Err(InvalidAssertion::RelationWithoutAbout {
            role,
            reference: reference.clone(),
        });
    }
    if kind == AssertionKind::Withdrawal {
        if !withdraws {
            return Err(InvalidAssertion::NoWithdrawalTarget);
        }
    } else if withdraws {
        return Err(InvalidAssertion::WithdrawalRole);
    }
    let expected = match kind.spec().about {
        About::Anything if about.is_empty() => Some("one or more references"),
        About::One if about.len() != 1 => Some("exactly one reference"),
        About::OptionalOne if about.len() > 1 => Some("at most one reference"),
        About::Span if !matches!(about.as_slice(), [Reference::Span { .. }]) => {
            Some("exactly one span")
        }
        About::Session if !matches!(about.as_slice(), [Reference::Session(_)]) => {
            Some("exactly one session")
        }
        About::ExactAssertion if about.len() != 1 || !exact_assertion(about[0]) => {
            Some("exactly one assertion revision")
        }
        About::Several if about.len() < 2 => Some("at least two references"),
        About::None if !about.is_empty() => Some("no reference"),
        _ => None,
    };
    if let Some(expected) = expected {
        return Err(InvalidAssertion::About { kind, expected });
    }
    Ok(())
}

impl NewAssertion {
    pub fn validate_links(&self) -> Result<()> {
        validate_links(
            self.kind,
            self.key.as_deref(),
            self.links.iter().map(|link| (link.role, &link.reference)),
        )
    }
}

#[cfg(test)]
mod tests;

//! Structural authored policy shared by commands, markup and the Python Polylogue import.
//! Existence, current revisions, quoted text and configured rubric checks belong to
//! the accepting store snapshot; this module never resolves a reference.
use super::{About, AnnotationKind, KeyShape, LinkRole, NewAnnotation};
use crate::{Reference, RevisionReference};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidAnnotation {
    #[error("{kind:?} key must be {expected}")]
    Key {
        kind: AnnotationKind,
        expected: &'static str,
    },
    #[error("{kind:?} must be about {expected}")]
    About {
        kind: AnnotationKind,
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
        "all revises links must name revisions of one annotation family, unless a single-valued kind resolves families holding one slot"
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

type Result<T> = std::result::Result<T, InvalidAnnotation>;

fn exact_annotation(reference: &Reference) -> bool {
    matches!(
        reference,
        Reference::Annotation(RevisionReference {
            revision: Some(_),
            ..
        })
    )
}
fn withdrawal_target(reference: &Reference) -> bool {
    matches!(
        reference,
        Reference::Annotation(_) | Reference::Alternate { part: None, .. }
    )
}

/// New saved names start with a lowercase letter and contain lowercase
/// letters, digits and hyphens. Product vocabulary is checked by the verb.
pub fn valid_saved_name(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && name
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
}

/// Checks the kind declaration and relation address shapes without changing
/// authored content. Waiting targets are valid; existence is not inferred.
pub fn validate_links<'a>(
    kind: AnnotationKind,
    key: Option<&str>,
    links: impl IntoIterator<Item = (LinkRole, &'a Reference)>,
) -> Result<()> {
    match (kind.spec().key, key) {
        (KeyShape::None, Some(_)) => {
            return Err(InvalidAnnotation::Key {
                kind,
                expected: "absent",
            });
        }
        (KeyShape::Required, None) | (_, Some("")) => {
            return Err(InvalidAnnotation::Key {
                kind,
                expected: "nonempty",
            });
        }
        _ => {}
    }
    if kind == AnnotationKind::SavedQuery
        && key.is_some_and(|key| key.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(InvalidAnnotation::Key {
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
            return Err(InvalidAnnotation::DuplicateLink {
                role,
                reference: reference.clone(),
            });
        }
        let valid = if role.requires_exact_annotation() {
            exact_annotation(reference)
        } else {
            match role {
                LinkRole::DerivedFrom => {
                    !matches!(reference, Reference::Annotation(_)) || exact_annotation(reference)
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
            return Err(InvalidAnnotation::Link {
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
            && let Reference::Annotation(revision) = reference
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
        return Err(InvalidAnnotation::RevisionFamilies);
    }
    // A relation is a statement about the revision it relates to.
    if let Some((role, reference)) = relations
        .into_iter()
        .find(|(_, reference)| !about.contains(reference))
    {
        return Err(InvalidAnnotation::RelationWithoutAbout {
            role,
            reference: reference.clone(),
        });
    }
    if kind == AnnotationKind::Withdrawal {
        if !withdraws {
            return Err(InvalidAnnotation::NoWithdrawalTarget);
        }
    } else if withdraws {
        return Err(InvalidAnnotation::WithdrawalRole);
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
        About::ExactAnnotation if about.len() != 1 || !exact_annotation(about[0]) => {
            Some("exactly one annotation revision")
        }
        About::Several if about.len() < 2 => Some("at least two references"),
        About::None if !about.is_empty() => Some("no reference"),
        _ => None,
    };
    if let Some(expected) = expected {
        return Err(InvalidAnnotation::About { kind, expected });
    }
    Ok(())
}

impl NewAnnotation {
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

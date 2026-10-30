//! Natural identity slots for kinds with constrained current values. The
//! accepting store finds indexed candidates and evaluates graph currency.
use super::{AssertionKind, Cardinality, NewAssertion, values};
use crate::{Author, Json, Reference};

#[derive(Debug, Clone)]
pub enum Key {
    About {
        kind: AssertionKind,
        about: Reference,
    },
    Name {
        kind: AssertionKind,
        name: String,
    },
    Tag {
        about: Reference,
        name: String,
        value: Option<Json>,
    },
    Annotation {
        about: Reference,
        rubric: Reference,
        name: String,
        author: Author,
    },
    Scope {
        kind: AssertionKind,
        scope: Option<Reference>,
        name: String,
    },
    Setting {
        name: String,
        machine: Option<String>,
    },
}
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Structure(#[from] super::policy::InvalidAssertion),
    #[error("annotation cardinality requires its resolved exact rubric revision")]
    Rubric,
    #[error("cardinality received a value of another kind")]
    Shape,
    #[error("host setting cardinality requires its resolved target machine")]
    SettingTarget,
    #[error(transparent)]
    Setting(#[from] crate::settings::Error),
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::About { kind: a, about: ar }, Self::About { kind: b, about: br }) => {
                a == b && ar == br
            }
            (Self::Name { kind: a, name: ak }, Self::Name { kind: b, name: bk }) => {
                a == b && ak == bk
            }
            // Confidence describes the observation, not the tagged value.
            (
                Self::Tag {
                    about: a,
                    name: ak,
                    value: av,
                },
                Self::Tag {
                    about: b,
                    name: bk,
                    value: bv,
                },
            ) => {
                a == b
                    && ak == bk
                    && match (av, bv) {
                        (None, None) => true,
                        (Some(a), Some(b)) => values::equal(a, b),
                        _ => false,
                    }
            }
            (
                Self::Annotation {
                    about: a,
                    rubric: ar,
                    name: ak,
                    author: aa,
                },
                Self::Annotation {
                    about: b,
                    rubric: br,
                    name: bk,
                    author: ba,
                },
            ) => a == b && ar == br && ak == bk && aa == ba,
            (
                Self::Scope {
                    kind: a,
                    scope: ar,
                    name: ak,
                },
                Self::Scope {
                    kind: b,
                    scope: br,
                    name: bk,
                },
            ) => a == b && ar == br && ak == bk,
            (
                Self::Setting {
                    name: a,
                    machine: am,
                },
                Self::Setting {
                    name: b,
                    machine: bm,
                },
            ) => a == b && am == bm,
            _ => false,
        }
    }
}
impl Eq for Key {}

impl NewAssertion {
    /// Returns one key per constrained target. Bulk writes resolve each target
    /// before applying these slots; free-form kinds have no cardinality key.
    /// Setting values must already have passed Setting::resolve; rubric is the
    /// exact linked rubric selected by the accepting snapshot.
    pub fn cardinality_keys(
        &self,
        value: &values::Value,
        author: &Author,
        rubric: Option<&Reference>,
    ) -> Result<Vec<Key>, Error> {
        self.validate_links()?;
        let name = || {
            self.key
                .clone()
                .expect("kind key checked by structural policy")
        };
        let mut about: Vec<_> = self
            .links
            .iter()
            .filter(|link| link.role == super::LinkRole::About)
            .map(|link| link.reference.clone())
            .collect();
        about.sort();
        Ok(match self.kind.spec().cardinality {
            Cardinality::Many => Vec::new(),
            Cardinality::OnePerAbout => about
                .into_iter()
                .map(|about| Key::About {
                    kind: self.kind,
                    about,
                })
                .collect(),
            Cardinality::OnePerName => vec![Key::Name {
                kind: self.kind,
                name: name(),
            }],
            Cardinality::Tag => {
                let values::Value::Tag(tag) = value else {
                    return Err(Error::Shape);
                };
                about
                    .into_iter()
                    .map(|about| Key::Tag {
                        about,
                        name: name(),
                        value: tag.fields.value.clone(),
                    })
                    .collect()
            }
            Cardinality::Annotation => {
                let Some(
                    rubric @ Reference::Assertion(crate::RevisionReference {
                        revision: Some(_),
                        ..
                    }),
                ) = rubric
                else {
                    return Err(Error::Rubric);
                };
                about
                    .into_iter()
                    .map(|about| Key::Annotation {
                        about,
                        rubric: rubric.clone(),
                        name: name(),
                        author: author.clone(),
                    })
                    .collect()
            }
            Cardinality::OnePerScopeKey => {
                let mut scopes: Vec<_> = self
                    .links
                    .iter()
                    .filter(|link| link.role == super::LinkRole::Scope)
                    .map(|link| Some(link.reference.clone()))
                    .collect();
                scopes.sort();
                if scopes.is_empty() {
                    scopes.push(None)
                }
                scopes
                    .into_iter()
                    .map(|scope| Key::Scope {
                        kind: self.kind,
                        scope,
                        name: name(),
                    })
                    .collect()
            }
            Cardinality::Setting => {
                let values::Value::Setting(setting) = value else {
                    return Err(Error::Shape);
                };
                if crate::settings::declaration(&name())?.scope == crate::settings::Scope::Host
                    && setting.fields.machine.is_none()
                {
                    return Err(Error::SettingTarget);
                }
                vec![Key::Setting {
                    name: name(),
                    machine: setting.fields.machine.clone(),
                }]
            }
        })
    }
}
#[cfg(test)]
mod tests;

//! The uniqueness slots an assertion's content occupies. Each kind declares
//! its uniqueness key as a list of components (`KindSpec::uniqueness`); this
//! module supplies each component's value from the write, and the shared
//! `authored` crate turns the values into slots. One function serves every
//! kind: one title per session, one tag per name and value, one annotation per
//! rubric field and author, one memory per path and scope, one setting per
//! name and host.
use super::{AssertionFact, AssertionKind, LinkRole, NewAssertion, values};
use crate::{Author, Reference, RevisionReference};
use authored::{Part, Unique};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Structure(#[from] super::policy::InvalidAssertion),
    #[error("an annotation's uniqueness key needs its resolved exact rubric revision")]
    Rubric,
    #[error("the uniqueness key received a value of another kind")]
    Shape,
    #[error("a host setting's uniqueness key needs its resolved target machine")]
    SettingTarget,
    #[error(transparent)]
    Setting(#[from] crate::settings::Error),
    #[error("uniqueness component `{0}` is not declared")]
    Component(&'static str),
}

/// The slots of one write of `kind`, or `None` for a kind with many. Setting
/// values must already be resolved to their target host; `rubric` is the
/// exact rubric revision an annotation was derived from.
pub fn unique<'a>(
    kind: AssertionKind,
    key: Option<&str>,
    value: &values::Value,
    links: impl IntoIterator<Item = (LinkRole, &'a Reference)>,
    author: &Author,
    rubric: Option<&Reference>,
) -> Result<Option<Unique>, Error> {
    let Some(uniqueness) = kind.spec().uniqueness else {
        return Ok(None);
    };
    let links: Vec<(LinkRole, &Reference)> = links.into_iter().collect();
    let references = |role: LinkRole| -> Vec<String> {
        links
            .iter()
            .filter(|(link, _)| *link == role)
            .map(|(_, reference)| reference.to_string())
            .collect()
    };
    let mut parts = Vec::with_capacity(uniqueness.components.len());
    for component in uniqueness.components {
        parts.push(match *component {
            "about" => Part::Each(references(LinkRole::About)),
            "scope" => Part::Each(references(LinkRole::Scope)),
            "key" => key.map_or(Part::Absent, |key| Part::One(key.to_owned())),
            "author" => Part::One(author.to_string()),
            "value" => {
                let values::Value::Tag(tag) = value else {
                    return Err(Error::Shape);
                };
                tag.fields
                    .value
                    .as_ref()
                    .map_or(Part::Absent, |value| Part::One(values::canonical(value)))
            }
            "rubric" => match rubric {
                Some(
                    rubric @ Reference::Assertion(RevisionReference {
                        revision: Some(_), ..
                    }),
                ) => Part::One(rubric.to_string()),
                _ => return Err(Error::Rubric),
            },
            "machine" => {
                let values::Value::Setting(setting) = value else {
                    return Err(Error::Shape);
                };
                let name = key.expect("a setting's key is checked by its kind");
                if crate::settings::declaration(name)?.scope == crate::settings::Scope::Host
                    && setting.fields.machine.is_none()
                {
                    return Err(Error::SettingTarget);
                }
                setting
                    .fields
                    .machine
                    .clone()
                    .map_or(Part::Absent, Part::One)
            }
            other => return Err(Error::Component(other)),
        });
    }
    let mut parts = parts.into_iter();
    Ok(Some(uniqueness.unique(kind.as_str(), |_| {
        parts.next().expect("one value per declared component")
    })))
}

impl NewAssertion {
    /// The slots this write occupies, after its value was decoded and, for a
    /// setting, resolved to its target host.
    pub fn unique(
        &self,
        value: &values::Value,
        author: &Author,
        rubric: Option<&Reference>,
    ) -> Result<Option<Unique>, Error> {
        self.validate_links()?;
        unique(
            self.kind,
            self.key.as_deref(),
            value,
            self.links.iter().map(|link| (link.role, &link.reference)),
            author,
            rubric,
        )
    }
}

impl AssertionFact {
    /// The slots a stored fact occupies; the same function a write uses.
    pub fn unique(&self, rubric: Option<&Reference>) -> Result<Option<Unique>, Error> {
        let value = self.validate().map_err(|error| match error {
            super::AssertionError::Structure(error) => Error::Structure(error),
            super::AssertionError::Setting(error) => Error::Setting(error),
            _ => Error::Shape,
        })?;
        unique(
            self.kind,
            self.key.as_deref(),
            &value,
            self.links.iter().map(|link| (link.role, &link.reference)),
            &self.author,
            rubric,
        )
    }
}

#[cfg(test)]
mod tests;

//! Honest attribution kept on authored records. The caller's facts are
//! gathered at the transport boundary and turned into an author by
//! `command_surface::author_of`, the same on every surface.
use crate::{IdError, Origin, SessionId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Honest attribution kept on every authored record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Author {
    Operator,
    /// An agent, by the session it wrote from when known.
    Agent(Option<SessionId>),
    /// The harness is known but no session identity was recorded.
    AgentOrigin(Origin),
    /// Nothing identifies a person, agent or named program.
    Unattributed,
    /// A program acting for the operator, by name.
    Script(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthorError {
    #[error("author {0:?}: {1}")]
    Id(String, IdError),
    #[error("author {0:?} is not a stored actor")]
    Unknown(String),
}

impl Author {
    pub fn is_agent(&self) -> bool {
        matches!(self, Self::Agent(_) | Self::AgentOrigin(_))
    }

    /// Parse stored attribution without inventing a missing harness or session.
    pub fn parse(text: &str) -> Result<Self, AuthorError> {
        match text.split_once(':') {
            None if text == "operator" => Ok(Self::Operator),
            None if text == "agent" => Ok(Self::Agent(None)),
            None if text == "unattributed" => Ok(Self::Unattributed),
            Some(("agent", origin)) if !origin.contains(':') => Ok(Self::AgentOrigin(
                Origin::new(origin).map_err(|e| AuthorError::Id(text.to_owned(), e))?,
            )),
            Some(("agent", s)) => Ok(Self::Agent(Some(
                SessionId::parse(s).map_err(|e| AuthorError::Id(text.to_owned(), e))?,
            ))),
            Some(("script", name)) if !name.is_empty() => Ok(Self::Script(name.to_owned())),
            _ => Err(AuthorError::Unknown(text.to_owned())),
        }
    }
}

impl fmt::Display for Author {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator => f.write_str("operator"),
            Self::Agent(None) => f.write_str("agent"),
            Self::Agent(Some(s)) => write!(f, "agent:{s}"),
            Self::AgentOrigin(origin) => write!(f, "agent:{origin}"),
            Self::Unattributed => f.write_str("unattributed"),
            Self::Script(n) => write!(f, "script:{n}"),
        }
    }
}

impl Serialize for Author {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Author {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Mcp,
    Cli,
    Web,
    Extension,
    Desktop,
    Hook,
    Markup,
    Import,
    Daemon,
}
stored_names!(Channel {Mcp=>"mcp",Cli=>"cli",Web=>"web",Extension=>"extension",Desktop=>"desktop",Hook=>"hook",Markup=>"markup",Import=>"import",Daemon=>"daemon"});

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Authorship {
    pub author: Author,
    pub via: Channel,
}

/// The shared crate computes authorship from the caller's facts; an agent
/// session that is not a Polylogue session ID is kept as its harness.
impl From<command_surface::Authorship> for Authorship {
    fn from(computed: command_surface::Authorship) -> Self {
        use command_surface::{Author as Computed, Via};
        let harness =
            |name: &str| Origin::new(name).map_or(Author::Agent(None), Author::AgentOrigin);
        let author = match computed.author {
            Computed::Operator => Author::Operator,
            Computed::Agent(session) => SessionId::parse(&session.to_string()).map_or_else(
                |_| harness(session.harness.as_str()),
                |id| Author::Agent(Some(id)),
            ),
            Computed::AgentOfHarness(name) => harness(name.as_str()),
            Computed::UnidentifiedAgent => Author::Agent(None),
            Computed::Script(unit) => Author::Script(unit),
            Computed::Unattributed => Author::Unattributed,
        };
        let via = match computed.via {
            Via::Cli => Channel::Cli,
            Via::Mcp => Channel::Mcp,
            Via::Web => Channel::Web,
            Via::Extension => Channel::Extension,
            Via::Desktop => Channel::Desktop,
            Via::Hook => Channel::Hook,
            Via::Markup => Channel::Markup,
            Via::Import => Channel::Import,
            Via::Daemon => Channel::Daemon,
        };
        Self { author, via }
    }
}

#[cfg(test)]
mod tests;

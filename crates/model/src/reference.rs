//! Natural archive addresses. Parsing does not read an archive or resolve a
//! shorthand; an address may name a target that has not arrived yet.
use crate::{DocumentId, IdError, MessageId, SessionId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpanUnit {
    Bytes,
    CodePoints,
    Utf16,
}
impl SpanUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bytes => "b",
            Self::CodePoints => "c",
            Self::Utf16 => "u16",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RevisionReference {
    pub root: String,
    pub revision: Option<String>,
}
impl RevisionReference {
    pub fn exact(root: String, revision: String) -> Self {
        Self {
            root,
            revision: Some(revision),
        }
    }
}
impl fmt::Display for RevisionReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.root)?;
        if let Some(revision) = &self.revision {
            write!(f, "@{revision}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reference {
    Session(SessionId),
    Message(MessageId),
    MessageRange {
        first: MessageId,
        last: MessageId,
    },
    Block {
        message: MessageId,
        n: u64,
    },
    ToolCall(SessionId, String),
    Document(DocumentId),
    Event(String),
    Attachment(String),
    Assertion(RevisionReference),
    Variant {
        revision: RevisionReference,
        part: Option<u64>,
    },
    Page(RevisionReference),
    Query {
        key: String,
        parameters: Vec<(String, String)>,
    },
    Result {
        id: String,
        row: Option<u64>,
    },
    Delivery(String),
    File {
        path: String,
        commit: String,
    },
    Repo(String),
    Cwd(String),
    Url(String),
    Sinex(String),
    Span {
        container: Box<Reference>,
        unit: SpanUnit,
        start: u64,
        end: u64,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReferenceError {
    #[error("reference {0:?} has no kind prefix")]
    NoKind(String),
    #[error("reference {0:?} has an unknown kind")]
    UnknownKind(String),
    #[error("reference {0:?}: {1}")]
    Id(String, IdError),
    #[error("reference {0:?} is incomplete")]
    Incomplete(String),
    #[error("reference {0:?}: {1}")]
    Invalid(String, &'static str),
}
impl Reference {
    pub fn parse(text: &str) -> Result<Self, ReferenceError> {
        let (kind, rest) = text
            .split_once(':')
            .ok_or_else(|| ReferenceError::NoKind(text.into()))?;
        let invalid = |why| ReferenceError::Invalid(text.into(), why);
        let nonempty = |value: &str| {
            if value.is_empty() {
                Err(ReferenceError::Incomplete(text.into()))
            } else {
                Ok(value.to_owned())
            }
        };
        let id = |error| ReferenceError::Id(text.into(), error);
        let number = |value: &str| -> Result<u64, ReferenceError> {
            if value.is_empty()
                || !value.bytes().all(|v| v.is_ascii_digit())
                || (value.len() > 1 && value.starts_with('0'))
            {
                return Err(invalid("expected a nonnegative integer"));
            }
            value
                .parse()
                .map_err(|_| invalid("integer is out of range"))
        };
        let revision = |value: &str| -> Result<RevisionReference, ReferenceError> {
            let (root, revision) = match value.rsplit_once('@') {
                Some((root, revision)) => (nonempty(root)?, Some(nonempty(revision)?)),
                None => (nonempty(value)?, None),
            };
            Ok(RevisionReference { root, revision })
        };
        Ok(match kind {
            "session" => Self::Session(SessionId::parse(rest).map_err(id)?),
            "message" => {
                if let Some((first, last)) = rest.split_once("..") {
                    Self::MessageRange {
                        first: MessageId::parse(first).map_err(id)?,
                        last: MessageId::parse(last).map_err(id)?,
                    }
                } else {
                    Self::Message(MessageId::parse(rest).map_err(id)?)
                }
            }
            "block" => {
                let (message, n) = rest
                    .rsplit_once('#')
                    .ok_or_else(|| ReferenceError::Incomplete(text.into()))?;
                Self::Block {
                    message: MessageId::parse(message).map_err(id)?,
                    n: number(n)?,
                }
            }
            "tool" => {
                let (session, call) = rest
                    .rsplit_once('#')
                    .ok_or_else(|| ReferenceError::Incomplete(text.into()))?;
                Self::ToolCall(SessionId::parse(session).map_err(id)?, nonempty(call)?)
            }
            "document" => Self::Document(DocumentId::parse(rest).map_err(id)?),
            "event" => Self::Event(nonempty(rest)?),
            "attachment" => Self::Attachment(nonempty(rest)?),
            "assertion" => Self::Assertion(revision(rest)?),
            "variant" => {
                let (reference, part) = match rest.rsplit_once('#') {
                    Some((reference, part)) => (reference, Some(number(part)?)),
                    None => (rest, None),
                };
                Self::Variant {
                    revision: revision(reference)?,
                    part,
                }
            }
            "page" => Self::Page(revision(rest)?),
            "query" => {
                let mut parameters = Vec::new();
                let key = if let Some((key, args)) = rest.split_once('(') {
                    let args = args
                        .strip_suffix(')')
                        .ok_or_else(|| invalid("query parameters need a closing parenthesis"))?;
                    if args.is_empty() {
                        return Err(invalid("query parameter list is empty"));
                    }
                    for arg in args.split(',') {
                        let (name, value) = arg
                            .split_once('=')
                            .ok_or_else(|| invalid("query parameter needs a name and value"))?;
                        let name = nonempty(name)?;
                        if parameters.iter().any(|(existing, _)| existing == &name) {
                            return Err(invalid("query parameter is repeated"));
                        }
                        parameters.push((name, nonempty(value)?));
                    }
                    nonempty(key)?
                } else {
                    nonempty(rest)?
                };
                Self::Query { key, parameters }
            }
            "result" => {
                let (value, row) = match rest.rsplit_once('#') {
                    Some((value, row)) => (value, Some(number(row)?)),
                    None => (rest, None),
                };
                Self::Result {
                    id: nonempty(value)?,
                    row,
                }
            }
            "delivery" => Self::Delivery(nonempty(rest)?),
            "file" => {
                let (path, commit) = rest
                    .rsplit_once('@')
                    .ok_or_else(|| ReferenceError::Incomplete(text.into()))?;
                Self::File {
                    path: nonempty(path)?,
                    commit: nonempty(commit)?,
                }
            }
            "repo" => Self::Repo(nonempty(rest)?),
            "cwd" => Self::Cwd(nonempty(rest)?),
            "url" => {
                let (scheme, _) = rest
                    .split_once(':')
                    .ok_or_else(|| invalid("URL must be absolute"))?;
                if !scheme.starts_with(|c: char| c.is_ascii_alphabetic())
                    || !scheme
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
                {
                    return Err(invalid("URL scheme is invalid"));
                }
                Self::Url(nonempty(rest)?)
            }
            "sx" => Self::Sinex(nonempty(rest)?),
            "span" => {
                let (container, range) = rest
                    .rsplit_once('#')
                    .ok_or_else(|| ReferenceError::Incomplete(text.into()))?;
                let (unit, range) = range
                    .split_once('=')
                    .ok_or_else(|| ReferenceError::Incomplete(text.into()))?;
                let unit = match unit {
                    "b" => SpanUnit::Bytes,
                    "c" => SpanUnit::CodePoints,
                    "u16" => SpanUnit::Utf16,
                    _ => return Err(invalid("unknown span unit")),
                };
                let (start, end) = range
                    .split_once('-')
                    .ok_or_else(|| ReferenceError::Incomplete(text.into()))?;
                let (start, end) = (number(start)?, number(end)?);
                if start > end {
                    return Err(invalid("span end precedes its start"));
                }
                let container = Box::new(Self::parse(container)?);
                if !matches!(
                    container.as_ref(),
                    Self::Message(_)
                        | Self::Block { .. }
                        | Self::Assertion(RevisionReference {
                            revision: Some(_),
                            ..
                        })
                        | Self::Variant {
                            revision: RevisionReference {
                                revision: Some(_),
                                ..
                            },
                            part: Some(_),
                            ..
                        }
                ) {
                    return Err(invalid(
                        "span container must be a message, block, exact assertion or exact variant part",
                    ));
                }
                Self::Span {
                    container,
                    unit,
                    start,
                    end,
                }
            }
            _ => return Err(ReferenceError::UnknownKind(text.into())),
        })
    }
    pub fn session(&self) -> Option<&SessionId> {
        match self {
            Self::Session(session) | Self::ToolCall(session, _) => Some(session),
            _ => None,
        }
    }
}
impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(id) => write!(f, "session:{id}"),
            Self::Message(id) => write!(f, "message:{id}"),
            Self::MessageRange { first, last } => write!(f, "message:{first}..{last}"),
            Self::Block { message, n } => write!(f, "block:{message}#{n}"),
            Self::ToolCall(session, call) => write!(f, "tool:{session}#{call}"),
            Self::Document(id) => write!(f, "document:{id}"),
            Self::Event(id) => write!(f, "event:{id}"),
            Self::Attachment(id) => write!(f, "attachment:{id}"),
            Self::Assertion(revision) => write!(f, "assertion:{revision}"),
            Self::Variant { revision, part } => {
                write!(f, "variant:{revision}")?;
                if let Some(n) = part {
                    write!(f, "#{n}")?;
                }
                Ok(())
            }
            Self::Page(revision) => write!(f, "page:{revision}"),
            Self::Query { key, parameters } => {
                write!(f, "query:{key}")?;
                if !parameters.is_empty() {
                    f.write_str("(")?;
                    for (n, (key, value)) in parameters.iter().enumerate() {
                        if n > 0 {
                            f.write_str(",")?;
                        }
                        write!(f, "{key}={value}")?;
                    }
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::Result { id, row } => {
                write!(f, "result:{id}")?;
                if let Some(n) = row {
                    write!(f, "#{n}")?;
                }
                Ok(())
            }
            Self::Delivery(id) => write!(f, "delivery:{id}"),
            Self::File { path, commit } => write!(f, "file:{path}@{commit}"),
            Self::Repo(value) => write!(f, "repo:{value}"),
            Self::Cwd(value) => write!(f, "cwd:{value}"),
            Self::Url(value) => write!(f, "url:{value}"),
            Self::Sinex(value) => write!(f, "sx:{value}"),
            Self::Span {
                container,
                unit,
                start,
                end,
            } => write!(f, "span:{container}#{}={start}-{end}", unit.as_str()),
        }
    }
}
impl std::str::FromStr for Reference {
    type Err = ReferenceError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}
impl Serialize for Reference {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_newtype_struct("polylogue_reference", &self.to_string())
    }
}
impl<'de> Deserialize<'de> for Reference {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Self::parse(&String::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}
#[cfg(test)]
mod tests;

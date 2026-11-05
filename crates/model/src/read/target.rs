//! The closed set of references a read selects directly. Conversion is the one
//! refusal point; every consumer matches this set without a remainder arm.
use crate::{DocumentId, MessageId, Reference, RevisionReference, SessionId};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReadTarget {
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
    Document(DocumentId),
    Event(String),
    /// One exact assertion revision; a family is resolved before conversion.
    Assertion {
        root: String,
        revision: String,
    },
    Delivery(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reference} is not a read target: {reason}")]
pub struct NotReadTarget {
    pub reference: String,
    pub reason: &'static str,
}

impl TryFrom<Reference> for ReadTarget {
    type Error = NotReadTarget;
    fn try_from(reference: Reference) -> Result<Self, NotReadTarget> {
        let refuse = |reference: &Reference, reason| NotReadTarget {
            reference: reference.to_string(),
            reason,
        };
        Ok(match reference {
            Reference::Session(id) => Self::Session(id),
            Reference::Message(id) => Self::Message(id),
            Reference::MessageRange { first, last } => Self::MessageRange { first, last },
            Reference::Block { message, n } => Self::Block { message, n },
            Reference::Document(id) => Self::Document(id),
            Reference::Event(id) => Self::Event(id),
            Reference::Delivery(id) => Self::Delivery(id),
            Reference::Assertion(RevisionReference {
                root,
                revision: Some(revision),
            }) => Self::Assertion { root, revision },
            Reference::Assertion(RevisionReference { revision: None, .. }) => {
                return Err(refuse(
                    &reference,
                    "an assertion family is read through its current revision",
                ));
            }
            Reference::Span { .. } => {
                return Err(refuse(
                    &reference,
                    "a span addresses text inside a record; read its container",
                ));
            }
            Reference::ToolCall(..)
            | Reference::Run(_)
            | Reference::Request(_)
            | Reference::Attachment(_)
            | Reference::Variant { .. }
            | Reference::Page(_)
            | Reference::Role(_)
            | Reference::Query { .. }
            | Reference::Result { .. }
            | Reference::File { .. }
            | Reference::Repo(_)
            | Reference::Cwd(_)
            | Reference::Url(_)
            | Reference::Sinex(_) => {
                return Err(refuse(&reference, "it has no selected-row reading"));
            }
        })
    }
}

impl ReadTarget {
    pub fn reference(&self) -> Reference {
        match self {
            Self::Session(id) => Reference::Session(id.clone()),
            Self::Message(id) => Reference::Message(id.clone()),
            Self::MessageRange { first, last } => Reference::MessageRange {
                first: first.clone(),
                last: last.clone(),
            },
            Self::Block { message, n } => Reference::Block {
                message: message.clone(),
                n: *n,
            },
            Self::Document(id) => Reference::Document(id.clone()),
            Self::Event(id) => Reference::Event(id.clone()),
            Self::Assertion { root, revision } => {
                Reference::Assertion(RevisionReference::exact(root.clone(), revision.clone()))
            }
            Self::Delivery(id) => Reference::Delivery(id.clone()),
        }
    }
}

impl fmt::Display for ReadTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.reference().fmt(f)
    }
}

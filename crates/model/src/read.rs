//! The read contract shared by selection, view projection, layout and delivery.
//! Canonical source records stay separate from presentation substitutions.
mod delivery;
mod evidence;
mod items;
mod options;
mod presentation;
mod projection;
mod tokens;
pub use delivery::*;
pub use evidence::*;
pub use items::*;
pub use options::*;
pub use presentation::*;
pub use projection::*;
pub use tokens::*;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReadError {
    #[error("view {view} does not read {target}; use transcript or raw")]
    UnsupportedTarget {
        view: View,
        target: String,
        accepted_views: Vec<View>,
    },
    #[error(
        "unknown view {name:?}; views: transcript, dialogue, summary, context, links, changes, live, raw"
    )]
    UnknownView { name: String },
    #[error("view {view} does not accept {option}; accepted by {accepted_by}")]
    UnsupportedOption {
        view: View,
        option: String,
        accepted_by: String,
    },
    #[error("{view} needs prepared {evidence} evidence for {reference}")]
    MissingEvidence {
        view: View,
        evidence: String,
        reference: String,
    },
    #[error("invalid read evidence: {0}")]
    InvalidEvidence(String),
}

#[cfg(test)]
mod tests;

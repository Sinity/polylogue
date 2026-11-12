//! The read contract shared by selection, view projection, layout and delivery.
//! Canonical source records stay separate from presentation substitutions.
mod composition;
mod cursor;
mod delivery;
mod evidence;
mod format;
mod fragments;
mod item_index;
mod items;
mod lineage;
mod options;
mod parents;

mod presentation;
mod projection;
mod target;
mod tokens;
pub use composition::*;
pub use cursor::*;
pub use delivery::*;
pub use evidence::*;
pub use format::*;
pub use fragments::*;
pub use item_index::*;
pub use items::*;
pub use lineage::*;
pub use options::*;
pub use parents::*;

pub use presentation::*;
pub use projection::*;
pub use target::*;
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
    #[error("view {view} needs {option}=full for {target}")]
    RequiredExpansion {
        view: View,
        target: String,
        option: String,
    },
    #[error("invalid read evidence: {0}")]
    InvalidEvidence(String),
    #[error("unknown fold {word:?}; kinds: {kinds}")]
    UnknownFold { word: String, kinds: String },
}

#[cfg(test)]
mod tests;

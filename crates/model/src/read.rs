//! The read contract shared by selection, view filter, layout and delivery.
//! Source records stay separate from presentation substitutions.
mod composition;
mod cursor;
mod delivery;
mod format;
mod fragments;
mod item_index;
mod items;
mod layout_facts;
mod lineage;
mod options;
mod parents;

mod presentation;
mod target;
mod tokens;
mod trace;
mod view_filter;
pub use composition::*;
pub use cursor::*;
pub use delivery::*;
pub use format::*;
pub use fragments::*;
pub use item_index::*;
pub use items::*;
pub use layout_facts::*;
pub use lineage::*;
pub use options::*;
pub use parents::*;

pub use presentation::*;
pub use target::*;
pub use tokens::*;
pub use trace::*;
pub use view_filter::*;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReadError {
    #[error("view {view} does not read {target}; use transcript or raw")]
    UnsupportedTarget {
        view: View,
        target: String,
        accepted_views: Vec<View>,
    },
    #[error("unknown view {name:?}; views: {}", View::NAMES.join(", "))]
    UnknownView { name: String },
    #[error("view {view} does not accept {option}; accepted by {accepted_by}")]
    UnsupportedOption {
        view: View,
        option: String,
        accepted_by: String,
    },
    #[error("{view} needs prepared {facts} for {reference}")]
    MissingPreparedData {
        view: View,
        facts: String,
        reference: String,
    },
    #[error("view {view} needs {option}=full for {target}")]
    RequiredExpansion {
        view: View,
        target: String,
        option: String,
    },
    #[error("invalid prepared read data: {0}")]
    InvalidPreparedData(String),
    #[error("unknown show or hide word {word:?}; kinds: {kinds}")]
    UnknownShowWord { word: String, kinds: String },
}

impl ReadError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidPreparedData(message.into())
    }
}

#[cfg(test)]
mod tests;

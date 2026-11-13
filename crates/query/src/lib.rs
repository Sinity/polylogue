//! The query language: the existing language ported, not a second one. A
//! read is a query, the columns it shows and the format it is printed in.
//! Parsing and binding produce a typed Plan. Store alone lowers and executes it.

pub mod ast;
pub mod catalog;
pub mod compile;
pub mod defaults;
pub mod explain;
pub mod fields;
pub mod filter;
pub mod parse;
pub mod plan;
pub mod search;
pub mod selection;
pub mod stages;
pub mod time;
pub mod units;

pub use compile::{
    Bindings, ColKind, Column, ColumnReference, Definition, Kept, Options, SavedQuery,
    SavedQueryBody, Shape, compile,
};
pub use parse::parse;
pub use time::Clock;

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, thiserror::Error, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueryError {
    #[error("{diagnostic}")]
    Language {
        error: Box<query_language::Error>,
        diagnostic: String,
    },
    /// With a caret under the position and, when a simple repair parses,
    /// a corrected query.
    #[error("{message} at position {position}\n  {line}\n  {caret}")]
    Syntax {
        message: String,
        position: usize,
        line: String,
        caret: String,
        suggestion: Option<String>,
    },
    #[error("unknown field {field:?} at position {position}{}", hint(.similar))]
    UnknownField {
        field: String,
        unit: String,
        position: usize,
        /// Known fields with similar names.
        similar: Vec<String>,
        suggestion: Option<String>,
    },
    /// A value the field cannot take: a closed set, or a malformed number,
    /// time, duration or amount.
    #[error("{message}")]
    BadValue {
        field: String,
        value: String,
        position: usize,
        message: String,
    },
    /// The first word is a command, or a near miss of a unit or command.
    #[error("{message}")]
    NotAQuery {
        word: String,
        message: String,
        suggestion: Option<String>,
    },
    #[error("{message}")]
    Invalid { message: String, position: usize },
    /// Valid, but the part it needs is not built yet.
    #[error("{what} is not built yet")]
    NotBuilt { what: String },
    #[error("{what} is unavailable")]
    Unavailable { what: String },
    #[error("{what} is ambiguous: {}", .candidates.join(", "))]
    Ambiguous {
        what: String,
        candidates: Vec<String>,
    },
}

fn hint(similar: &[String]) -> String {
    match similar {
        [] => String::new(),
        [one] => format!("; did you mean {one:?}?"),
        many => format!(
            "; did you mean one of {}?",
            many.iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

impl QueryError {
    pub fn language(error: query_language::Error, source: &str) -> Self {
        let diagnostic = error.diagnostic(source);
        Self::Language {
            error: Box::new(error),
            diagnostic,
        }
    }

    pub fn syntax(text: &str, at: usize, message: String, suggestion: Option<String>) -> Self {
        let at = at.min(text.len());
        let line = text.to_owned();
        let caret = format!("{}^", " ".repeat(text[..at].chars().count()));
        QueryError::Syntax {
            message,
            position: at,
            line,
            caret,
            suggestion,
        }
    }

    /// The stable machine name of the error, as surfaces report it.
    pub fn kind(&self) -> &'static str {
        match self {
            QueryError::Language { error, .. } => error.kind.code(),
            QueryError::Syntax { .. } | QueryError::Invalid { .. } => "invalid_query",
            QueryError::UnknownField { .. } => "unknown_field",
            QueryError::BadValue { .. } => "bad_value",
            QueryError::NotAQuery { .. } => "not_a_query",
            QueryError::NotBuilt { .. } => "not_built",
            QueryError::Unavailable { .. } => "not_yet_available",
            QueryError::Ambiguous { .. } => "ambiguous",
        }
    }

    /// Structured diagnostics preserve the shared language fields at the
    /// transport boundary instead of hiding them inside an enum wrapper.
    pub fn detail(&self) -> Option<serde_json::Value> {
        match self {
            Self::Language { error, diagnostic } => {
                let mut value = serde_json::to_value(error).ok()?;
                value["diagnostic"] = serde_json::Value::String(diagnostic.clone());
                Some(value)
            }
            _ => serde_json::to_value(self).ok(),
        }
    }

    /// The corrected query that runs as given, when one could be formed.
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            QueryError::Language { error, .. } => error.suggestion.as_deref(),
            QueryError::Syntax { suggestion, .. }
            | QueryError::UnknownField { suggestion, .. }
            | QueryError::NotAQuery { suggestion, .. } => suggestion.as_deref(),
            _ => None,
        }
    }
}

/// One literal string in the query language.
pub fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

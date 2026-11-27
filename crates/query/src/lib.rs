//! The shared query language bound to Polylogue's units and fields. A
//! read is a query, the columns it shows and the format it is printed in.
//! Parsing and binding produce a typed Plan. Store alone lowers and executes it.

pub mod ast;
pub mod bound;
pub mod catalog;
pub mod compile;
pub mod defaults;
pub mod fields;
pub mod filter;
pub mod money;
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
    #[error("{reason}")]
    Unavailable { reason: String },
    /// A thing the line names does not exist for this caller.
    #[error("{message}")]
    NotFound { message: String },
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
    /// This error, found while binding the body of the saved query `name`
    /// whose text is `text`, as the caller sees it: the position the
    /// caller wrote `query:NAME` at, and a message naming where in the
    /// saved text it was found.
    pub fn inside_saved(self, name: &str, text: &str, position: usize) -> Self {
        let within = |at: usize, message: &str| {
            format!("in query:{name} at position {at} of its text {text:?}: {message}")
        };
        match self {
            Self::Invalid {
                message,
                position: at,
            } => Self::Invalid {
                message: within(at, &message),
                position,
            },
            Self::BadValue {
                field,
                value,
                position: at,
                message,
            } => Self::BadValue {
                field,
                value,
                message: within(at, &message),
                position,
            },
            Self::Syntax {
                message,
                position: at,
                line,
                caret,
                ..
            } => Self::Syntax {
                message: within(at, &message),
                position,
                line,
                caret,
                suggestion: None,
            },
            // A saved body names only fields the catalog had when it was
            // saved; the caller's position is the one to show.
            Self::UnknownField {
                field,
                unit,
                similar,
                ..
            } => Self::UnknownField {
                field,
                unit,
                position,
                similar,
                suggestion: None,
            },
            Self::Language { error, diagnostic } => Self::Language {
                error,
                diagnostic: format!("in query:{name}: {diagnostic}"),
            },
            other => other,
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid {
            message: message.into(),
            position: 0,
        }
    }

    pub fn language(mut error: query_language::Error, source: &str) -> Self {
        if error.kind == query_language::ErrorKind::BadValue
            && error.message.to_ascii_lowercase().starts_with("`origin`")
        {
            let value = source
                .get(error.position..)
                .unwrap_or_default()
                .trim_start_matches(['\"', '\''])
                .split(|c: char| c.is_whitespace() || matches!(c, '\"' | '\'' | ',' | ')'))
                .next()
                .unwrap_or_default();
            let listed = error.candidates.join(", ");
            error
                .candidates
                .sort_by_key(|name| (strsim::damerau_levenshtein(value, name), name.clone()));
            error.message = error.message.replace(&listed, &error.candidates.join(", "));
        }
        if error.kind == query_language::ErrorKind::UnknownField
            && let Some((message, _)) = error.message.split_once("; fields: ")
        {
            error.message = message.to_owned();
        }
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
    pub fn kind(&self) -> polylogue_model::Code {
        use polylogue_model::Code;
        match self {
            QueryError::Language { error, .. } => error.kind.into(),
            QueryError::Syntax { .. } | QueryError::Invalid { .. } => Code::InvalidQuery,
            QueryError::UnknownField { .. } => Code::UnknownField,
            QueryError::BadValue { .. } => Code::BadValue,
            QueryError::NotAQuery { .. } => Code::NotAQuery,
            QueryError::NotBuilt { .. } => Code::NotBuilt,
            QueryError::Unavailable { .. } => Code::NotYetAvailable,
            QueryError::NotFound { .. } => Code::NotFound,
            QueryError::Ambiguous { .. } => Code::Ambiguous,
        }
    }

    /// Structured diagnostics preserve the shared language fields at the
    /// transport boundary instead of hiding them inside an enum wrapper. Other
    /// errors are described by `kind()` and their message alone.
    pub fn detail(&self) -> Option<serde_json::Value> {
        match self {
            Self::Language { error, diagnostic } => {
                let mut value = serde_json::to_value(error).ok()?;
                value["diagnostic"] = serde_json::Value::String(diagnostic.clone());
                Some(value)
            }
            _ => None,
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

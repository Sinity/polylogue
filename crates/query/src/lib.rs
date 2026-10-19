//! The query language (architecture 7; spec query.md): the existing
//! language ported, not a second one. A read is a query, the
//! columns it shows and the format it is printed in. Parsing, checking and compiling are pure: the
//! result is SQL with bound parameters that the daemon runs against
//! `index.db` (with `archive.db` attached as `a`) in one consistent read.

pub mod ast;
pub mod compile;
pub mod fields;
pub mod parse;
pub mod time;
pub mod units;

pub use compile::{
    Attach, ColKind, Column, IdCheck, Kept, Measure, Near, Options, Param, Program, Shape,
    WhyClause, compile,
};
pub use parse::parse;
pub use time::Clock;

use pest_derive::Parser;
use serde::Serialize;

#[derive(Parser)]
#[grammar = "query.pest"]
pub struct Grammar;

/// The CLI's command and verb names. A query whose first unquoted word is
/// one of these, or a near miss of one, is refused with a suggestion rather
/// than searched for (spec query.md, principle 2).
pub const COMMAND_WORDS: &[&str] = &[
    "status",
    "daemon",
    "import",
    "list",
    "read",
    "export",
    "note",
    "brief",
    "continue",
    "recall",
    "measure",
    "live",
    "notices",
    "sql",
    "excise",
    "embed",
    "sinex",
    "web",
    "mcp",
    "hook",
    "completions",
    "find",
    "takeover",
    "setting",
    "work",
    "changes",
    "keep",
    "hooks",
    "install-agent",
    "browser-host",
];

#[derive(Debug, Clone, PartialEq, thiserror::Error, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueryError {
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
    #[error("parentheses nest more than {max} deep")]
    TooDeep { max: usize },
    #[error("unknown field {field:?} at position {position}{}", hint(.candidates))]
    UnknownField {
        field: String,
        unit: String,
        position: usize,
        candidates: Vec<String>,
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
    #[error("{what} is not built yet ({section})")]
    NotBuilt { what: String, section: String },
}

fn hint(candidates: &[String]) -> String {
    match candidates {
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
    pub(crate) fn syntax(
        text: &str,
        at: usize,
        message: String,
        suggestion: Option<String>,
    ) -> Self {
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
            QueryError::Syntax { .. } | QueryError::TooDeep { .. } | QueryError::Invalid { .. } => {
                "invalid_query"
            }
            QueryError::UnknownField { .. } => "unknown_field",
            QueryError::BadValue { .. } => "bad_value",
            QueryError::NotAQuery { .. } => "not_a_query",
            QueryError::NotBuilt { .. } => "not_built",
        }
    }

    /// The corrected query that runs as given, when one could be formed.
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            QueryError::Syntax { suggestion, .. }
            | QueryError::UnknownField { suggestion, .. }
            | QueryError::NotAQuery { suggestion, .. } => suggestion.as_deref(),
            _ => None,
        }
    }
}

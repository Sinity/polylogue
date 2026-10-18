//! The query language (architecture 7): the existing language ported, not a
//! second one. One read program is query × projection × render. Parsing is
//! pure; the field table is data; compiling produces SQL for the store.

pub mod ast;
pub mod check;
pub mod compile;
pub mod fields;

use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "query.pest"]
pub struct Grammar;

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    /// With a caret under the position and the nearest alternative.
    #[error("{message}\n{line}\n{caret}")]
    Syntax {
        message: String,
        line: String,
        caret: String,
    },
    #[error("unknown field {field:?} for {unit}; did you mean {suggestion:?}?")]
    UnknownField {
        field: String,
        unit: String,
        suggestion: Option<String>,
    },
}

pub fn parse(_text: &str) -> Result<ast::Query, QueryError> {
    // Architecture 7: pest parse tree to AST, errors with caret.
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pest::Parser;

    #[test]
    fn the_grammar_accepts_both_entry_shapes() {
        for text in [
            "repo:polylogue since:7d \"json envelope\"",
            "origin:(codex|claude-code) -tag:stale",
            "messages:>=10 words:>=200",
            "date between 2026-01-01 and 2026-02-01",
            "sessions where (repo:polylogue OR origin:chatgpt) AND NOT tag:stale",
            "messages where role:assistant AND text:timeout | group by role | count | sort by count desc",
            "exists assertion(value.status:approved)",
            "timeout",
            "repo:polylogue OR repo:sinex",
            "",
        ] {
            Grammar::parse(Rule::query, text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        }
    }

    #[test]
    fn the_grammar_refuses_broken_input() {
        for text in ["repo:(a|", "messages where", "\"unclosed"] {
            assert!(Grammar::parse(Rule::query, text).is_err(), "{text:?}");
        }
    }
}

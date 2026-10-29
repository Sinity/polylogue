//! Product catalog binding for the shared parser.
use crate::{QueryError, ast::Query, catalog};

pub fn parse(text: &str) -> Result<Query, QueryError> {
    with_catalog(text, &catalog::catalog(&[], &[]))
}

pub fn with_catalog(text: &str, catalog: &query_language::catalog::Catalog) -> Result<Query, QueryError> {
    let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
    for c in text.chars() {
        if escaped { escaped = false; continue; }
        match c {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            '(' if !quoted => { depth += 1; if depth > 64 { return Err(QueryError::TooDeep { max: 64 }); } }
            ')' if !quoted => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    query_language::parse::parse(text, catalog).map_err(|error| QueryError::language(error, text))
}

#[cfg(test)]
mod tests;

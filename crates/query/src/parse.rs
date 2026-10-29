//! Product catalog binding for the shared parser.
use crate::{QueryError, ast::Query, catalog};

pub fn parse(text: &str) -> Result<Query, QueryError> {
    with_catalog(text, &catalog::catalog(&[], &[]))
}

pub fn with_catalog(text: &str, catalog: &query_language::catalog::Catalog) -> Result<Query, QueryError> {
    query_language::parse::parse(text, catalog).map_err(|error| QueryError::language(error, text))
}

#[cfg(test)]
mod tests;

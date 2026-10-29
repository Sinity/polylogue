//! Product catalog binding for the shared parser.
use crate::{QueryError, ast::Query, catalog};

pub fn parse(text: &str) -> Result<Query, QueryError> {
    with_catalog(text, &catalog::catalog(&[], &[]))
}

pub fn with_catalog(
    text: &str,
    catalog: &query_language::catalog::Catalog,
) -> Result<Query, QueryError> {
    let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
    for c in text.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            '(' if !quoted => {
                depth += 1;
                if depth > 64 {
                    return Err(QueryError::TooDeep { max: 64 });
                }
            }
            ')' if !quoted => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    query_language::parse::parse(text, catalog).map_err(|mut error| {
        if error.kind == query_language::ErrorKind::UnknownField
            && text.get(error.position..error.position + 5).is_some_and(|field| field.eq_ignore_ascii_case("near:")) {
            let corrected = format!("{}similar:{}", &text[..error.position], &text[error.position + 5..]);
            if query_language::parse::parse(&corrected, catalog).is_ok() { error.suggestion = Some(corrected); }
            error.hint = Some("use similar: for semantic ranking".into());
        }
        QueryError::language(error, text)
    })
}

#[cfg(test)]
mod tests;

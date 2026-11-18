//! Product catalog binding for the shared parser.
use crate::{
    QueryError,
    ast::{ConditionKind, Query, Source},
    catalog,
};

pub fn parse(text: &str) -> Result<Query, QueryError> {
    with_catalog(text, &catalog::catalog(&[]))
}

pub fn with_catalog(
    text: &str,
    catalog: &query_language::catalog::Catalog,
) -> Result<Query, QueryError> {
    let mut query = query_language::parse::parse(text, catalog).map_err(|mut error| {
        if error.kind == query_language::ErrorKind::UnknownField
            && text
                .get(error.position..error.position + error.length)
                .is_some_and(|kind| polylogue_model::SINEX_LOCAL_KINDS.contains(&kind))
            && let Some(correction) = sinex_local_error(text, catalog, error.position)
        {
            return correction;
        }
        if error.kind == query_language::ErrorKind::UnknownField
            && text
                .get(error.position..error.position + 5)
                .is_some_and(|field| field.eq_ignore_ascii_case("near:"))
        {
            let corrected = format!(
                "{}similar:{}",
                &text[..error.position],
                &text[error.position + 5..]
            );
            if query_language::parse::parse(&corrected, catalog).is_ok() {
                error.suggestion = Some(corrected);
            }
            error.hint = Some("use similar: for semantic ranking".into());
        }
        QueryError::language(error, text)
    })?;
    bind_lone_reference(&mut query);
    Ok(query)
}

/// A diagnostic-only catalog leaves accepted queries and captured bindings unchanged.
fn sinex_local_error(
    text: &str,
    catalog: &query_language::catalog::Catalog,
    position: usize,
) -> Option<QueryError> {
    let mut diagnostic = catalog.clone();
    diagnostic.references.extend(
        polylogue_model::SINEX_LOCAL_KINDS
            .into_iter()
            .map(|kind| query_language::catalog::ReferenceForm::prefix(format!("{kind}:"))),
    );
    let mut query = query_language::parse::parse(text, &diagnostic).ok()?;
    crate::plan::walk(&mut query, &mut |_| Ok(()), &mut |condition| {
        if condition.span.start == position
            && let ConditionKind::Reference { reference } = &condition.kind
            && let Err(error @ polylogue_model::ReferenceError::SinexLocal { .. }) =
                polylogue_model::Reference::parse(reference.as_str())
        {
            return Err(QueryError::Invalid {
                message: error.to_string(),
                position,
            });
        }
        Ok(())
    })
    .err()
}

/// A query that is only an exact assertion revision (`assertion:R@X`) reads
/// the annotations unit: the revision is one row whatever the default unit.
/// Every other lone reference stays a condition on the default unit.
fn bind_lone_reference(query: &mut Query) {
    if !query.transforms.is_empty() || !query.attach.is_empty() {
        return;
    }
    let Source::Unit {
        unit,
        implicit,
        condition: Some(condition),
    } = &mut query.source
    else {
        return;
    };
    if !*implicit {
        return;
    }
    let ConditionKind::Reference { reference: text } = &condition.kind else {
        return;
    };
    if let Ok(polylogue_model::Reference::Assertion(revision)) =
        polylogue_model::Reference::parse(text)
        && revision.revision.is_some()
    {
        // Explicit, so the unit survives formatting the query back to text.
        *unit = crate::units::Unit::Annotations.name().to_owned();
        *implicit = false;
    }
}

#[cfg(test)]
mod tests;

//! Assertions read through the single query backend for briefings, resume and
//! signal definitions. Every caller shares graph currency and visibility.

use super::{Ask, run};
use polylogue_model::{Reference, RevisionReference, assertion::Assertion};
use polylogue_store::{Reader, StoreError};

/// Every assertion `assertions where CONDITIONS` selects.
pub fn list(
    reader: &Reader<'_>,
    conditions: &str,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<Vec<Assertion>, StoreError> {
    let mut notes = Vec::new();
    for id in ids(reader, conditions, context)? {
        if let Some(note) = reader.assertion(&id)? {
            notes.push(note);
        }
    }
    Ok(notes)
}

/// How many assertions `assertions where CONDITIONS` selects.
pub fn count(
    reader: &Reader<'_>,
    conditions: &str,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<u64, StoreError> {
    let page = page(reader, &text(conditions), None, context)?;
    page.count.ok_or(StoreError::ReadDeadlineExceeded)
}

/// Exact association with at least one of `targets`.
pub fn about(targets: &[Reference]) -> String {
    let any: Vec<String> = targets
        .iter()
        .map(|t| format!("about:{}", quoted(&t.to_string())))
        .collect();
    format!("({})", any.join(" OR "))
}

/// A value as a quoted query string.
pub use polylogue_query::quoted;

fn text(conditions: &str) -> String {
    if conditions.trim().is_empty() {
        "assertions".into()
    } else {
        format!("assertions where {conditions}")
    }
}

pub(crate) fn ids(
    reader: &Reader<'_>,
    conditions: &str,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<Vec<RevisionReference>, StoreError> {
    let text = text(conditions);
    let mut ids = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let page = page(reader, &text, cursor.as_deref(), context)?;
        ids.extend(page.rows.iter().filter_map(|row| {
            Some(RevisionReference::exact(
                row.get("root")?.as_str()?.to_owned(),
                row.get("id")?.as_str()?.to_owned(),
            ))
        }));
        match page.next {
            Some(next) => cursor = Some(next),
            None => return Ok(ids),
        }
    }
}

fn page(
    reader: &Reader<'_>,
    text: &str,
    cursor: Option<&str>,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<crate::protocol::query::Rows, StoreError> {
    run(
        reader,
        Ask {
            text,
            cursor,
            page_size: None,
            why: false,
            session: None,
            context: std::sync::Arc::clone(context),
            embeddings: false,
            ranking_identity: None,
            near: None,
            fields: &["root".to_owned(), "id".to_owned()],
        },
    )
    .and_then(|answer| answer.value.into_rows())
    .map_err(|f| StoreError::Invalid(format!("assertions query {text:?}: {}", f.message)))
}

#[cfg(test)]
mod tests;

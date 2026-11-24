//! Annotations read through the single query backend for briefings, resume and
//! signal definitions. Every caller sees the same current revisions and visibility.

use super::{Ask, run};
use polylogue_model::{Reference, RevisionReference, annotation::Annotation};
use polylogue_store::{Reader, StoreError};

/// Every annotation `annotations where CONDITIONS` selects.
pub fn list(
    reader: &Reader<'_>,
    conditions: &str,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Vec<Annotation>, StoreError> {
    let mut notes = Vec::new();
    for id in ids(reader, conditions, context)? {
        if let Some(note) = reader.annotation(&id)? {
            notes.push(note);
        }
    }
    Ok(notes)
}

/// How many annotations `annotations where CONDITIONS` selects.
pub fn count(
    reader: &Reader<'_>,
    conditions: &str,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
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
        "annotations".into()
    } else {
        format!("annotations where {conditions}")
    }
}

pub(crate) fn ids(
    reader: &Reader<'_>,
    conditions: &str,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
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
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
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
    .map_err(|f| store_error(&f, text))
}

/// A failed annotations query as a store error that keeps what the caller
/// decides on: a deadline or stop request stays retryable, and any other
/// refusal keeps its product code.
pub(crate) fn store_error(f: &crate::daemon::Failure, text: &str) -> StoreError {
    let message = format!("annotations query {text:?}: {}", f.message);
    match f.code() {
        "deadline_exceeded" => StoreError::ReadDeadlineExceeded,
        "cancelled" => StoreError::ReadCancelled,
        code => match polylogue_model::Code::from_name(code) {
            Some(code) => StoreError::Selection { code, message },
            None => StoreError::internal(message),
        },
    }
}

#[cfg(test)]
mod tests;

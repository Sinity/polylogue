//! Notes read whole through a `notes` query, for the briefings, resume and
//! signal rules: one listing path, so a note is current, withdrawn or hidden
//! by the same rules everywhere.

use super::{Ask, run};
use polylogue_model::{Note, Target, Timestamp};
use polylogue_store::{Reader, StoreError};

/// Every note `notes where CONDITIONS` lists, newest first.
pub fn list(
    reader: &Reader<'_>,
    conditions: &str,
    now: Timestamp,
) -> Result<Vec<Note>, StoreError> {
    let mut notes = Vec::new();
    for id in ids(reader, conditions, now)? {
        if let Some(note) = reader.note(&id)? {
            notes.push(note);
        }
    }
    Ok(notes)
}

/// How many notes `notes where CONDITIONS` lists.
pub fn count(reader: &Reader<'_>, conditions: &str, now: Timestamp) -> Result<u64, StoreError> {
    let page = page(reader, &text(conditions), None, now)?;
    Ok(page["count"].as_u64().unwrap_or(0))
}

/// The condition that a note is about at least one of `targets`.
pub fn about(targets: &[Target]) -> String {
    let any: Vec<String> = targets
        .iter()
        .map(|t| format!("target:{}", quoted(&t.to_string())))
        .collect();
    format!("({})", any.join(" OR "))
}

/// A value as a quoted query string.
pub fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn text(conditions: &str) -> String {
    if conditions.trim().is_empty() {
        "notes".into()
    } else {
        format!("notes where {conditions}")
    }
}

fn ids(reader: &Reader<'_>, conditions: &str, now: Timestamp) -> Result<Vec<String>, StoreError> {
    let text = text(conditions);
    let mut ids = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let page = page(reader, &text, cursor.as_deref(), now)?;
        ids.extend(
            page["rows"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|row| row["id"].as_str().map(str::to_owned)),
        );
        match page["next"].as_str() {
            Some(next) => cursor = Some(next.to_owned()),
            None => return Ok(ids),
        }
    }
}

fn page(
    reader: &Reader<'_>,
    text: &str,
    cursor: Option<&str>,
    now: Timestamp,
) -> Result<serde_json::Value, StoreError> {
    run(
        reader,
        Ask {
            text,
            cursor,
            page_size: None,
            why: false,
            session: None,
            now: now.unix_ms(),
            day: None,
            embeddings: false,
            near: None,
            fields: &[],
        },
    )
    .map_err(|f| StoreError::Invalid(format!("notes query {text:?}: {}", f.message)))
}

#[cfg(test)]
mod tests;

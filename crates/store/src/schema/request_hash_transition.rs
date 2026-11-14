//! Validate recorded request encodings before replacing their digests.
use crate::{Result, StoreError};
use polylogue_model::user::{UserRequest, UserWrite};
use rusqlite::{Transaction, params};

// The stored JSON retains the predecessor serializer's field and nested-map
// order. Re-serializing decoded structs could introduce defaults or reorder
// maps. Change only the fields its intent function changed. This recipe is
// used at conversion; current reads and writes have one canonical recipe.
fn prior_hash(request: &UserRequest, packed: &[u8]) -> Result<[u8; 32]> {
    let mut frozen: polylogue_model::Json = crate::encoding::unpack_json(packed)?;
    let mut intent = frozen["write"].take();
    match &request.write {
        UserWrite::Assert { .. } => {
            if let Some(assertions) = intent["assertions"].as_array_mut() {
                for assertion in assertions {
                    if let Some(value) = assertion.get_mut("value") {
                        value.sort_all_objects();
                    }
                }
            }
        }
        UserWrite::AssertQuery { .. } => {
            intent["resolved"] = serde_json::json!([]);
            if let Some(value) = intent["assertion"].get_mut("value") {
                value.sort_all_objects();
            }
        }
        UserWrite::ReviseAssertions { patch, .. } => {
            if matches!(patch.value, polylogue_model::assertion::Change::Set(_)) {
                intent["patch"]["value"]["value"].sort_all_objects();
            }
        }
        UserWrite::KeepResult { .. } => {
            intent["rows_json"] = serde_json::json!("");
            intent["read_at"] = serde_json::json!("");
        }
        _ => {}
    }
    Ok(*blake3::hash(&serde_json::to_vec(&(
        frozen["machine"].take(),
        frozen["authorship"].take(),
        intent,
    ))?)
    .as_bytes())
}

pub(super) fn apply(tx: &Transaction<'_>) -> Result<()> {
    let source = crate::source::Source::Ledger;
    let mut statement = tx.prepare(
        &source.sql("SELECT id,content_hash,request FROM @source.client_request ORDER BY id"),
    )?;
    let mut rows = statement.query([])?;
    let file: String = tx.query_row(
        "SELECT file FROM pragma_database_list WHERE name='main'",
        [],
        |row| row.get(0),
    )?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let packed: Vec<u8> = row.get(2)?;
        let request: UserRequest = crate::encoding::unpack_json(&packed)?;
        if request.id != id {
            return Err(StoreError::Conflict(id));
        }
        let canonical = crate::primary::authored::request_hash(&request)?;
        if hash.as_slice() == canonical {
            continue;
        }
        if hash.as_slice() != prior_hash(&request, &packed)? {
            return Err(StoreError::Conflict(id));
        }
        tx.execute(
            &source.sql("UPDATE @source.client_request SET content_hash=?1 WHERE id=?2"),
            params![&canonical[..], id],
        )?;
        crate::fault::check("archive.request_hash_transition", &file)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

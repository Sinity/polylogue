//! Numbered representation changes. Source JSON and custody hashes remain
//! immutable; only the executable representation stored alongside them changes.
use crate::{Result, StoreError, encoding, primary::products::Product};
use polylogue_model::query_objects::{QueryObject, QueryRun};
use rusqlite::{Connection, params};
use serde_json::Value;

fn invalid(message: impl Into<String>) -> StoreError {
    StoreError::Invalid(message.into())
}
fn historical_object(value: &Value) -> bool {
    value["identity_ast"].get("stages").is_some()
}
fn historical_run(value: &Value) -> bool {
    value["binding"]["execution"]["replay"]["bound_ast"]
        .get("stages")
        .is_some()
}
fn object(value: Value) -> Result<QueryObject> {
    let object = if historical_object(&value) {
        QueryObject::from_structured_v1(value).map_err(invalid)?
    } else {
        serde_json::from_value(value)?
    };
    super::validate_object(&object)?;
    Ok(object)
}
fn run(value: Value) -> Result<QueryRun> {
    let run = if value["binding"]["kind"] == "continuation"
        || value["binding"]["execution"]["presentation"]["options"]
            .get("output_lines")
            .is_some()
    {
        QueryRun::from_retired_read_v2(value).map_err(invalid)?
    } else if historical_run(&value) {
        crate::primary::products::query_run_from_structured_v1(value)?
    } else {
        serde_json::from_value(value)?
    };
    super::validate_run(&run)?;
    Ok(run)
}

/// Version 1 -> 2. Called inside the schema owner's immediate transaction.
pub(crate) fn archive_v2(conn: &Connection) -> Result<()> {
    let mut statement = conn.prepare("SELECT hash,normal_form,object FROM query ORDER BY hash")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let form: String = row.get(1)?;
        let bytes: Vec<u8> = row.get(2)?;
        let source: Value = encoding::unpack_json(&bytes)?;
        let object = object(source)?;
        if object.id.to_string() != id
            || object.normal_form != serde_json::from_str::<Value>(&form)?
        {
            return Err(invalid(
                "stored query columns disagree with its immutable identity",
            ));
        }
        conn.execute(
            "UPDATE query SET object=?1 WHERE hash=?2",
            params![encoding::pack_json(&object)?, id],
        )?;
    }
    let mut statement = conn.prepare(
        "SELECT id,binding_hash,run_hash,run FROM query_run WHERE run IS NOT NULL ORDER BY id",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let binding: String = row.get(1)?;
        let record: String = row.get(2)?;
        let bytes: Vec<u8> = row.get(3)?;
        let source: Value = encoding::unpack_json(&bytes)?;
        let run = run(source)?;
        if run.id.to_string() != id
            || run.binding_hash()?.to_string() != binding
            || run.record_hash()?.to_string() != record
        {
            return Err(invalid(
                "stored run differs from original binding or invocation custody",
            ));
        }
        conn.execute(
            "UPDATE query_run SET run=?1 WHERE id=?2",
            params![encoding::pack_json(&run)?, id],
        )?;
    }
    // Completion rows, graph edges, first_run and all hashes are deliberately
    // untouched. The ordinary lookup path still cross-checks their custody.
    Ok(())
}

fn product(value: Value) -> Result<Product> {
    let mut value = value;
    match value["product"].as_str() {
        Some("query") => {
            value["object"] = serde_json::to_value(object(value["object"].take())?)?;
        }
        Some("query_run") => {
            value["run"] = serde_json::to_value(run(value["run"].take())?)?;
        }
        _ => {}
    }
    Ok(serde_json::from_value(value)?)
}

/// The primary cache keeps native context in separate columns. Updating its
/// Product body cannot change event identity, captured schema, or payload hash.
pub(crate) fn index_v2(conn: &Connection, primary: bool) -> Result<()> {
    if !primary {
        return Ok(());
    }
    for (table, key) in [("primary_fact", "event"), ("primary_pending", "n")] {
        let mut statement = conn.prepare(&format!(
            "SELECT {key},product FROM {table} WHERE product IS NOT NULL"
        ))?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let id: rusqlite::types::Value = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            let source: Value = encoding::unpack_json(&bytes)?;
            if !matches!(source["product"].as_str(), Some("query" | "query_run")) {
                continue;
            }
            let converted = product(source)?;
            conn.execute(
                &format!("UPDATE {table} SET product=?1 WHERE {key}=?2"),
                params![encoding::pack_json(&converted)?, id],
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

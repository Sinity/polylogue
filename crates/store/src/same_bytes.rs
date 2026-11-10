//! Chains of one input that hold exactly the same bytes. A file read again
//! after a restore or a remount could start a chain whose extents repeat an
//! older chain's; parsing such a chain gives that chain's reading again, so
//! its reading is recorded from the other chain's rows instead of parsed.
use crate::{Result, StoreError, source::Source};
use rusqlite::{Connection, OptionalExtension, params};

/// The newest chain of `input` after `chain` with the same length and the
/// same extents (offset, length, blob and blob offset) as `chain`.
pub(crate) fn newest_identical(conn: &Connection, input: &str, chain: i64) -> Result<Option<i64>> {
    let sql = Source::of(conn)?.sql(
        "WITH mine AS (SELECT offset, length, blob, blob_offset FROM @source.extent
                       WHERE input = ?1 AND chain = ?2),
              size AS (SELECT max(length) AS length FROM @source.revision
                       WHERE input = ?1 AND chain = ?2),
              later AS (SELECT chain, max(length) AS length FROM @source.revision
                        WHERE input = ?1 AND chain > ?2 GROUP BY chain)
         SELECT later.chain FROM later, size
         WHERE later.length = size.length AND size.length > 0
           AND (SELECT count(*) FROM @source.extent e WHERE e.input = ?1 AND e.chain = later.chain)
             = (SELECT count(*) FROM mine)
           AND NOT EXISTS (SELECT 1 FROM mine m WHERE NOT EXISTS (
                 SELECT 1 FROM @source.extent e WHERE e.input = ?1 AND e.chain = later.chain
                   AND e.offset = m.offset AND e.length = m.length
                   AND e.blob = m.blob AND e.blob_offset = m.blob_offset))
         ORDER BY later.chain DESC LIMIT 1",
    );
    Ok(conn
        .prepare_cached(&sql)?
        .query_row(params![input, chain], |row| row.get(0))
        .optional()?)
}

/// The index tables that hold one chain's reading, keyed by `input_pk` and
/// `chain`. Copying a reading copies each table's rows of the other chain.
const READING_TABLES: [&str; 18] = [
    "parse_state",
    "parsed",
    "session_record",
    "message_source",
    "document_source",
    "event_source",
    "link_source",
    "text_fragment_source",
    "message_injection_field",
    "unused",
    "parser_context",
    "parser_context_state",
    "wire_reading",
    "wire_receipt",
    "wire_shape",
    "wire_type",
    "attachment_binding",
    "export_reading",
];

/// An older chain whose reading is recorded from a newer chain of the same
/// input holding the same bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadingCopy {
    pub input: String,
    /// The newer chain, read under `reading_version`.
    pub from: i64,
    /// The older chain.
    pub to: i64,
    pub reading_version: String,
}

/// Records chain `to`'s reading as a copy of chain `from`'s, both of
/// `input` and holding the same bytes. `false` when the copy would not equal
/// parsing `to`: the bytes differ, `from` is not read to its end under
/// `reading_version`, `to` already has a reading, or the input's reading
/// writes rows shared by all its chains (a repository lookup, tool rows).
///
/// A message, document, event or link shows its source input's newest
/// chain, and on equal values a newer chain wins, so an older chain's copy
/// changes no shown row, and a copied document source is not the shown one. Its sessions are marked changed, as a parse of the
/// chain would mark them.
pub(crate) fn copy_reading(tx: &Connection, copy: &ReadingCopy, counter: i64) -> Result<bool> {
    let ReadingCopy {
        input,
        from,
        to,
        reading_version,
    } = copy;
    let (input, from, to) = (input.as_str(), *from, *to);
    if to >= from || newest_identical(tx, input, to)? != Some(from) {
        return Ok(false);
    }
    let Some(input_pk) = tx
        .query_row("SELECT pk FROM input_ref WHERE id = ?1", [input], |row| {
            row.get::<_, i64>(0)
        })
        .optional()?
    else {
        return Ok(false);
    };
    // Read to its end, so every source receipt of the reading was taken at
    // the end of its byte footprint and is taken again for `to` below.
    let length_sql = Source::of(tx)?
        .sql("(SELECT max(length) FROM @source.revision WHERE input = ?5 AND chain = ?2)");
    let ready: bool = tx.query_row(
        &format!(
            "SELECT EXISTS(SELECT 1 FROM parse_state WHERE input_pk = ?1 AND chain = ?2
             AND reading_version = ?3 AND status = 'done' AND parsed_upto = {length_sql})
         AND NOT EXISTS(SELECT 1 FROM parse_state WHERE input_pk = ?1 AND chain = ?4)
         AND NOT EXISTS(SELECT 1 FROM repo_lookup WHERE input_pk = ?1)
         AND NOT EXISTS(SELECT 1 FROM tool_row WHERE input_pk = ?1)
         AND NOT EXISTS(SELECT 1 FROM tool_database WHERE input_pk = ?1)"
        ),
        params![input_pk, from, reading_version, to, input],
        |row| row.get(0),
    )?;
    if !ready {
        return Ok(false);
    }
    for table in READING_TABLES {
        let columns: Vec<String> = tx
            .prepare_cached(&format!("SELECT name FROM pragma_table_info('{table}')"))?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let (key, select): (&str, Vec<String>) = if table == "attachment_binding" {
            // Keyed by the input's ID; a binding declared in the same chain
            // is declared in the copy's chain.
            (
                "input = ?4 AND chain = ?2",
                columns
                    .iter()
                    .map(|column| match column.as_str() {
                        "chain" => "?3".to_owned(),
                        "declared_chain" => {
                            "CASE WHEN declared_input = ?4 AND declared_chain = ?2 THEN ?3 ELSE declared_chain END"
                                .to_owned()
                        }
                        other => other.to_owned(),
                    })
                    .collect(),
            )
        } else {
            (
                "input_pk = ?1 AND chain = ?2",
                columns
                    .iter()
                    .map(|column| match column.as_str() {
                        "chain" => "?3".to_owned(),
                        // The shown document source is the newest chain's
                        // (documents.rs `refresh`); the older copy never is.
                        "shown" if table == "document_source" => "0".to_owned(),
                        // When this write admitted the row, as a parse
                        // of the chain would record its own write.
                        "commit_seq" | "admitted" => {
                            format!("CASE WHEN {column} IS NULL THEN NULL ELSE ?5 END")
                        }
                        other => other.to_owned(),
                    })
                    .collect(),
            )
        };
        let sql = format!(
            "INSERT INTO {table} ({}) SELECT {} FROM {table} WHERE {key}",
            columns.join(","),
            select.join(","),
        );
        let mut statement = tx.prepare_cached(&sql)?;
        let parameters = statement.parameter_count();
        let values: [&dyn rusqlite::ToSql; 5] = [&input_pk, &from, &to, &input, &counter];
        statement
            .execute(&values[..parameters])
            .map_err(|error| StoreError::Invalid(format!("copy {table}: {error}")))?;
    }
    let mut receipts = std::collections::HashMap::new();
    for (table, column, key) in [
        ("wire_reading", "state", "reading_version"),
        (
            "wire_receipt",
            "receipt",
            "reading_version || char(0) || number",
        ),
    ] {
        let rows: Vec<(String, String)> = tx
            .prepare(&format!(
                "SELECT {key}, {column} FROM {table} WHERE input_pk = ?1 AND chain = ?2"
            ))?
            .query_map(params![input_pk, to], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (row, text) in rows {
            let mut value: serde_json::Value = serde_json::from_str(&text)?;
            if !retake_receipts(tx, &mut value, input, from, to, &mut receipts)? {
                return Ok(false);
            }
            tx.prepare_cached(&format!(
                "UPDATE {table} SET {column} = ?4 WHERE input_pk = ?1 AND chain = ?2 AND {key} = ?3"
            ))?
            .execute(params![input_pk, to, row, serde_json::to_string(&value)?])?;
        }
    }
    tx.execute(
        "UPDATE session SET changed = ?3 WHERE pk IN (SELECT session_pk FROM parsed WHERE input_pk = ?1 AND chain = ?2)",
        params![input_pk, to, counter],
    )?;
    Ok(true)
}

/// Wire-shape evidence names where each shape was seen by source receipt:
/// the chain, the receipt order and the observation that confirmed the bytes
/// through the footprint's end. A receipt of chain `from` is taken again
/// for chain `to` through the same end. `false` for a receipt over anything
/// but stored bytes, whose copy would not be exact.
fn retake_receipts(
    tx: &Connection,
    value: &mut serde_json::Value,
    input: &str,
    from: i64,
    to: i64,
    receipts: &mut std::collections::HashMap<u64, crate::wire_shapes::ReadingReceipt>,
) -> Result<bool> {
    match value {
        serde_json::Value::Object(object) => {
            let receipt = object.get("input").and_then(serde_json::Value::as_str) == Some(input)
                && object.get("chain").and_then(serde_json::Value::as_i64) == Some(from)
                && object.contains_key("sequence")
                && object.contains_key("footprint");
            if receipt {
                let footprint = &object["footprint"];
                let (Some("bytes"), Some(end)) = (
                    footprint.get("kind").and_then(serde_json::Value::as_str),
                    footprint.get("end").and_then(serde_json::Value::as_u64),
                ) else {
                    return Ok(false);
                };
                let taken = match receipts.get(&end) {
                    Some(taken) => taken.clone(),
                    None => {
                        let taken = Source::reading_receipt(tx, input, to, end)?;
                        receipts.insert(end, taken.clone());
                        taken
                    }
                };
                object.insert("sequence".into(), taken.sequence.into());
                object.insert("id".into(), taken.id.into());
                object.insert("chain".into(), to.into());
                object.insert("observation".into(), taken.observation.into());
                object.insert("observed_at_ms".into(), taken.observed_at_ms.into());
                return Ok(true);
            }
            for child in object.values_mut() {
                if !retake_receipts(tx, child, input, from, to, receipts)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        serde_json::Value::Array(items) => {
            for item in items {
                if !retake_receipts(tx, item, input, from, to, receipts)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(true),
    }
}

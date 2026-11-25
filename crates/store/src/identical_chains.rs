//! Chains of one input whose bytes begin alike. A rewritten file starts a
//! chain that often repeats an older chain's bytes, wholly (a file read
//! again after a restore or a remount) or up to a point (a file that was
//! appended to, or edited after its first part). Parsing those bytes again
//! gives the reading the other chain had at that point, so the older chain's
//! reading is recorded from the newer chain's rows instead of parsed.
use crate::ConnExt as _;
use crate::{Result, StoreError, source::Source};
use polylogue_model::JsonExt;
use rusqlite::{Connection, params};
mod planning;
pub use planning::{ChainLayouts, PrefixCopies, PrefixCopy, SharedPrefixes};

/// One chain's stored length and how far its reading has parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainLayout {
    pub chain: i64,
    pub length: u64,
    /// The chain's reading, when it has one.
    pub reading: Option<StoredParseState>,
}

/// How far a chain's reading has read, under which parse version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredParseState {
    pub parse_version: String,
    /// The stored parse status word (`parsing`, `waiting`, `done`, ...).
    pub status: String,
    pub parsed_upto: u64,
    /// The stored length a waiting, refused or failed reading stands at.
    pub at_length: Option<u64>,
}

impl StoredParseState {
    /// Read through the end of a chain of `length` bytes: nothing remains
    /// for a later parse but a record still being written.
    pub fn settled(&self, length: u64) -> bool {
        self.status == "done" || (self.status == "waiting" && self.at_length == Some(length))
    }

    /// A reading another chain's reading can continue from or be copied
    /// from: current and not stopped.
    pub fn continues(&self, parse_version: &str) -> bool {
        self.parse_version == parse_version
            && matches!(self.status.as_str(), "parsing" | "waiting" | "done")
    }
}

/// The index tables that hold one chain's reading, keyed by `input_pk` and
/// `chain`. Copying a reading copies each table's rows of the other chain.
const READING_TABLES: [&str; 22] = [
    "parse_state",
    "session_record",
    "session_alias",
    "message_parse",
    "message_parse_body",
    "parse_text",
    "message_alias",
    "event_parse",
    "link_parse",
    "document_parse",
    "call_outcome",
    "text_fragment_source",
    "message_injection_field",
    "unused",
    "wire_reading",
    "wire_occurrence",
    "wire_shape",
    "wire_type",
    "attachment_binding",
    "export_reading",
    "prefix_copy",
    "prefix_member",
];

/// An older chain whose reading is recorded from a newer chain of the same
/// input whose bytes begin with the same `upto` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadingCopy {
    pub input: String,
    /// The newer chain, read under `parse_version` exactly to `upto`.
    pub from: i64,
    /// The older chain.
    pub to: i64,
    pub parse_version: String,
    /// The parse position of `from` the copy is taken at: the end of its
    /// last complete record within the bytes the two chains share.
    pub upto: u64,
    /// Every byte of `to` is one `from` was given when it reached `upto`, so
    /// the copy reads `to` to its end, as a parse of `to` alone would. Else
    /// `to` has bytes of its own after `upto`, parsed from there.
    pub reaches_end: bool,
}

/// Records chain `to`'s reading as a copy of chain `from`'s at `upto`,
/// both of `input`. The caller has compared the bytes: the two chains hold
/// the same bytes before the point the copy is taken at. `false` when the
/// copy would not equal parsing `to`: `from` is not read exactly to `upto`
/// under `parse_version`, `to` already has a reading, or the input's
/// reading writes rows shared by all its chains (a repository lookup, tool
/// rows), or copied membership names its own chain and must be resolved by
/// parsing.
///
/// A message, document, event or link shows its source input's newest
/// chain, and on equal values a newer chain wins. The copy holds exactly
/// the rows `from` holds at that moment, so it changes no shown row, and a
/// copied document source is not the shown one. Its sessions are marked
/// changed, as a parse of the chain would mark them. The copied parse state
/// stands where a parse of `to` stands once it has read those bytes: at its
/// end when it `reaches_end`, else parsing, with the parser checkpoint that
/// lets the parse of `to`'s own bytes continue from `upto`.
pub(crate) fn copy_reading(tx: &Connection, copy: &ReadingCopy, counter: i64) -> Result<bool> {
    let ReadingCopy {
        input,
        from,
        to,
        parse_version,
        upto,
        reaches_end,
    } = copy;
    let (input, from, to, upto) = (input.as_str(), *from, *to, *upto);
    let Ok(upto_sql) = i64::try_from(upto) else {
        return Ok(false);
    };
    let length_sql = Source::of(tx)?
        .sql("SELECT max(length) FROM @source.revision WHERE input = ?1 AND chain = ?2");
    let length = |chain: i64| -> Result<Option<u64>> {
        Ok(tx
            .prepare_cached(&length_sql)?
            .query_row(params![input, chain], |row| row.get::<_, Option<u64>>(0))?)
    };
    let (Some(from_length), Some(to_length)) = (length(from)?, length(to)?) else {
        return Ok(false);
    };
    if to >= from || upto > from_length || upto > to_length {
        return Ok(false);
    }
    let Some(input_pk) =
        tx.scalar_optional::<i64>("SELECT pk FROM input_ref WHERE id = ?1", [input])?
    else {
        return Ok(false);
    };
    // Every source commit of the reading so far was taken through at most
    // `upto`, which `to` holds too, and is taken again for `to` below.
    let ready: bool = tx.exists(
        "SELECT EXISTS(SELECT 1 FROM parse_state WHERE input_pk = ?1 AND chain = ?2
             AND parse_version = ?3 AND status IN ('parsing','waiting','done')
             AND parsed_upto = ?6)
         AND NOT EXISTS(SELECT 1 FROM parse_state WHERE input_pk = ?1 AND chain = ?4)
         AND NOT EXISTS(SELECT 1 FROM repo_lookup WHERE input_pk = ?1)
         AND NOT EXISTS(SELECT 1 FROM tool_row WHERE input_pk = ?1)
         AND NOT EXISTS(SELECT 1 FROM tool_database WHERE input_pk = ?1)
         AND NOT EXISTS(SELECT 1 FROM prefix_member WHERE input_pk = ?1 AND chain = ?2
             AND source_input = ?5 AND source_chain = ?2)",
        params![input_pk, from, parse_version, to, input, upto_sql],
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
                        // When this write committed the row, as a parse
                        // of the chain would record its own write.
                        "commit_seq" | "committed" => {
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
        crate::fault::check("write.copy_table", table)?;
        statement.execute(&values[..parameters]).map_err(|error| {
            if error.sqlite_error_code() == Some(rusqlite::ErrorCode::OperationInterrupted) {
                StoreError::Sqlite(error)
            } else {
                StoreError::internal(format!("copy {table}: {error}"))
            }
        })?;
    }
    let mut commits = std::collections::HashMap::new();
    for (table, column, number) in [
        ("wire_reading", "state", "NULL"),
        ("wire_occurrence", "occurrence", "number"),
    ] {
        let rows: Vec<(String, Option<i64>, String)> = tx
            .prepare(&format!(
                "SELECT parse_version, {number}, {column} FROM {table} WHERE input_pk = ?1 AND chain = ?2"
            ))?
            .query_map(params![input_pk, to], |row| <(_, _, _)>::try_from(row))?
            .collect::<rusqlite::Result<_>>()?;
        for (version, number, text) in rows {
            let mut value: serde_json::Value = serde_json::from_str(&text)?;
            if !retake_occurrences(tx, &mut value, input, from, to, &mut commits)? {
                return Ok(false);
            }
            let text = serde_json::to_string(&value)?;
            if let Some(number) = number {
                // The occurrence's complete primary key avoids scanning every
                // occurrence in this chain again for each retaken one.
                tx.prepare_cached("UPDATE wire_occurrence SET occurrence = ?5 WHERE input_pk = ?1 AND chain = ?2 AND parse_version = ?3 AND number = ?4")?
                    .execute(params![input_pk, to, version, number, text])?;
            } else {
                tx.prepare_cached("UPDATE wire_reading SET state = ?4 WHERE input_pk = ?1 AND chain = ?2 AND parse_version = ?3")?
                    .execute(params![input_pk, to, version, text])?;
            }
        }
    }
    // `from` may have more bytes after the copy point than `to`, or bytes
    // of its own: the standing is `to`'s, over `to`'s own length.
    let progress = if *reaches_end {
        polylogue_model::batch::ParseProgress::after(
            polylogue_model::batch::StepProgress::AtEnd,
            upto,
            to_length,
        )
    } else {
        polylogue_model::batch::ParseProgress::Parsing
    };
    tx.prepare_cached(
        "UPDATE parse_state SET status = ?3, at_length = ?4 WHERE input_pk = ?1 AND chain = ?2",
    )?
    .execute(params![
        input_pk,
        to,
        progress.status(),
        progress
            .at_length()
            .map(i64::try_from)
            .transpose()
            .map_err(|_| {
                StoreError::internal("a stored length exceeds SQLite's integer range")
            })?
    ])?;
    let commit = Source::reading_commit(tx, input, to, upto)?;
    crate::wire_shapes::copied(
        tx,
        crate::wire_shapes::Source {
            input_pk,
            input,
            format: "",
            chain: to,
            length: to_length,
            commit_sequence: commit.sequence,
            commit_id: commit.id.as_deref(),
            observation: commit.observation.as_deref(),
            observed_at_ms: commit.observed_at_ms,
        },
        parse_version,
        *reaches_end,
    )?;
    tx.execute(
        "UPDATE session SET changed = ?3 WHERE id IN (SELECT session_id FROM session_record WHERE input_pk = ?1 AND chain = ?2)",
        params![input_pk, to, counter],
    )?;
    Ok(true)
}

/// A wire-shape occurrence names where a shape was seen by source commit:
/// the chain, the commit order and the observation that confirmed the bytes
/// through the covered range's end. An occurrence in chain `from` is taken
/// again for chain `to` through the same end. `false` for an occurrence that
/// covers anything but stored bytes, whose copy would not be identical.
fn retake_occurrences(
    tx: &Connection,
    value: &mut serde_json::Value,
    input: &str,
    from: i64,
    to: i64,
    commits: &mut std::collections::HashMap<u64, crate::wire_shapes::ReadingCommit>,
) -> Result<bool> {
    match value {
        serde_json::Value::Object(object) => {
            let occurrence = object.get("input").and_then(serde_json::Value::as_str) == Some(input)
                && object.get("chain").and_then(serde_json::Value::as_i64) == Some(from)
                && object.contains_key("sequence")
                && object.contains_key("covered");
            if occurrence {
                let covered = &object["covered"];
                let (Some("bytes"), Some(end)) = (covered.str("kind"), covered.u64("end")) else {
                    return Ok(false);
                };
                let taken = match commits.get(&end) {
                    Some(taken) => taken.clone(),
                    None => {
                        let taken = Source::reading_commit(tx, input, to, end)?;
                        commits.insert(end, taken.clone());
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
                if !retake_occurrences(tx, child, input, from, to, commits)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        serde_json::Value::Array(items) => {
            for item in items {
                if !retake_occurrences(tx, item, input, from, to, commits)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(true),
    }
}

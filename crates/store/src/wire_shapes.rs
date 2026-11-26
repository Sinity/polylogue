//! Wire shape observations from the parser's existing full-depth walk.
//!
//! A reading's coverage and counts commit with its parsed rows. The saved
//! cursor accepts each byte range once; a new parse version gets separate
//! rows. Occurrences state order, identity and optional time.
use crate::{OptionExt as _, ResultExt as _};
use crate::{ParseStep, Result, StoreError};
use polylogue_model::WireShape;
use polylogue_model::batch::{ExportCursor, ExportDeliverySource, ReadRange};
mod export;
pub(crate) use export::admit_export;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA: &str = include_str!("schema/wire_shapes.sql");

/// The commit that confirmed the reading's source. Sequence is the archive
/// commit order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReadingCommit {
    pub sequence: i64,
    pub id: Option<String>,
    pub observation: Option<String>,
    pub observed_at_ms: Option<i64>,
}

/// The committed source described by this parse step. Time may be unknown;
/// occurrence order decides even when observation clocks disagree.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    pub input_pk: i64,
    pub input: &'a str,
    pub format: &'a str,
    pub chain: i64,
    pub length: u64,
    pub commit_sequence: i64,
    pub commit_id: Option<&'a str>,
    pub observation: Option<&'a str>,
    pub observed_at_ms: Option<i64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Occurrence {
    pub write_counter: i64,
    pub id: Option<String>,
    pub input: String,
    pub chain: i64,
    pub covered: ReadRange,
    pub observation: Option<String>,
    pub observed_at: Option<i64>,
}
impl From<Source<'_>> for Occurrence {
    fn from(source: Source<'_>) -> Self {
        Self {
            write_counter: source.commit_sequence,
            id: source.commit_id.map(str::to_owned),
            input: source.input.to_owned(),
            chain: source.chain,
            covered: ReadRange::Bytes {
                start: 0,
                end: source.length,
            },
            observation: source.observation.map(str::to_owned),
            observed_at: source.observed_at_ms,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TypeObservation {
    pub count: u64,
    pub first: Occurrence,
    pub last: Occurrence,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PathShape {
    pub format: String,
    pub parse_version: String,
    pub record_kind: String,
    pub path: String,
    pub table: Option<String>,
    pub readings: u64,
    pub types: BTreeMap<String, TypeObservation>,
    pub expected: Vec<String>,
    pub dynamic_map: bool,
    pub known_kinds: Vec<String>,
    pub unknown_kind_count: u64,
    pub unknown_table_count: u64,
    pub unused_count: u64,
    pub first: Occurrence,
    pub last: Occurrence,
    pub complete_presence: Option<Occurrence>,
    pub present_in_latest: bool,
}
impl PathShape {
    /// The record kind this path belongs to, within its format, version and table.
    fn record_key(&self) -> (String, String, String, Option<String>) {
        (
            self.format.clone(),
            self.parse_version.clone(),
            self.record_kind.clone(),
            self.table.clone(),
        )
    }
}

polylogue_model::stored_names! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum CoverageStatus {
        Complete => "complete",
        Incomplete => "incomplete",
        Partial => "partial",
        Unreadable => "unreadable",
        Refused => "refused",
        Failed => "failed",
        Unknown => "unknown",
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Coverage {
    pub format: String,
    pub parse_version: String,
    pub readings: u64,
    pub inputs: u64,
    pub current_readings: u64,
    pub states: BTreeMap<CoverageStatus, u64>,
    pub parsed_bytes: u64,
    pub known_source_bytes: u64,
    pub unknown_lengths: u64,
    pub first: Option<Occurrence>,
    pub last: Option<Occurrence>,
}
impl Coverage {
    pub fn status(&self) -> CoverageStatus {
        if self.states.len() == 1 {
            self.states
                .keys()
                .next()
                .copied()
                .unwrap_or(CoverageStatus::Unknown)
        } else {
            CoverageStatus::Incomplete
        }
    }
}
/// Reads the active readings of each chain by default; history is an explicit request.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scope<'a> {
    pub format: Option<&'a str>,
    pub parse_version: Option<&'a str>,
    pub history: bool,
}
impl<'a> Scope<'a> {
    pub fn current(format: Option<&'a str>) -> Self {
        Self {
            format,
            parse_version: None,
            history: false,
        }
    }
    pub fn history(format: Option<&'a str>, parse_version: Option<&'a str>) -> Self {
        Self {
            format,
            parse_version,
            history: true,
        }
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    NewPath,
    TypeChanged,
    StoppedField,
    UnknownKind,
    UnknownTable,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DriftReason {
    /// Present bytes directly establish a new path or a declaration mismatch.
    Seen,
    /// An absence in comparable complete records does not establish that an
    /// optional field was removed from the provider's format.
    MaybeOptional,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Drift {
    pub kind: Kind,
    pub shape: PathShape,
    pub coverage: CoverageStatus,
    pub reason: DriftReason,
    pub compared_with: Vec<Coverage>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Report {
    pub shapes: Vec<PathShape>,
    pub drift: Vec<Drift>,
    pub coverage: Vec<Coverage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct Reading {
    input: String,
    parsed_to: u64,
    length: u64,
    /// The last step read through the end of the bytes it was given.
    reached_end: bool,
    /// The input was refused by its origin's shape check.
    #[serde(default)]
    refused: bool,
    unreadable: u64,
    prefix_missing: bool,
    first: Occurrence,
    last: Occurrence,
    last_start: u64,
    last_end: u64,
    last_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    export_cursor: Option<ExportCursor>,
    #[serde(default)]
    refused_export: bool,
    /// The latest occurrence at which this reading, since its last reset,
    /// covered its whole source: a path counted now and first seen at or
    /// before it was present in a complete reading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completed: Option<Occurrence>,
    /// The latest completion before the last reset: a path the reset left
    /// uncounted and last seen at or before it was present in a complete reading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completed_before_reset: Option<Occurrence>,
}
/// The number of an occurrence among its reading's occurrences, numbered from 1 in
/// the order they were first cited.
fn occurrence_number(
    tx: &Connection,
    source: Source<'_>,
    version: &str,
    occurrence: &Occurrence,
) -> Result<i64> {
    let json = serde_json::to_string(occurrence)?;
    let key = params![source.input_pk, source.chain, version, json];
    if let Some(number) = tx
        .prepare_cached("SELECT position FROM wire_occurrence WHERE input_pk=?1 AND chain=?2 AND parser_version=?3 AND occurrence=?4")?
        .query_row(key, |row| row.get(0))
        .optional()?
    {
        return Ok(number);
    }
    Ok(tx
        .prepare_cached("INSERT INTO wire_occurrence(input_pk,chain,parser_version,position,occurrence) SELECT ?1,?2,?3,coalesce(max(position),0)+1,?4 FROM wire_occurrence WHERE input_pk=?1 AND chain=?2 AND parser_version=?3 RETURNING position")?
        .query_row(key, |row| row.get(0))?)
}

impl Reading {
    /// Records that the reading now covers its whole source.
    fn complete(&mut self, source: Source<'_>, occurrence: &Occurrence) {
        self.completed = Some(if matches!(occurrence.covered, ReadRange::Bytes { .. }) {
            Occurrence::from(source)
        } else {
            occurrence.clone()
        });
    }

    fn status(&self) -> CoverageStatus {
        if self.unreadable > 0 {
            CoverageStatus::Unreadable
        } else if self.refused {
            CoverageStatus::Refused
        } else if self.reached_end && !self.prefix_missing && self.parsed_to >= self.length {
            CoverageStatus::Complete
        } else {
            CoverageStatus::Partial
        }
    }
}

/// The stored state of one chain's reading under a parse version.
fn stored_reading(tx: &Connection, source: Source<'_>, version: &str) -> Result<Option<Reading>> {
    let state: Option<String> = tx
        .prepare_cached(
            "SELECT state FROM wire_parse WHERE input_pk=?1 AND chain=?2 AND parser_version=?3",
        )?
        .query_row(params![source.input_pk, source.chain, version], |row| {
            row.get(0)
        })
        .optional()?;
    Ok(state
        .map(|json| serde_json::from_str::<Reading>(&json))
        .transpose()?)
}

/// Apply before or after the parse checkpoint, in the same transaction.
/// A complete reset replaces current counts, keeping first/last occurrences and
/// the rows of older parse versions. A step that ends inside the committed
/// frontier is ignored as already counted (only the last range's hash is kept,
/// so its bytes are not compared); a step that starts inside the frontier and
/// extends past it, or repeats the last range with different bytes, fails.
pub fn apply(tx: &Transaction<'_>, source: Source<'_>, step: &ParseStep) -> Result<()> {
    apply_inner(tx, source, step, false)
}

/// A whole-chain replacement starts a new aggregate in its existing
/// transaction. Parser reset flags alone do not grant this operation.
pub(crate) fn replace(tx: &Transaction<'_>, source: Source<'_>, step: &ParseStep) -> Result<()> {
    if !step.reset || step.delivery.is_some() {
        return Err(StoreError::internal(
            "wire replacement requires its first reset step",
        ));
    }
    apply_inner(tx, source, step, true)
}

fn apply_inner(
    tx: &Transaction<'_>,
    source: Source<'_>,
    step: &ParseStep,
    replacement: bool,
) -> Result<()> {
    // A failed parse observed no records; its standing is the parse state's.
    if step
        .failure
        .as_ref()
        .is_some_and(|failure| failure.kind != polylogue_model::ChainFailureKind::Refused)
    {
        return Ok(());
    }
    if step.parsed_upto < step.start || step.parsed_upto > source.length {
        return Err(StoreError::internal(
            "wire step lies outside its committed source",
        ));
    }
    let mut occurrence = Occurrence::from(source);
    occurrence.covered = match step.delivery {
        Some(delivery) => ReadRange::Export {
            source: delivery.source,
        },
        None => ReadRange::Bytes {
            start: step.start,
            end: step.parsed_upto,
        },
    };
    let export_delivery = step.delivery.is_some();
    if !admit_export(tx, source.input, source.chain, source.length, step)? {
        return Ok(());
    }
    let reset = if export_delivery {
        step.reset
    } else {
        step.reset || step.start == 0
    };
    let computed_hash = blake3::hash(&serde_json::to_vec(&(
        step.start,
        step.parsed_upto,
        reset,
        &step.counts.wire_shapes,
        step.counts.records,
        step.counts
            .unreadable
            .values()
            .copied()
            .try_fold(0u64, add)?,
    ))?)
    .to_hex()
    .to_string();
    let hash = step
        .delivery
        .map_or(computed_hash, |delivery| delivery.body_hash.to_string());
    let previous = stored_reading(tx, source, &step.parse_version)?;
    // The observations are of the reading's rows and follow them. A step
    // from the chain's first byte that ends before the observations do
    // starts the reading again: the row writer drops the chain's rows and
    // takes its parse position back to this step's end, so the observations
    // start again here too. Taken as seen already, they would stay ahead of
    // the rows they describe.
    let restarts = step.reset
        && previous
            .as_ref()
            .is_some_and(|old| step.parsed_upto < old.parsed_to);
    let duplicate = !replacement
        && !export_delivery
        && !restarts
        && previous.as_ref().is_some_and(|old| {
            step.parsed_upto <= old.parsed_to
                && (step.parsed_upto < old.parsed_to
                    || step.start != old.last_start
                    || hash == old.last_hash)
        });
    if let Some(old) = &previous
        && !export_delivery
        && !replacement
        && step.parsed_upto == old.last_end
        && step.start == old.last_start
        && hash != old.last_hash
    {
        return Err(StoreError::internal(
            "the same wire source range has different observations under one parse version",
        ));
    }
    if duplicate {
        if let Some(mut old) = previous {
            let prefix_missing = !step.reset && old.prefix_missing;
            let coverage_changed = prefix_missing != old.prefix_missing;
            old.prefix_missing = prefix_missing;
            let reaches_end = step.at_end && step.parsed_upto >= source.length;
            if coverage_changed
                || (step.parsed_upto == old.parsed_to && reaches_end && !old.reached_end)
            {
                if reaches_end && !old.refused {
                    old.reached_end = true;
                }
                old.length = source.length;
                old.last = occurrence.clone();
                if old.status() == CoverageStatus::Complete {
                    old.complete(source, &occurrence);
                }
                tx.prepare_cached("UPDATE wire_parse SET state=?4 WHERE input_pk=?1 AND chain=?2 AND parser_version=?3")?.execute(params![source.input_pk,source.chain,step.parse_version,serde_json::to_string(&old)?])?;
            }
        }
        return Ok(());
    }
    let export_refusal = export_delivery && export::is_refusal(step);
    let mut reading = Reading {
        input: source.input.to_owned(),
        parsed_to: step.parsed_upto,
        length: source.length,
        reached_end: step.at_end && step.failure.is_none(),
        refused: step.failure.is_some(),
        prefix_missing: if reset {
            false
        } else {
            previous.as_ref().is_none_or(|old| {
                old.prefix_missing || (!export_delivery && step.start != old.parsed_to)
            })
        },
        unreadable: step
            .counts
            .unreadable
            .values()
            .copied()
            .try_fold(0u64, add)?,
        first: if export_delivery && step.reset {
            occurrence.clone()
        } else {
            previous
                .as_ref()
                .map_or_else(|| occurrence.clone(), |old| old.first.clone())
        },
        last: occurrence.clone(),
        last_start: step.start,
        last_end: step.parsed_upto,
        last_hash: hash,
        refused_export: export_refusal,
        completed: previous
            .as_ref()
            .filter(|_| !reset)
            .and_then(|old| old.completed.clone()),
        completed_before_reset: previous.as_ref().and_then(|old| {
            if reset {
                old.completed
                    .clone()
                    .or_else(|| old.completed_before_reset.clone())
            } else {
                old.completed_before_reset.clone()
            }
        }),
        export_cursor: if export_refusal {
            None
        } else {
            step.delivery
                .as_ref()
                .map(|delivery| delivery.cursor())
                .transpose()
                .map_err(crate::StoreError::internal)?
        },
    };
    if let Some(old) = &previous
        && !reset
    {
        reading.unreadable = add(old.unreadable, reading.unreadable)?;
    }
    // A missing prefix is incomplete even if this suffix reached EOF.
    if previous.is_none() && step.start != 0 && !step.reset && !export_delivery {
        reading.reached_end = false;
    }
    if reading.status() == CoverageStatus::Complete {
        reading.complete(source, &occurrence);
    }
    tx.prepare_cached(
        "UPDATE wire_parse SET active=0 WHERE input_pk=?1 AND chain=?2 AND active=1",
    )?
    .execute(params![source.input_pk, source.chain])?;
    tx.prepare_cached("INSERT INTO wire_parse(input_pk,chain,parser_version,origin,active,state) VALUES (?1,?2,?3,?4,1,?5) ON CONFLICT(input_pk,chain,parser_version) DO UPDATE SET origin=excluded.origin,active=1,state=excluded.state")?.execute(params![source.input_pk,source.chain,step.parse_version,source.format,serde_json::to_string(&reading)?])?;
    let reading_key = params![source.input_pk, source.chain, step.parse_version];
    if reset {
        // A reset replaces the counts; where each path was first and last
        // seen stays.
        tx.prepare_cached("UPDATE wire_shape SET unknown_kinds=0,unknown_tables=0,unused=0 WHERE input_pk=?1 AND chain=?2 AND parser_version=?3")?
            .execute(reading_key)?;
        tx.prepare_cached(
            "UPDATE wire_type SET count=0 WHERE input_pk=?1 AND chain=?2 AND parser_version=?3",
        )?
        .execute(reading_key)?;
    }
    if step.counts.wire_shapes.is_empty() {
        return Ok(());
    }
    let number = occurrence_number(tx, source, &step.parse_version, &occurrence)?;
    for observation in step.counts.wire_shapes.values() {
        validate(observation)?;
        let table_key = serde_json::to_string(&observation.table)?;
        declare(
            tx,
            source.format,
            &step.parse_version,
            observation,
            &table_key,
        )?;
        tx.prepare_cached("INSERT INTO wire_shape(input_pk,chain,parser_version,record_kind,path,table_key,unknown_kinds,unknown_tables,unused,first_position,last_position) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?10) ON CONFLICT(input_pk,chain,parser_version,record_kind,path,table_key) DO UPDATE SET unknown_kinds=unknown_kinds+excluded.unknown_kinds,unknown_tables=unknown_tables+excluded.unknown_tables,unused=unused+excluded.unused,last_position=excluded.last_position")?
            .execute(params![
                source.input_pk,
                source.chain,
                step.parse_version,
                observation.record_kind,
                observation.path,
                table_key,
                stored_count(observation.unknown_kind_count)?,
                stored_count(observation.unknown_table_count)?,
                stored_count(observation.unused_count)?,
                number,
            ])?;
        for (json_type, &count) in &observation.types {
            if count == 0 {
                continue;
            }
            tx.prepare_cached("INSERT INTO wire_type(input_pk,chain,parser_version,record_kind,path,table_key,json_type,count,first_position,last_position) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?9) ON CONFLICT(input_pk,chain,parser_version,record_kind,path,table_key,json_type) DO UPDATE SET count=count+excluded.count,last_position=excluded.last_position")?
                .execute(params![
                    source.input_pk,
                    source.chain,
                    step.parse_version,
                    observation.record_kind,
                    observation.path,
                    table_key,
                    json_type,
                    stored_count(count)?,
                    number,
                ])?;
        }
    }
    Ok(())
}

/// Records what a parse version declares about a path, once per format;
/// the same version declaring otherwise is refused.
fn declare(
    tx: &Connection,
    format: &str,
    version: &str,
    observation: &WireShape,
    table_key: &str,
) -> Result<()> {
    let expected = serde_json::to_string(&observation.expected)?;
    let known_kinds = serde_json::to_string(&observation.known_kinds)?;
    let stored: Option<(String, bool, String)> = tx
        .prepare_cached("SELECT expected,dynamic_map,known_kinds FROM wire_declaration WHERE origin=?1 AND parser_version=?2 AND record_kind=?3 AND path=?4 AND table_key=?5")?
        .query_row(
            params![format, version, observation.record_kind, observation.path, table_key],
            |row| <(_, _, _)>::try_from(row),
        )
        .optional()?;
    match stored {
        Some((stored_expected, stored_dynamic_map, stored_known_kinds))
            if stored_expected == expected
                && stored_dynamic_map == observation.dynamic_map
                && stored_known_kinds == known_kinds =>
        {
            Ok(())
        }
        Some(_) => Err(StoreError::internal(format!(
            "wire declarations changed without a parse version change: {format} {version} \
             declares {} {} {table_key} otherwise",
            observation.record_kind, observation.path
        ))),
        None => {
            tx.prepare_cached("INSERT INTO wire_declaration(origin,parser_version,record_kind,path,table_key,expected,dynamic_map,known_kinds) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)")?
                .execute(params![
                    format,
                    version,
                    observation.record_kind,
                    observation.path,
                    table_key,
                    expected,
                    observation.dynamic_map,
                    known_kinds,
                ])?;
            Ok(())
        }
    }
}

/// A count as SQLite stores it; sums that leave the range are refused there.
fn stored_count(count: u64) -> Result<i64> {
    crate::to_i64(count, "wire count exceeds its representation")
}
fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .or_internal("wire count exceeds its representation")
}

fn validate(shape: &WireShape) -> Result<()> {
    const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];
    if shape
        .types
        .keys()
        .chain(shape.expected.iter())
        .any(|kind| !TYPES.contains(&kind.as_str()))
    {
        return Err(StoreError::stored_data(
            "wire shape has an unknown JSON type",
        ));
    }
    if !shape.path.is_empty() && !shape.path.starts_with('/') {
        return Err(StoreError::stored_data("wire path is not canonical"));
    }
    for part in shape.path.split('/').skip(1) {
        let mut chars = part.chars();
        while let Some(ch) = chars.next() {
            if ch == '~' && !matches!(chars.next(), Some('0' | '1' | '2' | '3')) {
                return Err(StoreError::stored_data("wire path has an invalid escape"));
            }
        }
    }
    Ok(())
}
mod read;
pub use read::read;

#[cfg(test)]
mod tests;

impl crate::Reader<'_> {
    pub fn wire_shapes(&self, scope: Scope<'_>) -> Result<Report> {
        read(self.conn()?, scope)
    }
}

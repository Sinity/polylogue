//! Wire shape observations from the parser's existing full-depth walk.
//!
//! A reading's coverage and counts commit with its parsed rows. The saved
//! cursor admits each byte range once; a new reading version gets a separate
//! projection. Receipts supply order, identity and optional time explicitly.
use crate::{ParseStep, Result, StoreError};
use polylogue_model::WireShape;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA: &str = include_str!("schema/wire_shapes.sql");

/// Product-native confirmation of the reading's source. Sequence is the
/// standalone archive commit order or the primary confirmed fact order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadingReceipt {
    pub sequence: i64,
    pub id: Option<String>,
    pub observation: Option<String>,
    pub observed_at_ms: Option<i64>,
}

/// The committed source described by this parse step. Time may be unknown;
/// receipt order is the authority even when observation clocks disagree.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    pub input_pk: i64,
    pub input: &'a str,
    pub format: &'a str,
    pub chain: i64,
    pub length: u64,
    pub receipt: i64,
    pub receipt_id: Option<&'a str>,
    pub observation: Option<&'a str>,
    pub observed_at_ms: Option<i64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub sequence: i64,
    pub id: Option<String>,
    pub input: String,
    pub chain: i64,
    pub start: u64,
    pub end: u64,
    pub observation: Option<String>,
    pub observed_at_ms: Option<i64>,
}
impl From<Source<'_>> for Receipt {
    fn from(source: Source<'_>) -> Self {
        Self {
            sequence: source.receipt,
            id: source.receipt_id.map(str::to_owned),
            input: source.input.to_owned(),
            chain: source.chain,
            start: 0,
            end: source.length,
            observation: source.observation.map(str::to_owned),
            observed_at_ms: source.observed_at_ms,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeObservation {
    pub count: u64,
    pub first: Receipt,
    pub last: Receipt,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub format: String,
    pub reading_version: String,
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
    pub first: Receipt,
    pub last: Receipt,
    pub complete_presence: Option<Receipt>,
    pub present_in_latest: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageStatus {
    Complete,
    Incomplete,
    Partial,
    Unreadable,
    Refused,
    NoParserYet,
    Failed,
    Unknown,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coverage {
    pub format: String,
    pub reading_version: String,
    pub readings: u64,
    pub inputs: u64,
    pub current_readings: u64,
    pub states: BTreeMap<CoverageStatus, u64>,
    pub parsed_bytes: u64,
    pub known_source_bytes: u64,
    pub unknown_lengths: u64,
    pub first: Option<Receipt>,
    pub last: Option<Receipt>,
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
/// Default reads active chain projections; history is an explicit request.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scope<'a> {
    pub format: Option<&'a str>,
    pub reading_version: Option<&'a str>,
    pub history: bool,
}
impl<'a> Scope<'a> {
    pub fn current(format: Option<&'a str>) -> Self {
        Self {
            format,
            reading_version: None,
            history: false,
        }
    }
    pub fn history(format: Option<&'a str>, reading_version: Option<&'a str>) -> Self {
        Self {
            format,
            reading_version,
            history: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    NewPath,
    TypeChanged,
    StoppedField,
    UnknownKind,
    UnknownTable,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Certainty {
    /// Present bytes directly establish a new path or a declaration mismatch.
    Observed,
    /// An absence in comparable complete records does not establish that an
    /// optional field was removed from the provider's format.
    OptionalPresence,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Drift {
    pub kind: Kind,
    pub shape: Observation,
    pub coverage: CoverageStatus,
    pub certainty: Certainty,
    pub compared_with: Vec<Coverage>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub shapes: Vec<Observation>,
    pub drift: Vec<Drift>,
    pub coverage: Vec<Coverage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Reading {
    input: String,
    parsed_upto: u64,
    length: u64,
    at_end: bool,
    unreadable: u64,
    no_parser_yet: bool,
    prefix_missing: bool,
    first: Receipt,
    last: Receipt,
    last_start: u64,
    last_end: u64,
    last_hash: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Shape {
    types: BTreeMap<String, TypeObservation>,
    expected: Vec<String>,
    dynamic_map: bool,
    known_kinds: Vec<String>,
    unknown_kind_count: u64,
    table: Option<String>,
    unknown_table_count: u64,
    unused_count: u64,
    first: Receipt,
    last: Receipt,
    complete_presence: Option<Receipt>,
}
impl Reading {
    fn status(&self) -> CoverageStatus {
        if self.no_parser_yet {
            CoverageStatus::NoParserYet
        } else if self.unreadable > 0 {
            CoverageStatus::Unreadable
        } else if !self.prefix_missing && self.at_end && self.parsed_upto >= self.length {
            CoverageStatus::Complete
        } else {
            CoverageStatus::Partial
        }
    }
}

/// Apply before or after the ordinary parse checkpoint, in the same transaction.
/// A complete reset replaces current counts, retaining first/last evidence and
/// the rows of older reading versions. Overlapping non-identical ranges fail;
/// callers must resume from their committed checkpoint rather than double-count.
pub fn apply(tx: &Transaction<'_>, source: Source<'_>, step: &ParseStep) -> Result<()> {
    if step.parsed_upto < step.start || step.parsed_upto > source.length {
        return invalid("wire step lies outside its committed source");
    }
    let mut receipt = Receipt::from(source);
    receipt.start = step.start;
    receipt.end = step.parsed_upto;
    let reset = step.reset || step.start == 0;
    let hash = blake3::hash(&serde_json::to_vec(&(
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
        step.no_parser_yet,
    ))?)
    .to_hex()
    .to_string();
    let previous: Option<String> = tx
        .prepare_cached(
            "SELECT state FROM wire_reading WHERE input_pk=?1 AND chain=?2 AND reading_version=?3",
        )?
        .query_row(
            params![source.input_pk, source.chain, step.reading_version],
            |row| row.get(0),
        )
        .optional()?;
    let previous = previous
        .map(|json| serde_json::from_str::<Reading>(&json))
        .transpose()?;
    let duplicate = previous.as_ref().is_some_and(|old| {
        step.parsed_upto <= old.parsed_upto
            && (step.parsed_upto < old.parsed_upto
                || step.start != old.last_start
                || hash == old.last_hash)
    });
    if let Some(old) = &previous {
        if step.parsed_upto == old.last_end && step.start == old.last_start && hash != old.last_hash
        {
            return invalid(
                "the same wire source range has different observations under one reading version",
            );
        }
        if !duplicate && !step.reset && step.start < old.parsed_upto {
            return invalid("wire step overlaps a committed source range");
        }
    }
    if duplicate {
        if let Some(mut old) = previous {
            let prefix_missing = if step.reset {
                step.source_prefix_missing
            } else {
                old.prefix_missing || step.source_prefix_missing
            };
            let coverage_changed = prefix_missing != old.prefix_missing;
            old.prefix_missing = prefix_missing;
            if coverage_changed
                || (step.parsed_upto == old.parsed_upto
                    && step.at_end
                    && !old.at_end
                    && step.parsed_upto >= source.length)
            {
                old.at_end |= step.at_end && step.parsed_upto >= source.length;
                old.length = source.length;
                old.last = receipt.clone();
                tx.prepare_cached("UPDATE wire_reading SET state=?4 WHERE input_pk=?1 AND chain=?2 AND reading_version=?3")?.execute(params![source.input_pk,source.chain,step.reading_version,serde_json::to_string(&old)?])?;
                if old.status() == CoverageStatus::Complete {
                    complete(tx, source, &step.reading_version)?;
                }
            }
        }
        return Ok(());
    }
    let mut reading = Reading {
        input: source.input.to_owned(),
        parsed_upto: step.parsed_upto,
        length: source.length,
        at_end: step.at_end,
        prefix_missing: step.source_prefix_missing
            || if reset {
                false
            } else {
                previous
                    .as_ref()
                    .is_none_or(|old| old.prefix_missing || step.start != old.parsed_upto)
            },
        no_parser_yet: step.no_parser_yet,
        unreadable: step
            .counts
            .unreadable
            .values()
            .copied()
            .try_fold(0u64, add)?,
        first: previous
            .as_ref()
            .map_or_else(|| receipt.clone(), |old| old.first.clone()),
        last: receipt.clone(),
        last_start: step.start,
        last_end: step.parsed_upto,
        last_hash: hash,
    };
    if let Some(old) = &previous {
        if !reset {
            reading.unreadable = add(old.unreadable, reading.unreadable)?;
        }
    }
    // A missing prefix is incomplete even if this suffix reached EOF.
    if previous.is_none() && step.start != 0 && !step.reset {
        reading.at_end = false;
    }
    tx.prepare_cached(
        "UPDATE wire_reading SET active=0 WHERE input_pk=?1 AND chain=?2 AND active=1",
    )?
    .execute(params![source.input_pk, source.chain])?;
    tx.prepare_cached("INSERT INTO wire_reading(input_pk,chain,reading_version,format,active,state) VALUES (?1,?2,?3,?4,1,?5) ON CONFLICT(input_pk,chain,reading_version) DO UPDATE SET format=excluded.format,active=1,state=excluded.state")?.execute(params![source.input_pk,source.chain,step.reading_version,source.format,serde_json::to_string(&reading)?])?;
    if reset {
        let old = stored_shapes(tx, source.input_pk, source.chain, &step.reading_version)?;
        for (kind, path, mut shape) in old {
            for observation in shape.types.values_mut() {
                observation.count = 0;
            }
            shape.unknown_kind_count = 0;
            shape.unknown_table_count = 0;
            shape.unused_count = 0;
            save(tx, source, &step.reading_version, &kind, &path, &shape)?;
        }
    }
    for observation in step.counts.wire_shapes.values() {
        validate(observation)?;
        let previous:Option<String>=tx.prepare_cached("SELECT shape FROM wire_shape WHERE input_pk=?1 AND chain=?2 AND reading_version=?3 AND record_kind=?4 AND path=?5 AND table_key=?6")?.query_row(params![source.input_pk,source.chain,step.reading_version,observation.record_kind,observation.path,serde_json::to_string(&observation.table)?],|row|row.get(0)).optional()?;
        let mut shape = previous
            .map(|json| serde_json::from_str::<Shape>(&json))
            .transpose()?
            .unwrap_or_else(|| Shape {
                types: BTreeMap::new(),
                expected: observation.expected.clone(),
                dynamic_map: observation.dynamic_map,
                known_kinds: observation.known_kinds.clone(),
                unknown_kind_count: 0,
                table: observation.table.clone(),
                unknown_table_count: 0,
                unused_count: 0,
                first: receipt.clone(),
                last: receipt.clone(),
                complete_presence: None,
            });
        if shape.expected != observation.expected
            || shape.dynamic_map != observation.dynamic_map
            || shape.known_kinds != observation.known_kinds
            || shape.table != observation.table
        {
            return invalid("wire declarations changed without a reading version change");
        }
        shape.last = receipt.clone();
        shape.unknown_kind_count = add(shape.unknown_kind_count, observation.unknown_kind_count)?;
        shape.unknown_table_count =
            add(shape.unknown_table_count, observation.unknown_table_count)?;
        shape.unused_count = add(shape.unused_count, observation.unused_count)?;
        for (kind, count) in &observation.types {
            if *count == 0 {
                continue;
            }
            let entry = shape
                .types
                .entry(kind.clone())
                .or_insert_with(|| TypeObservation {
                    count: 0,
                    first: receipt.clone(),
                    last: receipt.clone(),
                });
            entry.count = add(entry.count, *count)?;
            entry.last = receipt.clone();
        }
        if reading.status() == CoverageStatus::Complete
            && shape.types.values().any(|kind| kind.count > 0)
        {
            shape.complete_presence = Some(receipt.clone());
        }
        save(
            tx,
            source,
            &step.reading_version,
            &observation.record_kind,
            &observation.path,
            &shape,
        )?;
    }
    if reading.status() == CoverageStatus::Complete {
        complete(tx, source, &step.reading_version)?;
    }
    Ok(())
}
fn complete(tx: &Transaction<'_>, source: Source<'_>, version: &str) -> Result<()> {
    for (kind, path, mut shape) in stored_shapes(tx, source.input_pk, source.chain, version)? {
        if shape.types.values().any(|kind| kind.count > 0)
            && shape.complete_presence.as_ref() != Some(&Receipt::from(source))
        {
            shape.complete_presence = Some(Receipt::from(source));
            save(tx, source, version, &kind, &path, &shape)?;
        }
    }
    Ok(())
}
fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| StoreError::Invalid("wire count exceeds its representation".into()))
}
fn invalid<T>(what: &str) -> Result<T> {
    Err(StoreError::Invalid(what.into()))
}
fn validate(shape: &WireShape) -> Result<()> {
    const TYPES: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];
    if shape
        .types
        .keys()
        .chain(shape.expected.iter())
        .any(|kind| !TYPES.contains(&kind.as_str()))
    {
        return invalid("wire shape has an unknown JSON type");
    }
    if !shape.path.is_empty() && !shape.path.starts_with('/') {
        return invalid("wire path is not canonical");
    }
    for part in shape.path.split('/').skip(1) {
        let mut chars = part.chars();
        while let Some(ch) = chars.next() {
            if ch == '~' && !matches!(chars.next(), Some('0' | '1' | '2' | '3')) {
                return invalid("wire path has an invalid escape");
            }
        }
    }
    Ok(())
}
fn stored_shapes(
    conn: &Connection,
    input: i64,
    chain: i64,
    version: &str,
) -> Result<Vec<(String, String, Shape)>> {
    let mut statement=conn.prepare_cached("SELECT record_kind,path,shape FROM wire_shape WHERE input_pk=?1 AND chain=?2 AND reading_version=?3 ORDER BY record_kind,path,table_key")?;
    let mut rows = statement.query(params![input, chain, version])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let json: String = row.get(2)?;
        out.push((row.get(0)?, row.get(1)?, serde_json::from_str(&json)?));
    }
    Ok(out)
}
fn save(
    tx: &Transaction<'_>,
    source: Source<'_>,
    version: &str,
    kind: &str,
    path: &str,
    shape: &Shape,
) -> Result<()> {
    tx.prepare_cached("INSERT INTO wire_shape(input_pk,chain,reading_version,record_kind,path,shape,table_key) VALUES (?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(input_pk,chain,reading_version,record_kind,path,table_key) DO UPDATE SET shape=excluded.shape")?.execute(params![source.input_pk,source.chain,version,kind,path,serde_json::to_string(shape)?,serde_json::to_string(&shape.table)?])?;
    Ok(())
}

mod read;
pub use read::read;

#[cfg(test)]
mod tests;

impl crate::Reader<'_> {
    pub fn wire_shapes(&self, scope: Scope<'_>) -> Result<Report> {
        {
            let conn = self.conn()?;
            read(&conn, scope)
        }
    }
}

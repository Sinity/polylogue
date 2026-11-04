//! Wire shape observations from the parser's existing full-depth walk.
//!
//! A reading's coverage and counts commit with its parsed rows. The saved
//! cursor admits each byte range once; a new reading version gets a separate
//! projection. Receipts supply order, identity and optional time explicitly.
use crate::{ParseStep, Result, StoreError};
use polylogue_model::WireShape;
use polylogue_model::batch::{ExportCursor, ExportDeliverySource, ReadingFootprint, StepProgress};
mod export;
pub(crate) use export::{ExportPrefix, admit_export, advance_prefix, is_refusal, load_prefix};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Retained facts without their reading receipt cannot establish current coverage.
pub(crate) fn unobserved(tx: &Transaction<'_>, input_pk: i64, chain: i64) -> Result<()> {
    tx.execute(
        "UPDATE wire_reading SET active=0 WHERE input_pk=?1 AND chain=?2",
        params![input_pk, chain],
    )?;
    Ok(())
}

pub const SCHEMA: &str = include_str!("schema/wire_shapes.sql");

/// Product-native confirmation of the reading's source. Sequence is the
/// standalone archive commit order or the primary confirmed fact order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Receipt {
    pub sequence: i64,
    pub id: Option<String>,
    pub input: String,
    pub chain: i64,
    pub footprint: ReadingFootprint,
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
            footprint: ReadingFootprint::Bytes {
                start: 0,
                end: source.length,
            },
            observation: source.observation.map(str::to_owned),
            observed_at_ms: source.observed_at_ms,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TypeObservation {
    pub count: u64,
    pub first: Receipt,
    pub last: Receipt,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
pub enum Certainty {
    /// Present bytes directly establish a new path or a declaration mismatch.
    Observed,
    /// An absence in comparable complete records does not establish that an
    /// optional field was removed from the provider's format.
    OptionalPresence,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Drift {
    pub kind: Kind,
    pub shape: Observation,
    pub coverage: CoverageStatus,
    pub certainty: Certainty,
    pub compared_with: Vec<Coverage>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Report {
    pub shapes: Vec<Observation>,
    pub drift: Vec<Drift>,
    pub coverage: Vec<Coverage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct Reading {
    input: String,
    parsed_upto: u64,
    length: u64,
    progress: StepProgress,
    unreadable: u64,
    prefix_missing: bool,
    first: Receipt,
    last: Receipt,
    last_start: u64,
    last_end: u64,
    last_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    export_cursor: Option<ExportCursor>,
    #[serde(default)]
    refused_export: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    export_refusals: Option<ExportRefusals>,
}
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct ExportRefusals {
    count: u64,
    first: Receipt,
    last: Receipt,
}
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
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
        match &self.progress {
            StepProgress::NoParserYet { .. } => CoverageStatus::NoParserYet,
            _ if self.unreadable > 0 => CoverageStatus::Unreadable,
            StepProgress::AtEnd if !self.prefix_missing && self.parsed_upto >= self.length => {
                CoverageStatus::Complete
            }
            StepProgress::AtEnd | StepProgress::Partway => CoverageStatus::Partial,
            StepProgress::Refused { .. } => CoverageStatus::Refused,
            StepProgress::Failed { .. } => CoverageStatus::Failed,
        }
    }
}

/// Apply before or after the ordinary parse checkpoint, in the same transaction.
/// A complete reset replaces current counts, retaining first/last evidence and
/// the rows of older reading versions. Overlapping non-identical ranges fail;
/// callers must resume from their committed checkpoint rather than double-count.
pub fn apply(tx: &Transaction<'_>, source: Source<'_>, step: &ParseStep) -> Result<()> {
    apply_inner(tx, source, step, false)
}

/// The complete chain-replacement owner starts a new aggregate in its existing
/// transaction. Parser reset flags alone do not grant this operation.
pub(crate) fn replace(tx: &Transaction<'_>, source: Source<'_>, step: &ParseStep) -> Result<()> {
    if !step.reset || step.delivery.is_some() {
        return invalid("wire replacement requires its first ordinary reset step");
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
    if matches!(step.progress, StepProgress::Failed { .. }) {
        return Ok(());
    }
    if step.parsed_upto < step.start || step.parsed_upto > source.length {
        return invalid("wire step lies outside its committed source");
    }
    let mut receipt = Receipt::from(source);
    receipt.footprint = match step.delivery {
        Some(delivery) => ReadingFootprint::Export {
            source: delivery.source,
        },
        None => ReadingFootprint::Bytes {
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
    let ordinary_hash = blake3::hash(&serde_json::to_vec(&(
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
        matches!(step.progress, StepProgress::NoParserYet { .. }),
    ))?)
    .to_hex()
    .to_string();
    let hash = step
        .delivery
        .map_or(ordinary_hash, |delivery| delivery.body_hash.to_string());
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
    let duplicate = !replacement
        && !export_delivery
        && previous.as_ref().is_some_and(|old| {
            step.parsed_upto <= old.parsed_upto
                && (step.parsed_upto < old.parsed_upto
                    || step.start != old.last_start
                    || hash == old.last_hash)
        });
    if let Some(old) = &previous
        && !export_delivery
    {
        if !replacement
            && step.parsed_upto == old.last_end
            && step.start == old.last_start
            && hash != old.last_hash
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
            let reaches_end = step.progress.reached_end() && step.parsed_upto >= source.length;
            if coverage_changed
                || (step.parsed_upto == old.parsed_upto
                    && reaches_end
                    && !old.progress.reached_end())
            {
                if reaches_end {
                    old.progress = old.progress.clone().with_end(true);
                }
                old.length = source.length;
                old.last = receipt.clone();
                tx.prepare_cached("UPDATE wire_reading SET state=?4 WHERE input_pk=?1 AND chain=?2 AND reading_version=?3")?.execute(params![source.input_pk,source.chain,step.reading_version,serde_json::to_string(&old)?])?;
                if old.status() == CoverageStatus::Complete {
                    complete(tx, source, &step.reading_version, &receipt)?;
                }
            }
        }
        return Ok(());
    }
    let mut export_refusals = previous
        .as_ref()
        .and_then(|old| old.export_refusals.clone());
    if export::is_refusal(step) {
        if let Some(ref mut history) = export_refusals {
            history.count = add(history.count, 1)?;
            history.last = receipt.clone();
        } else {
            export_refusals = Some(ExportRefusals {
                count: 1,
                first: receipt.clone(),
                last: receipt.clone(),
            });
        }
    }
    let mut reading = Reading {
        input: source.input.to_owned(),
        parsed_upto: step.parsed_upto,
        length: source.length,
        progress: step.progress.clone(),
        prefix_missing: step.source_prefix_missing
            || if reset {
                false
            } else {
                previous.as_ref().is_none_or(|old| {
                    old.prefix_missing || (!export_delivery && step.start != old.parsed_upto)
                })
            },
        unreadable: step
            .counts
            .unreadable
            .values()
            .copied()
            .try_fold(0u64, add)?,
        first: if export_delivery && step.reset {
            receipt.clone()
        } else {
            previous
                .as_ref()
                .map_or_else(|| receipt.clone(), |old| old.first.clone())
        },
        last: receipt.clone(),
        last_start: step.start,
        last_end: step.parsed_upto,
        last_hash: hash,
        refused_export: export::is_refusal(step),
        export_refusals,
        export_cursor: if export::is_refusal(step) {
            None
        } else {
            step.delivery
                .as_ref()
                .map(|delivery| delivery.cursor())
                .transpose()
                .map_err(|error| StoreError::Invalid(error.to_string()))?
        },
    };
    if let Some(old) = &previous
        && !reset
    {
        reading.unreadable = add(old.unreadable, reading.unreadable)?;
    }
    // A missing prefix is incomplete even if this suffix reached EOF.
    if previous.is_none() && step.start != 0 && !step.reset && !export_delivery {
        reading.progress = reading.progress.clone().with_end(false);
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
        complete(tx, source, &step.reading_version, &receipt)?;
    }
    Ok(())
}
fn complete(
    tx: &Transaction<'_>,
    source: Source<'_>,
    version: &str,
    receipt: &Receipt,
) -> Result<()> {
    let whole;
    let receipt = if matches!(receipt.footprint, ReadingFootprint::Bytes { .. }) {
        whole = Receipt::from(source);
        &whole
    } else {
        receipt
    };
    for (kind, path, mut shape) in stored_shapes(tx, source.input_pk, source.chain, version)? {
        if shape.types.values().any(|kind| kind.count > 0)
            && shape.complete_presence.as_ref() != Some(receipt)
        {
            shape.complete_presence = Some(receipt.clone());
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
        read(self.conn()?, scope)
    }
}

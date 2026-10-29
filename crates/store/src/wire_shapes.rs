//! Wire shape observations from the parser's existing full-depth walk.
//!
//! A reading's coverage and counts commit with its parsed rows. The saved
//! cursor admits each byte range once; a new reading version gets a separate
//! projection. Receipts supply order, identity and optional time explicitly.
use crate::{ParseStep, Result, StoreError};
use polylogue_model::WireShape;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = include_str!("schema/wire_shapes.sql");

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
    pub observation: Option<&'a str>,
    pub observed_at_ms: Option<i64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub sequence: i64,
    pub observation: Option<String>,
    pub observed_at_ms: Option<i64>,
}
impl From<Source<'_>> for Receipt {
    fn from(source: Source<'_>) -> Self {
        Self {
            sequence: source.receipt,
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
    pub input: String,
    pub format: String,
    pub chain: i64,
    pub reading_version: String,
    pub active: bool,
    pub record_kind: String,
    pub path: String,
    pub types: BTreeMap<String, TypeObservation>,
    pub expected: Vec<String>,
    pub dynamic_map: bool,
    pub known_kinds: Vec<String>,
    pub unknown_kind_count: u64,
    pub table: Option<String>,
    pub unknown_table_count: u64,
    pub first: Receipt,
    pub last: Receipt,
    pub complete_presence: Option<Receipt>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageStatus {
    Complete,
    Partial,
    Unreadable,
    Refused,
    NoParserYet,
    Failed,
    Unknown,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coverage {
    pub input: String,
    pub format: String,
    pub chain: i64,
    pub reading_version: String,
    pub active: bool,
    pub status: CoverageStatus,
    pub parsed_upto: u64,
    pub length: Option<u64>,
    pub first: Option<Receipt>,
    pub last: Option<Receipt>,
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
    let receipt = Receipt::from(source);
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
            if step.parsed_upto == old.parsed_upto
                && step.at_end
                && !old.at_end
                && step.parsed_upto >= source.length
            {
                old.at_end = true;
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
        prefix_missing: if reset {
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
            save(tx, source, &step.reading_version, &kind, &path, &shape)?;
        }
    }
    for (key, observation) in &step.counts.wire_shapes {
        let decoded: (String, String) = serde_json::from_str(key)?;
        if decoded.0 != observation.record_kind || decoded.1 != observation.path {
            return invalid("wire shape key differs from its record kind and path");
        }
        validate(observation)?;
        let previous:Option<String>=tx.prepare_cached("SELECT shape FROM wire_shape WHERE input_pk=?1 AND chain=?2 AND reading_version=?3 AND record_kind=?4 AND path=?5")?.query_row(params![source.input_pk,source.chain,step.reading_version,observation.record_kind,observation.path],|row|row.get(0)).optional()?;
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
    let mut statement=conn.prepare_cached("SELECT record_kind,path,shape FROM wire_shape WHERE input_pk=?1 AND chain=?2 AND reading_version=?3 ORDER BY record_kind,path")?;
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
    tx.prepare_cached("INSERT INTO wire_shape(input_pk,chain,reading_version,record_kind,path,shape) VALUES (?1,?2,?3,?4,?5,?6) ON CONFLICT(input_pk,chain,reading_version,record_kind,path) DO UPDATE SET shape=excluded.shape")?.execute(params![source.input_pk,source.chain,version,kind,path,serde_json::to_string(shape)?])?;
    Ok(())
}

/// Read derived observations and drift without rereading source bytes. Historical
/// reading versions are present in `shapes` and `coverage`; drift uses the active
/// reading of each input chain. Absence is reported only within comparable,
/// complete receipt groups and remains an optional-presence observation.
pub fn read(conn: &Connection, format: Option<&str>) -> Result<Report> {
    let mut report = Report::default();
    let mut keys = BTreeMap::new();
    let mut statement=conn.prepare_cached("SELECT input_pk,chain,reading_version,format,active,state FROM wire_reading WHERE (?1 IS NULL OR format=?1) ORDER BY format,input_pk,chain,reading_version")?;
    let mut rows = statement.query([format])?;
    while let Some(row) = rows.next()? {
        let json: String = row.get(5)?;
        let state: Reading = serde_json::from_str(&json)?;
        let input_pk: i64 = row.get(0)?;
        let chain: i64 = row.get(1)?;
        let version: String = row.get(2)?;
        let status = state.status();
        let coverage = Coverage {
            input: state.input,
            format: row.get(3)?,
            chain,
            reading_version: version.clone(),
            active: row.get(4)?,
            status,
            parsed_upto: state.parsed_upto,
            length: Some(state.length),
            first: Some(state.first),
            last: Some(state.last),
        };
        keys.insert((input_pk, chain, version), report.coverage.len());
        report.coverage.push(coverage);
    }
    // Refusals and failures can occur before a parser produces any shapes. They
    // remain coverage facts of the ordinary parse state, never empty success.
    let mut statement=conn.prepare_cached("SELECT p.input_pk,i.id,i.format,p.chain,p.reading_version,p.status,p.parsed_upto FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE (?1 IS NULL OR i.format=?1) ORDER BY i.format,i.id,p.chain")?;
    let mut rows = statement.query([format])?;
    while let Some(row) = rows.next()? {
        let input_pk: i64 = row.get(0)?;
        let chain: i64 = row.get(3)?;
        let version: String = row.get(4)?;
        let status: String = row.get(5)?;
        let status = match status.as_str() {
            "done" => CoverageStatus::Complete,
            "waiting" | "parsing" => CoverageStatus::Partial,
            "unreadable" => CoverageStatus::Unreadable,
            "refused" => CoverageStatus::Refused,
            "failed" => CoverageStatus::Failed,
            "no_parser_yet" => CoverageStatus::NoParserYet,
            _ => CoverageStatus::Unknown,
        };
        if let Some(&index) = keys.get(&(input_pk, chain, version.clone())) {
            if report.coverage[index].active && status != CoverageStatus::Complete {
                report.coverage[index].status = status;
            }
        } else {
            report.coverage.push(Coverage {
                input: row.get(1)?,
                format: row.get(2)?,
                chain,
                reading_version: version,
                active: true,
                status,
                parsed_upto: row.get::<_, u64>(6)?,
                length: None,
                first: None,
                last: None,
            });
        }
    }
    let mut statement=conn.prepare_cached("SELECT s.input_pk,s.chain,s.reading_version,s.record_kind,s.path,s.shape FROM wire_shape s JOIN wire_reading r ON r.input_pk=s.input_pk AND r.chain=s.chain AND r.reading_version=s.reading_version WHERE (?1 IS NULL OR r.format=?1) ORDER BY r.format,s.reading_version,s.record_kind,s.path,s.input_pk,s.chain")?;
    let mut rows = statement.query([format])?;
    while let Some(row) = rows.next()? {
        let input: i64 = row.get(0)?;
        let chain: i64 = row.get(1)?;
        let version: String = row.get(2)?;
        let json: String = row.get(5)?;
        let shape: Shape = serde_json::from_str(&json)?;
        let source = &report.coverage[keys[&(input, chain, version.clone())]];
        report.shapes.push(Observation {
            input: source.input.clone(),
            format: source.format.clone(),
            chain,
            reading_version: version,
            active: source.active,
            record_kind: row.get(3)?,
            path: row.get(4)?,
            types: shape.types,
            expected: shape.expected,
            dynamic_map: shape.dynamic_map,
            known_kinds: shape.known_kinds,
            unknown_kind_count: shape.unknown_kind_count,
            table: shape.table,
            unknown_table_count: shape.unknown_table_count,
            first: shape.first,
            last: shape.last,
            complete_presence: shape.complete_presence,
        });
    }
    for shape in report.shapes.iter().filter(|shape| shape.active) {
        let coverage = report
            .coverage
            .iter()
            .find(|source| {
                source.input == shape.input
                    && source.chain == shape.chain
                    && source.reading_version == shape.reading_version
            })
            .map_or(CoverageStatus::Unknown, |source| source.status);
        let present = shape.types.values().any(|kind| kind.count > 0);
        let mut kinds = Vec::new();
        if present && !shape.path.is_empty() && shape.expected.is_empty() && !shape.dynamic_map {
            kinds.push(Kind::NewPath);
        }
        if shape.types.iter().any(|(kind, observed)| {
            observed.count > 0 && !shape.expected.is_empty() && !shape.expected.contains(kind)
        }) {
            kinds.push(Kind::TypeChanged);
        }
        if shape.unknown_kind_count > 0 {
            kinds.push(Kind::UnknownKind);
        }
        if shape.unknown_table_count > 0 {
            kinds.push(Kind::UnknownTable);
        }
        for kind in kinds {
            report.drift.push(Drift {
                kind,
                shape: shape.clone(),
                coverage,
                certainty: Certainty::Observed,
                compared_with: Vec::new(),
            });
        }
    }
    stopped(&mut report);
    report.drift.sort_by(|left, right| {
        drift_receipt(right)
            .cmp(&drift_receipt(left))
            .then_with(|| left.shape.format.cmp(&right.shape.format))
            .then_with(|| left.shape.reading_version.cmp(&right.shape.reading_version))
            .then_with(|| left.shape.record_kind.cmp(&right.shape.record_kind))
            .then_with(|| left.shape.path.cmp(&right.shape.path))
            .then_with(|| kind_order(left.kind).cmp(&kind_order(right.kind)))
            .then_with(|| left.shape.input.cmp(&right.shape.input))
            .then_with(|| left.shape.chain.cmp(&right.shape.chain))
    });
    Ok(report)
}
fn drift_receipt(drift: &Drift) -> i64 {
    drift
        .compared_with
        .iter()
        .filter_map(|source| source.last.as_ref().map(|receipt| receipt.sequence))
        .max()
        .unwrap_or(drift.shape.last.sequence)
}
fn kind_order(kind: Kind) -> u8 {
    match kind {
        Kind::NewPath => 0,
        Kind::TypeChanged => 1,
        Kind::StoppedField => 2,
        Kind::UnknownKind => 3,
        Kind::UnknownTable => 4,
    }
}
fn stopped(report: &mut Report) {
    let groups: BTreeSet<_> = report
        .coverage
        .iter()
        .filter(|source| source.active)
        .map(|source| (source.format.clone(), source.reading_version.clone()))
        .collect();
    for (format, version) in groups {
        let sources: Vec<_> = report
            .coverage
            .iter()
            .filter(|source| {
                source.active && source.format == format && source.reading_version == version
            })
            .collect();
        // A population with inaccessible or unreadable coverage cannot establish
        // field absence. Its coverage remains visible in the report instead.
        if sources
            .iter()
            .any(|source| source.status != CoverageStatus::Complete)
        {
            continue;
        }
        let Some(latest) = sources
            .iter()
            .filter_map(|source| source.last.as_ref().map(|receipt| receipt.sequence))
            .max()
        else {
            continue;
        };
        let latest_sources: Vec<_> = sources
            .into_iter()
            .filter(|source| {
                source
                    .last
                    .as_ref()
                    .is_some_and(|receipt| receipt.sequence == latest)
            })
            .collect();
        let shapes: Vec<_> = report
            .shapes
            .iter()
            .filter(|shape| {
                shape.active && shape.format == format && shape.reading_version == version
            })
            .collect();
        let record_kinds: BTreeSet<_> = shapes
            .iter()
            .filter(|shape| {
                shape.path.is_empty()
                    && latest_sources
                        .iter()
                        .any(|source| source.input == shape.input && source.chain == shape.chain)
                    && shape.types.values().any(|kind| kind.count > 0)
            })
            .map(|shape| shape.record_kind.as_str())
            .collect();
        let mut candidates: BTreeMap<(&str, &str), &Observation> = BTreeMap::new();
        for shape in &shapes {
            if shape.path.is_empty()
                || !record_kinds.contains(shape.record_kind.as_str())
                || shape.complete_presence.is_none()
            {
                continue;
            }
            let key = (shape.record_kind.as_str(), shape.path.as_str());
            if candidates.get(&key).is_none_or(|old| {
                old.complete_presence
                    .as_ref()
                    .map(|receipt| receipt.sequence)
                    < shape
                        .complete_presence
                        .as_ref()
                        .map(|receipt| receipt.sequence)
            }) {
                candidates.insert(key, shape);
            }
        }
        for ((kind, path), shape) in candidates {
            if shape
                .complete_presence
                .as_ref()
                .is_some_and(|receipt| receipt.sequence >= latest)
                || shapes.iter().any(|other| {
                    other.record_kind == kind
                        && other.path == path
                        && latest_sources.iter().any(|source| {
                            source.input == other.input && source.chain == other.chain
                        })
                        && other.types.values().any(|kind| kind.count > 0)
                })
            {
                continue;
            }
            report.drift.push(Drift {
                kind: Kind::StoppedField,
                shape: shape.clone(),
                coverage: CoverageStatus::Complete,
                certainty: Certainty::OptionalPresence,
                compared_with: latest_sources
                    .iter()
                    .map(|source| (**source).clone())
                    .collect(),
            });
        }
    }
}

#[cfg(test)]
mod tests;

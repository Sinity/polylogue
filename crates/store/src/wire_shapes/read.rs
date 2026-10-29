//! SQLite groups source observations before rows cross the store boundary.
use super::*;
use rusqlite::functions::{Aggregate, Context, FunctionFlags};
use serde_json::Value;
const FLAGS: FunctionFlags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);
const READINGS: &str = r#"
WITH readings AS (
 SELECT r.input_pk,r.chain,r.reading_version,r.format,r.active,r.state,
   CASE WHEN p.status IS NOT NULL AND p.status<>'done' THEN p.status
        ELSE pl_wire_status(r.state) END AS status
 FROM wire_reading r LEFT JOIN parse_state p ON r.active=1 AND p.input_pk=r.input_pk
   AND p.chain=r.chain AND p.reading_version=r.reading_version
 WHERE (?1 IS NULL OR r.format=?1) AND (?2 IS NULL OR r.reading_version=?2)
   AND (?3 OR r.active=1)
 UNION ALL
 SELECT p.input_pk,p.chain,p.reading_version,i.format,1,
   json_object('input',i.id,'parsed_upto',p.parsed_upto,'length',NULL,'first',NULL,'last',NULL),
   CASE WHEN p.status='done' THEN 'unknown' ELSE p.status END
 FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk
 WHERE (?1 IS NULL OR i.format=?1) AND (?2 IS NULL OR p.reading_version=?2)
   AND NOT EXISTS(SELECT 1 FROM wire_reading r WHERE r.input_pk=p.input_pk AND r.chain=p.chain AND r.active=1)
), latest AS (
 SELECT format,reading_version,MAX(json_extract(state,'$.last.sequence')) AS receipt
 FROM readings GROUP BY format,reading_version
)
"#;
type Key = (String, String, String, String, Option<String>);

/// Aggregate active readings by default, retaining only schema-shaped results
/// and first/last source references in memory. SQLite owns grouping and spill.
/// Full source provenance remains in the tables; history must be requested.
pub fn read(conn: &Connection, scope: Scope<'_>) -> Result<Report> {
    register(conn)?;
    let mut report = Report::default();
    let mut coverage_keys = BTreeMap::new();
    let sql = format!(
        "{READINGS} SELECT format,reading_version,COUNT(*),COUNT(DISTINCT input_pk),SUM(active),pl_wire_sum(state,'parsed_upto'),pl_wire_sum(state,'length'),SUM(json_extract(state,'$.length') IS NULL),pl_wire_first(state,'first'),pl_wire_last(state,'last') FROM readings GROUP BY format,reading_version ORDER BY format,reading_version"
    );
    let mut statement = conn.prepare_cached(&sql)?;
    let mut rows = statement.query(params![scope.format, scope.reading_version, scope.history])?;
    while let Some(row) = rows.next()? {
        let format: String = row.get(0)?;
        let version: String = row.get(1)?;
        coverage_keys.insert((format.clone(), version.clone()), report.coverage.len());
        report.coverage.push(Coverage {
            format,
            reading_version: version,
            readings: integer(row, 2)?,
            inputs: integer(row, 3)?,
            current_readings: integer(row, 4)?,
            states: BTreeMap::new(),
            parsed_bytes: decimal(row, 5)?,
            known_source_bytes: decimal(row, 6)?,
            unknown_lengths: integer(row, 7)?,
            first: receipt(row, 8)?,
            last: receipt(row, 9)?,
        });
    }
    let sql = format!(
        "{READINGS} SELECT format,reading_version,status,COUNT(*) FROM readings GROUP BY format,reading_version,status ORDER BY format,reading_version,status"
    );
    let mut statement = conn.prepare_cached(&sql)?;
    let mut rows = statement.query(params![scope.format, scope.reading_version, scope.history])?;
    while let Some(row) = rows.next()? {
        let format: String = row.get(0)?;
        let version: String = row.get(1)?;
        let status: String = row.get(2)?;
        let index = coverage_keys[&(format, version)];
        *report.coverage[index]
            .states
            .entry(status_of(&status))
            .or_default() += integer(row, 3)?;
    }
    let sql = format!(
        r#"{READINGS}
SELECT r.format,s.reading_version,s.record_kind,s.path,json_extract(s.shape,'$.table'),COUNT(*),
 json_group_array(DISTINCT json(json_extract(s.shape,'$.expected'))),
 MAX(json_extract(s.shape,'$.dynamic_map')),
 json_group_array(DISTINCT json(json_extract(s.shape,'$.known_kinds'))),
 pl_wire_sum(s.shape,'unknown_kind_count'),pl_wire_sum(s.shape,'unknown_table_count'),
 pl_wire_first(s.shape,'first'),pl_wire_last(s.shape,'last'),pl_wire_last(s.shape,'complete_presence'),
 MAX(CASE WHEN json_extract(r.state,'$.last.sequence')=l.receipt AND EXISTS(
   SELECT 1 FROM json_each(s.shape,'$.types') t WHERE CAST(json_extract(t.value,'$.count') AS REAL)>0)
   THEN 1 ELSE 0 END),pl_wire_sum(s.shape,'unused_count')
FROM wire_shape s JOIN readings r ON r.input_pk=s.input_pk AND r.chain=s.chain AND r.reading_version=s.reading_version
JOIN latest l ON l.format=r.format AND l.reading_version=r.reading_version
GROUP BY r.format,s.reading_version,s.record_kind,s.path,json_extract(s.shape,'$.table')
ORDER BY r.format,s.reading_version,s.record_kind,s.path,json_extract(s.shape,'$.table')"#
    );
    let mut statement = conn.prepare_cached(&sql)?;
    let mut rows = statement.query(params![scope.format, scope.reading_version, scope.history])?;
    let mut shape_keys = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let format: String = row.get(0)?;
        let version: String = row.get(1)?;
        let kind: String = row.get(2)?;
        let path: String = row.get(3)?;
        let table: Option<String> = row.get(4)?;
        shape_keys.insert(
            (
                format.clone(),
                version.clone(),
                kind.clone(),
                path.clone(),
                table.clone(),
            ),
            report.shapes.len(),
        );
        report.shapes.push(Observation {
            format,
            reading_version: version,
            record_kind: kind,
            path,
            table,
            readings: integer(row, 5)?,
            types: BTreeMap::new(),
            expected: declarations(row, 6)?,
            dynamic_map: row.get(7)?,
            known_kinds: declarations(row, 8)?,
            unknown_kind_count: decimal(row, 9)?,
            unknown_table_count: decimal(row, 10)?,
            first: receipt(row, 11)?
                .ok_or_else(|| StoreError::Invalid("wire shape has no first receipt".into()))?,
            last: receipt(row, 12)?
                .ok_or_else(|| StoreError::Invalid("wire shape has no last receipt".into()))?,
            complete_presence: receipt(row, 13)?,
            present_in_latest: row.get(14)?,
            unused_count: decimal(row, 15)?,
        });
    }
    let sql = format!(
        r#"{READINGS}
SELECT r.format,s.reading_version,s.record_kind,s.path,json_extract(s.shape,'$.table'),t.key,
 pl_wire_sum(t.value,'count'),pl_wire_first(t.value,'first'),pl_wire_last(t.value,'last')
FROM wire_shape s JOIN readings r ON r.input_pk=s.input_pk AND r.chain=s.chain AND r.reading_version=s.reading_version,
 json_each(s.shape,'$.types') t
GROUP BY r.format,s.reading_version,s.record_kind,s.path,json_extract(s.shape,'$.table'),t.key
ORDER BY r.format,s.reading_version,s.record_kind,s.path,json_extract(s.shape,'$.table'),t.key"#
    );
    let mut statement = conn.prepare_cached(&sql)?;
    let mut rows = statement.query(params![scope.format, scope.reading_version, scope.history])?;
    while let Some(row) = rows.next()? {
        let key: Key = (
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
        );
        let index = shape_keys[&key];
        report.shapes[index].types.insert(
            row.get(5)?,
            TypeObservation {
                count: decimal(row, 6)?,
                first: receipt(row, 7)?
                    .ok_or_else(|| StoreError::Invalid("wire type has no first receipt".into()))?,
                last: receipt(row, 8)?
                    .ok_or_else(|| StoreError::Invalid("wire type has no last receipt".into()))?,
            },
        );
    }
    let latest_kinds: std::collections::BTreeSet<_> = report
        .shapes
        .iter()
        .filter(|shape| shape.path.is_empty() && shape.present_in_latest)
        .map(|shape| {
            (
                shape.format.clone(),
                shape.reading_version.clone(),
                shape.record_kind.clone(),
                shape.table.clone(),
            )
        })
        .collect();
    for shape in &report.shapes {
        let source =
            &report.coverage[coverage_keys[&(shape.format.clone(), shape.reading_version.clone())]];
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
                coverage: source.status(),
                certainty: Certainty::Observed,
                compared_with: Vec::new(),
            });
        }
        if !shape.path.is_empty()
            && shape.complete_presence.is_some()
            && !shape.present_in_latest
            && source.status() == CoverageStatus::Complete
            && latest_kinds.contains(&(
                shape.format.clone(),
                shape.reading_version.clone(),
                shape.record_kind.clone(),
                shape.table.clone(),
            ))
        {
            report.drift.push(Drift {
                kind: Kind::StoppedField,
                shape: shape.clone(),
                coverage: CoverageStatus::Complete,
                certainty: Certainty::OptionalPresence,
                compared_with: vec![source.clone()],
            });
        }
    }
    report.drift.sort_by(|left, right| {
        event_receipt(right)
            .cmp(&event_receipt(left))
            .then_with(|| left.shape.format.cmp(&right.shape.format))
            .then_with(|| left.shape.reading_version.cmp(&right.shape.reading_version))
            .then_with(|| left.shape.record_kind.cmp(&right.shape.record_kind))
            .then_with(|| left.shape.path.cmp(&right.shape.path))
            .then_with(|| kind_order(left.kind).cmp(&kind_order(right.kind)))
            .then_with(|| left.shape.table.cmp(&right.shape.table))
    });
    Ok(report)
}
fn status_of(status: &str) -> CoverageStatus {
    match status {
        "complete" | "done" => CoverageStatus::Complete,
        "partial" | "waiting" | "parsing" => CoverageStatus::Partial,
        "unreadable" => CoverageStatus::Unreadable,
        "refused" => CoverageStatus::Refused,
        "no_parser_yet" => CoverageStatus::NoParserYet,
        "failed" => CoverageStatus::Failed,
        _ => CoverageStatus::Unknown,
    }
}
fn name_of(status: CoverageStatus) -> &'static str {
    match status {
        CoverageStatus::Complete => "complete",
        CoverageStatus::Incomplete => "incomplete",
        CoverageStatus::Partial => "partial",
        CoverageStatus::Unreadable => "unreadable",
        CoverageStatus::Refused => "refused",
        CoverageStatus::NoParserYet => "no_parser_yet",
        CoverageStatus::Failed => "failed",
        CoverageStatus::Unknown => "unknown",
    }
}
fn event_receipt(drift: &Drift) -> i64 {
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
fn integer(row: &rusqlite::Row<'_>, at: usize) -> Result<u64> {
    u64::try_from(row.get::<_, i64>(at)?)
        .map_err(|_| StoreError::Invalid("negative wire count in index".into()))
}
fn decimal(row: &rusqlite::Row<'_>, at: usize) -> Result<u64> {
    row.get::<_, String>(at)?
        .parse()
        .map_err(|_| StoreError::Invalid("invalid wire count in index".into()))
}
fn receipt(row: &rusqlite::Row<'_>, at: usize) -> Result<Option<Receipt>> {
    row.get::<_, Option<String>>(at)?
        .map(|json| serde_json::from_str(&json))
        .transpose()
        .map_err(StoreError::from)
}
fn declarations(row: &rusqlite::Row<'_>, at: usize) -> Result<Vec<String>> {
    let values: Vec<Vec<String>> = serde_json::from_str(&row.get::<_, String>(at)?)?;
    Ok(values
        .into_iter()
        .flatten()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect())
}
fn error(error: impl std::error::Error + Send + Sync + 'static) -> rusqlite::Error {
    rusqlite::Error::UserFunctionError(Box::new(error))
}
fn field(ctx: &Context<'_>) -> rusqlite::Result<Value> {
    let json: String = ctx.get(0)?;
    let key: String = ctx.get(1)?;
    let value: Value = serde_json::from_str(&json).map_err(error)?;
    Ok(value.get(&key).cloned().unwrap_or(Value::Null))
}
struct Sum;
impl Aggregate<u64, String> for Sum {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<u64> {
        Ok(0)
    }
    fn step(&self, ctx: &mut Context<'_>, sum: &mut u64) -> rusqlite::Result<()> {
        let value = field(ctx)?;
        if value.is_null() {
            return Ok(());
        }
        let count = value.as_u64().ok_or_else(|| {
            error(std::io::Error::other(
                "wire count is not an unsigned integer",
            ))
        })?;
        *sum = sum.checked_add(count).ok_or_else(|| {
            error(std::io::Error::other(
                "wire count exceeds its representation",
            ))
        })?;
        Ok(())
    }
    fn finalize(&self, _: &mut Context<'_>, sum: Option<u64>) -> rusqlite::Result<String> {
        Ok(sum.unwrap_or(0).to_string())
    }
}
struct ReceiptOrder {
    first: bool,
}
impl Aggregate<Option<Receipt>, Option<String>> for ReceiptOrder {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<Option<Receipt>> {
        Ok(None)
    }
    fn step(&self, ctx: &mut Context<'_>, stored: &mut Option<Receipt>) -> rusqlite::Result<()> {
        let value = field(ctx)?;
        if value.is_null() {
            return Ok(());
        }
        let next: Receipt = serde_json::from_value(value).map_err(error)?;
        let key = |receipt: &Receipt| {
            (
                receipt.sequence,
                receipt.input.clone(),
                receipt.chain,
                receipt.observation.clone(),
            )
        };
        if stored.as_ref().is_none_or(|current| {
            if self.first {
                key(&next) < key(current)
            } else {
                key(&next) > key(current)
            }
        }) {
            *stored = Some(next);
        }
        Ok(())
    }
    fn finalize(
        &self,
        _: &mut Context<'_>,
        stored: Option<Option<Receipt>>,
    ) -> rusqlite::Result<Option<String>> {
        stored
            .flatten()
            .map(|receipt| serde_json::to_string(&receipt))
            .transpose()
            .map_err(error)
    }
}
fn register(conn: &Connection) -> Result<()> {
    conn.create_aggregate_function("pl_wire_sum", 2, FLAGS, Sum)?;
    conn.create_aggregate_function("pl_wire_first", 2, FLAGS, ReceiptOrder { first: true })?;
    conn.create_aggregate_function("pl_wire_last", 2, FLAGS, ReceiptOrder { first: false })?;
    conn.create_scalar_function("pl_wire_status", 1, FLAGS, |ctx| {
        let json: String = ctx.get(0)?;
        let reading: Reading = serde_json::from_str(&json).map_err(error)?;
        Ok(name_of(reading.status()).to_owned())
    })?;
    Ok(())
}

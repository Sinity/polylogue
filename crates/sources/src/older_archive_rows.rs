//! Notes in the verbatim row stream written by `import --from-archive`.
//! Events retain the complete source envelope and exact record range. Their
//! ordinary reading facts rebuild the imported-note projection from stored bytes.
use crate::parse::{Output, Parse, Parser};
use crate::{Record, RecordOutcome, Records};
use polylogue_model::{Event, Json, Precision, TimeQuality, Timestamp};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-archive-rows";
pub const VERSION: u32 = 1;
pub const NOTE: &str = "imported_note";
pub const TARGET: &str = "imported_note_target";
pub const RELATION: &str = "imported_note_relation";

pub fn is_member(path: &Path) -> bool {
    (path.file_name().is_some_and(|name| name == "rows.jsonl")
        && path
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "archive.db!"))
        || path
            .to_str()
            .is_some_and(|path| path.replace('\\', "/").ends_with("/archive.db!/rows.jsonl"))
}

pub fn recognize(path: &Path, _: &[u8]) -> bool {
    is_member(path)
}
pub fn open(path: &Path) -> Option<Parse> {
    is_member(path).then(|| Parse::Lines(Box::new(Rows)))
}

pub struct Rows;
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct State {}

fn text(row: &Json, field: &str) -> bool {
    row.get(field).is_some_and(Json::is_string)
}
fn optional_text(row: &Json, field: &str) -> bool {
    row.get(field).is_some_and(|v| v.is_null() || v.is_string())
}
fn valid(table: &str, row: &Json) -> bool {
    match table {
        "note" => {
            ["id", "first_note", "kind", "author", "machine"]
                .into_iter()
                .all(|f| text(row, f))
                && ["key", "value"].into_iter().all(|f| optional_text(row, f))
                && ["created_at", "commit_seq"]
                    .into_iter()
                    .all(|f| row.get(f).is_some_and(Json::is_i64))
                && row
                    .get("valid_until")
                    .is_some_and(|v| v.is_null() || v.is_i64())
        }
        "note_target" => text(row, "note_id") && text(row, "ref"),
        "note_relation" => text(row, "note_id") && text(row, "kind") && text(row, "other"),
        _ => false,
    }
}
impl Parser for Rows {
    type ParserState = State;
    fn kind(&self, value: &Json) -> String {
        value
            .get("table")
            .and_then(Json::as_str)
            .unwrap_or("invalid")
            .into()
    }
    fn parse_record(
        &self,
        _: &mut State,
        record: Record<'_>,
        value: &Json,
        output: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let (Some(table), Some(row)) = (
            value.get("table").and_then(Json::as_str),
            value.get("row").filter(|v| v.is_object()),
        ) else {
            return RecordOutcome::unreadable("archive row needs table and row fields");
        };
        let kind = match table {
            "note" => NOTE,
            "note_target" => TARGET,
            "note_relation" => RELATION,
            _ => {
                records.unknown_table("archive row", table, row);
                return RecordOutcome::skipped("other archive table retained in original input");
            }
        };
        if !valid(table, row) {
            return RecordOutcome::unreadable(format!("invalid {table} row in older archive"));
        }
        let at = row
            .get("created_at")
            .and_then(Json::as_i64)
            .and_then(Timestamp::from_unix_ms);
        output.events.push(Event {
            session: None,
            kind: kind.into(),
            at,
            at_quality: if at.is_some() {
                TimeQuality::Intrinsic
            } else {
                TimeQuality::Unknown
            },
            at_precision: at.map(|_| Precision::Instant),
            at_until: None,
            after: None,
            range: Some((record.offset, record.offset + record.bytes.len() as u64)),
            data: value.clone(),
        });
        RecordOutcome::Used
    }
    fn flush(&self, _: &mut State, _: &mut Output) {}
}

#[cfg(test)]
mod tests;

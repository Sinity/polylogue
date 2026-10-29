//! Metadata for native tool-database row dumps whose origin has no domain parser.
//! The archive owns original rows. This parser emits only counts and wire shapes.
use crate::fields::{ExpectedType as T, Fields, Rule};
use crate::parse::{Output, Parse, Parser};
use crate::{Record, RecordOutcome, Records};
use polylogue_model::Json;
use serde::{Deserialize, Serialize};

pub const VERSION: u32 = 2;

pub fn open() -> Parse {
    Parse::Lines(Box::new(RowDump))
}

pub struct RowDump;
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

static HEADER: Fields = Fields::new(&[
    ("db", Rule::Read(T::String)),
    ("at_ms", Rule::Read(T::Number)),
    ("schema", Rule::Read(T::Object)),
    ("table_metadata", Rule::Read(T::Object)),
])
.for_kind("schema_header");
static INVALID: Fields = Fields::new(&[]).for_kind("record");

impl Parser for RowDump {
    type ParserState = NoState;
    fn kind(&self, value: &Json) -> String {
        if value.get("schema").is_some() {
            "schema"
        } else if value.get("gone").is_some() {
            "gone"
        } else {
            "row"
        }
        .into()
    }
    fn parse_record(
        &self,
        _: &mut NoState,
        _: Record<'_>,
        value: &Json,
        _: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        if let Some(schema) = value.get("schema") {
            records.fields(
                "schema",
                &crate::fields::header(value, &["schema", "table_metadata"]),
                &HEADER,
            );
            let Some(tables) = schema.as_object() else {
                return RecordOutcome::unreadable("row-dump schema is not an object");
            };
            for (table, definition) in tables {
                records.unknown_table_schema(table, definition);
            }
            if let Some(metadata) = value.get("table_metadata") {
                let Some(tables) = metadata.as_object() else {
                    records.fields("table metadata", metadata, &INVALID);
                    return RecordOutcome::unreadable("row-dump table metadata is not an object");
                };
                for (table, metadata) in tables {
                    records.table_metadata(table, metadata);
                }
            }
            return RecordOutcome::Used;
        }
        let Some(table) = value.get("t").and_then(Json::as_str) else {
            records.fields("row", value, &INVALID);
            return RecordOutcome::unreadable("row-dump record has no table identifier");
        };
        if let Some(gone) = value.get("gone") {
            records.unknown_table_gone(table, value);
            return if gone == &Json::Bool(true) {
                RecordOutcome::skipped("upstream row disappeared; original rows remain archived")
            } else {
                RecordOutcome::unreadable("row-dump gone marker is not true")
            };
        }
        let Some(row) = value.get("r").filter(|row| row.is_object()) else {
            records.fields("row", value, &INVALID);
            return RecordOutcome::unreadable("row-dump record has no row object");
        };
        records.unknown_table_row_envelope(table, value);
        records.unknown_table("row", table, row);
        RecordOutcome::skipped("tool table has no domain parser")
    }
    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}

#[cfg(test)]
mod tests;

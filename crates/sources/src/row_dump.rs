//! Metadata for tool-database row dumps whose origin has no domain parser.
//! The archive owns original rows. This parser emits only counts and wire shapes.
use crate::fields::{ExpectedType as T, Fields, Rule};
use crate::parse::{Output, Parse, Parser};
use crate::{Record, RecordOutcome, Records};
use polylogue_model::Json;
use polylogue_model::JsonExt;
use serde::{Deserialize, Serialize};

pub const VERSION: u32 = 3;

parser_modules! {}

pub fn open() -> Parse {
    Parse::Lines(Box::new(RowDump))
}

pub struct RowDump;
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

/// The header line the store writes when a database's schema changes.
static HEADER: Fields = Fields::new(
    "schema_header",
    &[
        ("db", Rule::Read(T::String)),
        ("at_ms", Rule::Read(T::Number)),
        ("schema", Rule::Read(T::Object)),
        ("table_metadata", Rule::Read(T::Object)),
        (
            "source",
            Rule::Kept("the database file's identity; stays with the stored rows"),
        ),
    ],
);
/// A row line as the store writes it: table, key, the row's columns, when
/// the snapshot was read, and how the columns are encoded.
pub static ROW_ENVELOPE: Fields = Fields::new(
    "row_wrapper",
    &[
        ("t", Rule::Read(T::String)),
        (
            "k",
            Rule::ReadWhole("the row's key: its primary-key values, or its rowid"),
        ),
        ("r", Rule::Read(T::Object)),
        ("at_ms", Rule::Read(T::Number)),
        ("encoding", Rule::Read(T::String)),
    ],
);
/// A line saying a row is gone from the database.
static GONE: Fields = Fields::new(
    "gone",
    &[
        ("t", Rule::Read(T::String)),
        (
            "k",
            Rule::ReadWhole("the row's key: its primary-key values, or its rowid"),
        ),
        ("gone", Rule::Read(T::Boolean)),
        ("at_ms", Rule::Read(T::Number)),
    ],
);
static INVALID: Fields = Fields::new("record", &[]);

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
            let mut invalid = None;
            if let Some(tables) = schema.as_object() {
                for (table, definition) in tables {
                    records.unknown_table_schema(table, definition);
                }
            } else {
                invalid = Some("row-dump schema is not an object");
            }
            if let Some(metadata) = value.get("table_metadata") {
                if let Some(tables) = metadata.as_object() {
                    for (table, metadata) in tables {
                        records.table_metadata(table, metadata);
                    }
                } else {
                    records.fields("table metadata", metadata, &INVALID);
                    invalid.get_or_insert("row-dump table metadata is not an object");
                }
            }
            return invalid.map_or(RecordOutcome::Used, RecordOutcome::unreadable);
        }
        let Some(table) = value.str("t") else {
            records.fields("row", value, &INVALID);
            return RecordOutcome::unreadable("row-dump record has no table identifier");
        };
        if let Some(gone) = value.get("gone") {
            records.unknown_table_gone(table, value, &GONE);
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
        records.unknown_table_row_wrapper(table, value, &ROW_ENVELOPE);
        records.unknown_table("row", table, row);
        RecordOutcome::skipped("tool table has no domain parser")
    }
    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}

#[cfg(test)]
mod tests;

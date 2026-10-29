//! Record and wire-field counts shared by line and document parsers.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::fields::{Fields, inspect};
use polylogue_model::{CountMismatch, Json, RecordCounts, RecordOutcome};

/// A parse step's record counts. Parsers supply record outcomes and field
/// declarations; only the framework can finish and expose the counts.
#[derive(Default)]
pub struct Records {
    counts: RecordCounts,
    /// Line inputs count their records while splitting them. Document parsers discover
    /// their records while walking the document instead.
    line_count_supplied: bool,
    field_checks: u64,
}

impl Records {
    pub(crate) fn lines(records: u64) -> Self {
        Self {
            counts: RecordCounts {
                records,
                ..RecordCounts::default()
            },
            line_count_supplied: true,
            field_checks: 0,
        }
    }

    /// Counts one record. For a document this also counts it as read;
    /// splitting lines has already counted the records read.
    pub fn record(&mut self, kind: &str, outcome: &RecordOutcome) {
        if !self.line_count_supplied {
            self.counts.records += 1;
        }
        self.counts.add(kind, outcome);
    }

    /// Count unused leaves and derived shapes in one walk.
    pub fn fields(&mut self, kind: &str, value: &Json, declared: &Fields) {
        self.account(kind, value, declared, None, false);
    }

    fn account(
        &mut self,
        kind: &str,
        value: &Json,
        declared: &Fields,
        table: Option<&str>,
        unknown: bool,
    ) {
        self.field_checks += 1;
        let accounting = inspect(value, declared);
        for path in accounting.unused {
            self.counts.unread(kind, &path);
        }
        let public_kind = declared.kind(kind);
        for (path, mut shape) in accounting.shapes {
            shape.record_kind = public_kind.to_owned();
            shape.table = table.map(str::to_owned);
            if unknown && path.is_empty() {
                shape.unknown_table_count += 1;
            }
            let key = match table {
                Some(table) => serde_json::to_string(&(public_kind, table, &path)),
                None => serde_json::to_string(&(public_kind, &path)),
            }
            .expect("string tuple");
            self.counts
                .wire_shapes
                .entry(key)
                .or_default()
                .merge(&shape);
        }
    }

    /// Native database table names are schema identifiers. Row scalar bodies
    /// remain absent from these derived observations.
    pub fn unknown_table(&mut self, kind: &str, table: &str, row: &Json) {
        static ROW: Fields = Fields::new(&[(
            "",
            crate::fields::Rule::Read(crate::fields::ExpectedType::Object),
        )])
        .for_kind("row");
        self.account(kind, row, &ROW, Some(table), true);
    }
    /// A row disappearance carries table identity and framing fields rather
    /// than a replacement sample of that table's columns.
    pub fn unknown_table_gone(&mut self, table: &str, record: &Json) {
        static GONE: Fields = Fields::new(&[
            (
                "t",
                crate::fields::Rule::Read(crate::fields::ExpectedType::String),
            ),
            (
                "k",
                crate::fields::Rule::Read(crate::fields::ExpectedType::Any),
            ),
            (
                "gone",
                crate::fields::Rule::Read(crate::fields::ExpectedType::Boolean),
            ),
        ])
        .for_kind("gone");
        self.account("gone", record, &GONE, Some(table), true);
    }

    pub fn unknown_table_schema(&mut self, table: &str, schema: &Json) {
        static SCHEMA: Fields = Fields::new(&[(
            "",
            crate::fields::Rule::Read(crate::fields::ExpectedType::Any),
        )])
        .for_kind("schema");
        self.account("schema", schema, &SCHEMA, Some(table), true);
    }

    pub(crate) fn field_checks(&self) -> u64 {
        self.field_checks
    }
    /// A complete retained event is evidenced by the emitted row. A record
    /// without that row needs its own declaration or its leaves remain unused.
    pub(crate) fn remaining_fields(
        &mut self,
        kind: &str,
        value: &Json,
        root_kept: bool,
        payload_kept: bool,
    ) {
        use crate::fields::{ExpectedType as T, Rule};
        static RETAINED: Fields = Fields::new(&[
            ("", Rule::Kept("complete event payload")),
            ("type", Rule::Read(T::String)),
            ("timestamp", Rule::Read(T::StringOrNumber)),
        ])
        .for_kind("event");
        static PAYLOAD: Fields = Fields::new(&[
            ("type", Rule::Read(T::String)),
            ("timestamp", Rule::Read(T::StringOrNumber)),
            ("payload", Rule::Kept("complete native event payload")),
        ])
        .for_kind("event");
        static UNREAD: Fields = Fields::new(&[]).for_kind("*");
        self.fields(
            kind,
            value,
            if root_kept {
                &RETAINED
            } else if payload_kept {
                &PAYLOAD
            } else {
                &UNREAD
            },
        );
    }

    pub(crate) fn finish(self) -> Result<RecordCounts, CountMismatch> {
        self.counts.check()?;
        Ok(self.counts)
    }
}

#[cfg(test)]
mod tests;

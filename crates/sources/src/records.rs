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

    /// Record fields neither read nor excluded by this format's declaration.
    pub fn fields(&mut self, kind: &str, value: &Json, declared: &Fields) {
        self.field_checks += 1;
        let accounting = inspect(value, declared);
        for path in accounting.unused {
            self.counts.unread(kind, &path);
        }
        for (path, mut shape) in accounting.shapes {
            shape.record_kind = kind.to_owned();
            let key = serde_json::to_string(&(kind, &path)).expect("string tuple");
            let observed = self.counts.wire_shapes.entry(key).or_default();
            observed.record_kind.clone_from(&shape.record_kind);
            observed.path.clone_from(&shape.path);
            observed.merge(&shape);
        }
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
        ]);
        static PAYLOAD: Fields = Fields::new(&[
            ("type", Rule::Read(T::String)),
            ("timestamp", Rule::Read(T::StringOrNumber)),
            ("payload", Rule::Kept("complete native event payload")),
        ]);
        static UNREAD: Fields = Fields::new(&[]);
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

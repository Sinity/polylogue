//! Record and wire-field counts shared by line and document parsers.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::fields::{Fields, unaccounted};
use polylogue_model::{CountMismatch, Json, RecordCounts, RecordOutcome};

/// A parse step's record counts. Parsers supply record outcomes and field
/// declarations; only the framework can finish and expose the counts.
#[derive(Default)]
pub struct Records {
    counts: RecordCounts,
    /// Line inputs count their records while splitting them. Document parsers discover
    /// their records while walking the document instead.
    line_count_supplied: bool,
}

impl Records {
    pub(crate) fn lines(records: u64) -> Self {
        Self {
            counts: RecordCounts {
                records,
                ..RecordCounts::default()
            },
            line_count_supplied: true,
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
        for path in unaccounted(value, declared) {
            self.counts.unread(kind, &path);
        }
    }

    pub(crate) fn finish(self) -> Result<RecordCounts, CountMismatch> {
        self.counts.check()?;
        Ok(self.counts)
    }
}

#[cfg(test)]
mod tests;

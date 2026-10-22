//! Record and wire-field accounting shared by line and document parsers.

use crate::fields::{Fields, unaccounted};
use polylogue_model::{Accounting, CountMismatch, Disposition, Json};

/// A parse step's accounting sink. Parsers supply dispositions and field
/// declarations; only the framework can finish and expose the counts.
#[derive(Default)]
pub struct Records {
    accounting: Accounting,
    /// Lines have an independently framed count. Document parsers discover
    /// their records while walking the document instead.
    line_count_supplied: bool,
}

impl Records {
    pub(crate) fn lines(framed: u64) -> Self {
        Self {
            accounting: Accounting {
                framed,
                ..Accounting::default()
            },
            line_count_supplied: true,
        }
    }

    /// Account for one record. For a document this also frames its item;
    /// line framing has already supplied the independent expected count.
    pub fn record(&mut self, kind: &str, disposition: &Disposition) {
        if !self.line_count_supplied {
            self.accounting.framed += 1;
        }
        self.accounting.add(kind, disposition);
    }

    /// Record fields neither read nor excluded by this format's declaration.
    pub fn fields(&mut self, kind: &str, value: &Json, declared: &Fields) {
        for path in unaccounted(value, declared) {
            self.accounting.unread(kind, &path);
        }
    }

    pub(crate) fn finish(self) -> Result<Accounting, CountMismatch> {
        self.accounting.check()?;
        Ok(self.accounting)
    }
}

#[cfg(test)]
mod tests;

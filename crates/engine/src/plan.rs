//! What the loop knows about each input between passes, and one input's
//! work in a pass.

use crate::pool::{self, Done, Job, Priority};
use polylogue_model::Json;
use polylogue_sources::OriginSpec;
use polylogue_store::{Change, FileFacts, InputKey, Observation};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// What the loop knows about one input between passes. Its parser state
/// (`parser_state`) is `None` until the input is next parsed after a restart.
pub(crate) struct InputState {
    pub(crate) input_id: String,
    pub(crate) origin: &'static OriginSpec,
    pub(crate) facts: Option<FileFacts>,
    pub(crate) stored: u64,
    pub(crate) parsed_upto: u64,
    /// Whether the current chain was parsed at all.
    pub(crate) parsed: bool,
    /// A successful parse reached source EOF with an incomplete last record.
    pub(crate) waiting: bool,
    pub(crate) parser_state: Option<Json>,
    pub(crate) tail: Option<Vec<u8>>,
    /// The revision chain whose bytes are stored, when known from the
    /// archive; `None` once this run stored a new chain.
    pub(crate) chain: Option<i64>,
    /// The file facts a parse failed at: the input is tried again only when
    /// its bytes change or the daemon restarts.
    pub(crate) failed: Option<FileFacts>,
    /// Failed immutable chains wait for changed bytes or a restart.
    pub(crate) failed_history: HashMap<i64, FailedChain>,
}

#[derive(PartialEq, Eq)]
pub(crate) struct FailedChain {
    pub(crate) length: u64,
}

/// One input's work in this pass, advanced job by job.
pub(crate) struct Plan {
    pub(crate) path: PathBuf,
    pub(crate) origin: &'static OriginSpec,
    pub(crate) input: InputKey,
    pub(crate) change: Change,
    pub(crate) observation: Option<Observation>,
    pub(crate) facts: Option<FileFacts>,
    pub(crate) size: u64,
    pub(crate) stored_upto: u64,
    pub(crate) parse_pos: u64,
    pub(crate) storing_started: bool,
    pub(crate) parser_state: Option<Json>,
    pub(crate) tail: Option<Vec<u8>>,
    pub(crate) span: u64,
    pub(crate) chunk: u64,
    pub(crate) priority: Priority,
    /// A location the loop keeps watching (not a one-off import).
    pub(crate) tracked: bool,
    pub(crate) failed: bool,
    pub(crate) replace: bool,
    /// The bytes after `parse_pos` are an incomplete last record.
    pub(crate) waiting: bool,
    /// Where the input's bytes come from.
    pub(crate) source: pool::Bytes,
}

impl Plan {
    /// A plan that parses bytes the archive stored, from `parse_pos`, and
    /// stores nothing: an index behind its archive (a rebuild, a file that
    /// moved or went, an import).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stored(
        path: PathBuf,
        origin: &'static OriginSpec,
        input: InputKey,
        change: Change,
        facts: Option<FileFacts>,
        size: u64,
        parse_pos: u64,
        source: pool::Bytes,
        chunk: u64,
        tracked: bool,
    ) -> Plan {
        Plan {
            span: span(origin, &path, chunk),
            path,
            origin,
            input,
            change,
            observation: None,
            facts,
            size,
            stored_upto: size,
            parse_pos,
            storing_started: false,
            parser_state: None,
            tail: None,
            chunk,
            priority: Priority::Bulk,
            tracked,
            failed: false,
            replace: false,
            waiting: false,
            source,
        }
    }

    pub(crate) fn finished(&self) -> bool {
        self.failed
            || (self.observation.is_none()
                && self.stored_upto >= self.size
                && (self.waiting || self.parse_pos >= self.size))
    }

    pub(crate) fn next_job(&mut self) -> Job {
        let start = self.parse_pos;
        let end = self.size.min(start + self.span);
        let store_from = (end > self.stored_upto).then_some(self.stored_upto);
        let (change, observation, continues, length) = match store_from {
            None if self.size == 0 && self.observation.is_some() => {
                // An empty file still commits its revision and file facts.
                (self.change.clone(), self.observation.take(), false, end)
            }
            Some(_) if !self.storing_started => {
                (self.change.clone(), self.observation.take(), false, end)
            }
            Some(_) => (self.change.clone(), None, true, end),
            None => (
                match self.change {
                    Change::Reparse { chain } => Change::Reparse { chain },
                    _ => Change::Unchanged,
                },
                None,
                false,
                self.stored_upto,
            ),
        };
        Job {
            path: self.path.clone(),
            origin: self.origin,
            input: self.input.clone(),
            change,
            observation,
            continues,
            start,
            store_from,
            end,
            at_end: end == self.size,
            length,
            parser_state: self.parser_state.take(),
            priority: self.priority,
            source: self.source.clone(),
        }
    }

    pub(crate) fn apply(&mut self, d: Done, parsed: bool) {
        // Parsing can fail after new source bytes were sent to the writer.
        // Its rejected writes reload their checkpoints through the existing
        // write report; a parse failure must still remember the stored prefix.
        if d.error.is_none() || d.stored > 0 {
            if d.end > self.stored_upto {
                self.stored_upto = d.end;
                self.storing_started = true;
            }
            self.tail = Some(d.tail);
        }
        if d.error.is_some() {
            self.failed = true;
            self.parser_state = d.parser_state;
            return;
        }
        self.parser_state = d.parser_state;
        if !parsed {
            self.parse_pos = d.end;
        } else if d.parsed_upto > d.start {
            self.parse_pos = d.parsed_upto;
            self.span = self.chunk;
        } else if d.end < self.size {
            // One record is longer than the range read: read more of it.
            self.span = (d.end - d.start).saturating_mul(2);
        } else {
            // Only an incomplete last record remains; it is parsed when the
            // file grows, from `parse_pos`, which stays before it.
            self.waiting = self.parse_pos < self.size;
        }
    }
}

/// Whether `origin` reads the input at `path` whole, as one document.
pub(crate) fn whole(origin: &OriginSpec, path: &Path) -> bool {
    (origin.open)(path).is_some_and(|p| p.is_document())
}

/// The range a plan's first job reads: a document whole, lines by chunk.
pub(crate) fn span(origin: &OriginSpec, path: &Path, chunk: u64) -> u64 {
    if whole(origin, path) { u64::MAX } else { chunk }
}

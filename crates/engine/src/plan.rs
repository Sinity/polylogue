//! What the loop knows about each input between passes, and one input's
//! work in a pass.

use crate::pool::{self, Done, Job, Priority};
use polylogue_model::Json;
use polylogue_sources::OriginSpec;
use polylogue_store::{Change, FileFacts, InputKey, Observation, read::ChainProgress};
use std::path::{Path, PathBuf};

/// What the loop knows about one input between passes. Its parser state
/// (`parser_state`) is restored lazily from the admitted cursor after a restart.
pub(crate) struct InputState {
    pub(crate) input_id: String,
    pub(crate) origin: &'static OriginSpec,
    pub(crate) facts: Option<FileFacts>,
    pub(crate) stored: u64,
    pub(crate) parsed_upto: u64,
    /// Whether the current chain was parsed at all.
    pub(crate) parsed: bool,
    /// The chain is not read again until its bytes grow: a parse reached
    /// source EOF with an incomplete last record, or this engine recorded a
    /// refusal or failure at the stored length (a restart reads it again).
    pub(crate) waiting: bool,
    pub(crate) parser_state: Option<Json>,
    pub(crate) tail: Option<Vec<u8>>,
    /// The revision chain whose bytes are stored, when known from the
    /// archive; `None` once this run stored a new chain.
    pub(crate) chain: Option<i64>,
    /// The file facts a read failed at when no parse state could record the
    /// failure (nothing of the file was stored, or the record itself failed):
    /// the file is tried again only when its bytes change or the daemon
    /// restarts. A failure the parse state records is that table's fact.
    pub(crate) unread: Option<FileFacts>,
    /// A cache of actual SDK working state, separate from confirmed progress.
    pub(crate) working: Option<WorkingState>,
}

impl InputState {
    pub(crate) fn new(input_id: String, origin: &'static OriginSpec) -> Self {
        Self {
            input_id,
            origin,
            facts: None,
            stored: 0,
            parsed_upto: 0,
            parsed: false,
            waiting: false,
            parser_state: None,
            tail: None,
            chain: None,
            unread: None,
            working: None,
        }
    }

    /// Adopt committed logical progress. Private custody never supplies a cursor.
    /// Cached continuation is useful only at the exact same reading checkpoint.
    pub(crate) fn adopt_confirmed(
        &mut self,
        input_id: &str,
        origin: &'static OriginSpec,
        facts: Option<FileFacts>,
        progress: Option<&ChainProgress>,
    ) {
        let chain = progress.map(|p| p.chain);
        let stored = progress.map_or(0, |p| p.length);
        let parsed_upto = progress.map_or(0, |p| p.parsed_upto);
        let parsed = progress.is_some_and(|p| p.parsed);
        let waiting = progress.is_some_and(|p| p.waiting);
        let same_input = self.input_id == input_id && self.origin.name == origin.name;
        let same_cursor = same_input
            && self.chain == chain
            && self.parsed_upto == parsed_upto
            && self.parsed == parsed
            && self.waiting == waiting;
        if !same_cursor {
            self.parser_state = None;
        }
        if !same_input || self.chain != chain || self.stored != stored {
            self.tail = None;
        }
        if self.facts != facts || !same_cursor {
            self.unread = None;
        }
        if !same_input
            || self
                .working
                .as_ref()
                .is_some_and(|working| working.facts == facts && working.stored <= stored)
        {
            self.working = None;
        }
        self.input_id = input_id.into();
        self.origin = origin;
        self.facts = facts;
        self.chain = chain;
        self.stored = stored;
        self.parsed_upto = parsed_upto;
        self.parsed = parsed;
        self.waiting = waiting;
    }

    pub(crate) fn needs_reading(&self) -> bool {
        (!self.waiting && self.parsed_upto < self.stored) || !self.parsed
    }
}

pub(crate) struct WorkingState {
    pub(crate) capture: String,
    pub(crate) group: String,
    pub(crate) facts: Option<FileFacts>,
    pub(crate) stored: u64,
}

/// Physical custody has its own continuation and cannot update source state.
pub(crate) struct RawState {
    pub(crate) cursor: Option<polylogue_store::primary::RawAcquisitionCursor>,
    pub(crate) custody: Option<polylogue_store::primary::RawCustodyReceipt>,
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
    /// Capture continues while a missing same-version continuation defers parsing.
    pub(crate) parser_deferred: bool,
    pub(crate) parser_state: Option<Json>,
    pub(crate) tail: Option<Vec<u8>>,
    pub(crate) span: u64,
    pub(crate) chunk: u64,
    pub(crate) priority: Priority,
    /// A location the loop keeps watching (not a one-off import).
    pub(crate) tracked: bool,
    pub(crate) failed: bool,
    /// The chain's parse state records this plan's failure or refusal.
    pub(crate) failure_recorded: bool,
    pub(crate) replace: bool,
    /// The bytes after `parse_pos` are an incomplete last record.
    pub(crate) waiting: bool,
    /// Where the input's bytes come from.
    pub(crate) source: pool::Bytes,
    pub(crate) basis: Option<polylogue_store::primary::SourceBasis>,
    /// The latest step has durable private custody but no confirmed visibility.
    pub(crate) pending: Option<String>,
    pub(crate) checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    pub(crate) raw: Option<RawState>,
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
            basis: None,
            pending: None,
            checkpoint: None,
            raw: None,
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
            parser_deferred: false,
            parser_state: None,
            tail: None,
            chunk,
            priority: Priority::Bulk,
            tracked,
            failed: false,
            failure_recorded: false,
            replace: false,
            waiting: false,
            source,
        }
    }

    pub(crate) fn finished(&self) -> bool {
        if let Some(raw) = &self.raw {
            // Missing proof stops this attempt with unknown completeness.
            return raw
                .cursor
                .as_ref()
                .is_none_or(|cursor| cursor.range().end >= self.size);
        }
        self.failed
            || (self.parser_deferred && self.observation.is_none() && self.stored_upto >= self.size)
            || (self.observation.is_none()
                && self.stored_upto >= self.size
                && (self.waiting || self.parse_pos >= self.size))
    }

    pub(crate) fn next_job(&mut self) -> Job {
        let start = if let Some(raw) = &self.raw {
            raw.cursor
                .as_ref()
                .expect("a continuing physical acquisition has a verified cursor")
                .range()
                .end
        } else if self.parser_deferred {
            self.stored_upto
        } else {
            self.parse_pos
        };
        let end = self.size.min(start + self.span);
        let store_from = if self.raw.is_some() {
            (end > start).then_some(start)
        } else {
            (end > self.stored_upto
                || self.size == 0 && self.observation.is_some() && !self.storing_started)
                .then_some(self.stored_upto)
        };
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
            acquisition: None,
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
            parser_deferred: self.parser_deferred,
            priority: self.priority,
            source: self.source.clone(),
        }
    }

    pub(crate) fn apply(&mut self, d: Done, parsed: bool) {
        self.pending = d.pending.clone();
        if d.checkpoint.is_some() {
            self.checkpoint = d.checkpoint.clone();
        }
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
        if d.continuation_missing {
            self.parser_deferred = true;
            self.waiting = true;
            return;
        }
        if d.error.is_some() {
            self.failure_recorded |= d.failure_recorded && !d.write_failed;
            if d.stored > 0 && !d.write_failed {
                // The parser failed after custody succeeded. Preserve its last
                // valid cursor while continuing raw acquisition through EOF.
                self.parser_deferred = true;
                self.parser_state = d.parser_state;
                return;
            }
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

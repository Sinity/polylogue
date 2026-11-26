//! What the loop knows about each input between passes, and one input's
//! work in a pass.

use crate::pool::{self, Done, Job, Priority};
use polylogue_model::Json;
use polylogue_sources::OriginSpec;
use polylogue_store::{Change, FileFacts, InputKey, Observation, read::ChainProgress};
use std::path::{Path, PathBuf};

/// File facts and cursors between passes. Active plans restore parser state
/// from the committed checkpoint when they start.
pub(crate) struct InputState {
    pub(crate) input_id: String,
    pub(crate) origin: &'static OriginSpec,
    pub(crate) facts: Option<FileFacts>,
    pub(crate) stored: u64,
    pub(crate) parsed_upto: u64,
    /// Whether the current chain was parsed at all.
    pub(crate) parsed: bool,
    /// Current progress waits at this length: an incomplete record at source
    /// EOF, or a recorded stop. The stored-chain planner selects stop retries.
    pub(crate) waiting: bool,
    /// The revision chain whose bytes are stored, from confirmed metadata
    /// or a writer acknowledgment. Unknown until one supplies it.
    pub(crate) chain: Option<i64>,
    /// The file facts a read failed at when no parse state could record the
    /// failure (nothing of the file was stored, or the record itself failed):
    /// the file is tried again only when its bytes change or the daemon
    /// restarts. A failure the parse state records is that table's fact.
    pub(crate) unread: Option<FileFacts>,
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
            chain: None,
            unread: None,
        }
    }

    /// Copy the cursor fields of committed progress.
    pub(crate) fn take_progress(&mut self, progress: &ChainProgress) {
        self.chain = Some(progress.chain);
        self.stored = progress.length;
        self.parsed_upto = progress.parsed_upto;
        self.parsed = progress.parsed;
        self.waiting = progress.waiting;
    }

    /// Adopt committed logical progress.
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
        if self.facts != facts || !same_cursor {
            self.unread = None;
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

    /// A new scope reads progress again from the index.
    pub(crate) fn discard_confirmed(&mut self) {
        *self = Self::new(std::mem::take(&mut self.input_id), self.origin);
    }

    pub(crate) fn needs_reading(&self) -> bool {
        (!self.waiting && self.parsed_upto < self.stored) || !self.parsed
    }
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
    /// Capture continues and parsing waits: a byte read, a source not yet
    /// confirmed, or a parse that failed after its bytes were stored.
    pub(crate) storing_only: bool,
    pub(crate) parser_state: Option<Json>,
    /// The chain the parse continues, kept across jobs.
    pub(crate) parser_chain: Option<i64>,
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
    /// The input will not grow (an import, an export member): the end of its
    /// bytes ends its last record.
    pub(crate) input_finished: bool,
    /// Where the input's bytes come from.
    pub(crate) source: pool::Bytes,
    pub(crate) checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    /// Older chains of the input recorded from this chain's reading as the
    /// parse passes the end of the bytes they share with it, by stop
    /// ([`crate::prefix_copies`]). No range reads past the first stop.
    pub(crate) prefix_copies: Option<polylogue_store::PrefixCopies>,
    /// The end of the last range parsed: once it reaches a copy's stop, the
    /// parse has read every complete record the shared bytes hold.
    pub(crate) read_to: u64,
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
        let parser_chain = match (&change, &source) {
            (Change::First | Change::Rewritten, _) => None,
            (Change::Reparse { chain }, _) | (_, pool::Bytes::Unloaded { chain, .. }) => {
                Some(*chain)
            }
            _ => None,
        };
        Plan {
            checkpoint: None,
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
            storing_only: false,
            parser_state: None,
            parser_chain,
            tail: None,
            chunk,
            priority: Priority::Bulk,
            tracked,
            failed: false,
            failure_recorded: false,
            replace: false,
            waiting: false,
            input_finished: false,
            source,
            prefix_copies: None,
            read_to: parse_pos,
        }
    }

    pub(crate) fn finished(&self) -> bool {
        self.failed
            || (self.storing_only && self.observation.is_none() && self.stored_upto >= self.size)
            || (self.observation.is_none()
                && self.stored_upto >= self.size
                && (self.waiting || self.parse_pos >= self.size))
    }

    /// The byte range the next job reads.
    pub(crate) fn next_range(&self) -> (u64, u64) {
        let start = if self.storing_only {
            self.stored_upto
        } else {
            self.parse_pos
        };
        let end = self.size.min(start.saturating_add(self.span));
        match self
            .prefix_copies
            .as_ref()
            .and_then(|copies| copies.first())
        {
            Some(copy) if copy.stop > start => (start, end.min(copy.stop)),
            _ => (start, end),
        }
    }

    pub(crate) fn next_job(
        &mut self,
        stopped_attempt: &polylogue_model::batch::StoppedAttempt,
    ) -> Job {
        let (start, end) = self.next_range();
        let store_from = (end > self.stored_upto
            || self.size == 0 && self.observation.is_some() && !self.storing_started)
            .then_some(self.stored_upto);
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
            stopped_attempt: Some(*stopped_attempt),
            source_read: None,
            proven_tail: self.tail.clone().map(|bytes| crate::read::ProvenTail {
                end: self.stored_upto,
                bytes,
            }),
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
            finished: self.input_finished && end == self.size,
            length,
            parser_state: self.parser_state.take(),
            parser_chain: self.parser_chain,
            store_only: self.storing_only,
            priority: self.priority,
            source: self.source.clone(),
        }
    }

    pub(crate) fn apply(&mut self, d: Done, parsed: bool) {
        if let Some(chain) = d.chain {
            self.parser_chain = Some(chain);
        }
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
        if d.restart {
            // Without the reading it continues, the chain is read again
            // from its first byte, which gives the same reading.
            self.parse_pos = 0;
            self.read_to = 0;
            self.parser_state = None;
            self.replace = true;
            return;
        }
        if d.error.is_some() {
            self.failure_recorded |= d.failure_recorded && !d.write_failed;
            if d.stored > 0 && !d.write_failed {
                // The parser failed after its bytes were stored. Keep its last
                // valid cursor while continuing byte cursor through EOF.
                self.storing_only = true;
                self.parser_state = d.parser_state;
                return;
            }
            self.failed = true;
            self.parser_state = d.parser_state;
            return;
        }
        self.parser_state = d.parser_state;
        // A range that ends at a copy's stop is cut short only for the copy.
        let at_stop = self
            .prefix_copies
            .as_ref()
            .and_then(|copies| copies.first())
            .is_some_and(|copy| copy.stop == d.end);
        if parsed {
            self.read_to = d.end;
        }
        if !parsed {
            self.parse_pos = d.end;
        } else if d.parsed_upto > d.start {
            self.parse_pos = d.parsed_upto;
            self.span = self.chunk;
        } else if at_stop {
            // No record completed before the stop: the copy is taken where
            // the parse stands, and the next range reads on from there.
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

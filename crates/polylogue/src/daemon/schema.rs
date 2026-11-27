//! Read-only schema operations over pinned stored revisions. Selection and
//! payload observation release their SQL readers independently.
use super::service::{Daemon, error, store_error};
use command_surface::Answer;
use polylogue_model::LockExt as _;
use polylogue_sources::{
    Parse,
    schema::{Framing, Observer, Schema, Selection},
};
use polylogue_store::{
    ReadPool, StoreError,
    blobs::{ChainReader, PlainBlobs},
    read::limit::ReadLimit,
    schema_inputs::InputRevision,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, BufRead, BufReader, Read},
    path::Path,
    sync::{Arc, Condvar, Mutex},
};

/// Inputs read per store page while inferring a schema.
const INPUTS_PER_PAGE: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Infer {
    pub origin: Option<String>,
    #[serde(default)]
    pub inputs: BTreeSet<String>,
    pub revision: Option<i64>,
    pub since: Option<String>,
    #[serde(default)]
    pub fields: BTreeSet<String>,
}
impl Infer {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .origin
            .as_deref()
            .is_some_and(|origin| polylogue_sources::origin::by_name(origin).is_none())
        {
            return Err("unknown origin".into());
        }
        if self.revision.is_some_and(|r| r < 1) {
            return Err("revision must be positive".into());
        }
        self.since_ms()?;
        if !self.inputs.is_empty() && self.since.is_some() {
            return Err("select explicit inputs or --since".into());
        }
        Ok(())
    }
    fn since_ms(&self) -> Result<Option<i64>, String> {
        self.since
            .as_deref()
            .map(|s| {
                polylogue_model::Timestamp::parse_rfc3339(s)
                    .map(|t| t.unix_ms())
                    .ok_or_else(|| "since needs RFC 3339 with a UTC offset".into())
            })
            .transpose()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SelectionSummary {
    pub origin: Option<String>,
    pub explicit_inputs: bool,
    pub revision: Option<i64>,
    pub since_ms: Option<i64>,
    /// Counted only for a selection by origin; `None` when the inputs were named.
    pub unknown_time_inputs: Option<u64>,
    pub missing_revisions: Vec<String>,
    pub selected_inputs: u64,
    pub selected_bytes: u64,
    pub read_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Inference {
    pub selection: SelectionSummary,
    pub origins: BTreeMap<String, Schema>,
}

/// Ordinary work enters immediately. One idle-priority job (a schema
/// inference or a verification) runs at a time and waits for ordinary work
/// at each streaming read and before opening a SQL reader; waiting holds no
/// reader and no chain read transaction.
#[derive(Default)]
pub(super) struct IdleGate(Arc<Shared>);
#[derive(Default)]
pub(super) struct Shared {
    state: Mutex<Activity>,
    changed: Condvar,
}
impl std::ops::Deref for IdleGate {
    type Target = Shared;
    fn deref(&self) -> &Shared {
        &self.0
    }
}
#[derive(Default)]
struct Activity {
    foreground: usize,
    background: bool,
}
pub(super) struct Foreground<'a>(&'a IdleGate);
impl Drop for Foreground<'_> {
    fn drop(&mut self) {
        let mut s = self.0.state.locked();
        s.foreground -= 1;
        self.0.changed.notify_all();
    }
}
pub(super) struct BackgroundTurn<'a>(&'a IdleGate);
impl Drop for BackgroundTurn<'_> {
    fn drop(&mut self) {
        self.0.state.locked().background = false;
        self.0.changed.notify_all();
    }
}
impl IdleGate {
    pub(super) fn foreground(&self) -> Foreground<'_> {
        self.state.locked().foreground += 1;
        Foreground(self)
    }
    /// Waits on this thread until `ready` holds, then runs `claim` under the
    /// same lock. A cancellation wakes the wait and the request's deadline
    /// bounds it; activity changes notify it. Nothing polls.
    fn wait(
        &self,
        ready: impl Fn(&Activity) -> bool,
        before_waiting: impl FnOnce(),
        claim: impl FnOnce(&mut Activity),
    ) -> Result<(), StoreError> {
        let limit = ReadLimit::current();
        let _wake = limit.as_ref().map(|limit| {
            let shared = Arc::clone(&self.0);
            limit.cancellation.on_cancel(move || {
                let _held = shared.state.locked();
                shared.changed.notify_all();
            })
        });
        let mut state = self.state.locked();
        let mut before_waiting = Some(before_waiting);
        while !ready(&state) {
            if let Some(release) = before_waiting.take() {
                release();
            }
            check()?;
            state = match limit.as_ref().and_then(ReadLimit::remaining) {
                Some(left) => polylogue_model::unpoisoned(self.changed.wait_timeout(state, left)).0,
                None => polylogue_model::unpoisoned(self.changed.wait(state)),
            };
        }
        check()?;
        claim(&mut state);
        Ok(())
    }
    pub(super) fn background(&self) -> Result<BackgroundTurn<'_>, StoreError> {
        // Test and claim under the same lock: two waiters cannot both enter.
        self.wait(
            |s| !s.background && s.foreground == 0,
            || (),
            |s| s.background = true,
        )?;
        Ok(BackgroundTurn(self))
    }
    pub(super) fn idle(&self) -> Result<(), StoreError> {
        self.idle_releasing(|| ())
    }
    /// Waits like [`IdleGate::idle`]; when it has to wait, `release` runs
    /// first, so a streaming read gives up its read transaction for the wait.
    pub(super) fn idle_releasing(&self, release: impl FnOnce()) -> Result<(), StoreError> {
        self.wait(|s| s.foreground == 0, release, |_| ())
    }
}
fn check() -> Result<(), StoreError> {
    ReadLimit::current().map_or(Ok(()), |limit| limit.check())
}

struct IdleRead<'a, B> {
    source: ChainReader<B>,
    gate: &'a IdleGate,
    read: &'a std::cell::Cell<u64>,
}
impl<B: PlainBlobs> Read for IdleRead<'_, B> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let source = &self.source;
        self.gate
            .idle_releasing(|| source.release())
            .map_err(io::Error::other)?;
        let n = self.source.read(bytes)?;
        self.read.set(self.read.get() + n as u64);
        Ok(n)
    }
}

pub(super) fn infer(d: &Daemon, selection: Infer) -> Answer<Inference> {
    if let Err(message) = selection.validate() {
        return error(polylogue_model::Code::BadRequest, message);
    }
    let _turn = match d.idle_gate.background() {
        Ok(turn) => turn,
        Err(e) => return store_error(&e),
    };
    match observe(&d.reads, &d.idle_gate, selection) {
        Ok(report) => Answer::ok(report),
        Err(e) => store_error(&e),
    }
}

fn observe(pool: &ReadPool, gate: &IdleGate, selection: Infer) -> Result<Inference, StoreError> {
    let mut summary = SelectionSummary {
        origin: selection.origin.clone(),
        explicit_inputs: !selection.inputs.is_empty(),
        revision: selection.revision,
        since_ms: selection
            .since_ms()
            .map_err(|problem| polylogue_model::Reason::BadTime {
                text: selection.since.clone().unwrap_or_default(),
                problem,
            })?,
        unknown_time_inputs: None,
        missing_revisions: Vec::new(),
        selected_inputs: 0,
        selected_bytes: 0,
        read_bytes: 0,
    };
    let mut observers = BTreeMap::<String, (Observer, Vec<Selection>)>::new();
    if let Some(origin) = &selection.origin {
        observers.insert(
            origin.clone(),
            (Observer::new(selection.fields.clone()), Vec::new()),
        );
    }
    if !selection.inputs.is_empty() {
        for input in &selection.inputs {
            gate.idle()?;
            let stored = pool.reader()?.schema_input(input, selection.revision)?;
            match stored {
                Some(stored) => {
                    observe_input(pool, gate, &selection, stored, &mut observers, &mut summary)?
                }
                None => summary.missing_revisions.push(input.clone()),
            }
        }
    } else {
        let mut after = None;
        loop {
            gate.idle()?;
            let page = pool.reader()?.schema_inputs_page(
                selection.origin.as_deref(),
                selection.revision,
                summary.since_ms,
                after.as_deref(),
                INPUTS_PER_PAGE,
            )?;
            if let Some(unknown) = page.unknown_time_inputs {
                summary.unknown_time_inputs = Some(unknown);
            }
            summary.missing_revisions.extend(page.missing_revisions);
            for stored in page.inputs {
                observe_input(pool, gate, &selection, stored, &mut observers, &mut summary)?;
            }
            after = page.next;
            if after.is_none() {
                break;
            }
        }
    }
    let sample = summary.explicit_inputs
        || summary.since_ms.is_some()
        || !summary.missing_revisions.is_empty();
    let origins = observers
        .into_iter()
        .map(|(name, (observer, selected))| {
            let mut schema = observer.finish(name.clone(), selected);
            schema.sample = sample;
            (name, schema)
        })
        .collect();
    Ok(Inference {
        selection: summary,
        origins,
    })
}

fn observe_input(
    pool: &ReadPool,
    gate: &IdleGate,
    selection: &Infer,
    stored: InputRevision,
    observers: &mut BTreeMap<String, (Observer, Vec<Selection>)>,
    summary: &mut SelectionSummary,
) -> Result<(), StoreError> {
    check()?;
    summary.selected_inputs += 1;
    summary.selected_bytes += stored.length;
    let read = std::cell::Cell::new(0);
    let (observer, selected) = observers
        .entry(stored.format.clone())
        .or_insert_with(|| (Observer::new(selection.fields.clone()), Vec::new()));
    let mut entry = Selection {
        input: stored.input.clone(),
        revision: Some(stored.revision),
        records: 0,
        unreadable_records: 0,
        unreadable: None,
    };
    let result = (|| -> Result<(), &'static str> {
        if selection
            .origin
            .as_ref()
            .is_some_and(|o| *o != stored.format)
        {
            return Err("different_origin");
        }
        let spec =
            polylogue_sources::origin::by_name(&stored.format).ok_or("unsupported_origin")?;
        if stored.format == polylogue_sources::browser_capture::ORIGIN {
            return Err("browser_capture_format");
        }
        gate.idle().map_err(|_| "input_read")?;
        // Check the chain's extents and take a read-only blob handle, then
        // release the SQL reader before reading any payload bytes. The chain
        // reads its extents in its own transaction, which it ends whenever
        // the read waits for foreground work.
        let bytes = {
            let reader = pool.reader().map_err(|_| "input_read")?;
            let chain = reader
                .stored_chain(&stored.input, stored.chain, stored.length)
                .map_err(|_| "stored_bytes_unavailable")?;
            let blobs = Arc::new(
                reader
                    .blob_reader()
                    .map_err(|_| "stored_bytes_unavailable")?,
            );
            chain.reader(blobs)
        };
        let mut bytes = BufReader::new(IdleRead {
            source: bytes,
            gate,
            read: &read,
        });
        let mut one = Observer::new(selection.fields.clone());
        let compressed_form_of = spec
            .compressed_form_of
            .and_then(polylogue_sources::origin::by_name);
        let result = if let Some(decoded_origin) = compressed_form_of {
            // Stored Codex revisions already contain decoded bytes even when
            // their observation path ends in .zst. Only an origin declared as
            // another's compressed form is decompressed; framing never
            // follows the filename alone.
            let decoded =
                zstd::stream::read::Decoder::new(bytes).map_err(|_| "unreadable_input")?;
            observe_bytes(
                &mut one,
                BufReader::new(decoded),
                decoded_origin,
                &stored.path,
            )
        } else {
            observe_bytes(&mut one, &mut bytes, spec, &stored.path)
        };
        (entry.records, entry.unreadable_records) = one.counts();
        observer.merge(one)?;
        result
    })();
    if let Err(reason) = result {
        entry.unreadable = Some(reason.into());
    }
    selected.push(entry);
    summary.read_bytes += read.get();
    // Cancellation and deadlines are request refusals, not source coverage.
    check()
}

fn observe_bytes(
    observer: &mut Observer,
    mut bytes: impl BufRead,
    spec: &polylogue_sources::OriginSpec,
    path: &str,
) -> Result<(), &'static str> {
    let framing = if spec.capture.is_row_dump() {
        Framing::RowDump
    } else {
        match (spec.open)(Path::new(path)).ok_or("unsupported_framing")? {
            Parse::Lines(_) => Framing::Lines,
            Parse::CaptureLog(_) => Framing::CaptureLog,
            Parse::Document(_) | Parse::Export(_) => {
                loop {
                    let buf = bytes.fill_buf().map_err(|_| "input_read")?;
                    let spaces = buf.iter().take_while(|b| b.is_ascii_whitespace()).count();
                    if spaces == 0 {
                        break;
                    }
                    bytes.consume(spaces);
                }
                if bytes.fill_buf().map_err(|_| "input_read")?.first() == Some(&b'[') {
                    Framing::Array
                } else {
                    Framing::Json
                }
            }
        }
    };
    observer
        .read(bytes, framing)
        .map_err(|_| "unreadable_input")
}

pub(super) fn unread(
    reader: &polylogue_store::Reader<'_>,
    origin: Option<&str>,
) -> Answer<polylogue_store::wire_shapes::Report> {
    match reader.wire_shapes(polylogue_store::wire_shapes::Scope::current(origin)) {
        Ok(mut report) => {
            report.shapes.retain(|shape| shape.unused_count > 0);
            Answer::ok(report)
        }
        Err(e) => store_error(&e),
    }
}

#[cfg(test)]
mod tests;

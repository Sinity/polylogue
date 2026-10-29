//! Read-only schema operations over pinned stored revisions. Selection and
//! payload observation release their SQL readers independently.
use super::{
    Response,
    service::{Daemon, error, ok, store_error},
};
use polylogue_sources::{
    Capture, Parse,
    schema::{Framing, Observer, Schema, Selection},
};
use polylogue_store::{ReadPool, StoreError, read::limit::ReadLimit, schema_inputs::InputRevision};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, BufRead, BufReader, Read},
    path::Path,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

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
    pub unknown_time_inputs: u64,
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

/// Ordinary work enters immediately. One inference waits for it at each
/// streaming read and before opening a SQL reader; waiting holds no reader.
#[derive(Default)]
pub(super) struct Admission {
    state: Mutex<Activity>,
    changed: Condvar,
}
#[derive(Default)]
struct Activity {
    foreground: usize,
    inference: bool,
}
pub(super) struct Foreground<'a>(&'a Admission);
impl Drop for Foreground<'_> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        s.foreground -= 1;
        self.0.changed.notify_all();
    }
}
struct InferenceLease<'a>(&'a Admission);
impl Drop for InferenceLease<'_> {
    fn drop(&mut self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .inference = false;
        self.0.changed.notify_all();
    }
}
impl Admission {
    pub(super) fn foreground(&self) -> Foreground<'_> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .foreground += 1;
        Foreground(self)
    }
    fn wait(&self, predicate: impl Fn(&Activity) -> bool) -> Result<(), StoreError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while !predicate(&state) {
            check()?;
            // A disconnect/deadline does not notify this condvar. The timed
            // wait bounds cancellation latency while activity changes notify it.
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        check()
    }
    fn infer(&self) -> Result<InferenceLease<'_>, StoreError> {
        // Test and claim under the same lock: two waiters cannot both enter.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.inference || state.foreground > 0 {
            check()?;
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(20))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        check()?;
        state.inference = true;
        Ok(InferenceLease(self))
    }
    fn idle(&self) -> Result<(), StoreError> {
        self.wait(|s| s.foreground == 0)
    }
}
fn check() -> Result<(), StoreError> {
    ReadLimit::current().map_or(Ok(()), |limit| limit.check())
}

struct IdleRead<'a, R> {
    source: R,
    admission: &'a Admission,
    read: &'a std::cell::Cell<u64>,
}
impl<R: Read> Read for IdleRead<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.admission.idle().map_err(io::Error::other)?;
        let n = self.source.read(bytes)?;
        self.read.set(self.read.get() + n as u64);
        Ok(n)
    }
}

pub(super) fn infer(d: &Daemon, selection: Infer) -> Response {
    if let Err(message) = selection.validate() {
        return error("bad_request", message);
    }
    let _lease = match d.schema_admission.infer() {
        Ok(lease) => lease,
        Err(e) => return store_error(&e),
    };
    match observe(&d.reads, &d.schema_admission, selection) {
        Ok(report) => ok(report),
        Err(e) => store_error(&e),
    }
}

fn observe(
    pool: &ReadPool,
    admission: &Admission,
    selection: Infer,
) -> Result<Inference, StoreError> {
    let mut summary = SelectionSummary {
        origin: selection.origin.clone(),
        explicit_inputs: !selection.inputs.is_empty(),
        revision: selection.revision,
        since_ms: selection.since_ms().map_err(StoreError::Invalid)?,
        unknown_time_inputs: 0,
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
            admission.idle()?;
            let stored = pool.reader()?.schema_input(input, selection.revision)?;
            match stored {
                Some(stored) => observe_input(
                    pool,
                    admission,
                    &selection,
                    stored,
                    &mut observers,
                    &mut summary,
                )?,
                None => summary.missing_revisions.push(input.clone()),
            }
        }
    } else {
        let mut after = None;
        loop {
            admission.idle()?;
            let page = pool.reader()?.schema_inputs_page(
                selection.origin.as_deref(),
                selection.revision,
                summary.since_ms,
                after.as_deref(),
                128,
            )?;
            if let Some(unknown) = page.unknown_time_inputs {
                summary.unknown_time_inputs = unknown;
            }
            summary.missing_revisions.extend(page.missing_revisions);
            for stored in page.inputs {
                observe_input(
                    pool,
                    admission,
                    &selection,
                    stored,
                    &mut observers,
                    &mut summary,
                )?;
            }
            after = page.next;
            if after.is_none() {
                break;
            }
        }
    }
    let sample = summary.explicit_inputs || summary.since_ms.is_some();
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
    admission: &Admission,
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
            return Err("native_capture_format");
        }
        admission.idle().map_err(|_| "input_read")?;
        // Capture immutable extent descriptors and a read-only blob handle,
        // then release the SQL transaction before reading any payload bytes.
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
            admission,
            read: &read,
        });
        let mut one = Observer::new(selection.fields.clone());
        let result = if stored.format == "codex-compressed" {
            // Native codex revisions already contain decoded bytes even when
            // their observation path ends in .zst. Only this container origin
            // is decompressed; framing never follows the filename alone.
            let native = polylogue_sources::origin::by_name("codex").expect("codex origin");
            let decoded =
                zstd::stream::read::Decoder::new(bytes).map_err(|_| "unreadable_input")?;
            observe_bytes(&mut one, BufReader::new(decoded), native, &stored.path)
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
    let framing = if spec.capture == Capture::RowDump {
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

pub(super) fn unread(reader: &polylogue_store::Reader<'_>, origin: Option<&str>) -> Response {
    match reader.wire_shapes(polylogue_store::wire_shapes::Scope::current(origin)) {
        Ok(mut report) => {
            report.shapes.retain(|shape| shape.unused_count > 0);
            ok(report)
        }
        Err(e) => store_error(&e),
    }
}

#[cfg(test)]
mod tests;

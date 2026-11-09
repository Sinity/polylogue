//! The parse pool: N worker threads, limited by bytes in flight, feeding
//! the one writer. A worker reads its byte range,
//! hashes and stores or compresses the new bytes, parses, and hands the
//! writer one batch. Live appends go before bulk work. One input has at
//! most one job in flight, so its parser state passes from job to job.

use crate::read;
use polylogue_model::Json;
use polylogue_sources::OriginSpec;
use polylogue_store::staging::PreparedBatch;
use polylogue_store::writer::WriterSender;
use polylogue_store::{Batch, Change, InFlight, InputKey, Observation, ParseStep};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Live,
    Bulk,
}

/// One range of one input to store and parse.
pub struct Job {
    pub path: PathBuf,
    pub origin: &'static OriginSpec,
    pub input: InputKey,
    pub change: Change,
    pub observation: Option<Observation>,
    pub(crate) acquisition: Option<crate::acquisition::AcquisitionContext>,
    pub continues: bool,
    /// Where parsing starts: after the last complete record parsed, or 0
    /// when parser state is rebuilt by parsing the input again.
    pub start: u64,
    /// Where new bytes start; `None` when the range is only parsed again.
    pub store_from: Option<u64>,
    pub end: u64,
    /// This range reaches the observed source end, not merely a chunk boundary.
    pub at_end: bool,
    /// The input's stored length once this job is written.
    pub length: u64,
    pub parser_state: Option<Json>,
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub parser_chain: Option<i64>,
    /// One privately staged replacement keeps its own context between jobs.
    pub(crate) parser_context: Option<polylogue_store::parser_context::Overlay>,
    pub(crate) parser_deferred: bool,
    pub priority: Priority,
    /// Where the bytes come from.
    pub source: Bytes,
}

/// Where a job's bytes come from.
#[derive(Clone, Debug, Default)]
pub enum Bytes {
    /// The file at the job's path.
    #[default]
    File,
    /// One complete import, stored before parsing begins.
    Import(Arc<Mutex<read::Imported>>),
    /// What the archive stored for the input: its file changed, moved or
    /// went, or the index is being built again.
    Stored(Arc<polylogue_store::blobs::StoredChain>),
    /// A stored chain not loaded yet: a pass plans every stored input before
    /// it runs any, and a loaded chain holds its extent list in a database
    /// of its own. The engine loads it as the plan starts, as `Import` when
    /// `decoded` (read forward from the start), else as `Stored`.
    Unloaded {
        chain: i64,
        length: u64,
        decoded: bool,
    },
}

impl Job {
    fn bytes(&self) -> u64 {
        self.end - self.start
    }
}

/// What a finished job tells the loop.
#[derive(Debug)]
pub struct Done {
    /// The actual committed chain returned by the writer, never a guess.
    pub chain: Option<i64>,
    pub path: PathBuf,
    pub start: u64,
    pub end: u64,
    /// Offset after the last complete record parsed; the next job of this
    /// input starts here.
    pub parsed_upto: u64,
    pub parser_state: Option<Json>,
    /// The last bytes read, kept to check the next growth.
    pub tail: Vec<u8>,
    pub stored: u64,
    pub error: Option<String>,
    pub write_failed: bool,
    /// The batch carries a step that records the parser's failure.
    pub failure_recorded: bool,
    pub continuation_missing: bool,
    /// Accepted SDK custody; this is neither failure nor confirmed admission.
    pub pending: Option<String>,
    pub checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    pub raw_custody: Option<polylogue_store::primary::RawCustodyReceipt>,
    pub raw_acquisition: Option<polylogue_store::primary::AcquisitionReceipt>,
}

struct Queue {
    live: VecDeque<Job>,
    bulk: VecDeque<Job>,
    in_flight: u64,
    closed: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    work: Condvar,
    room: Condvar,
    limit: u64,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    acknowledgments: Option<JoinHandle<()>>,
}

impl Pool {
    pub fn start(
        workers: usize,
        bytes_in_flight: u64,
        writer: WriterSender,
        blobs: Arc<polylogue_store::blobs::Blobs>,
        archive: polylogue_store::Archive,
        done: Sender<Done>,
    ) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                live: VecDeque::new(),
                bulk: VecDeque::new(),
                in_flight: 0,
                closed: false,
            }),
            work: Condvar::new(),
            room: Condvar::new(),
            limit: bytes_in_flight.max(1),
        });
        let (acknowledge, pending) = channel();
        let acknowledged = done.clone();
        let acknowledgments = std::thread::Builder::new()
            .name("polylogue-acknowledgments".into())
            .spawn(move || acknowledge_groups(pending, acknowledged))
            .expect("spawn acknowledgment reader");
        let workers = (0..workers.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                let writer = writer.clone();
                let blobs = Arc::clone(&blobs);
                let archive = archive.clone();
                let acknowledge = acknowledge.clone();
                std::thread::Builder::new()
                    .name(format!("polylogue-parse-{i}"))
                    .spawn(move || work(&shared, &writer, &blobs, &archive, &acknowledge))
                    .expect("spawn a parse worker")
            })
            .collect();
        drop(acknowledge);
        Self {
            shared,
            workers,
            acknowledgments: Some(acknowledgments),
        }
    }

    /// Queues a job. Blocks while the bytes already in flight are at the
    /// limit; a job larger than the limit runs alone.
    pub fn submit(&self, job: Job) {
        let mut q = self.shared.queue.lock().expect("pool lock");
        while q.in_flight > 0 && q.in_flight + job.bytes() > self.shared.limit {
            q = self.shared.room.wait(q).expect("pool lock");
        }
        polylogue_store::timing::queue(polylogue_store::timing::Queue::Parse, 1);
        q.in_flight += job.bytes();
        match job.priority {
            Priority::Live => q.live.push_back(job),
            Priority::Bulk => q.bulk.push_back(job),
        }
        self.shared.work.notify_one();
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        if let Ok(mut q) = self.shared.queue.lock() {
            q.closed = true;
        }
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        if let Some(thread) = self.acknowledgments.take() {
            let _ = thread.join();
        }
    }
}

/// Jobs a worker groups under one blob commit, by bytes read.
const GROUP_BYTES: u64 = 64 << 20;

/// A bounded worker group whose prepared files stay private until the writer
/// stores them.
struct Group {
    waiting: Vec<(Option<PreparedBatch>, Done, u64)>,
    bytes: u64,
}

/// Gives `bytes` back to the in-flight budget and wakes waiting submitters.
fn release(shared: &Shared, bytes: u64) {
    if bytes == 0 {
        return;
    }
    if let Ok(mut q) = shared.queue.lock() {
        q.in_flight -= bytes;
    }
    shared.room.notify_all();
}

fn work(
    shared: &Arc<Shared>,
    writer: &WriterSender,
    store: &Arc<polylogue_store::blobs::Blobs>,
    archive: &polylogue_store::Archive,
    acknowledge: &Sender<PendingGroup>,
) {
    let _worker = polylogue_store::timing::worker();
    let mut group = Group {
        waiting: Vec::new(),
        bytes: 0,
    };
    loop {
        let job = {
            let mut q = shared.queue.lock().expect("pool lock");
            loop {
                if let Some(j) = q.live.pop_front().or_else(|| q.bulk.pop_front()) {
                    polylogue_store::timing::queue(polylogue_store::timing::Queue::Parse, -1);
                    break Some(j);
                }
                if !group.waiting.is_empty() {
                    // Nothing waiting: make the group durable and hand it on
                    // now rather than hold it.
                    break None;
                }
                if q.closed {
                    return;
                }
                q = shared.work.wait(q).expect("pool lock");
            }
        };
        let Some(job) = job else {
            if !flush(&mut group, shared, writer, store, acknowledge) {
                return;
            }
            continue;
        };
        let size = job.bytes();
        let live = job.priority == Priority::Live;
        let (batch, result) = run(job, archive, store, true);
        group.waiting.push((batch, result, size));
        group.bytes += size;
        if (live || group.bytes >= GROUP_BYTES)
            && !flush(&mut group, shared, writer, store, acknowledge)
        {
            return;
        }
    }
}

struct PendingGroup {
    results: polylogue_store::Result<
        Receiver<Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>>,
    >,
    waiting: Vec<Done>,
    queued: Vec<bool>,
}

fn acknowledge_groups(pending: Receiver<PendingGroup>, done: Sender<Done>) {
    for mut group in pending {
        let results = group.results.and_then(|rx| {
            rx.recv()
                .map_err(|_| polylogue_store::StoreError::WriterGone)
        });
        match results {
            Ok(results) => {
                let mut results = results.into_iter();
                for (result, queued) in group.waiting.iter_mut().zip(group.queued) {
                    if queued {
                        match results.next() {
                            Some(Err(polylogue_store::StoreError::PrimaryPending { identity })) => {
                                result.pending = Some(identity);
                            }
                            Some(Err(polylogue_store::StoreError::UnresolvedBasisConflict {
                                custody,
                            })) => {
                                result.raw_custody = Some(custody);
                                result.stored = 0;
                                result.checkpoint = None;
                                result.parser_state = None;
                            }
                            Some(Err(error)) => {
                                result.error = Some(error.to_string());
                                result.write_failed = true;
                                result.stored = 0;
                            }
                            Some(Ok(stored)) => {
                                result.chain = Some(stored.chain);
                            }
                            None => {
                                result.error = Some("the writer returned no result".into());
                                result.write_failed = true;
                                result.stored = 0;
                            }
                        }
                    }
                }
            }
            Err(error) => {
                for result in &mut group.waiting {
                    result.error.get_or_insert_with(|| error.to_string());
                    result.write_failed = true;
                    result.stored = 0;
                }
            }
        }
        for result in group.waiting {
            if done.send(result).is_err() {
                return;
            }
        }
    }
}

/// A job's bytes count against the pool until the writer has stored them.
fn flush(
    group: &mut Group,
    shared: &Arc<Shared>,
    writer: &WriterSender,
    store: &Arc<polylogue_store::blobs::Blobs>,
    acknowledge: &Sender<PendingGroup>,
) -> bool {
    let mut waiting = std::mem::take(&mut group.waiting);
    let queued: Vec<bool> = waiting
        .iter()
        .map(|(batch, _, _)| batch.is_some())
        .collect();
    let mut batches = Vec::new();
    for (batch, _, size) in &mut waiting {
        if let Some(mut batch) = batch.take() {
            let owner = Arc::clone(shared);
            let bytes = *size;
            batch.batch.release = InFlight::new(move || release(&owner, bytes));
            batches.push(batch);
        } else {
            release(shared, *size);
        }
    }
    let results = writer.enqueue_group(batches, Arc::clone(store));
    group.bytes = 0;
    let waiting = waiting.into_iter().map(|(_, result, _)| result).collect();
    acknowledge
        .send(PendingGroup {
            results,
            waiting,
            queued,
        })
        .is_ok()
}

/// The record of a parse that failed: no rows, and the chain's cursor stays
/// where this parse began.
fn failed_step(job: &Job, error: String, prepare: bool) -> ParseStep {
    let mut failed = ParseStep::new(
        Default::default(),
        Default::default(),
        job.start..job.start,
        false,
    );
    failed.reset = false;
    failed.progress = polylogue_model::batch::StepProgress::Failed { error };
    failed.reading_version = job.origin.reading_version();
    failed.stopped_attempt = job.stopped_attempt;
    if prepare {
        failed.prepare();
    }
    failed
}

pub(crate) fn run(
    mut job: Job,
    archive: &polylogue_store::Archive,
    store: &polylogue_store::blobs::Blobs,
    prepare: bool,
) -> (Option<PreparedBatch>, Done) {
    use polylogue_store::timing::{Stage, add, time};
    let mut out = Done {
        chain: None,
        path: job.path.clone(),
        start: job.start,
        end: job.end,
        parsed_upto: job.start,
        parser_state: job.parser_state.take(),
        tail: Vec::new(),
        stored: 0,
        error: None,
        write_failed: false,
        failure_recorded: false,
        continuation_missing: false,
        pending: None,
        checkpoint: None,
        raw_custody: None,
        raw_acquisition: None,
    };
    let started = std::time::Instant::now();
    let mut file_snapshot = None;
    let read = match &job.source {
        Bytes::Import(source) => source
            .lock()
            .expect("import reader lock")
            .read_range(job.start, job.end),
        Bytes::Stored(chain) => chain
            .read_range(store, job.start, job.end)
            .map_err(|e| std::io::Error::other(e.to_string())),
        Bytes::File => match &job.acquisition {
            Some(context) => read::read_native_range(&job.path, job.start, job.end, context.facts)
                .map(|(bytes, observed)| {
                    file_snapshot = Some(observed);
                    bytes
                }),
            None => read::read_range(&job.path, job.start, job.end),
        },
        Bytes::Unloaded { .. } => Err(std::io::Error::other(
            "stored bytes were not loaded before their job",
        )),
    };
    add(Stage::Read, started.elapsed());
    let bytes = match read {
        Ok(b) => b,
        Err(e) => {
            out.error = Some(format!("read {}: {e}", job.path.display()));
            return (None, out);
        }
    };
    out.tail = read::tail_of(&bytes);
    let mut files = Vec::new();
    let new_blob = match job.store_from {
        Some(from) => match time(Stage::Store, || {
            archive.prepare_blob(&bytes[(from - job.start) as usize..])
        }) {
            Ok((p, file)) => {
                files.extend(file);
                out.stored = p.size;
                Some(p)
            }
            Err(e) => {
                out.error = Some(format!("store {}: {e}", job.path.display()));
                return (None, out);
            }
        },
        None => None,
    };
    let acquisition = new_blob
        .as_ref()
        .zip(job.acquisition.as_ref())
        .map(|(blob, context)| {
            let range = polylogue_store::primary::RawRange {
                start: job.store_from.expect("new body has a source interval"),
                end: job.store_from.expect("new body has a source interval") + blob.size,
            };
            match &file_snapshot {
                Some(observed) => context.finish_file(
                    range,
                    blob.hash,
                    job.input.clone(),
                    observed.clone(),
                    polylogue_store::primary::RawStreamRole::ProviderBytes,
                    polylogue_store::primary::RawRange {
                        start: 0,
                        end: observed.size,
                    },
                ),
                None => context.finish(range, blob.hash),
            }
        });
    out.raw_acquisition = acquisition.clone();
    let context_overlay = if let Some(context) = job.parser_context.take() {
        match context.check_source(
            &job.input.id,
            job.parser_chain,
            &job.origin.reading_version(),
            job.start,
        ) {
            Ok(()) => Some(context),
            Err(error) => {
                out.error = Some(error.to_string());
                out.continuation_missing = true;
                None
            }
        }
    } else if job.origin.name == "claude-code"
        && !job.parser_deferred
        && (job.start == 0 || out.parser_state.is_some())
    {
        let chain = job.parser_chain.or(match job.change {
            Change::Reparse { chain } => Some(chain),
            _ => None,
        });
        match polylogue_store::parser_context::Overlay::open(
            archive,
            &job.input.id,
            chain,
            &job.origin.reading_version(),
            job.start,
        ) {
            Ok(context) => Some(context),
            Err(error) => {
                out.error = Some(error.to_string());
                out.continuation_missing = true;
                None
            }
        }
    } else {
        None
    };
    let context = context_overlay
        .as_ref()
        .map(|overlay| polylogue_model::parser_context::Handle::new(overlay.clone()))
        .unwrap_or_default();
    if let Some(chain) = context_overlay
        .as_ref()
        .and_then(|overlay| overlay.replay_chain())
    {
        match rebuild_context(&job, archive, store, chain, &context) {
            Ok(state) => {
                let mut expected = out.parser_state.clone().unwrap_or(Json::Null);
                if let Some(object) = expected.as_object_mut() {
                    object.remove(polylogue_model::parser_context::FIELD);
                }
                if expected != state {
                    out.error = Some(
                        "reconstructed context differs from its admitted parser checkpoint".into(),
                    );
                    out.continuation_missing = true;
                } else {
                    out.parser_state = Some(state);
                }
            }
            Err(error) => {
                out.error = Some(error);
                out.continuation_missing = true;
            }
        }
    }
    let mut extra = Vec::new();
    let parser = (job.origin.open)(&job.path).or_else(|| {
        job.origin
            .is_row_dump(&job.path)
            .then(polylogue_sources::row_dump::open)
    });
    let parse = parser
        .map(|parser| {
            if job.origin.support == polylogue_sources::Support::Parsed
                && (job.parser_deferred || out.continuation_missing || job.start > 0 && out.parser_state.is_none()) {
                out.continuation_missing = true;
                return Vec::new();
            }
            // A parser that panics loses only its own input: the panic
            // becomes this input's failure, and every other input is parsed.
            let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                polylogue_store::fault::check("parse", &job.path.to_string_lossy())
                    .map_err(|e| e.to_string())?;
                time(Stage::Parse, || {
                    parser.steps(out.parser_state.as_ref(), &bytes, job.start, &context)
                })
                .map_err(|e| e.to_string())
            }))
            .unwrap_or_else(|panic| {
                let why = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                Err(format!("the parser panicked: {why}"))
            });
            match step {
                Ok((steps, parser_state)) => {
                    out.parser_state = Some(parser_state);
                    let mut parsed = Vec::new();
                    for mut step in steps {
                        if step.consumed == step.start && !job.at_end {
                            continue;
                        }
                        for blob in std::mem::take(&mut step.output.blobs) {
                            match time(Stage::Store, || archive.prepare_decoded(&blob.bytes)) {
                                Ok((p, file)) => {
                                    if p.hash!=blob.hash || p.size!=blob.size {
                                        let error = "decoded bytes differ from their declared hash or length";
                                        out.error = Some(error.into());
                                        out.failure_recorded = true;
                                        return vec![failed_step(&job, error.into(), prepare)];
                                    }
                                    files.extend(file);
                                    extra.push(p);
                                }
                                Err(e) => out.error = Some(format!("store inline bytes: {e}")),
                            }
                        }
                        out.parsed_upto = job.start + step.consumed;
                        let mut p = ParseStep::new(
                            step.output,
                            step.counts,
                            job.start + step.start..job.start + step.consumed,
                            job.at_end,
                        );
                        p.order_from = parser.is_document().then_some(0);
                        if job.origin.support == polylogue_sources::Support::NoParser {
                            p.progress = polylogue_model::batch::StepProgress::NoParserYet {
                                at_end: job.at_end,
                            };
                        }
                        p.repeats_whole_conversations = step.repeats_whole_conversations;
                        if matches!(p.progress, polylogue_model::batch::StepProgress::Failed { .. } | polylogue_model::batch::StepProgress::Refused { .. }) {
                            p.stopped_attempt = job.stopped_attempt;
                        }
                        if job.origin.is_row_dump(&job.path) {
                            let start = step.start as usize;
                            let end = step.consumed as usize;
                            if let Err(error) = p.compare_records(&bytes[start..end]) {
                                out.error = Some(error.to_string());
                                out.failure_recorded = true;
                                return vec![failed_step(&job, error.to_string(), prepare)];
                            }
                        }
                        // Packing for the writer is store work, not parsing.
                        let packed = std::time::Instant::now();
                        p.reading_version = job.origin.reading_version();
                        if prepare { p.prepare(); }
                        add(Stage::Store, packed.elapsed());
                        parsed.push(p);
                    }
                    if let Some(last) = parsed.last_mut()
                        && !matches!(last.progress, polylogue_model::batch::StepProgress::NoParserYet { .. }) {
                        if let Some(overlay) = &context_overlay {
                            let Some(state) = out.parser_state.as_mut().and_then(Json::as_object_mut) else {
                                let error = "keyed parser continuation has no object state";
                                out.error = Some(error.into());
                                out.failure_recorded = true;
                                return vec![failed_step(&job, error.into(), prepare)];
                            };
                            match overlay.checkpoint_delta(out.parsed_upto).and_then(|delta| serde_json::to_value(delta).map_err(Into::into)) {
                                Ok(delta) => { state.insert(polylogue_model::parser_context::FIELD.into(), delta); }
                                Err(error) => {
                                    out.error = Some(error.to_string());
                                    out.failure_recorded = true;
                                    return vec![failed_step(&job, error.to_string(), prepare)];
                                }
                            }
                        }
                        last.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
                            last.reading_version.clone(), last.parsed_upto, out.parser_state.clone().filter(|state| !state.is_null())));
                    }
                    parsed
                }
                Err(e) => {
                    out.error = Some(format!("parse {}: {e}", job.path.display()));
                    out.failure_recorded = true;
                    vec![failed_step(&job, e, prepare)]
                }
            }
        })
        .unwrap_or_else(|| {
            out.parsed_upto = job.end;
            out.parser_state = None;
            if !crate::has_reading(job.origin) || job.origin.support != polylogue_sources::Support::NoParser {
                return Vec::new();
            }
            let mut step = ParseStep::new(
                Default::default(),
                Default::default(),
                job.start..job.end,
                job.at_end,
            );
            step.progress = polylogue_model::batch::StepProgress::NoParserYet {
                at_end: job.at_end,
            };
            step.reading_version = job.origin.reading_version();
            if prepare {
                step.prepare();
            }
            vec![step]
        });
    if let Bytes::Import(source) = &job.source {
        let upto = if job.origin.support != polylogue_sources::Support::Kept {
            out.parsed_upto
        } else {
            job.end
        };
        source.lock().expect("import reader lock").consume(upto);
    }
    out.checkpoint = parse.last().and_then(|step| step.checkpoint.clone());
    let batch = Batch {
        required_disposition: job
            .acquisition
            .as_ref()
            .and_then(|context| context.required_disposition),
        input: job.input,
        change: job.change,
        observation: job.observation,
        continues: job.continues,
        new_blob,
        blobs: extra,
        length: job.length,
        parse,
        acquisition,
        basis: job
            .acquisition
            .as_ref()
            .and_then(|context| context.basis.clone()),
        release: InFlight::default(),
    };
    (Some(PreparedBatch { batch, files }), out)
}

/// Exceptional reconstruction reads only archived, admitted bytes. Ordinary
/// resume opens the recorded keyed view and never enters this path.
fn rebuild_context(
    job: &Job,
    archive: &polylogue_store::Archive,
    blobs: &polylogue_store::blobs::Blobs,
    chain: i64,
    context: &polylogue_model::parser_context::Handle,
) -> Result<Json, String> {
    let reads = polylogue_store::ReadPool::open(archive).map_err(|e| e.to_string())?;
    let reader = reads.reader().map_err(|e| e.to_string())?;
    let source = reader
        .stored_chain(&job.input.id, chain, job.start)
        .map_err(|e| e.to_string())?;
    let parser = (job.origin.open)(&job.path)
        .ok_or_else(|| "context reconstruction has no provider parser".to_owned())?;
    let mut state = None;
    let mut pending = Vec::new();
    let mut read = 0u64;
    let mut cursor = 0u64;
    while read < job.start {
        let end = (read + 65536).min(job.start);
        pending.extend(
            source
                .read_range(blobs, read, end)
                .map_err(|e| e.to_string())?,
        );
        read = end;
        let (steps, next) = parser
            .steps(state.as_ref(), &pending, cursor, context)
            .map_err(|e| e.to_string())?;
        let consumed = steps.last().map_or(0, |step| step.consumed) as usize;
        if consumed > pending.len() {
            return Err("context reconstruction cursor exceeds its bytes".into());
        }
        pending.drain(..consumed);
        cursor += consumed as u64;
        state = Some(next);
    }
    if cursor != job.start || !pending.is_empty() {
        return Err("context reconstruction did not reach its admitted checkpoint".into());
    }
    Ok(state.unwrap_or(Json::Null))
}

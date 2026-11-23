//! The parse pool: N worker threads, limited by bytes in flight, feeding
//! the one writer. A worker reads a job's byte range, hashes and stores or
//! compresses the new bytes, parses, and adds one batch to its group; it
//! hands the writer the group once the group is large enough or a live job
//! is in it. A whole-chain replacement task is different: the worker parses
//! the stored chain end to end into a reading the writer commits as one
//! unit, after the worker's earlier groups. Live appends go before bulk
//! work. One input has at most one job in flight, so its parser state
//! passes from job to job.

use crate::{EngineError, plan::Plan, read};
use polylogue_model::{Json, LockExt, spawn_named};
use polylogue_sources::OriginSpec;
use polylogue_store::staging::PreparedBatch;
use polylogue_store::writer::ChainReading;
use polylogue_store::writer::WriterSender;
use polylogue_store::{Archive, ReadPool};
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
    pub(crate) source_read: Option<crate::source_read::SourceReadContext>,
    pub continues: bool,
    /// Where parsing starts: after the last complete record parsed, or 0
    /// when parser state is rebuilt by parsing the input again.
    pub start: u64,
    /// Where new bytes start; `None` when the range is only parsed again.
    pub store_from: Option<u64>,
    pub end: u64,
    /// This range reaches the observed source end, not merely a chunk boundary.
    pub at_end: bool,
    /// This range reaches the end of an input that will not grow (an import,
    /// an export member), so its last line is a record without a newline.
    pub finished: bool,
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
    /// of its own. The worker that runs the chain task loads it, as `Import`
    /// when `decoded` (read forward from the start), else as `Stored`.
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
    /// The committed chain the writer returned, never a guess.
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
    /// Bytes the job added to the archive, as the writer reports them.
    pub stored: u64,
    pub error: Option<String>,
    pub cancelled: bool,
    pub write_failed: bool,
    /// The writer refused the batch's content, so the same range is refused
    /// again; any other failed write is retried. When the bytes were stored
    /// before the rows were refused, `chain` names where they are.
    pub content_refused: bool,
    /// The batch carries a step that records the parser's failure.
    pub failure_recorded: bool,
    pub continuation_missing: bool,
    /// The capture is saved and waiting for Sinex's confirmation; neither a
    /// failure nor confirmed.
    pub pending: Option<String>,
    pub checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    pub spooled_group: Option<polylogue_store::primary::SpooledGroup>,
    pub source_read: Option<polylogue_store::primary::AcquisitionReceipt>,
}

pub(crate) enum Task {
    Range(Box<Job>),
    Chain { id: u64, plan: Box<Plan> },
}

pub(crate) enum Finished {
    Range(Box<Done>),
    Chain(Box<ChainDone>),
}

pub(crate) struct ChainDone {
    pub id: u64,
    pub plan: Plan,
    pub written: Result<Option<u64>, crate::EngineError>,
    pub bytes_read: u64,
}

pub(crate) struct ChainSettings {
    pub stopped_attempt: polylogue_model::batch::StoppedAttempt,
    pub read_memory: u64,
    pub yielding: Arc<std::sync::atomic::AtomicBool>,
}

pub(crate) struct Cancelled {
    pub ranges: usize,
    pub chains: Vec<(u64, Plan)>,
}

impl Task {
    fn bytes(&self) -> u64 {
        match self {
            Self::Range(job) => job.bytes(),
            Self::Chain { plan, .. } => plan.size - plan.parse_pos,
        }
    }

    fn priority(&self) -> Priority {
        match self {
            Self::Range(job) => job.priority,
            Self::Chain { plan, .. } => plan.priority,
        }
    }
}

struct Queue {
    live: VecDeque<Task>,
    bulk: VecDeque<Task>,
    chains_running: usize,
    in_flight: u64,
    closed: bool,
    /// Workers waiting for a job or handing their last group to the
    /// writer, the most recently idle last.
    idle: Vec<usize>,
}

/// A new job wakes the worker that went idle last. A worker's allocator
/// keeps what its last job freed for its next one; handing consecutive jobs
/// to one worker reuses that memory, where waking the longest-idle worker
/// would leave a job's worth of freed memory with every worker in turn.
pub(crate) struct Shared {
    queue: Mutex<Queue>,
    /// One per worker, so a job wakes the chosen one.
    wake: Vec<Condvar>,
    room: Condvar,
    limit: u64,
    chains_running_limit: usize,
    /// The archive's stop request: once made, workers start no further job.
    stop: polylogue_store::cancellation::Cancellation,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    acknowledgments: Option<JoinHandle<()>>,
    /// Wakes submitters waiting for room when the stop request is made.
    _stop_wake: polylogue_store::cancellation::CancellationWake,
}

impl Pool {
    pub(crate) fn start(
        workers: usize,
        bytes_in_flight: u64,
        writer: WriterSender,
        blobs: Arc<polylogue_store::blobs::Blobs>,
        archive: polylogue_store::Archive,
        done: Sender<Finished>,
        chains: ChainSettings,
    ) -> Self {
        let workers = workers.max(1);
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                live: VecDeque::new(),
                bulk: VecDeque::new(),
                in_flight: 0,
                chains_running: 0,
                closed: false,
                idle: Vec::with_capacity(workers),
            }),
            wake: (0..workers).map(|_| Condvar::new()).collect(),
            room: Condvar::new(),
            limit: bytes_in_flight.max(1),
            chains_running_limit: workers.saturating_sub(1).max(1),
            stop: archive.stop().clone(),
        });
        let stop_wake = {
            let shared = Arc::downgrade(&shared);
            archive.stop().on_cancel(move || {
                if let Some(shared) = shared.upgrade() {
                    shared.wake_for_stop();
                }
            })
        };
        let (acknowledge, pending) = channel();
        let acknowledged = done.clone();
        let acknowledgments = spawn_named("polylogue-acknowledgments", move || {
            acknowledge_groups(pending, acknowledged)
        })
        .expect("spawn acknowledgment reader");
        let chains = Arc::new(chains);
        let workers = (0..workers)
            .map(|i| {
                let shared = Arc::clone(&shared);
                let writer = writer.clone();
                let blobs = Arc::clone(&blobs);
                let archive = archive.clone();
                let acknowledge = acknowledge.clone();
                let chains = Arc::clone(&chains);
                spawn_named(format!("polylogue-parse-{i}"), move || {
                    polylogue_store::cancellation::under(archive.stop(), || {
                        work(i, &shared, &writer, &blobs, &archive, &acknowledge, &chains);
                    });
                })
                .expect("spawn a parse worker")
            })
            .collect();
        drop(acknowledge);
        Self {
            shared,
            workers,
            acknowledgments: Some(acknowledgments),
            _stop_wake: stop_wake,
        }
    }

    /// Whether submitting a job of `bytes` would return without waiting for
    /// bytes in flight to commit. A requested stop also makes submit return
    /// at once, not queued; this does not report that.
    pub fn has_room(&self, bytes: u64) -> bool {
        let q = self.shared.queue.locked();
        q.in_flight == 0 || q.in_flight + bytes <= self.shared.limit
    }

    /// Queues a job. Blocks while the bytes already in flight are at the
    /// limit; a job larger than the limit runs alone. Once a stop is
    /// requested the job is not queued and this returns false: queued bytes
    /// come back only through [`Self::cancel_queued`], so waiting for them
    /// would never end.
    #[must_use]
    pub fn submit(&self, job: Job) -> bool {
        self.submit_task(Task::Range(Box::new(job)))
    }

    #[must_use]
    pub(crate) fn submit_chain(&self, id: u64, plan: Plan) -> bool {
        self.submit_task(Task::Chain {
            id,
            plan: Box::new(plan),
        })
    }

    fn submit_task(&self, task: Task) -> bool {
        let bytes = task.bytes();
        let mut q = self.shared.queue.locked();
        loop {
            if self.shared.stop.is_cancelled() {
                return false;
            }
            if q.in_flight == 0 || q.in_flight.saturating_add(bytes) <= self.shared.limit {
                break;
            }
            q = self.shared.room.wait(q).expect("pool lock");
        }
        polylogue_store::timing::queue(polylogue_store::timing::Queue::Parse, 1);
        q.in_flight += bytes;
        match task.priority() {
            Priority::Live => q.live.push_back(task),
            Priority::Bulk => q.bulk.push_back(task),
        }
        if let Some(worker) = q.idle.pop() {
            self.shared.wake[worker].notify_one();
        }
        true
    }

    /// Takes back tasks no worker has started. Started tasks finish normally.
    pub(crate) fn cancel_queued(&self) -> Cancelled {
        let mut guard = self.shared.queue.locked();
        let q = &mut *guard;
        let tasks: Vec<Task> = q.live.drain(..).chain(q.bulk.drain(..)).collect();
        q.in_flight -= tasks.iter().map(Task::bytes).sum::<u64>();
        drop(guard);
        self.shared.room.notify_all();
        polylogue_store::timing::queue(
            polylogue_store::timing::Queue::Parse,
            -(tasks.len() as i64),
        );
        let mut cancelled = Cancelled {
            ranges: 0,
            chains: Vec::new(),
        };
        for task in tasks {
            match task {
                Task::Range(_) => cancelled.ranges += 1,
                Task::Chain { id, plan } => cancelled.chains.push((id, *plan)),
            }
        }
        cancelled
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.queue.locked().closed = true;
        for wake in &self.shared.wake {
            wake.notify_all();
        }
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

impl Shared {
    /// Wakes submitters waiting for room so they see a stop request. Taking
    /// the queue lock first means a submitter that checked the request
    /// before it was made is already waiting.
    fn wake_for_stop(&self) {
        drop(self.queue.locked());
        self.room.notify_all();
    }
}

/// Gives `bytes` back to the in-flight budget and wakes waiting submitters.
fn release(shared: &Shared, bytes: u64) {
    if bytes == 0 {
        return;
    }
    shared.queue.locked().in_flight -= bytes;
    shared.room.notify_all();
}

fn work(
    me: usize,
    shared: &Arc<Shared>,
    writer: &WriterSender,
    store: &Arc<polylogue_store::blobs::Blobs>,
    archive: &polylogue_store::Archive,
    acknowledge: &Sender<Pending>,
    chains: &ChainSettings,
) {
    let _worker = polylogue_store::timing::worker();
    let mut group = Group {
        waiting: Vec::new(),
        bytes: 0,
    };
    loop {
        let task = {
            let mut q = shared.queue.locked();
            let mut woken = false;
            loop {
                // After a stop request the queue is left for `cancel_queued`.
                if !shared.stop.is_cancelled()
                    && let Some(task) = q.take(shared.chains_running_limit)
                {
                    polylogue_store::timing::queue(polylogue_store::timing::Queue::Parse, -1);
                    break Some(task);
                }
                if !group.waiting.is_empty() {
                    // Nothing waiting: commit the group and hand it on
                    // now rather than hold it. The next job, which may well
                    // be this input's next range once the group commits, is
                    // this worker's; it looks at the queue again after the
                    // hand-over, so a wake sent meanwhile is not needed.
                    q.idle.retain(|&worker| worker != me);
                    q.idle.push(me);
                    break None;
                }
                if q.closed {
                    return;
                }
                // A worker woken for a job another worker took first goes
                // back as the least recently idle, so the one that ran the
                // last job stays first; a spurious wake leaves it listed.
                q.idle.retain(|&worker| worker != me);
                if woken {
                    q.idle.insert(0, me);
                } else {
                    q.idle.push(me);
                }
                q = shared.wake[me].wait(q).expect("pool lock");
                woken = true;
            }
        };
        let Some(task) = task else {
            if !flush(&mut group, shared, writer, store, acknowledge) {
                return;
            }
            continue;
        };
        match task {
            Task::Range(job) => {
                let size = job.bytes();
                let live = job.priority == Priority::Live;
                let (batch, result) = run(*job, archive, store, true);
                group.waiting.push((batch, result, size));
                group.bytes += size;
                if (live || group.bytes >= GROUP_BYTES)
                    && !flush(&mut group, shared, writer, store, acknowledge)
                {
                    return;
                }
            }
            Task::Chain { id, mut plan } => {
                let bytes = plan.size - plan.parse_pos;
                let charge = {
                    let shared = Arc::clone(shared);
                    InFlight::new(move || release(&shared, bytes))
                };
                // Earlier ranges leave this worker before its whole-chain write.
                if !flush(&mut group, shared, writer, store, acknowledge) {
                    return;
                }
                let mut bytes_read = 0;
                let reading = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_chain(&mut plan, archive, store, chains, shared, &mut bytes_read)
                }))
                .unwrap_or_else(|_| {
                    Err(EngineError::Parse("the replacement parser panicked".into()))
                })
                .and_then(|reading| {
                    if chains.yielding.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(None);
                    }
                    reading
                        .map(|mut reading| {
                            polylogue_store::timing::time(
                                polylogue_store::timing::Stage::Store,
                                || reading.finish(),
                            )?;
                            Ok(reading)
                        })
                        .transpose()
                });
                plan.source = Bytes::File;
                if !plan.tracked {
                    plan.parser_state = None;
                    plan.tail = None;
                }
                let reply = reading.and_then(|reading| {
                    if chains.yielding.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(None);
                    }
                    let Some(reading) = reading else {
                        return Ok(None);
                    };
                    let polylogue_store::Change::Reparse { chain } = plan.change else {
                        return Err(
                            std::io::Error::other("a replacement needs a stored chain").into()
                        );
                    };
                    writer
                        .queue_chain(plan.input.clone(), chain, plan.size, reading)
                        .map(Some)
                        .map_err(EngineError::from)
                });
                drop(charge);
                let sent = acknowledge
                    .send(Pending::Chain {
                        id,
                        plan,
                        reply,
                        bytes_read,
                    })
                    .is_ok();
                shared.queue.locked().chains_running -= 1;
                for wake in &shared.wake {
                    wake.notify_one();
                }
                if !sent {
                    return;
                }
            }
        }
    }
}

impl Queue {
    fn take(&mut self, chain_limit: usize) -> Option<Task> {
        let chains_allowed = self.chains_running < chain_limit;
        let eligible = |task: &Task| matches!(task, Task::Range(_)) || chains_allowed;
        let task = if let Some(index) = self.live.iter().position(eligible) {
            self.live.remove(index)
        } else {
            self.bulk
                .iter()
                .position(eligible)
                .and_then(|index| self.bulk.remove(index))
        }?;
        if matches!(task, Task::Chain { .. }) {
            self.chains_running += 1;
        }
        Some(task)
    }
}

enum Pending {
    Group(PendingGroup),
    Chain {
        id: u64,
        plan: Box<Plan>,
        reply: Result<Option<Receiver<polylogue_store::Result<u64>>>, EngineError>,
        bytes_read: u64,
    },
}

struct PendingGroup {
    results: polylogue_store::Result<
        Receiver<Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>>,
    >,
    waiting: Vec<Done>,
    queued: Vec<bool>,
}

fn acknowledge_groups(pending: Receiver<Pending>, done: Sender<Finished>) {
    for pending in pending {
        let mut group = match pending {
            Pending::Group(group) => group,
            Pending::Chain {
                id,
                plan,
                reply,
                bytes_read,
            } => {
                let written = polylogue_store::fault::check("pool.chain_reply", &plan.input.id)
                    .map_err(EngineError::from)
                    .and(reply)
                    .and_then(|reply| {
                        reply
                            .map(|reply| {
                                reply
                                    .recv()
                                    .map_err(|_| polylogue_store::StoreError::WriterGone)?
                                    .map_err(EngineError::from)
                            })
                            .transpose()
                    });
                if done
                    .send(Finished::Chain(Box::new(ChainDone {
                        id,
                        plan: *plan,
                        written,
                        bytes_read,
                    })))
                    .is_err()
                {
                    return;
                }
                continue;
            }
        };
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
                                spooled,
                            })) => {
                                result.spooled_group = Some(spooled);
                                result.stored = 0;
                                result.checkpoint = None;
                                result.parser_state = None;
                            }
                            Some(Err(polylogue_store::StoreError::Stopped)) => {
                                result.write_failed = true;
                                result.stored = 0;
                            }
                            Some(Err(error)) => {
                                result.error = Some(error.to_string());
                                result.write_failed = true;
                                result.content_refused = error.refuses_content();
                                result.stored = 0;
                                if let polylogue_store::StoreError::RowsRefused { stored, .. } =
                                    &error
                                {
                                    result.chain = Some(stored.chain);
                                }
                            }
                            Some(Ok(stored)) => {
                                result.chain = Some(stored.chain);
                                result.stored = stored.bytes_written;
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
            if done.send(Finished::Range(Box::new(result))).is_err() {
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
    acknowledge: &Sender<Pending>,
) -> bool {
    if group.waiting.is_empty() {
        return true;
    }
    let mut waiting = std::mem::take(&mut group.waiting);
    let queued: Vec<bool> = waiting
        .iter()
        .map(|(batch, _, _)| batch.is_some())
        .collect();
    let mut batches = Vec::new();
    for (batch, _, size) in &mut waiting {
        if let Some(mut batch) = batch.take() {
            let shared = Arc::clone(shared);
            let bytes = *size;
            batch.batch.release = InFlight::new(move || release(&shared, bytes));
            batches.push(batch);
        } else {
            release(shared, *size);
        }
    }
    let results = writer.enqueue_group(batches, Arc::clone(store));
    group.bytes = 0;
    let waiting = waiting.into_iter().map(|(_, result, _)| result).collect();
    acknowledge
        .send(Pending::Group(PendingGroup {
            results,
            waiting,
            queued,
        }))
        .is_ok()
}

fn run_chain(
    plan: &mut Plan,
    archive: &Archive,
    blobs: &Arc<polylogue_store::blobs::Blobs>,
    settings: &ChainSettings,
    shared: &Arc<Shared>,
    bytes_read: &mut u64,
) -> Result<Option<ChainReading>, EngineError> {
    let stop = archive.stop();
    let yielding = &settings.yielding;
    let stopped_attempt = &settings.stopped_attempt;
    if yielding.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(None);
    }
    if let Bytes::Unloaded {
        chain,
        length,
        decoded,
    } = plan.source
    {
        let reads = ReadPool::with_budget(archive, 1, settings.read_memory)?;
        let stored = reads
            .reader()?
            .stored_chain(&plan.input.id, chain, length)?;
        plan.source = if decoded {
            Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                stored,
                Arc::clone(blobs),
            ))))
        } else {
            Bytes::Stored(Arc::new(stored))
        };
    }
    polylogue_store::fault::check("reparse.prepare", &plan.input.id)?;
    let mut reading = ChainReading::new()?;
    if let Some(polylogue_sources::Parse::Export(mut parser)) = (plan.origin.open)(&plan.path) {
        let read = crate::import::exports::parse_export(plan, &mut *parser, blobs, |mut item| {
            // A reading commits whole, so a yield drops what was
            // prepared; the export's own loop observes the stop.
            if yielding.load(std::sync::atomic::Ordering::Acquire) {
                return Err(EngineError::Stopped);
            }
            let mut decoded = Vec::new();
            let mut files = Vec::new();
            for blob in std::mem::take(&mut item.output.blobs) {
                let (new, file) = archive.prepare_blob(&blob.bytes)?;
                if new.hash != blob.hash || new.size != blob.size {
                    return Err(std::io::Error::other(
                        "decoded bytes differ from their declared hash or length",
                    )
                    .into());
                }
                decoded.push(new);
                files.extend(file);
            }
            let (mut step, cursor) = item.into_parse(plan, blobs, stopped_attempt)?;
            polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                reading.push(&mut step)
            })?;
            keep_decoded(
                &mut reading,
                shared,
                PreparedBatch {
                    batch: Batch {
                        required_outcome: None,
                        input: plan.input.clone(),
                        change: plan.change.clone(),
                        observation: None,
                        continues: false,
                        new_blob: None,
                        blobs: decoded,
                        length: plan.size,
                        parse: Vec::new(),
                        acquisition: None,
                        basis: None,
                        release: Default::default(),
                    },
                    files,
                },
            );
            Ok(cursor)
        });
        let read = match read {
            Err(EngineError::Stopped)
                if !stop.is_cancelled() && yielding.load(std::sync::atomic::Ordering::Acquire) =>
            {
                return Ok(None);
            }
            result => result?,
        };
        *bytes_read += read.bytes_read;
        if read.complete {
            plan.parse_pos = plan.size;
        } else {
            plan.failed = true;
            plan.failure_recorded = true;
        }
    } else {
        let context = if (plan.origin.open)(&plan.path).is_some_and(|parser| parser.uses_context())
        {
            Some(polylogue_store::parser_context::Overlay::open(
                archive,
                &plan.input.id,
                plan.parser_chain,
                &plan.origin.reading_version(),
                plan.parse_pos,
            )?)
        } else {
            None
        };
        while !plan.finished() {
            if yielding.load(std::sync::atomic::Ordering::Acquire) {
                return Ok(None);
            }
            stop.check()?;
            let mut job = plan.next_job(stopped_attempt);
            job.parser_context = context.clone();
            let bytes = job.end - job.start;
            *bytes_read += bytes;
            let (batch, done) = run(job, archive, blobs, false);
            if done.cancelled {
                return Err(EngineError::Stopped);
            }
            if done.continuation_missing {
                return Err(EngineError::Parse(done.error.unwrap_or_else(|| {
                    "parser continuation is unavailable at the committed cursor".into()
                })));
            }
            if let Some(error) = &done.error {
                return Err(if error.starts_with("parse ") {
                    EngineError::Parse(error.clone())
                } else {
                    std::io::Error::other(error.clone()).into()
                });
            }
            let Some(mut batch) = batch else {
                return Err(std::io::Error::other("a replacement returned no rows").into());
            };
            for step in &mut batch.batch.parse {
                polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                    reading.push(step)
                })?;
            }
            batch.batch.parse = Vec::new();
            keep_decoded(&mut reading, shared, batch);
            plan.apply(done, true);
        }
    }
    Ok(Some(reading))
}

fn keep_decoded(reading: &mut ChainReading, shared: &Arc<Shared>, mut batch: PreparedBatch) {
    if !batch.batch.blobs.is_empty() {
        // Decoded payloads are files; only their batch and chunk descriptors stay in memory.
        let bytes = (std::mem::size_of::<PreparedBatch>()
            + batch.batch.input.id.capacity()
            + batch.batch.input.format.capacity()
            + batch.batch.blobs.capacity() * std::mem::size_of::<polylogue_store::NewBlob>()
            + batch
                .batch
                .blobs
                .iter()
                .map(|blob| blob.chunks.capacity() * std::mem::size_of::<polylogue_store::Chunk>())
                .sum::<usize>()
            + batch.files.capacity()
                * std::mem::size_of::<polylogue_store::staging::PreparedBytes>())
            as u64;
        shared.queue.locked().in_flight += bytes;
        let shared = Arc::clone(shared);
        batch.batch.release = InFlight::new(move || release(&shared, bytes));
        reading.decoded(batch);
    }
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

/// Reads, stores and parses one job. The archive's stop request ends it
/// between steps and inside each read; a stopped job keeps only the original
/// bytes it already prepared and makes no parser progress.
pub(crate) fn run(
    mut job: Job,
    archive: &polylogue_store::Archive,
    store: &polylogue_store::blobs::Blobs,
    prepare: bool,
) -> (Option<PreparedBatch>, Done) {
    let stop = archive.stop();
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
        cancelled: false,
        write_failed: false,
        content_refused: false,
        failure_recorded: false,
        continuation_missing: false,
        pending: None,
        checkpoint: None,
        spooled_group: None,
        source_read: None,
    };
    if stop.is_cancelled() {
        out.cancelled = true;
        return (None, out);
    }
    let started = std::time::Instant::now();
    let mut file_snapshot = None;
    let read = match &job.source {
        Bytes::Import(source) => source.locked().read_range(job.start, job.end),
        Bytes::Stored(chain) => chain
            .read_range(store, job.start, job.end)
            .map_err(std::io::Error::other),
        Bytes::File => match &job.source_read {
            Some(context) => read::read_source_range(&job.path, job.start, job.end, context.facts)
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
            if e.get_ref()
                .and_then(|e| e.downcast_ref::<polylogue_store::StoreError>())
                .is_some_and(|e| matches!(e, polylogue_store::StoreError::Stopped))
            {
                out.cancelled = true;
                return (None, out);
            }
            out.error = Some(format!("read {}: {e}", job.path.display()));
            return (None, out);
        }
    };
    out.tail = read::tail_of(&bytes);
    let mut files = Vec::new();
    // A document its source saves whole is stored as content-defined chunks,
    // so a revision that changes one place stores only the chunks around it.
    let rewritten_whole = job.store_from == Some(0)
        && !job.continues
        && matches!(job.change, Change::First | Change::Rewritten)
        && !job.input.route.is_imported()
        && crate::plan::whole(job.origin, &job.path);
    let new_blob = match job.store_from {
        Some(from) => match time(Stage::Store, || {
            let new = &bytes[(from - job.start) as usize..];
            if rewritten_whole {
                archive.prepare_chunked(new)
            } else {
                archive
                    .prepare_blob(new)
                    .map(|(blob, file)| (blob, file.into_iter().collect()))
            }
        }) {
            Ok((p, prepared)) => {
                files.extend(prepared);
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
    let source_read = new_blob
        .as_ref()
        .zip(job.source_read.as_ref())
        .map(|(blob, context)| {
            let range = polylogue_store::primary::RawRange {
                start: job.store_from.expect("new body has a source interval"),
                end: job.store_from.expect("new body has a source interval") + blob.size,
            };
            let mut record = match &file_snapshot {
                Some(observed) => context.finish_file(
                    range,
                    blob.hash,
                    job.input.clone(),
                    observed.clone(),
                    polylogue_store::primary::RawStreamRole::ProviderBytes,
                    polylogue_store::primary::RawRange {
                        // The earlier prefix belongs to its confirmed input;
                        // this file snapshot holds the newly read tail.
                        start: match job.change {
                            Change::Appended { from } => from,
                            _ => 0,
                        },
                        end: observed.size,
                    },
                ),
                None => context.finish(range, blob.hash),
            };
            record.chunks = blob.chunks.clone();
            record
        });
    out.source_read = source_read.clone();
    out.cancelled = stop.is_cancelled();
    let context_overlay = if out.cancelled {
        None
    } else if let Some(context) = job.parser_context.take() {
        match context.check_source(
            &job.input.id,
            job.parser_chain,
            &job.origin.reading_version(),
            job.start,
        ) {
            Ok(()) => Some(context),
            Err(polylogue_store::StoreError::Stopped) => {
                out.cancelled = true;
                None
            }
            Err(error) => {
                out.error = Some(error.to_string());
                out.continuation_missing = true;
                None
            }
        }
    } else if job.origin.support == polylogue_sources::Support::Parsed
        && (job.origin.open)(&job.path).is_some_and(|parser| parser.uses_context())
        && !job.parser_deferred
        && (job.start == 0 || out.parser_state.as_ref().is_some_and(Json::is_object))
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
            Err(polylogue_store::StoreError::Stopped) => {
                out.cancelled = true;
                None
            }
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
                        "reconstructed context differs from its committed parser checkpoint".into(),
                    );
                    out.continuation_missing = true;
                } else {
                    out.parser_state = Some(state);
                }
            }
            Err(crate::EngineError::Stopped) => out.cancelled = true,
            Err(_) if context.is_cancelled() => out.cancelled = true,
            Err(error) => {
                out.error = Some(error.to_string());
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
    let mut parse = parser
        .map(|parser| {
            if out.cancelled || stop.is_cancelled() && out.error.is_none() { out.cancelled = true; return Vec::new(); }
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
                    parser.steps(out.parser_state.as_ref(), &bytes, job.start, job.finished, &context)
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
                    if context.is_cancelled() { out.cancelled = true; return Vec::new(); }
                    out.parser_state = Some(parser_state);
                    let mut parsed = Vec::new();
                    for mut step in steps {
                        if stop.is_cancelled() && out.error.is_none() { out.cancelled = true; return Vec::new(); }
                        if step.consumed == step.start && !job.at_end {
                            continue;
                        }
                        for blob in std::mem::take(&mut step.output.blobs) {
                            match time(Stage::Store, || archive.prepare_blob(&blob.bytes)) {
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
                                Err(e) => {
                                    // The step would cite a blob the batch does
                                    // not carry; the bytes stay unparsed, the
                                    // position stays behind, and the parser state
                                    // that read past it is dropped.
                                    out.error = Some(format!("store inline bytes: {e}"));
                                    out.parser_state = None;
                                    return Vec::new();
                                }
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
                            match overlay.checkpoint_delta(out.parsed_upto) {
                                Ok(delta) => {
                                    if let Some(state) = out.parser_state.as_mut().and_then(Json::as_object_mut) {
                                        match serde_json::to_value(delta) {
                                            Ok(delta) => { state.insert(polylogue_model::parser_context::FIELD.into(),delta); }
                                            Err(error) => {
                                                out.error=Some(error.to_string());out.failure_recorded=true;
                                                return vec![failed_step(&job,error.to_string(),prepare)];
                                            }
                                        }
                                    } else if !delta.entries.is_empty() || !delta.cleared_states.is_empty() {
                                        let error="keyed parser continuation has no object state";
                                        out.error=Some(error.into());out.failure_recorded=true;
                                        return vec![failed_step(&job,error.into(),prepare)];
                                    }
                                }
                                Err(polylogue_store::StoreError::Stopped) => {
                                    out.cancelled = true;
                                    return Vec::new();
                                }
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
                    if context.is_cancelled() { out.cancelled = true; return Vec::new(); }
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
            let mut step = ParseStep::without_parser(
                job.start..job.end,
                job.at_end,
                job.origin.reading_version(),
            );
            if prepare {
                step.prepare();
            }
            vec![step]
        });
    if !out.cancelled
        && let Bytes::Import(source) = &job.source
    {
        let upto = if job.origin.support != polylogue_sources::Support::Kept {
            out.parsed_upto
        } else {
            job.end
        };
        source.locked().consume(upto);
    }
    if out.cancelled {
        parse.clear();
        out.parsed_upto = job.start;
        out.parser_state = None;
        out.failure_recorded = false;
        out.continuation_missing = false;
        if new_blob.is_none() && extra.is_empty() {
            return (None, out);
        }
    }
    out.checkpoint = parse.last().and_then(|step| step.checkpoint.clone());
    for state in [
        out.parser_state.as_mut(),
        out.checkpoint
            .as_mut()
            .and_then(|checkpoint| checkpoint.parser_state.as_mut()),
    ]
    .into_iter()
    .flatten()
    {
        crate::parser_checkpoint::without_context_changes(state);
    }
    let batch = Batch {
        required_outcome: job
            .source_read
            .as_ref()
            .and_then(|context| context.required_outcome),
        input: job.input,
        change: job.change,
        observation: job.observation,
        continues: job.continues,
        new_blob,
        blobs: extra,
        length: job.length,
        parse,
        acquisition: source_read,
        basis: job
            .source_read
            .as_ref()
            .and_then(|context| context.basis.clone()),
        release: InFlight::default(),
    };
    (Some(PreparedBatch { batch, files }), out)
}

/// Exceptional reconstruction reads only archived, committed bytes. Normal
/// resume opens the recorded keyed view and never enters this path.
fn rebuild_context(
    job: &Job,
    archive: &polylogue_store::Archive,
    blobs: &polylogue_store::blobs::Blobs,
    chain: i64,
    context: &polylogue_model::parser_context::Handle,
) -> Result<Json, crate::EngineError> {
    let reads = polylogue_store::ReadPool::open(archive)?;
    let reader = reads.reader()?;
    let source = reader.stored_chain(&job.input.id, chain, job.start)?;
    let parser = (job.origin.open)(&job.path).ok_or_else(|| {
        crate::EngineError::Parse("context reconstruction has no provider parser".into())
    })?;
    polylogue_store::fault::check("parse.context_prefix", &job.input.id)?;
    let mut state = None;
    let mut pending = Vec::new();
    let mut read = 0u64;
    let mut cursor = 0u64;
    while read < job.start {
        archive.stop().check()?;
        let end = (read + 65536).min(job.start);
        pending.extend(source.read_range(blobs, read, end)?);
        read = end;
        let (steps, next) = parser
            .steps(state.as_ref(), &pending, cursor, false, context)
            .map_err(|e| crate::EngineError::Parse(e.to_string()))?;
        let consumed = steps.last().map_or(0, |step| step.consumed) as usize;
        if consumed > pending.len() {
            return Err(crate::EngineError::Parse(
                "context reconstruction cursor exceeds its bytes".into(),
            ));
        }
        pending.drain(..consumed);
        cursor += consumed as u64;
        state = Some(next);
    }
    if cursor != job.start || !pending.is_empty() {
        return Err(crate::EngineError::Parse(
            "context reconstruction did not reach its committed checkpoint".into(),
        ));
    }
    Ok(state.unwrap_or(Json::Null))
}

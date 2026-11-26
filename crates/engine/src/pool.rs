//! The parse pool: N worker threads, limited by bytes in flight, feeding
//! the one writer. A worker reads a job's byte range, hashes and stores or
//! compresses the new bytes, parses, and adds one batch to its group; it
//! hands the writer the group once the group is large enough or a live job
//! is in it. A whole-chain replacement task is different: the worker parses
//! the stored chain end to end into a reading the writer commits as one
//! unit, after the worker's earlier groups. A rollout capture prepares one
//! rollout's bytes and hands them to the writer as a group; its answer is
//! awaited with the groups', off the thread that hands out work, which then
//! plans the rollout's reading. Live appends and
//! captures go before bulk work. One input has at most one job in flight, so
//! its parser state passes from job to job.

use crate::{EngineError, plan::Plan, read};
use polylogue_model::{Caused as _, Json, LockExt, spawn_named};
use polylogue_sources::OriginSpec;
use polylogue_sources::chain_names::SettleError;
use polylogue_store::parse::ParseSink as _;
use polylogue_store::staging::PreparedBatch;
use polylogue_store::writer::Replacement;
use polylogue_store::writer::WriterSender;
use polylogue_store::{Archive, ReadPool};
use polylogue_store::{Batch, Change, InFlight, InputKey, Observation, ParsePart};
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
    pub(crate) source_read: Option<crate::capture_read::SourceReadContext>,
    /// Bytes of the file this plan already holds, ending where its stored
    /// bytes end: a file that grows while it is read must still hold them.
    pub(crate) proven_tail: Option<read::ProvenTail>,
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
    pub parser_chain: Option<i64>,
    /// The bytes are stored and not parsed: parsing waits ([`Plan`]'s
    /// `storing_only`).
    pub(crate) store_only: bool,
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
    pub error: Option<EngineError>,
    pub cancelled: bool,
    pub write_failed: bool,
    /// The writer refused the batch for its bytes' content under this
    /// reading version, so the chain fails until either changes; any other
    /// failed write, a stale or inconsistent step among them, leaves the
    /// chain due and it is planned again from the record. When the bytes
    /// were stored before the rows were refused, `chain` names where they
    /// are.
    pub content_refused: bool,
    /// The batch carries a step that records the parser's failure.
    pub failure_recorded: bool,
    /// The chain's committed reading does not stand at this job's start,
    /// or its carried parser state is gone: it is read from its first byte.
    pub restart: bool,
}

pub(crate) enum Task {
    Range(Box<Job>),
    Chain {
        id: u64,
        plan: Box<Plan>,
    },
    /// A rollout's bytes to store; its reading is planned once they are.
    Capture(Box<crate::compressed::RolloutCapture>),
}

pub(crate) enum Finished {
    Range(Box<Done>),
    Chain(Box<ChainDone>),
    Captured(Box<crate::compressed::CapturedRollout>),
}

pub(crate) struct ChainDone {
    pub id: u64,
    pub plan: Plan,
    pub written: Result<Option<u64>, crate::EngineError>,
    pub bytes_read: u64,
}

pub(crate) struct ChainSettings {
    pub read_memory: u64,
    pub yielding: Arc<std::sync::atomic::AtomicBool>,
}

pub(crate) struct Cancelled {
    pub ranges: usize,
    pub chains: Vec<(u64, Plan)>,
    pub captures: Vec<crate::compressed::RolloutCapture>,
}

impl Task {
    fn bytes(&self) -> u64 {
        match self {
            Self::Range(job) => job.bytes(),
            Self::Chain { plan, .. } => plan.size - plan.parse_pos,
            // Stored bytes stream to their staging file; no parsed batch
            // waits on the writer.
            Self::Capture(_) => 0,
        }
    }

    fn priority(&self) -> Priority {
        match self {
            Self::Range(job) => job.priority,
            Self::Chain { plan, .. } => plan.priority,
            // Storing goes ahead of parsing; the feeder bounds how many
            // captures are in flight.
            Self::Capture(_) => Priority::Live,
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
    /// Nanoseconds each worker has spent on a CPU, as of its last task.
    cpu: Arc<Vec<std::sync::atomic::AtomicU64>>,
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
        let cpu: Arc<Vec<std::sync::atomic::AtomicU64>> = Arc::new(
            (0..workers)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
        );
        let workers = (0..workers)
            .map(|i| {
                let cpu = Arc::clone(&cpu);
                let shared = Arc::clone(&shared);
                let writer = writer.clone();
                let blobs = Arc::clone(&blobs);
                let archive = archive.clone();
                let acknowledge = acknowledge.clone();
                let chains = Arc::clone(&chains);
                spawn_named(format!("polylogue-parse-{i}"), move || {
                    polylogue_store::cancellation::under(archive.stop(), || {
                        work(
                            i,
                            &shared,
                            &writer,
                            &blobs,
                            &archive,
                            &acknowledge,
                            &chains,
                            &cpu[i],
                        );
                    });
                })
                .expect("spawn a parse worker")
            })
            .collect();
        drop(acknowledge);
        Self {
            shared,
            cpu,
            workers,
            acknowledgments: Some(acknowledgments),
            _stop_wake: stop_wake,
        }
    }

    /// Seconds each worker has spent on a CPU so far, as of its last task.
    pub(crate) fn worker_cpu_seconds(&self) -> Vec<f64> {
        self.cpu
            .iter()
            .map(|nanos| nanos.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9)
            .collect()
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

    #[must_use]
    pub(crate) fn submit_capture(&self, capture: crate::compressed::RolloutCapture) -> bool {
        self.submit_task(Task::Capture(Box::new(capture)))
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
            captures: Vec::new(),
        };
        for task in tasks {
            match task {
                Task::Range(_) => cancelled.ranges += 1,
                Task::Chain { id, plan } => cancelled.chains.push((id, *plan)),
                Task::Capture(capture) => cancelled.captures.push(*capture),
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

#[allow(clippy::too_many_arguments)]
fn work(
    me: usize,
    shared: &Arc<Shared>,
    writer: &WriterSender,
    store: &Arc<polylogue_store::blobs::Blobs>,
    archive: &polylogue_store::Archive,
    acknowledge: &Sender<Pending>,
    chains: &ChainSettings,
    cpu: &std::sync::atomic::AtomicU64,
) {
    let _worker = polylogue_store::timing::worker();
    let started = polylogue_store::timing::thread_cpu_nanos();
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
        if let (Some(started), Some(now)) = (started, polylogue_store::timing::thread_cpu_nanos()) {
            cpu.store(
                now.saturating_sub(started),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let Some(task) = task else {
            if !flush(&mut group, shared, writer, acknowledge) {
                return;
            }
            continue;
        };
        let _doing = match &task {
            Task::Range(job) => polylogue_store::activity::doing_on(
                "parsing a file",
                job.path.display().to_string(),
            ),
            Task::Chain { plan, .. } => {
                polylogue_store::activity::doing_on("parsing a chain again", plan.input.id.clone())
            }
            Task::Capture(capture) => polylogue_store::activity::doing_on(
                "storing a captured rollout",
                capture.path.display().to_string(),
            ),
        };
        match task {
            Task::Range(job) => {
                let size = job.bytes();
                let live = job.priority == Priority::Live;
                let (batch, result) = run(*job, archive, store, true, None);
                group.waiting.push((batch, result, size));
                group.bytes += size;
                if (live || group.bytes >= GROUP_BYTES)
                    && !flush(&mut group, shared, writer, acknowledge)
                {
                    return;
                }
            }
            Task::Capture(capture) => {
                // Earlier ranges go to the writer before this worker waits on it.
                if !flush(&mut group, shared, writer, acknowledge) {
                    return;
                }
                let captured = capture.run(archive, store, writer, chains.read_memory);
                if acknowledge
                    .send(Pending::Captured(Box::new(captured)))
                    .is_err()
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
                if !flush(&mut group, shared, writer, acknowledge) {
                    return;
                }
                let mut bytes_read = 0;
                let reading = polylogue_model::caught(|| {
                    run_chain(
                        &mut plan,
                        archive,
                        store,
                        chains,
                        shared,
                        writer,
                        &mut bytes_read,
                    )
                })
                .unwrap_or_else(|panic| Err(polylogue_model::ChainFailure::from(panic).into()));
                plan.source = Bytes::File;
                if !plan.tracked {
                    plan.parser_state = None;
                    plan.tail = None;
                }
                // A replacement that is not ended is dropped: nothing of it
                // reaches the index.
                let reply = reading.and_then(|reading| {
                    if chains.yielding.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(None);
                    }
                    let Some((reading, committed)) = reading else {
                        return Ok(None);
                    };
                    polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                        Box::new(reading).end()
                    })?;
                    Ok(Some(committed))
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
    Captured(Box<crate::compressed::CapturedRollout>),
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
            Pending::Captured(mut captured) => {
                captured.answered();
                if done.send(Finished::Captured(captured)).is_err() {
                    return;
                }
                continue;
            }
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
                            Some(Err(polylogue_store::StoreError::Stopped)) => {
                                result.write_failed = true;
                                result.stored = 0;
                            }
                            Some(Err(error)) => {
                                result.write_failed = true;
                                result.content_refused =
                                    error.cause() == polylogue_model::Cause::Content;
                                result.stored = 0;
                                if let polylogue_store::StoreError::RowsRefused { stored, .. } =
                                    &error
                                {
                                    result.chain = Some(stored.chain);
                                    result.stored = stored.bytes_written;
                                }
                                result.error = Some(error.into());
                            }
                            Some(Ok(stored)) => {
                                result.chain = Some(stored.chain);
                                result.stored = stored.bytes_written;
                            }
                            None => {
                                result.error =
                                    Some(EngineError::internal("the writer returned no result"));
                                result.write_failed = true;
                                result.stored = 0;
                            }
                        }
                    }
                }
            }
            Err(error) => {
                for result in &mut group.waiting {
                    if result.error.is_none() {
                        result.error = Some(
                            polylogue_store::StoreError::Failed(polylogue_model::Failure::of(
                                &error,
                            ))
                            .into(),
                        );
                    }
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
    let results = writer.enqueue_group(batches);
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

/// Parses a stored chain end to end into a reading staged for one
/// replacement transaction. Decoded bytes stay with the reading, counted
/// against the pool's bytes in flight.
fn run_chain(
    plan: &mut Plan,
    archive: &Archive,
    blobs: &Arc<polylogue_store::blobs::Blobs>,
    settings: &ChainSettings,
    shared: &Arc<Shared>,
    writer: &WriterSender,
    bytes_read: &mut u64,
) -> Result<Option<(Replacement, Receiver<polylogue_store::Result<u64>>)>, EngineError> {
    let polylogue_store::Change::Reparse { chain } = plan.change else {
        return Err(EngineError::internal("a replacement needs a stored chain"));
    };
    let (mut reading, committed) = Replacement::of(writer, plan.input.clone(), chain, plan.size)?;
    let read = read_chain(
        plan,
        archive,
        blobs,
        settings,
        bytes_read,
        &mut reading,
        |reading, batch| {
            keep_decoded(reading, shared, batch);
            Ok(())
        },
    )?;
    Ok(read.then_some((reading, committed)))
}

/// Parses a stored chain end to end into `sink`, step by step; decoded
/// bytes go to `keep` before the steps that name them. False when the read
/// yields to a request before the chain's end.
pub(crate) fn read_chain<S: polylogue_store::parse::ParseSink>(
    plan: &mut Plan,
    archive: &Archive,
    blobs: &Arc<polylogue_store::blobs::Blobs>,
    settings: &ChainSettings,
    bytes_read: &mut u64,
    sink: &mut S,
    mut keep: impl FnMut(&mut S, PreparedBatch) -> polylogue_store::Result<()>,
) -> Result<bool, EngineError> {
    let stop = archive.stop();
    let yielding = &settings.yielding;
    if yielding.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(false);
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
                    return Err(polylogue_model::ChainFailure::bytes_differ().into());
                }
                decoded.push(new);
                files.extend(file);
            }
            let (step, cursor) = item.into_parse(plan, blobs)?;
            keep(
                sink,
                PreparedBatch {
                    batch: Batch {
                        input: plan.input.clone(),
                        change: plan.change.clone(),
                        observation: None,
                        continues: false,
                        new_blob: None,
                        blobs: decoded,
                        length: plan.size,
                        parse: Vec::new(),
                        capture_read: None,
                        release: Default::default(),
                    },
                    files,
                },
            )?;
            polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                sink.part(step)
            })?;
            Ok(cursor)
        });
        let read = match read {
            Err(EngineError::Stopped)
                if !stop.is_cancelled() && yielding.load(std::sync::atomic::Ordering::Acquire) =>
            {
                return Ok(false);
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
        let mut names = None;
        while !plan.finished() {
            if yielding.load(std::sync::atomic::Ordering::Acquire) {
                return Ok(false);
            }
            stop.check()?;
            let job = plan.next_job();
            if names.is_none() {
                names = crate::chain_names::open(
                    archive,
                    &plan.input.id,
                    plan.parser_chain,
                    &plan.origin.parse_version(),
                    job.start,
                )?;
            }
            let bytes = job.end - job.start;
            *bytes_read += bytes;
            let (batch, done) = run(job, archive, blobs, false, names.as_mut());
            if done.cancelled {
                return Err(EngineError::Stopped);
            }
            if done.restart {
                // Nothing was staged yet: the reading starts again at byte 0.
                names = None;
                plan.apply(done, true);
                continue;
            }
            if let Some(error) = done.error {
                return Err(error);
            }
            let Some(mut batch) = batch else {
                return Err(EngineError::internal("a replacement returned no rows"));
            };
            let steps = std::mem::take(&mut batch.batch.parse);
            keep(sink, batch)?;
            for step in steps {
                polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                    sink.part(step)
                })?;
            }
            plan.apply(done, true);
        }
    }
    Ok(true)
}

fn keep_decoded(reading: &mut Replacement, shared: &Arc<Shared>, mut batch: PreparedBatch) {
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

/// A job whose chain's bytes cannot be read: the failure is its error, and
/// its one step records it as the chain's parse state.
fn failed(
    out: &mut Done,
    job: &Job,
    prepare: bool,
    failure: polylogue_model::ChainFailure,
) -> Vec<ParsePart> {
    let step = failed_step(job, failure.clone(), prepare);
    out.error = Some(failure.into());
    out.failure_recorded = true;
    vec![step]
}

/// The record of a parse that failed: no rows, and the chain's cursor stays
/// where this parse began.
fn failed_step(job: &Job, failure: polylogue_model::ChainFailure, prepare: bool) -> ParsePart {
    let mut failed = ParsePart::failed(job.start..job.start, failure);
    failed.reset = false;
    failed.parse_version = job.origin.parse_version();
    if prepare {
        failed.prepare();
    }
    failed
}

/// Reads, stores and parses one job. The archive's stop request ends it
/// between steps and inside each read; a stopped job keeps only the original
/// bytes it already prepared and makes no parser progress.
/// `names` settles the chain-local names of a run of jobs over one chain;
/// a job on its own (`None`) settles against the chain's committed rows.
pub(crate) fn run(
    mut job: Job,
    archive: &polylogue_store::Archive,
    store: &polylogue_store::blobs::Blobs,
    prepare: bool,
    names: Option<&mut crate::chain_names::Names>,
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
        restart: false,
    };
    if stop.is_cancelled() {
        out.cancelled = true;
        return (None, out);
    }
    let started = std::time::Instant::now();
    let mut file_snapshot = None;
    let read: Result<_, EngineError> = match &job.source {
        Bytes::Import(source) => source
            .locked()
            .read_range(job.start, job.end)
            .map_err(EngineError::from),
        Bytes::Stored(chain) => chain
            .read_range(store, job.start, job.end)
            .map_err(EngineError::from),
        Bytes::File => match &job.source_read {
            Some(context) => read::read_source_range(
                &job.path,
                job.start,
                job.end,
                context.facts,
                job.proven_tail.as_ref(),
            )
            .map(|(bytes, observed)| {
                file_snapshot = Some(observed);
                bytes
            })
            .map_err(EngineError::from),
            None => read::read_range(&job.path, job.start, job.end).map_err(EngineError::from),
        },
        Bytes::Unloaded { .. } => Err(EngineError::internal(
            "stored bytes were not loaded before their job",
        )),
    };
    add(Stage::Read, started.elapsed());
    let bytes = match read {
        Ok(b) => b,
        Err(e) => {
            if e.cause() == polylogue_model::Cause::Stopping {
                out.cancelled = true;
                return (None, out);
            }
            out.error = Some(e);
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
                out.error = Some(e.into());
                return (None, out);
            }
        },
        None => None,
    };
    let source_read = new_blob
        .as_ref()
        .zip(job.source_read.as_ref())
        .map(|(blob, context)| {
            let range = polylogue_store::capture_read::RawRange {
                start: job.store_from.expect("new body has a source interval"),
                end: job.store_from.expect("new body has a source interval") + blob.size,
            };
            let mut record = match &file_snapshot {
                Some(observed) => context.finish_file(
                    range,
                    blob.hash,
                    job.input.clone(),
                    observed.clone(),
                    polylogue_store::capture_read::RawStreamRole::ProviderBytes,
                    polylogue_store::capture_read::RawRange {
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
    out.cancelled = stop.is_cancelled();
    let parsed = job.origin.support == polylogue_sources::Support::Parsed;
    let mut own_names;
    let names = match names {
        Some(names) => Some(names),
        None if out.cancelled || job.store_only || !parsed => None,
        None => match crate::chain_names::open(
            archive,
            &job.input.id,
            job.parser_chain.or(match job.change {
                Change::Reparse { chain } => Some(chain),
                _ => None,
            }),
            &job.origin.parse_version(),
            job.start,
        ) {
            Ok(names) => {
                own_names = names;
                own_names.as_mut()
            }
            Err(polylogue_store::StoreError::Stopped) => {
                out.cancelled = true;
                None
            }
            Err(error) => {
                out.error = Some(error.into());
                return (None, out);
            }
        },
    };
    if parsed
        && !out.cancelled
        && !job.store_only
        && (names.is_none() || job.start > 0 && out.parser_state.is_none())
    {
        out.restart = true;
    }
    let mut names = names;
    let mut extra = Vec::new();
    let parser = (job.origin.open)(&job.path).or_else(|| {
        job.origin
            .is_row_dump(&job.path)
            .then(polylogue_sources::row_dump::open)
    });
    let mut parse = parser
        .map(|parser| {
            if out.cancelled || stop.is_cancelled() && out.error.is_none() {
                out.cancelled = true;
                return Vec::new();
            }
            if parsed && (job.store_only || out.restart) {
                return Vec::new();
            }
            // A parser that panics loses only its own input: the panic
            // becomes this input's failure, and every other input is parsed.
            let step = polylogue_model::caught(|| {
                polylogue_store::fault::check("parse", &job.path.to_string_lossy())?;
                Ok(time(Stage::Parse, || {
                    parser.steps(out.parser_state.as_ref(), &bytes, job.start, job.finished)
                })
                .map_err(polylogue_model::ChainFailure::from))
            });
            let step = match step {
                Ok(Ok(step)) => step,
                // An injected failure stands for a step that could not
                // complete, never for the bytes: the position stays behind
                // and the chain is read again.
                Ok(Err(error)) => {
                    let error: polylogue_store::StoreError = error;
                    out.error = Some(error.into());
                    out.parser_state = None;
                    return Vec::new();
                }
                Err(panic) => Err(panic.into()),
            };
            match step {
                Ok((steps, parser_state)) => {
                    out.parser_state = Some(parser_state);
                    let mut parsed = Vec::new();
                    for mut step in steps {
                        if stop.is_cancelled() && out.error.is_none() {
                            out.cancelled = true;
                            return Vec::new();
                        }
                        if step.consumed == step.start && !job.at_end {
                            continue;
                        }
                        for blob in std::mem::take(&mut step.output.blobs) {
                            match time(Stage::Store, || archive.prepare_blob(&blob.bytes)) {
                                Ok((p, file)) => {
                                    if p.hash != blob.hash || p.size != blob.size {
                                        return failed(
                                            &mut out,
                                            &job,
                                            prepare,
                                            polylogue_model::ChainFailure::bytes_differ(),
                                        );
                                    }
                                    files.extend(file);
                                    extra.push(p);
                                }
                                Err(e) => {
                                    // The step would cite a blob the batch does
                                    // not carry; the bytes stay unparsed, the
                                    // position stays behind, and the parser state
                                    // that read past it is dropped.
                                    out.error = Some(e.into());
                                    out.parser_state = None;
                                    return Vec::new();
                                }
                            }
                        }
                        out.parsed_upto = job.start + step.consumed;
                        // Names that belong to the chain are settled from its
                        // rows; a frame restating whole conversations starts
                        // from none.
                        let settled = match (&mut names, step.repeats_whole_conversations) {
                            (Some(names), false) => names.settle(&mut step.output),
                            _ => polylogue_sources::chain_names::settle(
                                &mut step.output,
                                &mut polylogue_sources::chain_names::NoEarlierRows,
                            )
                            .map(|_| ())
                            .map_err(|error| SettleError::Content(error.problem())),
                        };
                        match settled {
                            Ok(()) => {}
                            Err(SettleError::Content(problem)) => {
                                return failed(
                                    &mut out,
                                    &job,
                                    prepare,
                                    polylogue_model::ChainFailure::malformed(problem),
                                );
                            }
                            // The chain's earlier rows could not be read: the
                            // bytes stay unparsed and are read again.
                            Err(SettleError::Lookup(error)) => {
                                out.error = Some(error.into());
                                out.parser_state = None;
                                return Vec::new();
                            }
                        }
                        let mut p = ParsePart::new(
                            step.output,
                            step.counts,
                            job.start + step.start..job.start + step.consumed,
                            job.at_end,
                        );
                        p.order_from = parser.is_document().then_some(0);
                        p.repeats_whole_conversations = step.repeats_whole_conversations;
                        if job.origin.is_row_dump(&job.path) {
                            let start = step.start as usize;
                            let end = step.consumed as usize;
                            if let Err(error) = p.compare_records(&bytes[start..end]) {
                                return failed(
                                    &mut out,
                                    &job,
                                    prepare,
                                    polylogue_model::ChainFailure::malformed(error.to_string()),
                                );
                            }
                        }
                        // Packing for the writer is store work, not parsing.
                        let packed = std::time::Instant::now();
                        p.parse_version = job.origin.parse_version();
                        if prepare {
                            p.prepare();
                        }
                        add(Stage::Store, packed.elapsed());
                        parsed.push(p);
                    }
                    if let Some(last) = parsed.last_mut() {
                        // A finished input never grows, so nothing continues it.
                        if !job.finished {
                            last.resume = Some(polylogue_model::chain::ParserState(
                                out.parser_state.clone().filter(|state| !state.is_null()),
                            ));
                        }
                    }
                    parsed
                }
                Err(failure) => failed(&mut out, &job, prepare, failure),
            }
        })
        .unwrap_or_else(|| {
            out.parsed_upto = job.end;
            out.parser_state = None;
            if !crate::has_reading(job.origin)
                || job.origin.support != polylogue_sources::Support::NoParser
            {
                return Vec::new();
            }
            // A row-dump origin's file that is not a row dump: its bytes
            // are kept, and its parse holds no records.
            let mut step = ParsePart::new(
                Default::default(),
                Default::default(),
                job.start..job.end,
                job.at_end,
            );
            step.parse_version = job.origin.parse_version();
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
        out.restart = false;
        if new_blob.is_none() && extra.is_empty() {
            return (None, out);
        }
    }
    let batch = Batch {
        input: job.input,
        change: job.change,
        observation: job.observation,
        continues: job.continues,
        new_blob,
        blobs: extra,
        length: job.length,
        parse,
        capture_read: source_read,
        release: InFlight::default(),
    };
    (Some(PreparedBatch { batch, files }), out)
}

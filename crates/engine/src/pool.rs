//! The parse pool: N worker threads, admitted by bytes in flight, feeding
//! the one writer (architecture 2, 5.3). A worker reads its byte range,
//! hashes and stores or compresses the new bytes, parses, and hands the
//! writer one batch. Live appends go before bulk work. One input has at
//! most one job in flight, so its parser state passes from job to job.

use crate::read;
use polylogue_model::{BlobHash, Json};
use polylogue_sources::OriginSpec;
use polylogue_store::retention::{Extraction, PreparedBatch};
use polylogue_store::writer::WriterSender;
use polylogue_store::{Batch, Change, InFlight, InputKey, Observation, ParseStep};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
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
    pub continues: bool,
    /// Where parsing starts: after the last complete record parsed, or 0
    /// when parser state is rebuilt by parsing the input again.
    pub start: u64,
    /// Where new bytes start; `None` when the range is only parsed again.
    pub piece_from: Option<u64>,
    pub end: u64,
    /// This range reaches the observed source end, not merely a chunk boundary.
    pub at_end: bool,
    /// The input's stored length once this job is written.
    pub length: u64,
    pub carry: Option<Json>,
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
    /// One completely acquired import, retained before parsing begins.
    Import(Arc<Mutex<read::Imported>>),
    /// What the archive stored for the input: its file changed, moved or
    /// went, or the index is being built again.
    Stored(Arc<polylogue_store::blobs::StoredChain>),
}

impl Job {
    fn bytes(&self) -> u64 {
        self.end - self.start
    }
}

/// What a finished job tells the loop.
#[derive(Debug)]
pub struct Done {
    pub path: PathBuf,
    pub start: u64,
    pub end: u64,
    /// Offset after the last complete record parsed; the next job of this
    /// input starts here.
    pub parsed_upto: u64,
    pub carry: Option<Json>,
    /// The last bytes read, kept to check the next growth.
    pub tail: Vec<u8>,
    pub stored: u64,
    pub error: Option<String>,
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
}

impl Pool {
    pub fn start(
        workers: usize,
        bytes_in_flight: u64,
        writer: WriterSender,
        blobs: Arc<hashed_blobs::Store>,
        archive: polylogue_store::Archive,
        roots: Vec<PathBuf>,
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
        let workers = (0..workers.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                let writer = writer.clone();
                let blobs = Arc::clone(&blobs);
                let done = done.clone();
                let archive = archive.clone();
                let roots = roots.clone();
                std::thread::Builder::new()
                    .name(format!("polylogue-parse-{i}"))
                    .spawn(move || work(&shared, &writer, &blobs, &archive, &roots, &done))
                    .expect("spawn a parse worker")
            })
            .collect();
        Self { shared, workers }
    }

    /// Queues a job. Blocks while the bytes already in flight are at the
    /// limit; a job larger than the limit is admitted alone.
    pub fn submit(&self, job: Job) {
        let mut q = self.shared.queue.lock().expect("pool lock");
        while q.in_flight > 0 && q.in_flight + job.bytes() > self.shared.limit {
            q = self.shared.room.wait(q).expect("pool lock");
        }
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
    }
}

/// Jobs a worker groups under one blob commit, by bytes read.
const GROUP_BYTES: u64 = 64 << 20;

/// A bounded worker group whose file stages remain private until Writer admission.
struct Group {
    held: Vec<(Option<PreparedBatch>, Done, u64)>,
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
    store: &Arc<hashed_blobs::Store>,
    archive: &polylogue_store::Archive,
    roots: &[PathBuf],
    done: &Sender<Done>,
) {
    let mut group = Group {
        held: Vec::new(),
        bytes: 0,
    };
    loop {
        let job = {
            let mut q = shared.queue.lock().expect("pool lock");
            loop {
                if let Some(j) = q.live.pop_front().or_else(|| q.bulk.pop_front()) {
                    break Some(j);
                }
                if !group.held.is_empty() {
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
            if !flush(&mut group, shared, writer, store, done) {
                return;
            }
            continue;
        };
        let size = job.bytes();
        let live = job.priority == Priority::Live;
        let (batch, result) = run(job, archive, roots, store);
        group.held.push((batch, result, size));
        group.bytes += size;
        if (live || group.bytes >= GROUP_BYTES) && !flush(&mut group, shared, writer, store, done) {
            return;
        }
    }
}

/// Retention acknowledgements keep each job charged until the Writer commits.
fn flush(
    group: &mut Group,
    shared: &Arc<Shared>,
    writer: &WriterSender,
    store: &Arc<hashed_blobs::Store>,
    done: &Sender<Done>,
) -> bool {
    let mut held = std::mem::take(&mut group.held);
    let durable = held.iter().any(|(batch, _, _)| {
        batch
            .as_ref()
            .is_some_and(|b| !b.files.is_empty() || !b.extracted.is_empty())
    });
    if durable {
        let queued: Vec<bool> = held.iter().map(|(batch, _, _)| batch.is_some()).collect();
        let mut batches = Vec::new();
        for (batch, _, size) in &mut held {
            if let Some(mut batch) = batch.take() {
                let owner = Arc::clone(shared);
                let bytes = *size;
                batch.batch.release = InFlight::new(move || release(&owner, bytes));
                batches.push(batch);
            } else {
                release(shared, *size);
            }
        }
        let results = writer.retain_group(batches, Arc::clone(store));
        match results {
            Ok(results) => {
                let mut results = results.into_iter();
                for ((_, result, _), queued) in held.iter_mut().zip(queued) {
                    if queued {
                        match results.next() {
                            Some(Err(error)) => result.error = Some(error.to_string()),
                            Some(Ok(_)) => {}
                            None => {
                                result.error = Some("retention group returned no result".into())
                            }
                        }
                    }
                }
            }
            Err(error) => {
                for (_, result, _) in &mut held {
                    result.error.get_or_insert_with(|| error.to_string());
                }
            }
        }
    } else {
        for (batch, result, size) in &mut held {
            if let Some(mut batch) = batch.take() {
                let owner = Arc::clone(shared);
                let bytes = *size;
                batch.batch.release = InFlight::new(move || release(&owner, bytes));
                if let Err(error) = writer.send(batch.batch) {
                    result.error = Some(error.to_string());
                }
            } else {
                release(shared, *size);
            }
        }
    }
    group.bytes = 0;
    held.into_iter()
        .all(|(_, result, _)| done.send(result).is_ok())
}

fn run(
    mut job: Job,
    archive: &polylogue_store::Archive,
    roots: &[PathBuf],
    store: &hashed_blobs::Store,
) -> (Option<PreparedBatch>, Done) {
    use polylogue_store::timing::{Stage, add, time};
    let mut out = Done {
        path: job.path.clone(),
        start: job.start,
        end: job.end,
        parsed_upto: job.start,
        carry: job.carry.take(),
        tail: Vec::new(),
        stored: 0,
        error: None,
    };
    let started = std::time::Instant::now();
    let read = match &job.source {
        Bytes::Import(source) => source
            .lock()
            .expect("import reader lock")
            .read_range(job.start, job.end),
        Bytes::Stored(chain) => chain
            .read_range(store, job.start, job.end)
            .map_err(|e| std::io::Error::other(e.to_string())),
        Bytes::File => read::read_range(&job.path, job.start, job.end),
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
    let mut extracted = Vec::new();
    let piece = match job.piece_from {
        Some(from) => match time(Stage::Store, || {
            archive.prepare_piece(roots, &bytes[(from - job.start) as usize..])
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
    let mut extra = Vec::new();
    let parse = (job.origin.open)(&job.path)
        .map(|parser| {
            // A parser that panics loses only its own input (law 63): the panic
            // becomes this input's failure, and every other input is parsed.
            let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                polylogue_store::fault::check("parse", &job.path.to_string_lossy())
                    .map_err(|e| e.to_string())?;
                time(Stage::Parse, || {
                    parser.steps(out.carry.as_ref(), &bytes, job.start)
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
                Ok((steps, carry)) => {
                    out.carry = Some(carry);
                    let mut parsed = Vec::new();
                    for mut step in steps {
                        let range = job.start + step.start..job.start + step.consumed;
                        let source_hash =
                            BlobHash::of(&bytes[step.start as usize..step.consumed as usize]);
                        for blob in std::mem::take(&mut step.output.blobs) {
                            match time(Stage::Store, || archive.prepare_piece(roots, &blob.bytes)) {
                                Ok((p, file)) => {
                                    if p.hash!=blob.hash || p.size!=blob.size {
                                        out.error=Some("decoded bytes differ from their declared hash or length".into());
                                        return Vec::new();
                                    }
                                    files.extend(file);
                                    extracted.push(Extraction {
                                        range: range.clone(),
                                        source_hash,
                                        blob: p.hash,
                                    });
                                    extra.push(p);
                                }
                                Err(e) => out.error = Some(format!("store inline bytes: {e}")),
                            }
                        }
                        out.parsed_upto = job.start + step.consumed;
                        let mut p = ParseStep::new(
                            step.output,
                            step.accounting,
                            job.start + step.start..job.start + step.consumed,
                            job.at_end,
                        );
                        p.order_from = parser.is_document().then_some(0);
                        p.snapshot = step.restates;
                        // Packing for the writer is store work, not parsing.
                        let packed = std::time::Instant::now();
                        p.prepare();
                        add(Stage::Store, packed.elapsed());
                        parsed.push(p);
                    }
                    parsed
                }
                Err(e) => {
                    out.error = Some(format!("parse {}: {e}", job.path.display()));
                    Vec::new()
                }
            }
        })
        .unwrap_or_default();
    if let Bytes::Import(source) = &job.source {
        let upto = if job.origin.support == polylogue_sources::Support::Parsed {
            out.parsed_upto
        } else {
            job.end
        };
        source.lock().expect("import reader lock").consume(upto);
    }
    let batch = Batch {
        input: job.input,
        change: job.change,
        observation: job.observation,
        continues: job.continues,
        piece,
        blobs: extra,
        length: job.length,
        parse,
        release: InFlight::default(),
    };
    (
        Some(PreparedBatch {
            batch,
            files,
            extracted,
        }),
        out,
    )
}

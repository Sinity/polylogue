//! The parse pool: N worker threads, admitted by bytes in flight, feeding
//! the one writer (architecture 2, 5.3). A worker reads its byte range,
//! hashes and stores or compresses the new bytes, parses, and hands the
//! writer one batch. Live appends go before bulk work. One input has at
//! most one job in flight, so its parser state passes from job to job.

use crate::stage;
use polylogue_model::Json;
use polylogue_sources::OriginSpec;
use polylogue_store::writer::WriterSender;
use polylogue_store::{Batch, Change, InputKey, Observation, ParseStep, Piece};
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
    /// The input's stored length once this job is written.
    pub length: u64,
    pub carry: Option<Json>,
    pub priority: Priority,
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
                std::thread::Builder::new()
                    .name(format!("polylogue-parse-{i}"))
                    .spawn(move || work(&shared, &writer, &blobs, &done))
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

/// A worker's finished jobs whose large blobs are not yet durable.
struct Group<'s> {
    blobs: hashed_blobs::Batch<'s>,
    held: Vec<(Option<Batch>, Done)>,
    bytes: u64,
}

fn work(shared: &Shared, writer: &WriterSender, store: &hashed_blobs::Store, done: &Sender<Done>) {
    let mut group = Group {
        blobs: store.batch(hashed_blobs::Durability::Filesystem),
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
        let (batch, result) = run(job, &mut group.blobs);
        group.held.push((batch, result));
        group.bytes += size;
        if (live || group.bytes >= GROUP_BYTES) && !flush(&mut group, shared, writer, store, done) {
            return;
        }
    }
}

/// Commits the group's blobs (two filesystem syncs for all of them), then
/// sends its batches to the writer and its results to the loop. Returns
/// false when the loop has gone away.
fn flush<'s>(
    group: &mut Group<'s>,
    shared: &Shared,
    writer: &WriterSender,
    store: &'s hashed_blobs::Store,
    done: &Sender<Done>,
) -> bool {
    let blobs = std::mem::replace(&mut group.blobs, store.batch(hashed_blobs::Durability::Filesystem));
    let committed = if blobs.is_empty() {
        Ok(())
    } else {
        blobs.commit().map(drop)
    };
    for (batch, mut result) in group.held.drain(..) {
        match (&committed, batch) {
            (Err(e), _) => result.error = Some(format!("store blobs: {e}")),
            (Ok(()), Some(mut batch)) => {
                let settled = batch
                    .piece
                    .iter_mut()
                    .chain(batch.blobs.iter_mut())
                    .try_for_each(|p| p.settle(store));
                match settled.map_err(|e| e.to_string()).and_then(|()| {
                    writer.send(batch).map_err(|e| e.to_string())
                }) {
                    Ok(()) => {}
                    Err(e) => result.error = Some(e),
                }
            }
            (Ok(()), None) => {}
        }
        if done.send(result).is_err() {
            return false;
        }
    }
    {
        let mut q = shared.queue.lock().expect("pool lock");
        q.in_flight -= group.bytes;
    }
    group.bytes = 0;
    shared.room.notify_all();
    true
}

fn run(job: Job, blobs: &mut hashed_blobs::Batch<'_>) -> (Option<Batch>, Done) {
    let mut out = Done {
        path: job.path.clone(),
        start: job.start,
        end: job.end,
        parsed_upto: job.start,
        carry: job.carry.clone(),
        tail: Vec::new(),
        stored: 0,
        error: None,
    };
    let bytes = match stage::read_range(&job.path, job.start, job.end) {
        Ok(b) => b,
        Err(e) => {
            out.error = Some(format!("read {}: {e}", job.path.display()));
            return (None, out);
        }
    };
    out.tail = stage::tail_of(&bytes);
    let piece = match job.piece_from {
        Some(from) => match Piece::prepare_in(blobs, &bytes[(from - job.start) as usize..]) {
            Ok(p) => {
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
    let parse = (job.origin.open)(&job.path).and_then(|parser| {
        match parser.step(job.carry.as_ref(), &bytes, job.start) {
            Ok((mut step, carry)) => {
                for blob in std::mem::take(&mut step.output.blobs) {
                    match Piece::prepare_in(blobs, &blob.bytes) {
                        Ok(p) => extra.push(p),
                        Err(e) => out.error = Some(format!("store inline bytes: {e}")),
                    }
                }
                out.parsed_upto = job.start + step.consumed;
                out.carry = Some(carry);
                let mut p = ParseStep {
                    output: step.output,
                    accounting: step.accounting,
                    start: job.start,
                    parsed_upto: job.start + step.consumed,
                    parser_version: job.origin.parser_version,
                    hashes: Vec::new(),
                };
                p.hash_messages();
                Some(p)
            }
            Err(e) => {
                out.error = Some(format!("parse {}: {e}", job.path.display()));
                None
            }
        }
    });
    let batch = Batch {
        input: job.input,
        change: job.change,
        observation: job.observation,
        continues: job.continues,
        piece,
        blobs: extra,
        length: job.length,
        parse,
    };
    (Some(batch), out)
}

//! Time spent per stage of a build, summed over the process, so a benchmark
//! can say where a first build's time goes instead of only how long it took.
//!
//! Pool stages (read, store, parse) run on many threads at once: their sums
//! are thread-seconds and can exceed the wall time. Writer stages run on the
//! writer thread, so their sums are wall time spent there.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Reading input bytes from files or stored blobs (pool threads).
    Read,
    /// Hashing and compressing bytes and packing rows for the writer (pool).
    Store,
    /// Running the parser (pool).
    Parse,
    /// Writing a group of batches' rows (writer thread).
    Rows,
    /// Committing a replacement transaction, including its SQLite sync wait.
    Commit,
    /// A first build's full-text entries, written in bulk at its end.
    SearchIndex,
    /// A first build's trigram index rebuild.
    TrigramIndex,
    /// A first build's secondary indexes, created at its end.
    SecondaryIndexes,
    /// Making the index durable when a first build finishes.
    Sync,
    /// Deriving session facts from written rows.
    Derive,
}

const STAGES: [Stage; 10] = [
    Stage::Read,
    Stage::Store,
    Stage::Parse,
    Stage::Rows,
    Stage::Commit,
    Stage::SearchIndex,
    Stage::TrigramIndex,
    Stage::SecondaryIndexes,
    Stage::Sync,
    Stage::Derive,
];

static NANOS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Read => "read",
            Stage::Store => "store",
            Stage::Parse => "parse",
            Stage::Rows => "rows",
            Stage::Commit => "commit",
            Stage::SearchIndex => "search_index",
            Stage::TrigramIndex => "trigram_index",
            Stage::SecondaryIndexes => "secondary_indexes",
            Stage::Sync => "sync",
            Stage::Derive => "derive",
        }
    }

    /// Runs on a pool thread, so its sum is thread-seconds.
    pub fn parallel(self) -> bool {
        matches!(self, Stage::Read | Stage::Store | Stage::Parse)
    }

    fn index(self) -> usize {
        STAGES.iter().position(|s| *s == self).unwrap_or(0)
    }
}

pub fn add(stage: Stage, spent: Duration) {
    NANOS[stage.index()].fetch_add(spent.as_nanos() as u64, Ordering::Relaxed);
}

/// Times `f` as `stage`.
pub fn time<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = f();
    add(stage, started.elapsed());
    out
}

/// Seconds per stage since the process started (or the last [`reset`]).
pub fn stage_seconds() -> Vec<(Stage, f64)> {
    STAGES
        .iter()
        .map(|s| (*s, NANOS[s.index()].load(Ordering::Relaxed) as f64 / 1e9))
        .collect()
}

pub fn reset() {
    READING_WAITING.store(0, Ordering::Relaxed);
    READING_PREPARED.store(0, Ordering::Relaxed);
    READING_COMMITS.store(0, Ordering::Relaxed);
    READING_WORKERS.lock().expect("reading times").clear();
    for n in &NANOS {
        n.store(0, Ordering::Relaxed);
    }
}

static READING_WAITING: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static READING_PREPARED: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static READING_COMMITS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static READING_WORKERS: std::sync::Mutex<std::collections::BTreeMap<usize, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

pub fn reading_queue(waiting: i64, prepared: i64) {
    READING_WAITING.fetch_add(waiting, Ordering::Relaxed);
    READING_PREPARED.fetch_add(prepared, Ordering::Relaxed);
}

pub fn reading_commit_queue(delta: i64) {
    READING_COMMITS.fetch_add(delta, Ordering::Relaxed);
}

/// One parser worker's busy wall time and CPU ticks, accumulated by worker number.
pub struct ReadingWorker {
    worker: usize,
    started: Instant,
    cpu: u64,
}

fn cpu_ticks() -> u64 {
    std::fs::read_to_string("/proc/thread-self/stat")
        .ok()
        .and_then(|s| {
            let (_, fields) = s.rsplit_once(") ")?;
            let fields: Vec<_> = fields.split_whitespace().collect();
            Some(fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?)
        })
        .unwrap_or(0)
}

pub fn reading_worker(worker: usize) -> ReadingWorker {
    ReadingWorker {
        worker,
        started: Instant::now(),
        cpu: cpu_ticks(),
    }
}

impl Drop for ReadingWorker {
    fn drop(&mut self) {
        let mut workers = READING_WORKERS.lock().expect("reading times");
        let totals = workers.entry(self.worker).or_default();
        totals.0 += self.started.elapsed().as_nanos() as u64;
        totals.1 += cpu_ticks().saturating_sub(self.cpu);
    }
}

/// Queue depths and completed worker time for a benchmark's time series.
pub fn reading_pipeline() -> serde_json::Value {
    let workers = READING_WORKERS.lock().expect("reading times");
    serde_json::json!({
        "waiting": READING_WAITING.load(Ordering::Relaxed),
        "prepared": READING_PREPARED.load(Ordering::Relaxed),
        "commits": READING_COMMITS.load(Ordering::Relaxed),
        "workers": workers.iter().map(|(worker,(nanos,cpu))| serde_json::json!({
            "worker":worker,"busy_seconds":*nanos as f64 / 1e9,"cpu_ticks":cpu,
        })).collect::<Vec<_>>()
    })
}

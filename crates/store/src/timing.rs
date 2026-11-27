//! Time spent per stage of a build, summed over the process, so a benchmark
//! can say where a first build's time goes instead of only how long it took.
//!
//! Pool stages (read, store, parse) run on many threads at once: their sums
//! are thread-seconds and can exceed the wall time. Writer stages run on the
//! writer thread. The writer total includes its specific stages, and the
//! commit timer is part of rows; these timers must not be added together.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

polylogue_model::stored_names! {
    name, from_name;
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Stage {
        /// Reading input bytes from files or stored blobs (pool threads).
        Read => "read",
        /// Reading and comparing tool-database rows.
        Database => "database",
        /// Hashing and compressing bytes and packing rows for the writer (pool).
        Store => "store",
        /// Running the parser (pool).
        Parse => "parse",
        /// Writing a group of batches' rows (writer thread).
        Rows => "rows",
        /// All writer work between receiving a message and returning to its queue.
        Writer => "writer",
        /// Committing a replacement transaction, including its SQLite sync wait.
        Commit => "commit",
        /// A first build's full-text entries, read, folded and written on
        /// threads of their own while it composes (thread-seconds).
        SearchIndex => "search_index",
        /// A first build's trigram index rebuild.
        TrigramIndex => "trigram_index",
        /// A first build's secondary indexes, created at its end.
        SecondaryIndexes => "secondary_indexes",
        /// Syncing installed blob directories or a finished first-build index.
        Sync => "sync",
        /// Deriving session facts from written rows.
        Derive => "derive",
    }
}

static NANOS: [AtomicU64; Stage::ALL.len()] = [const { AtomicU64::new(0) }; Stage::ALL.len()];

impl Stage {
    /// Runs on a pool thread, so its sum is thread-seconds.
    pub fn parallel(self) -> bool {
        matches!(
            self,
            Stage::Read | Stage::Store | Stage::Parse | Stage::Database | Stage::SearchIndex
        )
    }

    fn index(self) -> usize {
        self as usize
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

/// Seconds per stage since the process started.
pub fn stage_seconds() -> Vec<(Stage, f64)> {
    Stage::ALL
        .iter()
        .map(|s| (*s, NANOS[s.index()].load(Ordering::Relaxed) as f64 / 1e9))
        .collect()
}

/// Times a stage from its creation until it is dropped, so the time includes
/// the final operation and early returns.
pub struct StageTimer(Stage, Instant);
pub fn start_stage(stage: Stage) -> StageTimer {
    StageTimer(stage, Instant::now())
}
impl Drop for StageTimer {
    fn drop(&mut self) {
        add(self.0, self.1.elapsed());
    }
}

/// The writer thread's busy seconds so far, the message it is working on
/// included: a first build is one message that runs for minutes, so its
/// busy time is read while it runs, from the writer's activity.
pub fn writer_seconds() -> f64 {
    NANOS[Stage::Writer.index()].load(Ordering::Relaxed) as f64 / 1e9
        + crate::activity::busy_seconds(crate::writer::WRITER_THREAD)
}

use std::sync::atomic::AtomicI64;
#[derive(Clone, Copy)]
pub enum Queue {
    Parse,
    Writer,
}
static PARSE_WAITING: AtomicI64 = AtomicI64::new(0);
static WRITER_WAITING: AtomicI64 = AtomicI64::new(0);
pub fn queue(queue: Queue, count: i64) {
    match queue {
        Queue::Parse => &PARSE_WAITING,
        Queue::Writer => &WRITER_WAITING,
    }
    .fetch_add(count, Ordering::Relaxed);
}
#[cfg(any(test, feature = "fixture"))]
pub fn queues() -> [(&'static str, i64); 2] {
    [
        ("parse", PARSE_WAITING.load(Ordering::Relaxed)),
        ("writer", WRITER_WAITING.load(Ordering::Relaxed)),
    ]
}

/// Completed worker lifetimes include threads shorter than the sampler period.
#[derive(Clone, serde::Serialize)]
pub struct WorkerTime {
    pub name: String,
    pub started: f64,
    pub seconds: f64,
    pub cpu_seconds: Option<f64>,
    pub run_queue_seconds: Option<f64>,
}
static WORKER_TIMES: std::sync::Mutex<Vec<WorkerTime>> = std::sync::Mutex::new(Vec::new());
static CLOCK: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
/// Seconds since the first call: the origin of [`WorkerTime::started`].
pub fn clock_seconds() -> f64 {
    CLOCK.get_or_init(Instant::now).elapsed().as_secs_f64()
}
/// Nanoseconds the calling thread has spent on a CPU.
pub fn thread_cpu_nanos() -> Option<u64> {
    scheduled().map(|(on_cpu, _)| on_cpu)
}

fn scheduled() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let mut fields = text.split_whitespace();
    Some((fields.next()?.parse().ok()?, fields.next()?.parse().ok()?))
}
struct WorkerStart {
    name: String,
    started: f64,
    scheduled: Option<(u64, u64)>,
}
pub struct WorkerTimeGuard(Option<WorkerStart>);
static WORKERS_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_workers_enabled(enabled: bool) {
    WORKERS_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn worker() -> WorkerTimeGuard {
    WorkerTimeGuard(WORKERS_ENABLED.load(Ordering::Relaxed).then(|| {
        WorkerStart {
            name: std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_owned(),
            started: clock_seconds(),
            scheduled: scheduled(),
        }
    }))
}
impl Drop for WorkerTimeGuard {
    fn drop(&mut self) {
        if let Some(WorkerStart {
            name,
            started,
            scheduled: before,
        }) = self.0.take()
        {
            let elapsed = scheduled().zip(before);
            let cpu_seconds =
                elapsed.map(|(after, before)| after.0.saturating_sub(before.0) as f64 / 1e9);
            let run_queue_seconds =
                elapsed.map(|(after, before)| after.1.saturating_sub(before.1) as f64 / 1e9);
            if let Ok(mut times) = WORKER_TIMES.lock() {
                times.push(WorkerTime {
                    name,
                    started,
                    seconds: clock_seconds() - started,
                    cpu_seconds,
                    run_queue_seconds,
                });
            }
        }
    }
}
pub fn workers() -> Vec<WorkerTime> {
    WORKER_TIMES.lock().map(|v| v.clone()).unwrap_or_default()
}

/// SQLite's own peak allocation counter, including its page caches and sort buffers.
#[allow(unsafe_code)]
pub fn sqlite_memory_peak() -> Option<u64> {
    let (mut current, mut peak) = (0, 0);
    // SAFETY: SQLite writes two live i64 outputs; the status code is a defined constant.
    let status = unsafe {
        rusqlite::ffi::sqlite3_status64(
            rusqlite::ffi::SQLITE_STATUS_MEMORY_USED,
            &mut current,
            &mut peak,
            0,
        )
    };
    (status == rusqlite::ffi::SQLITE_OK).then_some(peak as u64)
}

#[cfg(test)]
mod tests;

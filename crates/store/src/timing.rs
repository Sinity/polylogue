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
    /// A first build's full-text entries, written in bulk at its end.
    SearchIndex,
    /// A first build's trigram index rebuild.
    TrigramIndex,
    /// A first build's secondary indexes, created at its end.
    SecondaryIndexes,
    /// Making the index durable when a first build settles.
    Sync,
    /// Deriving session facts from written rows.
    Derive,
}

const STAGES: [Stage; 9] = [
    Stage::Read,
    Stage::Store,
    Stage::Parse,
    Stage::Rows,
    Stage::SearchIndex,
    Stage::TrigramIndex,
    Stage::SecondaryIndexes,
    Stage::Sync,
    Stage::Derive,
];

static NANOS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Read => "read",
            Stage::Store => "store",
            Stage::Parse => "parse",
            Stage::Rows => "rows",
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
    for n in &NANOS {
        n.store(0, Ordering::Relaxed);
    }
}

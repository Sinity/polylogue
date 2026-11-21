//! Temporary exclusive CPU attribution for the first-build experiment.
//! Each thread charges elapsed work to its innermost active part. The normal
//! worker lifetime remains the independent total against which to reconcile it.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Part {
    WriterOther,
    QueueWait,
    SearchTurnWait,
    RawOther,
    SegmentAppend,
    SegmentSyncWait,
    RegisterRows,
    RegisterCommit,
    RowTransaction,
    RowOther,
    VersionText,
    SessionFacts,
    MessageDispatch,
    MessageHeader,
    MessageClaim,
    SelectedMessage,
    SelectedBlocks,
    WireShapes,
    ParserContext,
    ParseState,
    Counts,
    Links,
    Items,
    SessionValues,
    Cleanup,
    IndexCommit,
    ChainProgress,
    WalRestart,
    SearchBuild,
    TrigramBuild,
    SecondaryBuild,
    Install,
    DeriveRead,
    DeriveApply,
}

const NAMES: [&str; 34] = [
    "writer_other",
    "queue_wait",
    "search_turn_wait",
    "raw_other",
    "segment_append",
    "segment_sync_wait",
    "register_rows",
    "register_commit",
    "row_transaction",
    "row_other",
    "version_text",
    "session_facts",
    "message_dispatch",
    "message_header",
    "message_claim",
    "selected_message",
    "selected_blocks",
    "wire_shapes",
    "parser_context",
    "parse_state",
    "counts",
    "links",
    "items",
    "session_values",
    "cleanup",
    "index_commit",
    "chain_progress",
    "wal_restart",
    "search_build",
    "trigram_build",
    "secondary_build",
    "install",
    "derive_read",
    "derive_apply",
];
static WALL: [AtomicU64; 34] = [const { AtomicU64::new(0) }; 34];
static CPU: [AtomicU64; 34] = [const { AtomicU64::new(0) }; 34];
static CALLS: [AtomicU64; 34] = [const { AtomicU64::new(0) }; 34];
static UNKNOWN: [AtomicBool; 34] = [const { AtomicBool::new(false) }; 34];

#[derive(Clone, Copy)]
struct Sample {
    wall: Instant,
    cpu: Option<u64>,
}

impl Sample {
    #[allow(unsafe_code)]
    fn now() -> Self {
        let mut time = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: clock_gettime writes one live timespec and the clock is local
        // to the calling thread. An unavailable clock stays unknown.
        let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
        let cpu = if status == 0 {
            u64::try_from(time.tv_sec)
                .ok()
                .and_then(|seconds| seconds.checked_mul(1_000_000_000))
                .zip(u64::try_from(time.tv_nsec).ok())
                .and_then(|(seconds, nanos)| seconds.checked_add(nanos))
        } else {
            None
        };
        Self {
            wall: Instant::now(),
            cpu,
        }
    }
}

#[derive(Default)]
struct Current {
    part: Option<Part>,
    since: Option<Sample>,
}

impl Current {
    fn switch(&mut self, next: Option<Part>, now: Sample) -> Option<(Part, u64, Option<u64>)> {
        let elapsed = self.part.zip(self.since).map(|(part, since)| {
            (
                part,
                now.wall.saturating_duration_since(since.wall).as_nanos() as u64,
                now.cpu
                    .zip(since.cpu)
                    .and_then(|(now, before)| now.checked_sub(before)),
            )
        });
        self.part = next;
        self.since = Some(now);
        elapsed
    }
}

thread_local! {
    static CURRENT: RefCell<Current> = const { RefCell::new(Current { part: None, since: None }) };
}

fn add(elapsed: Option<(Part, u64, Option<u64>)>) {
    if let Some((part, wall, cpu)) = elapsed {
        let index = part as usize;
        WALL[index].fetch_add(wall, Ordering::Relaxed);
        match cpu {
            Some(cpu) => {
                CPU[index].fetch_add(cpu, Ordering::Relaxed);
            }
            None => {
                UNKNOWN[index].store(true, Ordering::Relaxed);
            }
        }
    }
}

pub(crate) struct PartGuard {
    previous: Option<Part>,
    enabled: bool,
    _same_thread: PhantomData<*const ()>,
}

pub(crate) fn part(part: Part) -> PartGuard {
    let enabled = super::WORKERS_ENABLED.load(Ordering::Relaxed);
    let previous = enabled
        .then(|| {
            let now = Sample::now();
            CURRENT.with(|current| {
                let mut current = current.borrow_mut();
                let previous = current.part;
                add(current.switch(Some(part), now));
                CALLS[part as usize].fetch_add(1, Ordering::Relaxed);
                previous
            })
        })
        .flatten();
    PartGuard {
        previous,
        enabled,
        _same_thread: PhantomData,
    }
}

impl Drop for PartGuard {
    fn drop(&mut self) {
        if self.enabled {
            let now = Sample::now();
            CURRENT.with(|current| add(current.borrow_mut().switch(self.previous, now)));
        }
    }
}

#[derive(serde::Serialize)]
pub struct PartTime {
    pub calls: u64,
    pub seconds: f64,
    pub cpu_seconds: Option<f64>,
}

pub fn records() -> std::collections::BTreeMap<&'static str, PartTime> {
    NAMES
        .iter()
        .enumerate()
        .filter_map(|(index, &name)| {
            let calls = CALLS[index].load(Ordering::Relaxed);
            if calls == 0 {
                return None;
            }
            Some((
                name,
                PartTime {
                    calls,
                    seconds: WALL[index].load(Ordering::Relaxed) as f64 / 1e9,
                    cpu_seconds: (!UNKNOWN[index].load(Ordering::Relaxed))
                        .then(|| CPU[index].load(Ordering::Relaxed) as f64 / 1e9),
                },
            ))
        })
        .collect()
}

#[cfg(any(test, feature = "fixture"))]
pub(super) fn reset() {
    for counter in WALL.iter().chain(&CPU).chain(&CALLS) {
        counter.store(0, Ordering::Relaxed);
    }
    for unknown in &UNKNOWN {
        unknown.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;

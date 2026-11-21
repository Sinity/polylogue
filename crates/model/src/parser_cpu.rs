//! Unmerged parser CPU diagnostic. Reuses the existing exclusive writer guard.
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
static ENABLED: AtomicBool = AtomicBool::new(false);
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Part {
    Other,
    Read,
    Custody,
    Provider,
    JsonDecode,
    ContextGet,
    ContextPut,
    Checkpoint,
    Pack,
    Resolve,
}
const NAMES: [&str; 10] = [
    "parser_other",
    "input_read",
    "raw_custody",
    "provider_semantics",
    "json_decode",
    "context_get",
    "context_put",
    "checkpoint",
    "prepare_rows",
    "resolve_plan",
];
static WALL: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
static CPU: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
static CALLS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
static UNKNOWN: [AtomicBool; 10] = [const { AtomicBool::new(false) }; 10];
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

pub struct PartGuard {
    previous: Option<Part>,
    enabled: bool,
    _same_thread: PhantomData<*const ()>,
}

pub fn part(part: Part) -> PartGuard {
    let enabled = ENABLED.load(Ordering::Relaxed)
        && (part == Part::Other || CURRENT.with(|current| current.borrow().part.is_some()));
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
        .map(|(index, &name)| {
            let calls = CALLS[index].load(Ordering::Relaxed);
            (
                name,
                PartTime {
                    calls,
                    seconds: WALL[index].load(Ordering::Relaxed) as f64 / 1e9,
                    cpu_seconds: (!UNKNOWN[index].load(Ordering::Relaxed))
                        .then(|| CPU[index].load(Ordering::Relaxed) as f64 / 1e9),
                },
            )
        })
        .collect()
}

#[cfg(any(test, feature = "fixture"))]
pub fn reset() {
    for counter in WALL.iter().chain(&CPU).chain(&CALLS) {
        counter.store(0, Ordering::Relaxed);
    }
    for unknown in &UNKNOWN {
        unknown.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;

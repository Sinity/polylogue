//! The process's peak resident memory, per stage and over its life.
//!
//! Linux keeps one peak per process (`VmHWM` in `/proc/self/status`) and
//! lets a process reset it to its present resident size by writing `5` to
//! `/proc/self/clear_refs`. A stage that wants its own peak resets it when it
//! starts ([`Stage::start`]); since the reset is process-wide, the peak
//! before each reset is first folded into a lifetime figure kept here, and
//! every reader of "the process's peak" reads [`lifetime`], never `VmHWM`
//! directly. Where the reset is refused, a stage's peak is the lifetime
//! figure and says so.
use std::sync::atomic::{AtomicU64, Ordering};

/// The highest `VmHWM` seen before any reset, in bytes.
static LIFETIME: AtomicU64 = AtomicU64::new(0);

/// A memory figure of this process in bytes, by its field name in
/// `/proc/self/status` (`VmHWM:` the peak resident memory, `VmRSS:` the
/// resident memory now); zero where the system does not report it.
pub fn status_bytes(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(field))
                .and_then(|value| {
                    value
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse::<u64>()
                        .ok()
                })
        })
        .map_or(0, |kib| kib * 1024)
}

/// The process's peak resident memory over its whole life, in bytes.
pub fn lifetime() -> u64 {
    LIFETIME
        .fetch_max(status_bytes("VmHWM:"), Ordering::AcqRel)
        .max(status_bytes("VmHWM:"))
}

/// The peak of one stage, from its [`Stage::start`].
#[derive(Debug, Clone, Copy)]
pub struct Stage {
    /// Whether the process peak was reset when the stage started; if not,
    /// [`Stage::peak`] is the lifetime figure.
    own: bool,
}

impl Stage {
    /// Starts measuring a stage: folds the process peak into the lifetime
    /// figure, then resets it.
    pub fn start() -> Self {
        lifetime();
        let own = std::fs::write("/proc/self/clear_refs", "5").is_ok();
        Self { own }
    }

    /// The stage's peak resident memory in bytes, and whether it is the
    /// stage's own (false: the process's lifetime peak, the reset having
    /// been refused). Stages that overlap share one reset, so a stage's
    /// peak covers from the latest start of any stage.
    pub fn peak(&self) -> (u64, bool) {
        if self.own {
            let peak = status_bytes("VmHWM:");
            LIFETIME.fetch_max(peak, Ordering::AcqRel);
            (peak, true)
        } else {
            (lifetime(), false)
        }
    }
}

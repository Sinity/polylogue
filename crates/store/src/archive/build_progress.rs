//! How a build of `index.db` is advancing, for `status`: the phase it is
//! in, its rates over the last minutes and, from a steady rate, how long it
//! has left.
//!
//! Samples are taken where the build's progress is recorded, so every build
//! path that records progress is measured. They are at least
//! [`SAMPLE_EVERY`] apart and cover at most [`KEPT_FOR`].
use super::index_build::IndexBuild;
use polylogue_model::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub const SAMPLE_EVERY: Duration = Duration::from_secs(10);
pub const KEPT_FOR: Duration = Duration::from_secs(15 * 60);
/// The span the rates are measured over.
pub const RATE_OVER: Duration = Duration::from_secs(5 * 60);
/// Less measured time than this gives a rate but no estimate.
pub const ESTIMATE_AFTER: Duration = Duration::from_secs(60);
/// The span writer busy time and worker CPU are measured over.
const BUSY_OVER: Duration = Duration::from_secs(60);
/// How far the two halves' rates may differ, as a share of the whole
/// span's rate, for the rate to count as steady.
const STEADY_WITHIN: f64 = 0.2;

/// The build's progress as `status` reports it, beside its counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildProgress {
    /// The phase the build is in, as its log names it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_since: Option<Timestamp>,
    /// Bytes and chains read per second over the last five minutes; absent
    /// before two measurements exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chains_per_second: Option<f64>,
    /// The seconds the measured span covers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_seconds: Option<f64>,
    /// The rate moved by more than a fifth between the first and the second
    /// half of the span, so no estimate is given.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rate_varying: bool,
    /// The bytes left at the byte rate; only from at least a minute of
    /// measurements with a steady, positive rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    /// The share of the last minute the writer was busy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writer_busy: Option<f64>,
    /// Each parse worker's CPU seconds over the last minute it measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worker_cpu_seconds: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    at: Instant,
    bytes_read: u64,
    chains_read: u64,
    writer_seconds: f64,
}

/// One build's phase and samples.
#[derive(Debug, Default, Clone)]
pub(crate) struct Record {
    phase: Option<(String, Timestamp)>,
    samples: VecDeque<Sample>,
    /// When the workers' CPU was last measured, and each one's seconds
    /// per minute.
    workers: Option<(Instant, Vec<f64>)>,
}

impl Record {
    pub(crate) fn phase(&mut self, phase: &str, since: Timestamp) {
        self.phase = Some((phase.to_owned(), since));
    }

    pub(crate) fn workers(&mut self, at: Instant, cpu_seconds_per_minute: Vec<f64>) {
        self.workers = Some((at, cpu_seconds_per_minute));
    }

    /// Takes a sample unless the newest is younger than [`SAMPLE_EVERY`].
    pub(crate) fn sample(
        &mut self,
        at: Instant,
        bytes_read: u64,
        chains_read: u64,
        writer_seconds: impl FnOnce() -> f64,
    ) {
        if self
            .samples
            .back()
            .is_some_and(|newest| at.saturating_duration_since(newest.at) < SAMPLE_EVERY)
        {
            return;
        }
        while self
            .samples
            .front()
            .is_some_and(|oldest| at.saturating_duration_since(oldest.at) > KEPT_FOR)
        {
            self.samples.pop_front();
        }
        self.samples.push_back(Sample {
            at,
            bytes_read,
            chains_read,
            writer_seconds: writer_seconds(),
        });
    }

    /// The oldest sample within `span` of `now`, or the newest before it
    /// when none is that recent: a build that stalled still has a rate,
    /// measured over a longer span.
    fn baseline(&self, now: Instant, span: Duration) -> Option<&Sample> {
        let recent = self
            .samples
            .iter()
            .find(|sample| now.saturating_duration_since(sample.at) <= span);
        recent.or_else(|| self.samples.back())
    }

    /// The progress at `now`, with the build's current counts and the
    /// writer's busy seconds so far.
    pub(crate) fn report(
        &self,
        now: Instant,
        build: &IndexBuild,
        writer_seconds: f64,
    ) -> BuildProgress {
        let current = Sample {
            at: now,
            bytes_read: build.bytes_read,
            chains_read: build.chains_read,
            writer_seconds,
        };
        let rate = |from: &Sample, to: &Sample| {
            let seconds = to.at.saturating_duration_since(from.at).as_secs_f64();
            (seconds > 0.).then(|| {
                (
                    to.bytes_read.saturating_sub(from.bytes_read) as f64 / seconds,
                    to.chains_read.saturating_sub(from.chains_read) as f64 / seconds,
                    seconds,
                )
            })
        };
        let measured = self
            .baseline(now, RATE_OVER)
            .and_then(|from| rate(from, &current).map(|rate| (from, rate)));
        let mut progress = BuildProgress {
            phase: self.phase.as_ref().map(|(phase, _)| phase.clone()),
            phase_since: self.phase.as_ref().map(|(_, since)| *since),
            bytes_per_second: measured.map(|(_, (bytes, _, _))| bytes),
            chains_per_second: measured.map(|(_, (_, chains, _))| chains),
            rate_seconds: measured.map(|(_, (_, _, seconds))| seconds),
            rate_varying: false,
            eta_seconds: None,
            writer_busy: self.baseline(now, BUSY_OVER).and_then(|from| {
                let seconds = now.saturating_duration_since(from.at).as_secs_f64();
                (seconds > 0.)
                    .then(|| ((writer_seconds - from.writer_seconds) / seconds).clamp(0., 1.))
            }),
            worker_cpu_seconds: self
                .workers
                .as_ref()
                .filter(|(at, _)| now.saturating_duration_since(*at) <= 2 * BUSY_OVER)
                .map(|(_, cpu)| cpu.clone())
                .unwrap_or_default(),
        };
        if let Some((from, (bytes_per_second, _, seconds))) = measured
            && seconds >= ESTIMATE_AFTER.as_secs_f64()
            && bytes_per_second > 0.
        {
            // Progress is recorded in steps, and samples are taken as it is
            // recorded, so the halves are measured between samples: from
            // the first to the one nearest the middle, and from there to the
            // newest. A rate measured up to now would count the time since
            // the last step as slower work.
            let Some(newest) = self.samples.back() else {
                return progress;
            };
            let middle = from.at + newest.at.saturating_duration_since(from.at) / 2;
            let halves = self
                .samples
                .iter()
                .filter(|sample| sample.at > from.at && sample.at < newest.at)
                .min_by_key(|sample| {
                    if sample.at > middle {
                        sample.at - middle
                    } else {
                        middle - sample.at
                    }
                })
                .and_then(|split| rate(from, split).zip(rate(split, newest)));
            let steady = halves.is_some_and(|((first, _, _), (second, _, _))| {
                (first - second).abs() <= STEADY_WITHIN * bytes_per_second
            });
            let left = build.bytes.saturating_sub(build.bytes_read);
            if left == 0 {
                // Every byte is read; what remains is the writer's phases,
                // which the byte rate does not measure.
            } else if steady {
                progress.eta_seconds = Some((left as f64 / bytes_per_second).ceil() as u64);
            } else {
                progress.rate_varying = true;
            }
        }
        progress
    }
}

#[cfg(test)]
mod tests;

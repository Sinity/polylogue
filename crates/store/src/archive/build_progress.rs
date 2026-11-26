//! How a rebuild of `index.db` is advancing, for `status`: the steps
//! finished and how long each took, the step running with its count, rate
//! and estimate, and the steps still to come.
//!
//! The steps are named by the code that runs them: each announces itself
//! as it starts ([`Record::step`]), and the build says at its start which
//! steps follow ([`Record::plan`]). Nothing here lists them. Samples of the
//! running step's count are taken as the count is recorded, at least
//! [`SAMPLE_EVERY`] apart and covering at most [`KEPT_FOR`]; a new step
//! starts with none.
use polylogue_model::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

pub const SAMPLE_EVERY: Duration = Duration::from_secs(10);
pub const KEPT_FOR: Duration = Duration::from_secs(15 * 60);
/// The span the rate is measured over.
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
    /// The steps finished, in the order they ran.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub finished: Vec<FinishedStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<CurrentStep>,
    /// The steps still to come, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub to_come: Vec<String>,
    /// Seconds left for the whole build. Steps to come have no rate until
    /// they run, so this is given only once none is left to come: it is
    /// then the current step's estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    /// An earlier attempt of this build failed and was discarded; this one
    /// started again from the stored files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_attempt: Option<FailedAttempt>,
    /// The share of the last minute the database writer was busy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writer_busy: Option<f64>,
    /// Each parse worker's CPU seconds per minute, over the last minute it
    /// measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worker_cpu_seconds: Vec<f64>,
}

/// A step the build has finished.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FinishedStep {
    pub name: String,
    pub seconds: f64,
    /// What it handled, when the step counts its work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<StepCount>,
}

/// The step the build is running.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CurrentStep {
    pub name: String,
    pub since: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<StepCount>,
    /// Units of `count` done per second over the last five minutes; absent
    /// before two measurements exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_second: Option<f64>,
    /// The seconds the rate is measured over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_seconds: Option<f64>,
    /// The rate moved by more than a fifth between the first and the second
    /// half of the span, so no estimate is given.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rate_varying: bool,
    /// What is left of the count's total at the rate; only from at least a
    /// minute of measurements with a steady, positive rate and a known
    /// total.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
}

/// How much of its work a step has done, and of how much when that is known
/// before the step ends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StepCount {
    pub done: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    pub unit: CountUnit,
}

/// What a step counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CountUnit {
    Bytes,
    Files,
    Records,
    Indexes,
    Sessions,
}

impl CountUnit {
    /// The unit as a plural word: `records`.
    pub fn word(self) -> &'static str {
        match self {
            Self::Bytes => "bytes",
            Self::Files => "files",
            Self::Records => "records",
            Self::Indexes => "indexes",
            Self::Sessions => "sessions",
        }
    }
}

/// An attempt of the build that failed: the step it was in and the error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FailedAttempt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    pub error: String,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    at: Instant,
    done: u64,
    writer_seconds: f64,
}

/// One build's steps, counts and samples.
#[derive(Debug, Default, Clone)]
pub(crate) struct Record {
    /// The steps the build said follow, in order.
    plan: Vec<String>,
    finished: Vec<(String, f64)>,
    current: Option<(String, Timestamp)>,
    /// Each step's latest count. The writer counts a step as it works and
    /// the engine records that the step started, so a count can come first.
    counts: BTreeMap<String, StepCount>,
    /// Samples of the current step's count.
    samples: VecDeque<Sample>,
    /// When the workers' CPU was last measured, and each one's seconds
    /// per minute.
    workers: Option<(Instant, Vec<f64>)>,
    failed: Option<FailedAttempt>,
}

impl Record {
    pub(crate) fn plan(&mut self, steps: &[&str]) {
        self.plan = steps.iter().map(|step| (*step).to_owned()).collect();
    }

    /// The build started `step` at `since`; the step running before it is
    /// finished. A step announced again while it runs changes nothing.
    pub(crate) fn step(&mut self, step: &str, since: Timestamp) {
        if let Some((running, started)) = &self.current {
            if running == step {
                return;
            }
            let seconds = since.unix_ms().saturating_sub(started.unix_ms()) as f64 / 1000.;
            self.finished.push((running.clone(), seconds.max(0.)));
        }
        self.current = Some((step.to_owned(), since));
        self.samples.clear();
    }

    /// Records `step`'s count, sampling it while the step runs.
    pub(crate) fn count(
        &mut self,
        at: Instant,
        step: &str,
        count: StepCount,
        writer_seconds: impl FnOnce() -> f64,
    ) {
        let done = count.done;
        self.counts.insert(step.to_owned(), count);
        if self
            .current
            .as_ref()
            .is_some_and(|(running, _)| running == step)
        {
            self.sample(at, done, writer_seconds);
        }
    }

    pub(crate) fn workers(&mut self, at: Instant, cpu_seconds_per_minute: Vec<f64>) {
        self.workers = Some((at, cpu_seconds_per_minute));
    }

    /// The attempt failed and its files were discarded: the steps start
    /// again, the failure kept until the build ends.
    pub(crate) fn failed(&mut self, error: String) {
        self.failed = Some(FailedAttempt {
            step: self.current.as_ref().map(|(step, _)| step.clone()),
            error,
        });
        self.finished.clear();
        self.current = None;
        self.counts.clear();
        self.samples.clear();
    }

    /// Takes a sample unless the newest is younger than [`SAMPLE_EVERY`].
    fn sample(&mut self, at: Instant, done: u64, writer_seconds: impl FnOnce() -> f64) {
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
            done,
            writer_seconds: writer_seconds(),
        });
    }

    /// The oldest sample within `span` of `now`, or the newest before it
    /// when none is that recent: a step that stalled still has a rate,
    /// measured over a longer span.
    fn baseline(&self, now: Instant, span: Duration) -> Option<&Sample> {
        let recent = self
            .samples
            .iter()
            .find(|sample| now.saturating_duration_since(sample.at) <= span);
        recent.or_else(|| self.samples.back())
    }

    /// The progress at `now`, with the writer's busy seconds so far.
    pub(crate) fn report(&self, now: Instant, writer_seconds: f64) -> BuildProgress {
        let finished = self
            .finished
            .iter()
            .map(|(name, seconds)| FinishedStep {
                name: name.clone(),
                seconds: *seconds,
                count: self.counts.get(name).cloned(),
            })
            .collect::<Vec<_>>();
        let current = self
            .current
            .as_ref()
            .map(|(name, since)| self.current_step(now, name, *since));
        // The plan's steps after the running one; a planned step the build
        // passed over (nothing to sort when nothing was read) is not to come.
        let after = self
            .current
            .as_ref()
            .and_then(|(name, _)| self.plan.iter().position(|step| step == name))
            .map_or(0, |at| at + 1);
        let to_come: Vec<String> = self.plan[after.min(self.plan.len())..]
            .iter()
            .filter(|step| !self.finished.iter().any(|(done, _)| done == *step))
            .cloned()
            .collect();
        BuildProgress {
            eta_seconds: current
                .as_ref()
                .and_then(|step| step.eta_seconds)
                .filter(|_| to_come.is_empty()),
            finished,
            current,
            to_come,
            failed_attempt: self.failed.clone(),
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
        }
    }

    fn current_step(&self, now: Instant, name: &str, since: Timestamp) -> CurrentStep {
        let count = self.counts.get(name).cloned();
        let mut step = CurrentStep {
            name: name.to_owned(),
            since,
            count: count.clone(),
            per_second: None,
            rate_seconds: None,
            rate_varying: false,
            eta_seconds: None,
        };
        let Some(count) = count else {
            return step;
        };
        let rate = |from: &Sample, to: &Sample| {
            let seconds = to.at.saturating_duration_since(from.at).as_secs_f64();
            (seconds > 0.).then(|| (to.done.saturating_sub(from.done) as f64 / seconds, seconds))
        };
        let current = Sample {
            at: now,
            done: count.done,
            writer_seconds: 0.,
        };
        let Some((from, (per_second, seconds))) = self
            .baseline(now, RATE_OVER)
            .and_then(|from| rate(from, &current).map(|rate| (from, rate)))
        else {
            return step;
        };
        step.per_second = Some(per_second);
        step.rate_seconds = Some(seconds);
        let left = count.total.map(|total| total.saturating_sub(count.done));
        if seconds < ESTIMATE_AFTER.as_secs_f64() || per_second <= 0. || left == Some(0) {
            return step;
        }
        // Counts are recorded in steps, and samples are taken as they are
        // recorded, so the halves are measured between samples: from the
        // first to the one nearest the middle, and from there to the newest.
        // A rate measured up to now would count the time since the last
        // record as slower work.
        let Some(newest) = self.samples.back() else {
            return step;
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
        let steady = halves.is_some_and(|((first, _), (second, _))| {
            (first - second).abs() <= STEADY_WITHIN * per_second
        });
        if !steady {
            step.rate_varying = true;
        } else if let Some(left) = left {
            step.eta_seconds = Some((left as f64 / per_second).ceil() as u64);
        }
        step
    }
}

#[cfg(test)]
mod tests;

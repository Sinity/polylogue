//! Survival measures over the per-session facts the `session_span`
//! derivation keeps: time to a session's first failed tool call, and time
//! to a session ending on an unanswered prompt.

use super::Selection;
use crate::{Reader, Result};
use serde::Serialize;

/// One step of a Kaplan-Meier curve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Step {
    pub at: f64,
    pub at_risk: u64,
    pub events: u64,
    pub survival: f64,
    /// Greenwood standard error.
    pub std_error: f64,
}

/// A survival curve and the cohort it came from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Survival {
    /// What "survival" means here: the event, the time axis and its unit.
    pub event: &'static str,
    pub axis: &'static str,
    pub unit: &'static str,
    pub sessions: u64,
    pub events: u64,
    /// Sessions that ended or are still running without the event.
    pub censored: u64,
    /// Sessions left out because their duration is unknown.
    pub unmeasured: u64,
    pub pending: u64,
    pub quiet_period_ms: Option<i64>,
    /// Where survival first drops to one half; `None` when it never does.
    pub median: Option<f64>,
    pub curve: Vec<Step>,
}

/// Kaplan-Meier over `(time, event)` pairs; `event = false` is censored.
pub fn kaplan_meier(mut obs: Vec<(f64, bool)>) -> (Vec<Step>, Option<f64>) {
    obs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut at_risk = obs.len() as u64;
    let (mut s, mut greenwood) = (1.0_f64, 0.0_f64);
    let mut curve = vec![];
    let mut median = None;
    let mut i = 0;
    while i < obs.len() {
        let t = obs[i].0;
        let (mut d, mut c) = (0u64, 0u64);
        while i < obs.len() && obs[i].0 == t {
            if obs[i].1 {
                d += 1
            } else {
                c += 1
            }
            i += 1;
        }
        if d > 0 {
            let n = at_risk as f64;
            s *= 1.0 - d as f64 / n;
            if n > d as f64 {
                greenwood += d as f64 / (n * (n - d as f64));
            }
            curve.push(Step {
                at: t,
                at_risk,
                events: d,
                survival: s,
                std_error: s * greenwood.sqrt(),
            });
            if median.is_none() && s <= 0.5 {
                median = Some(t);
            }
        }
        at_risk -= d + c;
    }
    (curve, median)
}

/// A behind session is computed only for a single-session read.
fn facts(reader:&Reader<'_>,selection:&Selection)->Result<(Vec<super::time::Span>,u64)> {
    super::time::selected_spans(reader,selection)
}

fn survival(
    event: &'static str,
    axis: &'static str,
    unit: &'static str,
    obs: Vec<(f64, bool)>,
    unmeasured: u64,
) -> Survival {
    let sessions = obs.len() as u64;
    let events = obs.iter().filter(|o| o.1).count() as u64;
    let (curve, median) = kaplan_meier(obs);
    Survival {
        event,
        axis,
        unit,
        sessions,
        events,
        censored: sessions - events,
        unmeasured,
        pending:0,
        quiet_period_ms:(event=="ended on an unanswered prompt").then_some(QUIET),
        median,
        curve,
    }
}

/// Time from a session's first message to its first failed tool call, on
/// a wall-clock axis (minutes) or a turn axis (human prompts). A session
/// with no failure is censored at its last activity, whether it ended or is
/// still running: its last message is not an outcome.
pub fn first_failure(reader: &Reader<'_>, selection: &Selection, turns: bool) -> Result<Survival> {
    let mut obs = vec![];
    let mut unmeasured = 0;
    let (facts,pending)=facts(reader,selection)?;
    for f in facts {
        let (Some(first), Some(last)) = (f.first_at, f.last_at) else {
            unmeasured += 1;
            continue;
        };
        let (end, event) = match f.first_failure_at {
            Some(at) if at >= first => (at, true),
            _ => (last, false),
        };
        let x = if turns {
            f.prompts_to_failure as f64
        } else {
            (end - first) as f64 / 60_000.0
        };
        obs.push((x, event));
    }
    let mut result=if turns {
        survival(
            "first failed tool call",
            "turns",
            "human prompts",
            obs,
            unmeasured,
        )
    } else {
        survival(
            "first failed tool call",
            "wall clock",
            "minutes",
            obs,
            unmeasured,
        )
    };
    result.pending=pending;
    result.unmeasured+=pending;
    Ok(result)
}

/// How long a session must have been quiet before it counts as over.
/// Inside it, a session is censored as possibly still going.
pub const QUIET: i64 = 6 * 3_600_000;

/// Time to a session ending on an unanswered prompt: the
/// session went quiet for [`QUIET`] with its last human prompt followed by
/// no model message. A session that ended answered is censored at its end,
/// and so is one still within the quiet period. Structural only: what the
/// prompt asked is never read.
pub fn unanswered_end(reader: &Reader<'_>, selection: &Selection, now: i64) -> Result<Survival> {
    let mut obs = vec![];
    let mut unmeasured = 0;
    let (facts,pending)=facts(reader,selection)?;
    for f in facts {
        let (Some(first), Some(last)) = (f.first_at, f.last_at) else {
            unmeasured += 1;
            continue;
        };
        // A session no human prompted (a subagent, a compaction call) has no
        // prompt to leave unanswered.
        if !f.prompted {
            unmeasured += 1;
            continue;
        }
        let over = now - last >= QUIET;
        obs.push((
            (last - first) as f64 / 60_000.0,
            over && f.ends_with == "human_prompt",
        ));
    }
    let mut result=survival(
        "ended on an unanswered prompt",
        "wall clock",
        "minutes",
        obs,
        unmeasured+pending,
    );
    result.pending=pending;
    Ok(result)
}

#[cfg(test)]
mod tests;

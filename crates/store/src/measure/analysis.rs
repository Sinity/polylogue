//! Analyses over sessions and tool calls, descriptive, with the population
//! each one used stated beside it. Code selects and structures; it never
//! reads meaning into text.

use super::{Scope, usage};
use crate::{Reader, Result};
use polylogue_model::LogicalDay;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// Tool outcomes kept apart: they never collapse into one success figure.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Outcomes {
    pub ok: u64,
    pub error: u64,
    /// The call has no result record yet, and nothing later in its session
    /// rules one out; read as pending while the session is live.
    pub no_result: u64,
    /// The call has no result record and the session went on past it, so
    /// none will come.
    pub passed_without_result: u64,
    /// Any other unknown outcome.
    pub unknown: u64,
}

pub fn outcomes(reader: &Reader<'_>, scope: &Scope) -> Result<Outcomes> {
    let (cond, params) = scope.condition();
    Ok(reader.conn()?.query_row(
        &format!(
            "SELECT sum(t.outcome = 'ok'), sum(t.outcome = 'error'),
                    sum(t.outcome = 'unknown' AND t.outcome_reason = 'no_result'),
                    sum(t.outcome = 'unknown' AND t.outcome_reason = 'passed_without_result'),
                    sum(t.outcome = 'unknown'
                        AND coalesce(t.outcome_reason, '') NOT IN ('no_result', 'passed_without_result'))
             FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
             JOIN message m ON m.pk = b.message_pk
             WHERE t.inherited = 0 AND {cond}"
        ),
        rusqlite::params_from_iter(params),
        |r| {
            let n = |i| r.get::<_, Option<i64>>(i).map(|v| v.unwrap_or(0) as u64);
            Ok(Outcomes {
                ok: n(0)?,
                error: n(1)?,
                no_result: n(2)?,
                passed_without_result: n(3)?,
                unknown: n(4)?,
            })
        },
    )?)
}

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

/// Time from a session's first response to its first failed tool call, on
/// a wall-clock axis (minutes) or a turn axis (human prompts). A session
/// with no failure is censored at its last activity, whether it ended or is
/// still running: its last message is not an outcome.
pub fn first_failure(reader: &Reader<'_>, scope: &Scope, turns: bool) -> Result<Survival> {
    let (cond, params) = scope.condition();
    let sql = format!(
        "WITH s AS (
           SELECT m.session_pk AS pk, min(m.at) AS first_at, max(m.at) AS last_at
           FROM message m WHERE {cond} GROUP BY m.session_pk),
         f AS (
           SELECT b.session_pk AS pk, min(coalesce(t.at, fm.at)) AS fail_at
           FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
           JOIN message fm ON fm.pk = b.message_pk
           WHERE t.outcome = 'error' AND t.inherited = 0 GROUP BY b.session_pk)
         SELECT s.first_at, s.last_at, f.fail_at,
           (SELECT count(*) FROM message h WHERE h.session_pk = s.pk AND h.written_by = 'human'
              AND h.prompt_kind = 'prose' AND h.at <= coalesce(f.fail_at, s.last_at))
         FROM s LEFT JOIN f ON f.pk = s.pk"
    );
    let mut obs = vec![];
    let mut unmeasured = 0;
    let mut stmt = reader.conn()?.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
        Ok((
            r.get::<_, Option<i64>>(0)?,
            r.get::<_, Option<i64>>(1)?,
            r.get::<_, Option<i64>>(2)?,
            r.get::<_, i64>(3)?,
        ))
    })?;
    for row in rows {
        let (first, last, fail, prompts) = row?;
        let (Some(first), Some(last)) = (first, last) else {
            unmeasured += 1;
            continue;
        };
        let (end, event) = match fail {
            Some(f) if f >= first => (f, true),
            _ => (last, false),
        };
        let x = if turns {
            prompts as f64
        } else {
            (end - first) as f64 / 60_000.0
        };
        obs.push((x, event));
    }
    let sessions = obs.len() as u64;
    let events = obs.iter().filter(|o| o.1).count() as u64;
    let (curve, median) = kaplan_meier(obs);
    Ok(Survival {
        event: "first failed tool call",
        axis: if turns { "turns" } else { "wall clock" },
        unit: if turns { "human prompts" } else { "minutes" },
        sessions,
        events,
        censored: sessions - events,
        unmeasured,
        median,
        curve,
    })
}

/// A change in the level of a daily series.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Change {
    /// The first day of the new level.
    pub day: String,
    pub mean_before: f64,
    pub mean_after: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Series {
    pub metric: String,
    pub basis: &'static str,
    pub time_basis: &'static str,
    pub incomplete: bool,
    pub response_completeness_unknown: bool,
    pub unknown_spend_time: bool,
    /// Validated measured observations lacking a source record time.
    pub unplaced_observations: u64,
    pub days: Vec<(String, f64)>,
    /// Mean of the 7 days before each day; `None` for the first week.
    pub baseline_7d: Vec<Option<f64>>,
    pub changes: Vec<Change>,
}

/// Splits a series where its mean shifts, by binary segmentation: a split
/// is kept when it lowers the squared error by more than a BIC-style
/// penalty, and each side is searched again. Returns the split indices.
pub fn changepoints(xs: &[f64], min_len: usize) -> Vec<usize> {
    fn cost(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        xs.iter().map(|x| (x - mean).powi(2)).sum()
    }
    let n = xs.len();
    if n < 2 * min_len {
        return vec![];
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let penalty = 2.0 * var.max(f64::MIN_POSITIVE) * (n as f64).ln();
    let mut out = vec![];
    let mut stack = vec![(0usize, n)];
    while let Some((a, b)) = stack.pop() {
        if b - a < 2 * min_len {
            continue;
        }
        let whole = cost(&xs[a..b]);
        let best = (a + min_len..=b - min_len)
            .map(|k| (k, cost(&xs[a..k]) + cost(&xs[k..b])))
            .min_by(|x, y| x.1.total_cmp(&y.1));
        if let Some((k, c)) = best
            && whole - c > penalty
        {
            out.push(k);
            stack.push((a, k));
            stack.push((k, b));
        }
    }
    out.sort_unstable();
    out
}

/// The daily series: name, the value each message (or row) adds, and the
/// rows it is read from.
const DAILY: [(&str, &str, &str); 5] = [
    (
        "tokens",
        "coalesce(m.input_tokens,0)+coalesce(m.cache_read_tokens,0)+coalesce(m.cache_write_tokens,0)
         +coalesce(m.output_tokens,0)+coalesce(m.reasoning_tokens,0)",
        "message m",
    ),
    ("api_cost", "coalesce(m.cost_nusd, 0) / 1e9", "message m"),
    ("sessions", "0", "message m"),
    (
        "prompts",
        "(m.written_by = 'human' AND m.prompt_kind = 'prose')",
        "message m",
    ),
    (
        "tool_errors",
        "(t.outcome = 'error')",
        "tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk",
    ),
];

const fn daily_names() -> [&'static str; DAILY.len()] {
    let mut out = [""; DAILY.len()];
    let mut i = 0;
    while i < DAILY.len() {
        out[i] = DAILY[i].0;
        i += 1;
    }
    out
}

static DAILY_NAMES: [&str; DAILY.len()] = daily_names();

/// The metrics [`daily`] takes.
pub static DAILY_METRICS: &[&str] = &DAILY_NAMES;

/// A daily series of `metric` (one of [`DAILY_METRICS`]) over logical
/// days. A day with no activity is 0.
pub fn daily(reader: &Reader<'_>, scope: &Scope, metric: &str, day: &LogicalDay) -> Result<Series> {
    let (cond, params) = scope.condition();
    let Some((_, expr, from)) = DAILY.iter().find(|(name, _, _)| *name == metric) else {
        return Err(crate::StoreError::Invalid(format!(
            "no daily metric `{metric}`: {}",
            DAILY_METRICS.join(", ")
        )));
    };
    let measured = matches!(metric, "tokens" | "api_cost");
    let mut response_unknown = measured
        && (usage::unknown_scope(reader, scope)?
            || usage::unknown_items(reader, &cond, &params)? > 0);
    let mut incomplete = response_unknown;
    let mut unknown_spend_time = false;
    let mut unplaced_observations = 0;
    let known_rows = if measured { usage::RESPONSE_ROWS } else { "1" };
    let missing = match metric {
        "tokens" => {
            "m.input_tokens IS NULL OR m.output_tokens IS NULL OR m.cache_read_tokens IS NULL OR m.cache_write_tokens IS NULL OR m.reasoning_tokens IS NULL"
        }
        "api_cost" => "m.cost_nusd IS NULL",
        _ => "0",
    };
    if measured {
        let untimed = Scope {
            since: None,
            until: None,
            ..scope.clone()
        };
        let (condition, values) = untimed.condition();
        incomplete |= reader.conn()?.query_row(&format!("SELECT EXISTS(SELECT 1 FROM message m WHERE {known_rows} AND m.at IS NULL AND {condition})"),rusqlite::params_from_iter(values.iter()),|r|r.get::<_,bool>(0))?;
    }
    let mut by_day: BTreeMap<jiff::civil::Date, f64> = BTreeMap::new();
    let mut sessions: BTreeMap<jiff::civil::Date, std::collections::HashSet<i64>> = BTreeMap::new();
    let mut stmt = reader.conn()?.prepare(&format!(
        "SELECT m.at, {expr}, m.session_pk, ({missing}) FROM {from} WHERE m.at IS NOT NULL AND {known_rows} AND {cond}"
    ))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, f64>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, bool>(3)?,
        ))
    })?;
    for row in rows {
        let (at, v, session, missing) = row?;
        incomplete |= missing;
        let d = day.date_of(at);
        *by_day.entry(d).or_default() += v;
        if metric == "sessions" {
            sessions.entry(d).or_default().insert(session);
        }
    }
    if measured {
        let recorded = usage::recorded_with(reader, scope, |at, observation, cost| {
            let Some(at) = at else {
                unplaced_observations += 1;
                return Ok(());
            };
            let value = if metric == "api_cost" {
                cost.map(|cost| cost as f64 / 1e9)
            } else {
                let u = &observation.usage;
                let mut sum = None::<u64>;
                for n in [
                    u.input,
                    u.cached_input,
                    u.cache_write,
                    u.output,
                    u.reasoning,
                ]
                .into_iter()
                .flatten()
                {
                    sum = Some(sum.unwrap_or(0).checked_add(n).ok_or_else(|| {
                        crate::StoreError::Invalid("daily recorded token sum exceeds u64".into())
                    })?);
                }
                sum.map(|n| n as f64)
            };
            if let Some(value) = value {
                *by_day.entry(day.date_of(at)).or_default() += value;
            } else {
                incomplete = true;
            }
            Ok(())
        })?;
        unknown_spend_time = recorded.untimed_observations > 0;
        response_unknown |= recorded.observations > 0;
        incomplete |= response_unknown
            || unknown_spend_time
            || recorded.unconfirmed_observations > 0
            || unplaced_observations > 0;
    }
    if metric == "sessions" {
        for (d, s) in &sessions {
            by_day.insert(*d, s.len() as f64);
        }
    }
    let mut days = vec![];
    if let (Some(first), Some(last)) =
        (by_day.keys().next().copied(), by_day.keys().last().copied())
    {
        let mut d = first;
        while d <= last {
            days.push((d.to_string(), by_day.get(&d).copied().unwrap_or(0.0)));
            d = d
                .tomorrow()
                .map_err(|e| crate::StoreError::Invalid(e.to_string()))?;
        }
    }
    let xs: Vec<f64> = days.iter().map(|d| d.1).collect();
    let baseline_7d = (0..xs.len())
        .map(|i| (!incomplete && i >= 7).then(|| xs[i - 7..i].iter().sum::<f64>() / 7.0))
        .collect();
    let cuts = if incomplete {
        vec![]
    } else {
        changepoints(&xs, 7)
    };
    let mut edges = vec![0];
    edges.extend(&cuts);
    edges.push(xs.len());
    let mean = |a: usize, b: usize| xs[a..b].iter().sum::<f64>() / (b - a) as f64;
    let changes = cuts
        .iter()
        .enumerate()
        .map(|(i, &k)| Change {
            day: days[k].0.clone(),
            mean_before: mean(edges[i], k),
            mean_after: mean(k, edges[i + 2]),
        })
        .collect();
    Ok(Series {
        metric: metric.to_owned(),
        basis: if measured {
            "recorded_known_subset"
        } else {
            "recorded"
        },
        time_basis: "source_record_time",
        incomplete,
        response_completeness_unknown: response_unknown,
        unknown_spend_time,
        unplaced_observations,
        days,
        baseline_7d,
        changes,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Transition {
    pub from: String,
    pub to: String,
    pub count: u64,
    /// Of `count`, how many followed a failed `from` call.
    pub after_failure: u64,
}

/// A failed call and the call that came next in the same session, as both
/// tools recorded them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FailureExample {
    pub session: String,
    pub failed: String,
    pub next: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Process {
    pub sessions: u64,
    pub calls: u64,
    /// Calls with no time; order still comes from their place in the
    /// transcript, never from the clock.
    pub calls_without_time: u64,
    pub outcomes: Outcomes,
    pub transitions: Vec<Transition>,
    /// The most frequent runs of three consecutive actions.
    pub common_sequences: Vec<(String, u64)>,
    pub failure_examples: Vec<FailureExample>,
}

/// Directly-follows relations between tool calls within each session, in
/// transcript order (message order key, then block position): two calls
/// with the same time keep their transcript order and nothing is inferred
/// from timestamps. Calls are named by action and command or tool.
pub fn process(reader: &Reader<'_>, scope: &Scope, top: usize) -> Result<Process> {
    let (cond, params) = scope.condition();
    let mut stmt = reader.conn()?.prepare(&format!(
        "SELECT s.id, t.action, coalesce(t.command_norm, t.tool), t.outcome, t.at
         FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
         JOIN message m ON m.pk = b.message_pk JOIN session s ON s.pk = m.session_pk
         WHERE t.inherited = 0 AND {cond}
         ORDER BY m.session_pk, m.order_input_id, m.seq, b.pk"
    ))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, Option<i64>>(4)?,
        ))
    })?;
    let mut pairs: HashMap<(String, String), (u64, u64)> = HashMap::new();
    let mut triples: HashMap<String, u64> = HashMap::new();
    let mut examples = vec![];
    let (mut calls, mut untimed, mut sessions) = (0u64, 0u64, 0u64);
    let mut prev: Option<(String, String, String)> = None;
    let mut prev2: Option<String> = None;
    for row in rows {
        let (session, action, what, outcome, at) = row?;
        calls += 1;
        untimed += u64::from(at.is_none());
        let name = format!("{action}:{what}");
        match &prev {
            Some((s, p, o)) if *s == session => {
                let e = pairs.entry((p.clone(), name.clone())).or_default();
                e.0 += 1;
                if o == "error" {
                    e.1 += 1;
                    if examples.len() < top {
                        examples.push(FailureExample {
                            session: session.clone(),
                            failed: p.clone(),
                            next: name.clone(),
                        });
                    }
                }
                if let Some(p2) = &prev2 {
                    *triples.entry(format!("{p2} → {p} → {name}")).or_default() += 1;
                }
                prev2 = Some(p.clone());
            }
            _ => {
                sessions += 1;
                prev2 = None;
            }
        }
        prev = Some((session, name, outcome));
    }
    let mut transitions: Vec<Transition> = pairs
        .into_iter()
        .map(|((from, to), (count, after_failure))| Transition {
            from,
            to,
            count,
            after_failure,
        })
        .collect();
    transitions.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.from.cmp(&b.from)));
    transitions.truncate(top);
    let mut common_sequences: Vec<(String, u64)> = triples.into_iter().collect();
    common_sequences.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    common_sequences.truncate(top);
    Ok(Process {
        sessions,
        calls,
        calls_without_time: untimed,
        outcomes: outcomes(reader, scope)?,
        transitions,
        common_sequences,
        failure_examples: examples,
    })
}

/// Human prompts by weekday (Monday first) and hour of the operator's day.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rhythm {
    pub zone: Option<String>,
    pub prompts: u64,
    pub grid: [[u64; 24]; 7],
}

pub fn rhythm(reader: &Reader<'_>, scope: &Scope, day: &LogicalDay) -> Result<Rhythm> {
    let (cond, params) = scope.condition();
    let mut stmt = reader.conn()?.prepare(&format!(
        "SELECT m.at FROM message m WHERE m.at IS NOT NULL AND m.written_by = 'human'
           AND m.prompt_kind = 'prose' AND {cond}"
    ))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| r.get::<_, i64>(0))?;
    let zone = &day.zone;
    let mut grid = [[0u64; 24]; 7];
    let mut prompts = 0;
    for at in rows {
        let Ok(ts) = jiff::Timestamp::from_millisecond(at?) else {
            continue;
        };
        let local = ts.to_zoned(zone.clone());
        let wd = local.weekday().to_monday_zero_offset() as usize;
        grid[wd][local.hour() as usize] += 1;
        prompts += 1;
    }
    Ok(Rhythm {
        zone: day.zone_name().map(str::to_owned),
        prompts,
        grid,
    })
}

/// A command whose most recent run in a repository failed: what is left to
/// fix there. A run counts by its
/// decided outcome (ok or error); a run with an unknown outcome decides
/// nothing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FailingCommand {
    /// The repository's root commit, or `cwd:<dir>` for sessions in no
    /// known repository.
    pub repository: String,
    /// Its remote or name, when known.
    pub repo: Option<String>,
    pub command: String,
    /// The failing run: when, in which session, with which exit code.
    pub at: Option<i64>,
    pub session: String,
    pub exit_code: Option<i64>,
    /// Failed runs since the last run that succeeded.
    pub failures_since_ok: u64,
    /// The last run that succeeded, if any did.
    pub last_ok_at: Option<i64>,
}

/// Every repository's unresolved commands in the scope, most recent first.
pub fn failing_commands(reader: &Reader<'_>, scope: &Scope) -> Result<Vec<FailingCommand>> {
    let (cond, params) = scope.condition();
    let sql = format!(
        "WITH runs AS (
           SELECT coalesce(s.repo_root, 'cwd:' || s.cwd) AS repository, s.repo AS repo,
                  t.command_norm AS command, t.outcome, t.exit_code, coalesce(t.at, m.at) AS at,
                  t.pk AS call, s.id AS session
           FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
           JOIN message m ON m.pk = b.message_pk JOIN session s ON s.pk = t.session_pk
           WHERE t.command_norm IS NOT NULL AND t.inherited = 0 AND t.outcome IN ('ok', 'error')
             AND coalesce(s.repo_root, s.cwd) IS NOT NULL AND {cond}),
         ranked AS (
           SELECT *, row_number() OVER (PARTITION BY repository, command
                                         ORDER BY at DESC NULLS LAST, call DESC) AS rn,
                  max(CASE WHEN outcome = 'ok' THEN at END)
                    OVER (PARTITION BY repository, command) AS last_ok
           FROM runs)
         SELECT repository, repo, command, at, session, exit_code, last_ok,
                (SELECT count(*) FROM runs r WHERE r.repository = ranked.repository
                   AND r.command = ranked.command AND r.outcome = 'error'
                   AND (ranked.last_ok IS NULL OR r.at > ranked.last_ok))
         FROM ranked WHERE rn = 1 AND outcome = 'error'
         ORDER BY at DESC NULLS LAST, repository, command"
    );
    let mut stmt = reader.conn()?.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
        Ok(FailingCommand {
            repository: r.get(0)?,
            repo: r.get(1)?,
            command: r.get(2)?,
            at: r.get(3)?,
            session: r.get(4)?,
            exit_code: r.get(5)?,
            last_ok_at: r.get(6)?,
            failures_since_ok: r.get::<_, i64>(7)? as u64,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// How long a session must have been quiet before it counts as over.
/// Inside it, a session is censored as possibly still going.
pub const QUIET: i64 = 6 * 3_600_000;

/// Time to a session ending on an unanswered prompt (abandonment): the
/// session went quiet for [`QUIET`] with its last human
/// prompt followed by no model message. A session that ended answered is
/// censored at its end, and so is one still within the quiet period.
/// Structural only: what the prompt asked is never read.
pub fn unanswered_end(reader: &Reader<'_>, scope: &Scope, now: i64) -> Result<Survival> {
    let (cond, params) = scope.condition();
    let sql = format!(
        "WITH s AS (
           SELECT m.session_pk AS pk, min(m.at) AS first_at, max(m.at) AS last_at
           FROM message m WHERE m.counted_as IS NULL AND {cond} GROUP BY m.session_pk)
         SELECT s.first_at, s.last_at,
           EXISTS (SELECT 1 FROM message h WHERE h.session_pk = s.pk
                   AND h.written_by = 'human' AND h.prompt_kind = 'prose'),
           NOT EXISTS (
             SELECT 1 FROM message a WHERE a.session_pk = s.pk AND a.role = 'assistant'
               AND (a.order_input_id, a.seq) > (
                 SELECT h.order_input_id, h.seq FROM message h WHERE h.session_pk = s.pk
                   AND h.written_by = 'human' AND h.prompt_kind = 'prose'
                 ORDER BY h.order_input_id DESC, h.seq DESC LIMIT 1))
         FROM s"
    );
    let mut obs = vec![];
    let mut unmeasured = 0;
    let mut stmt = reader.conn()?.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
        Ok((
            r.get::<_, Option<i64>>(0)?,
            r.get::<_, Option<i64>>(1)?,
            r.get::<_, bool>(2)?,
            r.get::<_, bool>(3)?,
        ))
    })?;
    for row in rows {
        let (first, last, prompted, unanswered) = row?;
        let (Some(first), Some(last)) = (first, last) else {
            unmeasured += 1;
            continue;
        };
        // A session no human prompted (a subagent, a compaction call) has no
        // prompt to leave unanswered.
        if !prompted {
            unmeasured += 1;
            continue;
        }
        let over = now - last >= QUIET;
        obs.push(((last - first) as f64 / 60_000.0, over && unanswered));
    }
    let sessions = obs.len() as u64;
    let events = obs.iter().filter(|o| o.1).count() as u64;
    let (curve, median) = kaplan_meier(obs);
    Ok(Survival {
        event: "ended on an unanswered prompt",
        axis: "wall clock",
        unit: "minutes",
        sessions,
        events,
        censored: sessions - events,
        unmeasured,
        median,
        curve,
    })
}

/// Something a format's parser did not know, across every input of that
/// format: format drift (spec analysis.md, schema
/// inference). New record types and fields appear here first.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Drift {
    pub format: String,
    /// `record:<type>`, `field:<json path>`, `user:<tag>`, `invalid_utf8`,
    /// `time_unparsed` or `undecodable`.
    pub what: String,
    pub count: u64,
    pub inputs: u64,
    /// When it was first and last seen, by record time.
    pub first_at: Option<i64>,
    pub last_at: Option<i64>,
    /// Kept only as a count by design: known, not drift.
    pub by_design: bool,
}

/// Every unrecognized item, newest first; items kept as counts by design
/// come after the rest.
pub fn drift(reader: &Reader<'_>) -> Result<Vec<Drift>> {
    let mut stmt = reader.conn()?.prepare(
        "SELECT format, what, sum(count), count(DISTINCT input_pk), min(first_at), max(last_at),
                max(counted_only)
         FROM unrecognized GROUP BY format, what
         ORDER BY max(counted_only), min(first_at) DESC NULLS LAST, format, what",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(Drift {
            format: r.get(0)?,
            what: r.get(1)?,
            count: r.get::<_, i64>(2)? as u64,
            inputs: r.get::<_, i64>(3)? as u64,
            first_at: r.get(4)?,
            last_at: r.get(5)?,
            by_design: r.get::<_, i64>(6)? == 1,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests;

//! Series over the operator's logical days: a daily series with its
//! baseline and level changes, and prompts by weekday and hour.

use super::{Selection, usage};
use crate::{Reader, Result};
use polylogue_model::LogicalDay;
use serde::Serialize;
use std::collections::BTreeMap;

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
    /// `recorded`, or `recorded_known_values` when rows without a value
    /// are left out of the sum.
    pub covers: &'static str,
    /// The days are those of each source record's time.
    pub timed_by: &'static str,
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
pub fn daily(reader: &Reader<'_>, selection: &Selection, metric: &str, day: &LogicalDay) -> Result<Series> {
    let (cond, params) = selection.condition();
    let Some((_, expr, from)) = DAILY.iter().find(|(name, _, _)| *name == metric) else {
        return Err(crate::StoreError::Invalid(format!(
            "no daily metric `{metric}`: {}",
            DAILY_METRICS.join(", ")
        )));
    };
    let measured = matches!(metric, "tokens" | "api_cost");
    let mut response_unknown = measured
        && (usage::unknown_scope(reader, selection)?
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
        let untimed = Selection {
            since: None,
            until: None,
            ..selection.clone()
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
        let recorded = usage::recorded_with(reader, selection, |at, observation, cost| {
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
        covers: if measured {
            "recorded_known_values"
        } else {
            "recorded"
        },
        timed_by: "source_record_time",
        incomplete,
        response_completeness_unknown: response_unknown,
        unknown_spend_time,
        unplaced_observations,
        days,
        baseline_7d,
        changes,
    })
}

/// Human prompts by weekday (Monday first) and hour of the operator's day.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rhythm {
    pub zone: Option<String>,
    pub prompts: u64,
    pub grid: [[u64; 24]; 7],
}

pub fn rhythm(reader: &Reader<'_>, selection: &Selection, day: &LogicalDay) -> Result<Rhythm> {
    let (cond, params) = selection.condition();
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

#[cfg(test)]
mod tests;

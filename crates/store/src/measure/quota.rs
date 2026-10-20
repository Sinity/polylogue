//! Subscription quota: for each tool's
//! 5-hour and weekly windows, how much is used, how fast it is burning, and
//! when it runs out. Windows come from what the tool reported (Codex sends
//! its windows with every token count; Claude Code records a limit only
//! when it refuses a turn); a window nothing reported is inferred and
//! labelled so. A figure no source gives is `None`.

use super::{Scope, UsageReport, usage};
use crate::{Reader, Result};
use polylogue_model::{QuotaLimit, QuotaReading, Timestamp};
use serde::Serialize;

const MINUTE: i64 = 60_000;
const HOUR: i64 = 60 * MINUTE;
pub const FIVE_HOURS: i64 = 300;
pub const WEEK: i64 = 10_080;

/// The subscription a window belongs to: the tool's limit name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Observation {
    /// Unix ms of the response that carried it.
    pub at: i64,
    /// The origin that reported it (`claude-code`, `codex`).
    pub tool: String,
    /// The tool's name for the limit (`codex`, `five_hour`, `seven_day`, …).
    pub limit: String,
    pub window_minutes: i64,
    /// The window starts with the first response after the previous one
    /// ended.
    pub from_first_response: bool,
    pub used_percent: Option<f64>,
    /// Unix ms.
    pub resets_at: Option<i64>,
    /// Claude Code: `allowed`, `allowed_warning` or `rejected`.
    pub status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Credits {
    pub balance: String,
    pub observed_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowSource {
    /// The tool reported this window's reset.
    Reported,
    /// Projected from an earlier reported reset by the window's length.
    Projected,
    /// Built from message times alone.
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Window {
    pub tool: String,
    pub limit: String,
    pub window_minutes: i64,
    pub source: WindowSource,
    pub starts_at: i64,
    pub resets_at: i64,
    /// The latest observation inside this window.
    pub observed_at: Option<i64>,
    pub used_percent: Option<f64>,
    pub remaining_percent: Option<f64>,
    /// Percent per hour, from two or more observations in the window.
    pub burn_percent_per_hour: Option<f64>,
    /// When the window was or will be used up, if before its reset.
    pub exhausted_at: Option<i64>,
    /// Burn known and the window lasts to its reset.
    pub lasts_to_reset: bool,
    pub status: Option<String>,
    pub usage: UsageReport,
    /// Tokens of every kind per hour since the window's first response.
    pub tokens_per_hour: Option<f64>,
    /// API-equivalent USD per hour, over priced responses.
    pub api_usd_per_hour: Option<f64>,
    /// Tokens at reset if the burn so far continues.
    pub projected_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolQuota {
    pub tool: String,
    pub plan: Option<String>,
    pub credits: Option<Credits>,
    pub windows: Vec<Window>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quota {
    pub now: i64,
    pub tools: Vec<ToolQuota>,
}

/// Every limit reading of one response.
fn observations(tool: &str, at: i64, reading: &QuotaReading) -> Vec<Observation> {
    reading
        .windows
        .iter()
        .map(|w| Observation {
            at,
            tool: tool.to_owned(),
            limit: w.limit.limit.to_string(),
            window_minutes: w.limit.minutes,
            from_first_response: w.limit.from_first_response,
            used_percent: w.used_percent,
            resets_at: w.resets_at.map(Timestamp::unix_ms),
            status: w.status.clone(),
        })
        .collect()
}

/// The window containing `now` for one limit, from its observations
/// (sorted by time). `None` when nothing was ever reported for it.
pub fn current_window(obs: &[Observation], now: i64) -> Option<(WindowSource, i64, i64)> {
    let len = obs.first()?.window_minutes * MINUTE;
    let latest_reset = obs.iter().filter_map(|o| o.resets_at).max()?;
    if latest_reset > now {
        return Some((WindowSource::Reported, latest_reset - len, latest_reset));
    }
    // Windows of a fixed-schedule limit follow one another; step from the
    // last reported reset to the one after now.
    let steps = (now - latest_reset) / len + 1;
    let end = latest_reset + steps * len;
    Some((WindowSource::Projected, end - len, end))
}

/// Five-hour blocks from response times (sorted), the way the tools bill
/// them: a block starts at the hour of the first response after the
/// previous block ended and lasts `len`. Reported windows (`anchors`) are
/// fixed and win: a response inside one belongs to it, and an inferred
/// block ends where a reported one begins, so a response before a reported
/// reset never moves the windows after it.
pub fn blocks(times: &[i64], anchors: &[(i64, i64)], len: i64) -> Vec<(WindowSource, i64, i64)> {
    let mut anchors: Vec<(i64, i64)> = anchors.to_vec();
    anchors.sort_unstable();
    anchors.dedup();
    let mut out: Vec<(WindowSource, i64, i64)> = anchors
        .iter()
        .map(|&(s, e)| (WindowSource::Reported, s, e))
        .collect();
    let mut open: Option<(i64, i64)> = None;
    for &t in times {
        if anchors.iter().any(|&(s, e)| s <= t && t < e) {
            continue;
        }
        if let Some((_, end)) = open
            && t < end
        {
            continue;
        }
        let start = t - t.rem_euclid(HOUR);
        let next_anchor = anchors.iter().map(|&(s, _)| s).filter(|&s| s > start).min();
        let end = next_anchor.map_or(start + len, |a| a.min(start + len));
        open = Some((start, end));
        out.push((WindowSource::Inferred, start, end));
    }
    out.sort_by_key(|&(_, s, _)| s);
    out
}

/// Reads every quota reading and builds the report for `now`. `known` names
/// the tools that have a subscription quota and the limits each has
/// whether or not a reading named them.
pub fn quota(reader: &Reader<'_>, now: i64, known: &[(&str, &[QuotaLimit])]) -> Result<Quota> {
    let conn = reader.conn();
    let mut obs: Vec<Observation> = vec![];
    let mut plans: Vec<(String, String)> = vec![];
    let mut credits: Vec<(String, Credits)> = vec![];
    let mut tools: Vec<String> = known.iter().map(|(t, _)| (*t).to_owned()).collect();
    {
        let mut stmt = conn.prepare(
            "SELECT s.origin, m.at, json_extract(m.attrs_json, '$.production.quota')
             FROM message m JOIN session s ON s.pk = m.session_pk
             WHERE m.at IS NOT NULL
               AND json_extract(m.attrs_json, '$.production.quota') IS NOT NULL
             ORDER BY m.at",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (tool, at, reading) = row?;
            let reading: QuotaReading = serde_json::from_str(&reading)?;
            if let Some(plan) = &reading.plan {
                plans.push((tool.clone(), plan.clone()));
            }
            if let Some(balance) = &reading.credits {
                credits.push((
                    tool.clone(),
                    Credits {
                        balance: balance.clone(),
                        observed_at: at,
                    },
                ));
            }
            obs.extend(observations(&tool, at, &reading));
            if !tools.contains(&tool) {
                tools.push(tool);
            }
        }
    }
    let mut out = vec![];
    for tool in &tools {
        let tool = tool.as_str();
        let mine: Vec<&Observation> = obs.iter().filter(|o| o.tool == tool).collect();
        let mut limits: Vec<(String, i64, bool)> = mine
            .iter()
            .map(|o| (o.limit.clone(), o.window_minutes, o.from_first_response))
            .collect();
        limits.sort();
        limits.dedup();
        let declared = known
            .iter()
            .filter(|(t, _)| *t == tool)
            .flat_map(|(_, l)| *l);
        for want in declared {
            if !limits.iter().any(|(_, m, _)| *m == want.minutes) {
                limits.push((
                    want.limit.to_string(),
                    want.minutes,
                    want.from_first_response,
                ));
            }
        }
        let mut windows = vec![];
        for (limit, minutes, from_first_response) in limits {
            let these: Vec<Observation> = mine
                .iter()
                .filter(|o| o.limit == limit && o.window_minutes == minutes)
                .map(|o| (*o).clone())
                .collect();
            let window = match current_window(&these, now) {
                Some(w) if w.0 == WindowSource::Reported || !from_first_response => Some(w),
                // A window that starts with the first response after the
                // previous one ended.
                _ if from_first_response => {
                    let times = response_times(reader, tool, now - 2 * minutes * MINUTE, now)?;
                    let anchors: Vec<(i64, i64)> = these
                        .iter()
                        .filter_map(|o| o.resets_at)
                        .map(|r| (r - minutes * MINUTE, r))
                        .collect();
                    blocks(&times, &anchors, minutes * MINUTE)
                        .into_iter()
                        .find(|&(_, s, e)| s <= now && now < e)
                }
                _ => Some((WindowSource::Inferred, now - minutes * MINUTE, now)),
            };
            if let Some((source, start, end)) = window {
                windows.push(build(
                    reader, tool, &limit, minutes, source, start, end, now, &these,
                )?);
            }
        }
        out.push(ToolQuota {
            tool: tool.to_owned(),
            plan: plans
                .iter()
                .rev()
                .find(|p| p.0 == tool)
                .map(|p| p.1.clone()),
            credits: credits
                .iter()
                .rev()
                .find(|c| c.0 == tool)
                .map(|c| c.1.clone()),
            windows,
        });
    }
    Ok(Quota { now, tools: out })
}

fn response_times(reader: &Reader<'_>, tool: &str, since: i64, until: i64) -> Result<Vec<i64>> {
    let mut stmt = reader.conn().prepare_cached(
        "SELECT m.at FROM message m JOIN session s ON s.pk = m.session_pk
         WHERE s.origin = ?1 AND m.role = 'assistant' AND m.at >= ?2 AND m.at < ?3
         ORDER BY m.at",
    )?;
    let rows = stmt.query_map(rusqlite::params![tool, since, until], |r| r.get(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[allow(clippy::too_many_arguments)]
fn build(
    reader: &Reader<'_>,
    tool: &str,
    limit: &str,
    minutes: i64,
    source: WindowSource,
    start: i64,
    end: i64,
    now: i64,
    obs: &[Observation],
) -> Result<Window> {
    let inside: Vec<&Observation> = obs.iter().filter(|o| start <= o.at && o.at < end).collect();
    let last = inside.last();
    let used = last.and_then(|o| o.used_percent);
    let burn = percent_burn(&inside);
    let exhausted_at = inside
        .iter()
        .find(|o| o.used_percent.is_some_and(|p| p >= 100.0))
        .map(|o| o.at)
        .or_else(|| {
            let (rate, o) = (burn?, last?);
            let left = 100.0 - o.used_percent?;
            (rate > 0.0)
                .then(|| o.at + (left / rate * HOUR as f64) as i64)
                .filter(|&t| t < end)
        });
    let usage = usage::report(
        reader,
        &Scope {
            since: Some(start),
            until: Some(now.min(end)),
            ..Scope::default()
        }
        .for_tool(tool),
    )?;
    let first = response_times(reader, tool, start, now.min(end))?
        .first()
        .copied();
    let hours = first.map(|f| (now.min(end) - f) as f64 / HOUR as f64);
    let tokens = total_tokens(&usage);
    let per_hour = |x: f64| hours.filter(|&h| h > 0.0).map(|h| x / h);
    let tokens_per_hour = per_hour(tokens as f64);
    let api_usd_per_hour = usage.api_cost.known_usd.and_then(per_hour);
    let projected_tokens =
        tokens_per_hour.map(|r| tokens + (r * (end - now).max(0) as f64 / HOUR as f64) as u64);
    Ok(Window {
        tool: tool.to_owned(),
        limit: limit.to_owned(),
        window_minutes: minutes,
        source,
        starts_at: start,
        resets_at: end,
        observed_at: last.map(|o| o.at),
        used_percent: used,
        remaining_percent: used.map(|u| (100.0 - u).max(0.0)),
        burn_percent_per_hour: burn,
        lasts_to_reset: burn.is_some() && exhausted_at.is_none(),
        exhausted_at,
        status: last.and_then(|o| o.status.clone()),
        usage,
        tokens_per_hour,
        api_usd_per_hour,
        projected_tokens,
    })
}

/// Percent per hour between the first and last reported use in the window,
/// when they are at least five minutes apart.
fn percent_burn(inside: &[&Observation]) -> Option<f64> {
    let with: Vec<&&Observation> = inside.iter().filter(|o| o.used_percent.is_some()).collect();
    let (a, b) = (with.first()?, with.last()?);
    let span = b.at - a.at;
    (span >= 5 * MINUTE).then(|| {
        (b.used_percent.unwrap_or(0.0) - a.used_percent.unwrap_or(0.0)).max(0.0)
            / (span as f64 / HOUR as f64)
    })
}

fn total_tokens(u: &UsageReport) -> u64 {
    let l = &u.tokens;
    l.input.tokens
        + l.cached_input.tokens
        + l.cache_write.tokens
        + l.output.tokens
        + l.reasoning.tokens
}

#[cfg(test)]
mod tests;

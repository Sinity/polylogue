//! The subscription-window outlook (old u654v, knc7, 64g7): for each tool's
//! 5-hour and weekly windows, how much is used, how fast it is burning, and
//! when it runs out. Windows come from what the tool reported (Codex sends
//! its windows with every token count; Claude Code records a limit only
//! when it refuses a turn); a window nothing reported is inferred and
//! labelled so. A figure no source gives is `None`.

use super::{Scope, UsageReport, usage};
use crate::{Reader, Result};
use serde::Serialize;
use serde_json::Value;

const MINUTE: i64 = 60_000;
const HOUR: i64 = 60 * MINUTE;
pub const FIVE_HOURS: i64 = 300;
pub const WEEK: i64 = 10_080;

/// The subscription a window belongs to: the tool's limit name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Observation {
    /// Unix ms of the response that carried it.
    pub at: i64,
    /// `claude-code` or `codex`.
    pub tool: String,
    /// The tool's name for the limit (`codex`, `five_hour`, `seven_day`, …).
    pub limit: String,
    pub window_minutes: i64,
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
pub enum Basis {
    /// The tool reported this window's reset.
    Reported,
    /// Projected from an earlier reported reset by the window's length.
    Cadence,
    /// Built from message times alone.
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Window {
    pub tool: String,
    pub limit: String,
    pub window_minutes: i64,
    pub basis: Basis,
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
    /// Tokens of every lane per hour since the window's first response.
    pub tokens_per_hour: Option<f64>,
    /// API-equivalent USD per hour, over priced responses.
    pub api_usd_per_hour: Option<f64>,
    /// Tokens at reset if the burn so far continues.
    pub projected_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolOutlook {
    pub tool: String,
    pub plan: Option<String>,
    pub credits: Option<Credits>,
    pub windows: Vec<Window>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Outlook {
    pub now: i64,
    pub tools: Vec<ToolOutlook>,
}

/// Reads one response's quota record into observations.
pub fn observations(tool: &str, at: i64, quota: &Value) -> Vec<Observation> {
    let num = |v: &Value| {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    };
    let mut out = vec![];
    match tool {
        "codex" => {
            let limit = quota["limit_id"].as_str().unwrap_or("codex").to_owned();
            for slot in ["primary", "secondary"] {
                let w = &quota[slot];
                let Some(minutes) = w["window_minutes"].as_i64() else {
                    continue;
                };
                let resets_at = w["resets_at"]
                    .as_i64()
                    .map(|s| s * 1000)
                    .or_else(|| w["resets_in_seconds"].as_i64().map(|s| at + s * 1000));
                out.push(Observation {
                    at,
                    tool: tool.to_owned(),
                    limit: limit.clone(),
                    window_minutes: minutes,
                    used_percent: num(&w["used_percent"]),
                    resets_at,
                    status: None,
                });
            }
        }
        "claude-code" => {
            let Some(kind) = quota["rateLimitType"].as_str() else {
                return out;
            };
            let minutes = match kind {
                "five_hour" => FIVE_HOURS,
                k if k.starts_with("seven_day") => WEEK,
                _ => return out,
            };
            let status = quota["status"].as_str().map(str::to_owned);
            out.push(Observation {
                at,
                tool: tool.to_owned(),
                limit: kind.to_owned(),
                window_minutes: minutes,
                // A refused turn means the window is used up; other states
                // do not say how much.
                used_percent: (status.as_deref() == Some("rejected")).then_some(100.0),
                resets_at: quota["resetsAt"].as_i64().map(|s| s * 1000),
                status,
            });
        }
        _ => {}
    }
    out
}

/// The window containing `now` for one limit, from its observations
/// (sorted by time). `None` when nothing was ever reported for it.
pub fn current_window(obs: &[Observation], now: i64) -> Option<(Basis, i64, i64)> {
    let len = obs.first()?.window_minutes * MINUTE;
    let latest_reset = obs.iter().filter_map(|o| o.resets_at).max()?;
    if latest_reset > now {
        return Some((Basis::Reported, latest_reset - len, latest_reset));
    }
    // Windows of a fixed-schedule limit follow one another; step from the
    // last reported reset to the one after now.
    let steps = (now - latest_reset) / len + 1;
    let end = latest_reset + steps * len;
    Some((Basis::Cadence, end - len, end))
}

/// Five-hour blocks from response times (sorted), the way the tools bill
/// them: a block starts at the hour of the first response after the
/// previous block ended and lasts `len`. Reported windows (`anchors`) are
/// fixed and win: a response inside one belongs to it, and an inferred
/// block ends where a reported one begins, so a response before a reported
/// reset never moves the windows after it.
pub fn blocks(times: &[i64], anchors: &[(i64, i64)], len: i64) -> Vec<(Basis, i64, i64)> {
    let mut anchors: Vec<(i64, i64)> = anchors.to_vec();
    anchors.sort_unstable();
    anchors.dedup();
    let mut out: Vec<(Basis, i64, i64)> = anchors
        .iter()
        .map(|&(s, e)| (Basis::Reported, s, e))
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
        out.push((Basis::Inferred, start, end));
    }
    out.sort_by_key(|&(_, s, _)| s);
    out
}

/// Reads every observation and builds the outlook for `now`.
pub fn outlook(reader: &Reader<'_>, now: i64) -> Result<Outlook> {
    let conn = reader.conn();
    let mut obs: Vec<Observation> = vec![];
    let mut plans: Vec<(String, String)> = vec![];
    let mut credits: Vec<(String, Credits)> = vec![];
    {
        let mut stmt = conn.prepare(
            "SELECT s.origin, m.at, json_extract(m.attrs_json, '$.production.quota')
             FROM message m JOIN session s ON s.pk = m.session_pk
             WHERE m.at IS NOT NULL AND s.origin IN ('codex', 'claude-code')
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
            let (tool, at, quota) = row?;
            let Ok(quota) = serde_json::from_str::<Value>(&quota) else {
                continue;
            };
            if let Some(plan) = quota["plan_type"].as_str() {
                plans.push((tool.clone(), plan.to_owned()));
            }
            if let Some(balance) = quota["credits"]["balance"].as_str() {
                credits.push((
                    tool.clone(),
                    Credits {
                        balance: balance.to_owned(),
                        observed_at: at,
                    },
                ));
            }
            obs.extend(observations(&tool, at, &quota));
        }
    }
    let mut tools = vec![];
    for tool in ["claude-code", "codex"] {
        let mine: Vec<&Observation> = obs.iter().filter(|o| o.tool == tool).collect();
        let mut limits: Vec<(String, i64)> = mine
            .iter()
            .map(|o| (o.limit.clone(), o.window_minutes))
            .collect();
        limits.sort();
        limits.dedup();
        if tool == "claude-code" {
            // Claude Code's windows exist whether or not a turn was refused.
            for want in [("five_hour", FIVE_HOURS), ("seven_day", WEEK)] {
                if !limits.iter().any(|(_, m)| *m == want.1) {
                    limits.push((want.0.to_owned(), want.1));
                }
            }
        }
        let mut windows = vec![];
        for (limit, minutes) in limits {
            let these: Vec<Observation> = mine
                .iter()
                .filter(|o| o.limit == limit && o.window_minutes == minutes)
                .map(|o| (*o).clone())
                .collect();
            let window = match current_window(&these, now) {
                Some(w) if w.0 == Basis::Reported || tool == "codex" || minutes == WEEK => Some(w),
                // Claude Code's five-hour windows start with the first
                // response after the previous one ended.
                _ if minutes == FIVE_HOURS => {
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
                _ => Some((Basis::Inferred, now - minutes * MINUTE, now)),
            };
            if let Some((basis, start, end)) = window {
                windows.push(build(
                    reader, tool, &limit, minutes, basis, start, end, now, &these,
                )?);
            }
        }
        tools.push(ToolOutlook {
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
    Ok(Outlook { now, tools })
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
    basis: Basis,
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
        basis,
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
    let l = &u.lanes;
    l.input.tokens
        + l.cached_input.tokens
        + l.cache_write.tokens
        + l.output.tokens
        + l.reasoning.tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_790_000_000_000;

    #[test]
    fn codex_slots_are_classified_by_length_not_name() {
        let q = serde_json::json!({"limit_id": "codex", "plan_type": "pro",
            "primary": {"used_percent": 42.0, "window_minutes": 10080, "resets_at": 1_790_600_000},
            "secondary": null});
        let o = observations("codex", T0, &q);
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].window_minutes, WEEK);
        assert_eq!(o[0].resets_at, Some(1_790_600_000_000));
        assert_eq!(o[0].used_percent, Some(42.0));
        let q = serde_json::json!({"primary": {"used_percent": 1, "window_minutes": 300,
            "resets_in_seconds": 60}});
        assert_eq!(
            observations("codex", T0, &q)[0].resets_at,
            Some(T0 + 60_000)
        );
    }

    #[test]
    fn a_refused_claude_turn_means_the_window_is_used_up() {
        let q = serde_json::json!({"status": "rejected", "resetsAt": 1_790_010_000,
            "rateLimitType": "five_hour"});
        let o = observations("claude-code", T0, &q);
        assert_eq!(o[0].used_percent, Some(100.0));
        let q = serde_json::json!({"status": "allowed_warning", "resetsAt": 1_790_010_000,
            "rateLimitType": "seven_day"});
        let o = observations("claude-code", T0, &q);
        assert_eq!((o[0].window_minutes, o[0].used_percent), (WEEK, None));
    }

    #[test]
    fn a_past_reset_is_carried_forward_by_the_window_length() {
        let o = vec![Observation {
            at: T0,
            tool: "codex".into(),
            limit: "codex".into(),
            window_minutes: WEEK,
            used_percent: Some(10.0),
            resets_at: Some(T0 + HOUR),
            status: None,
        }];
        assert_eq!(
            current_window(&o, T0),
            Some((Basis::Reported, T0 + HOUR - WEEK * MINUTE, T0 + HOUR))
        );
        let later = T0 + HOUR + 3 * MINUTE;
        assert_eq!(
            current_window(&o, later),
            Some((Basis::Cadence, T0 + HOUR, T0 + HOUR + WEEK * MINUTE))
        );
    }

    /// Law 109: windows after a reported reset are unchanged when an earlier
    /// message is inserted, and windows nothing reported are labelled
    /// inferred.
    #[test]
    fn law_109_windows_anchor_to_reported_resets() {
        let five = 5 * HOUR;
        let reset = T0 + 10 * HOUR;
        let anchors = [(reset - five, reset)];
        let times = vec![
            T0 + 7 * HOUR,
            T0 + 11 * HOUR,
            T0 + 17 * HOUR,
            T0 + 23 * HOUR,
        ];
        let before = blocks(&times, &anchors, five);
        let mut more = times.clone();
        more.insert(0, T0 + 2 * HOUR + 30 * MINUTE);
        let after = blocks(&more, &anchors, five);
        let after_reset = |b: &[(Basis, i64, i64)]| -> Vec<(Basis, i64, i64)> {
            b.iter()
                .copied()
                .filter(|&(_, s, _)| s >= reset - five)
                .collect()
        };
        assert_eq!(after_reset(&before), after_reset(&after));
        for (basis, s, e) in &after {
            let reported = anchors.contains(&(*s, *e));
            assert_eq!(
                *basis == Basis::Reported,
                reported,
                "{s}..{e} is labelled {basis:?}"
            );
        }
        // The inserted message starts an inferred block cut short where the
        // reported window begins.
        let first = after[0];
        assert_eq!(first.0, Basis::Inferred);
        assert_eq!(first.2, reset - five);
    }

    #[test]
    fn blocks_start_on_the_hour_and_last_five_hours() {
        let times = vec![T0 + 30 * MINUTE, T0 + 4 * HOUR, T0 + 6 * HOUR];
        let b = blocks(&times, &[], 5 * HOUR);
        let h = T0 - T0.rem_euclid(HOUR);
        assert_eq!(
            b,
            vec![
                (Basis::Inferred, h, h + 5 * HOUR),
                (Basis::Inferred, h + 6 * HOUR, h + 11 * HOUR)
            ]
        );
    }

    #[test]
    fn burn_needs_two_reports_five_minutes_apart() {
        let o = |at, p| Observation {
            at,
            tool: "codex".into(),
            limit: "codex".into(),
            window_minutes: FIVE_HOURS,
            used_percent: Some(p),
            resets_at: None,
            status: None,
        };
        let (a, b) = (o(T0, 10.0), o(T0 + HOUR, 30.0));
        assert_eq!(percent_burn(&[&a, &b]), Some(20.0));
        let c = o(T0 + MINUTE, 11.0);
        assert_eq!(percent_burn(&[&a, &c]), None);
    }
}

//! Subscription quota: for each tool's
//! 5-hour and weekly windows, how much is used, how fast it is burning, and
//! when it runs out. Windows come from what the tool reported (Codex sends
//! its windows with every token count; Claude Code records a limit only
//! when it refuses a turn); a window nothing reported is inferred and
//! labelled so. A figure no source gives is `None`.

use super::{Selection, UsageMeasure, usage};
use crate::{Reader, Result};
use polylogue_model::{QuotaLimit, QuotaReading, Timestamp};
use serde::Serialize;

const MINUTE: i64 = 60_000;
const HOUR: i64 = 60 * MINUTE;
#[cfg(test)]
pub const FIVE_HOURS: i64 = 300;
#[cfg(test)]
pub const WEEK: i64 = 10_080;

/// The subscription a window belongs to: the tool's limit name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Observation {
    /// Record/emission time, not necessarily when the provider measured it.
    pub at: Option<i64>,
    pub measured_at: Option<i64>,
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
    pub measured_at: Option<i64>,
    pub recorded_at: Option<i64>,
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
    /// The latest report inside this window.
    pub measured_at: Option<i64>,
    pub used_percent: Option<f64>,
    pub remaining_percent: Option<f64>,
    /// Percent per hour, from two or more observations in the window.
    pub burn_percent_per_hour: Option<f64>,
    /// When the window was or will be used up, if before its reset.
    pub exhausted_at: Option<i64>,
    /// Burn known and the window lasts to its reset.
    pub lasts_to_reset: bool,
    pub status: Option<String>,
    pub usage: Option<UsageMeasure>,
    /// Tokens of every kind per hour since the window's first response.
    pub tokens_per_hour: Option<f64>,
    /// API-equivalent money per hour, over priced responses.
    pub api_cost_per_hour: Option<polylogue_model::money::Amount>,
    /// Tokens at reset if the burn so far continues.
    pub projected_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolQuota {
    pub tool: String,
    pub plan: Option<String>,
    pub credits: Option<Credits>,
    pub windows: Vec<Window>,
    /// Readings that cannot be placed in a provider window.
    pub unplaced: Vec<Observation>,
    /// Readings whose reset is earlier than the reading itself: ignored for
    /// windows and burn.
    pub stale_readings: usize,
    /// Recorded usage whose spend time cannot be assigned to a window.
    pub recorded_usage: usage::RecordedUsage,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quota {
    pub now: i64,
    pub tools: Vec<ToolQuota>,
}

/// Every limit reading of one response. A reading whose window length is
/// not a positive number of minutes whose arithmetic fits cannot place a
/// window and is left out.
fn observations(tool: &str, at: Option<i64>, reading: &QuotaReading) -> Vec<Observation> {
    reading
        .windows
        .iter()
        .filter(|w| w.limit.minutes > 0 && w.limit.minutes.checked_mul(4 * MINUTE).is_some())
        .map(|w| Observation {
            at,
            measured_at: at,
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
/// previous block ended and lasts `len`. Reported windows (`reported_windows`) are
/// fixed and win: a response inside one belongs to it, and an inferred
/// block ends where a reported one begins, so a response before a reported
/// reset never moves the windows after it.
pub fn blocks(
    times: &[i64],
    reported_windows: &[(i64, i64)],
    len: i64,
) -> Vec<(WindowSource, i64, i64)> {
    let mut reported_windows: Vec<(i64, i64)> = reported_windows.to_vec();
    reported_windows.sort_unstable();
    reported_windows.dedup();
    let mut out: Vec<(WindowSource, i64, i64)> = reported_windows
        .iter()
        .map(|&(s, e)| (WindowSource::Reported, s, e))
        .collect();
    let mut open: Option<(i64, i64)> = None;
    for &t in times {
        if reported_windows.iter().any(|&(s, e)| s <= t && t < e) {
            continue;
        }
        if let Some((_, end)) = open
            && t < end
        {
            continue;
        }
        let start = t - t.rem_euclid(HOUR);
        let next_reported_start = reported_windows
            .iter()
            .map(|&(s, _)| s)
            .filter(|&s| s > start)
            .min();
        let end = next_reported_start.map_or(start + len, |a| a.min(start + len));
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
    let conn = reader.conn()?;
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
        let rows = stmt.query_map([], |r| <(String, i64, String)>::try_from(r))?;
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
                        measured_at: Some(at),
                        recorded_at: Some(at),
                    },
                ));
            }
            obs.extend(observations(&tool, Some(at), &reading));
            if !tools.contains(&tool) {
                tools.push(tool);
            }
        }
    }
    super::reports::visit::<polylogue_model::QuotaReport>(
        reader,
        &Selection::default(),
        polylogue_model::reports::QUOTA_REPORT,
        |tool, recorded_at, value| {
            if let Some(plan) = &value.reading.plan {
                plans.push((tool.to_owned(), plan.clone()));
            }
            if let Some(balance) = &value.reading.credits {
                credits.push((
                    tool.to_owned(),
                    Credits {
                        balance: balance.clone(),
                        measured_at: value.measured_at.map(Timestamp::unix_ms),
                        recorded_at,
                    },
                ));
            }
            // Unknown emission time sorts before timed records; no provider
            // timing is inferred from it.
            let mut readings = observations(tool, recorded_at, &value.reading);
            for reading in &mut readings {
                reading.measured_at = value.measured_at.map(Timestamp::unix_ms);
            }
            obs.extend(readings);
            if !tools.iter().any(|t| t == tool) {
                tools.push(tool.to_owned());
            }
            Ok(())
        },
    )?;
    // Provider measurement time orders readings; emission time stands in
    // where the provider gave none.
    obs.sort_by_key(|o| o.measured_at.or(o.at));
    let (obs, stale): (Vec<Observation>, Vec<Observation>) = obs.into_iter().partition(
        |o| !matches!((o.resets_at, o.measured_at.or(o.at)), (Some(reset), Some(at)) if reset < at),
    );
    let mut out = vec![];
    for tool in &tools {
        let tool = tool.as_str();
        let mine: Vec<&Observation> = obs.iter().filter(|o| o.tool == tool).collect();
        let stale_readings = stale.iter().filter(|o| o.tool == tool).count();
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
        let boundaries_unknown =
            usage::unknown_scope(reader, &Selection::default().for_origin(tool))?;
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
                    // Each block starts after the one before it ended, so
                    // the phase depends on every response since the chain began.
                    let times = response_times(reader, tool, i64::MIN, now)?;
                    let reported_windows: Vec<(i64, i64)> = these
                        .iter()
                        .filter_map(|o| o.resets_at)
                        .map(|r| (r - minutes * MINUTE, r))
                        .collect();
                    blocks(&times, &reported_windows, minutes * MINUTE)
                        .into_iter()
                        .find(|&(_, s, e)| s <= now && now < e)
                }
                _ if boundaries_unknown => None,
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
            unplaced: mine
                .iter()
                .filter(|o| o.resets_at.is_none() && o.measured_at.is_none())
                .map(|o| (*o).clone())
                .collect(),
            stale_readings,
            recorded_usage: usage::recorded(reader, &Selection::default().for_origin(tool))?,
        });
    }
    Ok(Quota { now, tools: out })
}

fn response_times(reader: &Reader<'_>, tool: &str, since: i64, until: i64) -> Result<Vec<i64>> {
    let mut stmt = reader.conn()?.prepare_cached(&format!(
        "SELECT m.at FROM message m JOIN session s ON s.pk = m.session_pk
         WHERE s.origin = ?1 AND {} AND m.at >= ?2 AND m.at < ?3
         ORDER BY m.at",
        usage::RESPONSE_ROWS
    ))?;
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
    let inside: Vec<&Observation> = obs
        .iter()
        .filter(|o| match o.measured_at {
            Some(at) => start <= at && at < end,
            None => o.resets_at == Some(end),
        })
        .collect();
    let last = inside.last();
    let used = last.and_then(|o| o.used_percent);
    let burn = last
        .and_then(|o| o.measured_at)
        .and_then(|_| percent_burn(&inside));
    let exhausted_at = inside
        .iter()
        .find(|o| o.used_percent.is_some_and(|p| p >= 100.0))
        .and_then(|o| o.measured_at)
        .or_else(|| {
            let (rate, o) = (burn?, last?);
            let left = 100.0 - o.used_percent?;
            (rate > 0.0)
                .then(|| {
                    o.measured_at
                        .map(|at| at + (left / rate * HOUR as f64) as i64)
                })
                .flatten()
                .filter(|&t| t < end)
        });
    let usage = usage::report(
        reader,
        &Selection {
            since: Some(start),
            until: Some(now.min(end)),
            ..Selection::default()
        }
        .for_origin(tool),
    )?;
    let first = response_times(reader, tool, start, now.min(end))?
        .first()
        .copied();
    let hours = first.map(|f| (now.min(end) - f) as f64 / HOUR as f64);
    let tokens = total_tokens(&usage);
    let per_hour = |x: f64| hours.filter(|&h| h > 0.0).map(|h| x / h);
    let unknown_time =
        usage::recorded(reader, &Selection::default().for_origin(tool))?.untimed_observations > 0;
    let tokens_per_hour = (!unknown_time && usage.messages.is_some())
        .then(|| per_hour(tokens as f64))
        .flatten();
    // Integer nano-USD over whole milliseconds, rounded once.
    let elapsed_ms = first.map(|f| now.min(end) - f).filter(|&ms| ms > 0);
    let api_cost_per_hour = (!unknown_time && usage.messages.is_some())
        .then(|| {
            let (known, ms) = (usage.api_cost.known?, elapsed_ms?);
            let per_hour = polylogue_model::money::rounded_quotient(
                i128::from(known.nusd()).checked_mul(i128::from(HOUR))?,
                i128::from(ms),
            )?;
            i64::try_from(per_hour)
                .ok()
                .map(polylogue_model::money::Amount)
        })
        .flatten();
    let projected_tokens =
        tokens_per_hour.map(|r| tokens + (r * (end - now).max(0) as f64 / HOUR as f64) as u64);
    Ok(Window {
        tool: tool.to_owned(),
        limit: limit.to_owned(),
        window_minutes: minutes,
        source,
        starts_at: start,
        resets_at: end,
        measured_at: last.and_then(|o| o.measured_at),
        used_percent: used,
        remaining_percent: used.map(|u| (100.0 - u).max(0.0)),
        burn_percent_per_hour: burn,
        lasts_to_reset: burn.is_some() && exhausted_at.is_none(),
        exhausted_at,
        status: last.and_then(|o| o.status.clone()),
        usage: (!unknown_time).then_some(usage),
        tokens_per_hour,
        api_cost_per_hour,
        projected_tokens,
    })
}

/// Percent per hour between the first and last reported use in the window,
/// when they are at least five minutes apart. A reading lower than the one
/// before it starts the burn over from itself.
fn percent_burn(inside: &[&Observation]) -> Option<f64> {
    let mut with: Vec<&&Observation> = inside
        .iter()
        .filter(|o| o.used_percent.is_some() && o.measured_at.is_some())
        .collect();
    if let Some(restart) = (1..with.len())
        .rev()
        .find(|&i| with[i].used_percent < with[i - 1].used_percent)
    {
        with.drain(..restart);
    }
    let (a, b) = (with.first()?, with.last()?);
    let span = b.measured_at? - a.measured_at?;
    (span >= 5 * MINUTE).then(|| {
        (b.used_percent.unwrap_or(0.0) - a.used_percent.unwrap_or(0.0)).max(0.0)
            / (span as f64 / HOUR as f64)
    })
}

fn total_tokens(u: &UsageMeasure) -> u64 {
    let l = &u.tokens;
    l.input.tokens
        + l.cached_input.tokens
        + l.cache_write.tokens
        + l.output.tokens
        + l.reasoning.tokens
}

/// One subscription limit as status shows it: the latest reading in the
/// window that contains now, or why there is none.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CurrentWindow {
    /// The origin whose subscription it is (`claude-code`, `codex`).
    pub tool: String,
    /// The tool's name for the limit (`five_hour`, `seven_day`, …).
    pub limit: String,
    pub window_minutes: i64,
    #[serde(flatten)]
    pub state: CurrentUse,
}

#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CurrentUse {
    Measured {
        used_percent: f64,
        /// When the tool measured it.
        measured_at: Timestamp,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resets_at: Option<Timestamp>,
        /// The tool's own word for the state (`allowed`, `rejected`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    NotMeasured {
        reason: NotMeasured,
    },
}

/// Why a limit has no use to show.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NotMeasured {
    /// Nothing reported the limit within its window's length.
    NoReading,
    /// The latest reading says the state but not how much is used (Claude
    /// Code records a percent only when it refuses a turn).
    NoPercent,
    /// The window of the latest reading has ended; the next reading
    /// measures the new one.
    WindowReset,
}

impl CurrentWindow {
    /// The same limit at `now`: a measurement whose window has ended by
    /// then is no longer the current one.
    pub fn at(mut self, now: i64) -> Self {
        if let CurrentUse::Measured {
            measured_at,
            resets_at,
            ..
        } = &self.state
        {
            let ends = resets_at.map_or_else(
                || {
                    measured_at
                        .unix_ms()
                        .saturating_add(self.window_minutes.saturating_mul(MINUTE))
                },
                Timestamp::unix_ms,
            );
            if ends <= now {
                self.state = CurrentUse::NotMeasured {
                    reason: NotMeasured::WindowReset,
                };
            }
        }
        self
    }
}

/// The current window of every limit `known` declares and every limit a
/// reading names, from the newest reading of each. Only readings within the
/// longest window's length can be in a current window, so this reads the
/// quota events of that span (by kind and time) and the messages of the
/// sessions active in it, newest first, and skips a session once every
/// limit of its tool has a reading at least as new as its last activity:
/// its cost follows the recent readings, not the archive's size.
pub fn current(
    reader: &Reader<'_>,
    now: i64,
    known: &[(&str, &[QuotaLimit])],
) -> Result<Vec<CurrentWindow>> {
    type Key = (String, String, i64);
    let longest = known
        .iter()
        .flat_map(|(_, limits)| limits.iter().map(|limit| limit.minutes))
        .chain([polylogue_model::WEEK_MINUTES])
        .max()
        .unwrap_or(polylogue_model::WEEK_MINUTES);
    let since = now.saturating_sub(longest.saturating_mul(MINUTE));
    let mut newest: std::collections::BTreeMap<Key, Observation> = Default::default();
    // Keeps the newest reading of each limit; a reset earlier than its
    // reading is stale (as in `quota`).
    fn offer(newest: &mut std::collections::BTreeMap<Key, Observation>, reading: Observation) {
        let Some(at) = reading.measured_at.or(reading.at) else {
            return;
        };
        if reading.resets_at.is_some_and(|reset| reset < at) {
            return;
        }
        let key = (
            reading.tool.clone(),
            reading.limit.clone(),
            reading.window_minutes,
        );
        if newest
            .get(&key)
            .is_none_or(|held| held.measured_at.or(held.at) < Some(at))
        {
            newest.insert(key, reading);
        }
    }
    let conn = reader.conn()?;
    {
        let mut stmt = conn.prepare(&format!(
            "SELECT s.origin, m.at, {} FROM event m JOIN session s ON s.pk = m.session_pk
             WHERE m.kind = ?1 AND m.at >= ?2 ORDER BY m.at DESC, m.id DESC",
            crate::content::sql_for(crate::content::DATA_BYTES_SQL, "m"),
        ))?;
        let mut rows = stmt.query(rusqlite::params![
            polylogue_model::reports::QUOTA_REPORT,
            since
        ])?;
        while let Some(row) = rows.next()? {
            let tool: String = row.get(0)?;
            let at: Option<i64> = row.get(1)?;
            let value: serde_json::Value = crate::encoding::read_json(&row.get::<_, Vec<u8>>(2)?)?;
            let report: polylogue_model::QuotaReport = serde_json::from_value(value)?;
            for mut reading in observations(&tool, at, &report.reading) {
                reading.measured_at = report.measured_at.map(Timestamp::unix_ms).or(at);
                offer(&mut newest, reading);
            }
        }
    }
    let tools: Vec<&str> = known.iter().map(|(tool, _)| *tool).collect();
    if !tools.is_empty() {
        let placeholders = (0..tools.len())
            .map(|n| format!("?{}", n + 2))
            .collect::<Vec<_>>()
            .join(",");
        let mut sessions = conn.prepare(&format!(
            "SELECT pk, origin, last_activity_at FROM session
             WHERE last_activity_at >= ?1 AND origin IN ({placeholders})
             ORDER BY last_activity_at DESC"
        ))?;
        let mut params: Vec<rusqlite::types::Value> = vec![since.into()];
        params.extend(tools.iter().map(|tool| (*tool).to_owned().into()));
        let recent: Vec<(i64, String, i64)> = sessions
            .query_map(rusqlite::params_from_iter(params), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut readings = conn.prepare(
            "SELECT at, json_extract(attrs_json, '$.production.quota') FROM message
             WHERE session_pk = ?1 AND at >= ?2
               AND json_extract(attrs_json, '$.production.quota') IS NOT NULL",
        )?;
        for (session, tool, last_activity) in recent {
            // Each limit of the tool, declared or read, is settled once a
            // reading of it is at least as new as this session's last
            // activity, which bounds every reading the session holds.
            let as_new = |held: &Observation| {
                held.measured_at
                    .or(held.at)
                    .is_some_and(|at| at >= last_activity)
            };
            let mine = || newest.iter().filter(|((t, _, _), _)| *t == tool);
            let mut declared = known
                .iter()
                .filter(|(t, _)| *t == tool)
                .flat_map(|(_, limits)| limits.iter())
                .peekable();
            let any = declared.peek().is_some() || mine().next().is_some();
            let settled =
                any && declared.all(|limit| {
                    mine().any(|((_, _, minutes), held)| *minutes == limit.minutes && as_new(held))
                }) && mine().all(|(_, held)| as_new(held));
            if settled {
                continue;
            }
            let rows = readings.query_map(rusqlite::params![session, since], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (at, reading) = row?;
                let reading: QuotaReading = serde_json::from_str(&reading)?;
                for reading in observations(&tool, Some(at), &reading) {
                    offer(&mut newest, reading);
                }
            }
        }
    }
    let mut windows: Vec<CurrentWindow> = Vec::new();
    for (tool, limits) in known {
        for limit in *limits {
            if !newest
                .keys()
                .any(|(t, _, minutes)| t == tool && *minutes == limit.minutes)
            {
                windows.push(CurrentWindow {
                    tool: (*tool).to_owned(),
                    limit: limit.limit.to_string(),
                    window_minutes: limit.minutes,
                    state: CurrentUse::NotMeasured {
                        reason: NotMeasured::NoReading,
                    },
                });
            }
        }
    }
    for ((tool, limit, window_minutes), reading) in newest {
        let measured_at = reading
            .measured_at
            .or(reading.at)
            .and_then(Timestamp::from_unix_ms);
        let state = match (reading.used_percent, measured_at) {
            (Some(used_percent), Some(measured_at)) => CurrentUse::Measured {
                used_percent,
                measured_at,
                resets_at: reading.resets_at.and_then(Timestamp::from_unix_ms),
                status: reading.status,
            },
            _ => CurrentUse::NotMeasured {
                reason: NotMeasured::NoPercent,
            },
        };
        windows.push(CurrentWindow {
            tool,
            limit,
            window_minutes,
            state,
        });
    }
    windows.sort_by(|a, b| {
        (&a.tool, a.window_minutes, &a.limit).cmp(&(&b.tool, b.window_minutes, &b.limit))
    });
    Ok(windows.into_iter().map(|window| window.at(now)).collect())
}

#[cfg(test)]
mod tests;

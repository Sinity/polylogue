//! Billing estimates from explicit settings and recorded usage, in UTC days.
use super::{Selection, UsageReport, usage};
use crate::{Reader, Result, StoreError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const DAY: i64 = 86_400_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub origin: String,
    pub cycle_anchor_day: Option<i8>,
    pub quota_unit: Option<String>,
    pub quota: Option<f64>,
    pub price_usd: Option<f64>,
    pub overage_rate_usd_per_unit: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Window {
    pub start: i64,
    pub end: i64,
    pub now: i64,
    pub elapsed_days: f64,
    pub remaining_days: f64,
    pub total_days: f64,
}

#[derive(Debug, Serialize)]
pub struct Projection {
    pub used: f64,
    pub burn_per_day: f64,
    pub projected: f64,
}

#[derive(Debug, Serialize)]
pub struct Pressure {
    pub unit: String,
    pub quota: f64,
    pub used: Option<f64>,
    pub projected: Option<f64>,
    pub used_ratio: Option<f64>,
    pub projected_ratio: Option<f64>,
    pub breach_day: Option<i64>,
    pub actual_overage: Option<f64>,
    pub projected_overage: Option<f64>,
    pub projected_overage_cost_usd: Option<f64>,
    pub reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Outlook {
    pub plan_name: Option<String>,
    pub plan: Option<Plan>,
    pub reason: Option<&'static str>,
    pub estimate: bool,
    pub projection_method: String,
    pub timed_by: &'static str,
    pub window: Option<Window>,
    pub units: BTreeMap<String, Projection>,
    pub quota_pressure: Option<Pressure>,
    pub quota_reason: Option<&'static str>,
    pub coverage_ratio: Option<f64>,
    pub incomplete_days: Vec<i64>,
    pub incomplete_usage: bool,
    pub usage: Option<UsageReport>,
}

/// A setting is a JSON object keyed by plan name; there are no default plans.
pub fn outlook(
    reader: &Reader<'_>,
    selection: &Selection,
    now: i64,
    options: &str,
    machine: &str,
) -> Result<Outlook> {
    selection.session_condition("s.pk")?;
    let (name, method) = options.split_once(':').unwrap_or((options, "linear"));
    validate_method(method)?;
    let mut answer = Outlook {
        plan_name: (!name.is_empty()).then(|| name.to_owned()),
        plan: None,
        reason: None,
        estimate: true,
        projection_method: method.into(),
        timed_by: "source_record_time_utc_billing_days",
        window: None,
        units: BTreeMap::new(),
        quota_pressure: None,
        quota_reason: None,
        coverage_ratio: None,
        incomplete_days: vec![],
        incomplete_usage: false,
        usage: None,
    };
    let Some(setting) = reader
        .setting("cost.plans", machine)?
        .filter(|s| !s.is_empty())
    else {
        answer.reason = Some("plans_not_configured");
        return Ok(answer);
    };
    let plans: BTreeMap<String, Plan> = serde_json::from_str(&setting)
        .map_err(|e| StoreError::Invalid(format!("cost.plans: {e}")))?;
    if name.is_empty() {
        answer.reason = Some("plan_name_required");
        return Ok(answer);
    }
    let plan = plans
        .get(name)
        .ok_or_else(|| StoreError::Invalid(format!("unknown configured plan `{name}`")))?
        .clone();
    validate_plan(&plan)?;
    answer.plan = Some(plan.clone());
    let Some(anchor) = plan.cycle_anchor_day else {
        answer.reason = Some("no_cycle_anchor");
        return Ok(answer);
    };
    let window = window(now, anchor)?;
    let mut scope = selection.clone().for_tool(&plan.origin);
    if selection
        .origins
        .as_ref()
        .is_some_and(|origins| !origins.contains(&plan.origin))
    {
        scope.sessions = Some(vec![]);
    }
    scope.since = Some(window.start);
    scope.until = Some(now.min(window.end));
    let total = usage(reader, &scope)?;
    answer.incomplete_usage = total.api_cost.unpriced_messages > 0
        || total.api_cost.response_total_unknown
        || total.reported_cost.unpriced_messages > 0
        || total.recorded_usage.untimed_observations > 0
        || total.recorded_usage.unconfirmed_observations > 0;
    let unbounded = Selection {
        since: None,
        until: None,
        ..scope.clone()
    };
    let (condition, params) = unbounded.condition();
    let untimed:u64=reader.conn()?.query_row(&format!("SELECT count(*) FROM message m WHERE {condition} AND m.at IS NULL AND m.role='assistant'"),rusqlite::params_from_iter(params.iter()),|r|r.get(0))?;
    answer.incomplete_usage |= untimed > 0;
    let mut daily: BTreeMap<String, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut observed_days = std::collections::BTreeSet::new();
    let mut day = window.start;
    while day < now && day < window.end {
        scope.since = Some(day);
        scope.until = Some((day + DAY).min(now).min(window.end));
        let report = usage(reader, &scope)?;
        if report.sessions > 0 {
            observed_days.insert(day);
        }
        for (unit, amount) in amounts(&report) {
            daily.entry(unit).or_default().insert(day, amount);
        }
        day += DAY;
    }
    for (unit, days) in &daily {
        answer.units.insert(
            unit.clone(),
            project(
                &days.values().copied().collect::<Vec<_>>(),
                window.elapsed_days,
                window.total_days,
                method,
            )?,
        );
    }
    let expected = window.elapsed_days.floor() as i64;
    answer.incomplete_days = (0..expected)
        .map(|d| window.start + d * DAY)
        .filter(|d| !observed_days.contains(d))
        .collect();
    answer.coverage_ratio = Some(if expected == 0 {
        1.0
    } else {
        1.0 - answer.incomplete_days.len() as f64 / expected as f64
    });
    if let (Some(unit), Some(quota)) = (&plan.quota_unit, plan.quota) {
        let projection = answer.units.get(unit);
        let mut running = 0.0;
        let mut breach = None;
        if let Some(days) = daily.get(unit) {
            for (day, amount) in days {
                running += amount;
                if running >= quota && breach.is_none() {
                    breach = Some(*day);
                }
            }
        }
        if breach.is_none()
            && let Some(p) = projection
            && p.projected >= quota
            && p.burn_per_day > 0.0
        {
            let at = now + (((quota - p.used).max(0.0) / p.burn_per_day).floor() as i64) * DAY;
            if at < window.end {
                breach = Some(at.div_euclid(DAY) * DAY);
            }
        }
        let over = projection.map(|p| (p.projected - quota).max(0.0));
        answer.quota_pressure = Some(Pressure {
            unit: unit.clone(),
            quota,
            used: projection.map(|p| p.used),
            projected: projection.map(|p| p.projected),
            used_ratio: projection.and_then(|p| (quota > 0.0).then_some(p.used / quota)),
            projected_ratio: projection.and_then(|p| (quota > 0.0).then_some(p.projected / quota)),
            breach_day: breach,
            actual_overage: projection.map(|p| (p.used - quota).max(0.0)),
            projected_overage: over,
            projected_overage_cost_usd: over
                .zip(plan.overage_rate_usd_per_unit)
                .map(|(a, b)| a * b),
            reason: projection.is_none().then_some("quota_unit_unobserved"),
        });
    } else {
        answer.quota_reason = Some("no_quota_configured");
    }
    answer.window = Some(window);
    answer.usage = Some(total);
    Ok(answer)
}

fn validate_method(method: &str) -> Result<()> {
    if matches!(method, "linear" | "trailing-7d-mean" | "eom-naive") {
        Ok(())
    } else {
        Err(StoreError::Invalid(
            "outlook method is linear, trailing-7d-mean or eom-naive".into(),
        ))
    }
}

fn validate_plan(plan: &Plan) -> Result<()> {
    if plan.origin.is_empty()
        || plan
            .cycle_anchor_day
            .is_some_and(|d| !(1..=31).contains(&d))
        || [plan.quota, plan.price_usd, plan.overage_rate_usd_per_unit]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite() || v < 0.0)
        || plan.quota_unit.as_ref().is_some_and(|s| s.is_empty())
    {
        return Err(StoreError::Invalid(
            "cost.plans needs an origin, an anchor day from 1 to 31 and nonnegative finite amounts"
                .into(),
        ));
    }
    Ok(())
}

fn amounts(report: &UsageReport) -> BTreeMap<String, f64> {
    let mut values = BTreeMap::new();
    if let Some(n) = report.api_cost.known_usd {
        values.insert("api_usd".into(), n);
    }
    if let Some(n) = report.reported_cost.known_usd {
        values.insert("reported_usd".into(), n);
    }
    for (unit, tokens) in [
        ("input_tokens", &report.tokens.input),
        ("cached_input_tokens", &report.tokens.cached_input),
        ("cache_write_tokens", &report.tokens.cache_write),
        ("output_tokens", &report.tokens.output),
        ("reasoning_tokens", &report.tokens.reasoning),
    ] {
        if report.known_response_messages > tokens.unknown_messages {
            values.insert(unit.into(), tokens.tokens as f64);
        }
    }
    // The existing usage reader has already separated independent observations
    // from response rows and refused unconfirmed observations from known sums.
    if let Some(n) = report.recorded_usage.known_api_cost_usd {
        *values.entry("api_usd".into()).or_default() += n;
    }
    let recorded = &report.recorded_usage.known_usage;
    for (unit, amount) in [
        ("input_tokens", recorded.input),
        ("cached_input_tokens", recorded.cached_input),
        ("cache_write_tokens", recorded.cache_write),
        ("output_tokens", recorded.output),
        ("reasoning_tokens", recorded.reasoning),
    ] {
        if let Some(n) = amount {
            *values.entry(unit.into()).or_default() += n as f64;
        }
    }
    values
}

pub fn project(daily: &[f64], elapsed: f64, total: f64, method: &str) -> Result<Projection> {
    validate_method(method)?;
    let used = daily.iter().sum();
    let burn = if elapsed > 0.0 { used / elapsed } else { 0.0 };
    let projected = if elapsed <= 0.0 {
        used
    } else if method == "trailing-7d-mean" && !daily.is_empty() {
        let tail = &daily[daily.len().saturating_sub(7)..];
        used + tail.iter().sum::<f64>() / tail.len() as f64 * (total - elapsed).max(0.0)
    } else {
        burn * total
    };
    Ok(Projection {
        used,
        burn_per_day: burn,
        projected,
    })
}

fn window(now: i64, anchor: i8) -> Result<Window> {
    let invalid = |e| StoreError::Invalid(format!("billing window: {e}"));
    let date = jiff::Timestamp::from_millisecond(now)
        .map_err(invalid)?
        .to_zoned(jiff::tz::TimeZone::UTC)
        .date();
    let first = jiff::civil::date(date.year(), date.month(), 1);
    let anchor_in = |month: jiff::civil::Date| -> Result<i64> {
        let mut day = anchor;
        let date = loop {
            if let Ok(d) = jiff::civil::Date::new(month.year(), month.month(), day) {
                break d;
            }
            day -= 1;
        };
        Ok(date
            .to_zoned(jiff::tz::TimeZone::UTC)
            .map_err(invalid)?
            .timestamp()
            .as_millisecond())
    };
    let this = anchor_in(first)?;
    let start_month = if this > now {
        first
            .checked_sub(jiff::Span::new().months(1))
            .map_err(invalid)?
    } else {
        first
    };
    let start = anchor_in(start_month)?;
    let end = anchor_in(
        start_month
            .checked_add(jiff::Span::new().months(1))
            .map_err(invalid)?,
    )?;
    Ok(Window {
        start,
        end,
        now,
        elapsed_days: (now - start) as f64 / DAY as f64,
        remaining_days: (end - now) as f64 / DAY as f64,
        total_days: (end - start) as f64 / DAY as f64,
    })
}

#[cfg(test)]
mod tests;

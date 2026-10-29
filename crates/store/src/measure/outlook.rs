//! Calendar billing outlooks from dated plan settings and grouped recorded usage.
use super::{Selection, reports, usage};
use crate::{Reader, Result, StoreError};
use polylogue_model::settings::PLAN_UNITS;
use rusqlite::params;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const DAY: i64 = 86_400_000;
pub use polylogue_model::settings::{Budget, Plan, Population};
#[derive(Debug, Serialize)]
pub struct Window {
    pub start: i64,
    pub end: i64,
    pub now: i64,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Projection {
    pub used: Option<i64>,
    pub burn_per_day: Option<f64>,
    pub projected: Option<i64>,
    pub known_items: u64,
    pub unknown_items: u64,
    pub lower_bound: bool,
    pub observed_days: u64,
    pub expected_days: u64,
    pub coverage_ratio: Option<f64>,
    pub missing_days: Vec<i64>,
}

#[derive(Debug, Serialize)]
pub struct Pressure {
    pub label: &'static str,
    pub lower_bound: bool,
    pub population: Population,
    pub unit: String,
    pub budget: i64,
    pub used: Option<i64>,
    pub projected: Option<i64>,
    pub used_ratio: Option<f64>,
    pub projected_ratio: Option<f64>,
    pub breach_day: Option<i64>,
    pub actual_overage: Option<i64>,
    pub projected_overage: Option<i64>,
    pub projected_overage_cost_nusd: Option<i64>,
    pub reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Segment {
    pub plan: Plan,
    pub from: i64,
    pub until: i64,
    pub setting_at: i64,
    pub active: bool,
    pub subscription_price_share_nusd: Option<i64>,
    pub responses: BTreeMap<String, Projection>,
    pub recorded_reports: BTreeMap<String, Projection>,
    pub unknown_boundary_items: u64,
    pub unconfirmed_reports: u64,
    pub budget_pressure: Option<Pressure>,
}

#[derive(Debug, Serialize)]
pub struct Outlook {
    pub provider: Option<String>,
    pub reason: Option<&'static str>,
    pub estimate: bool,
    pub projection_method: String,
    pub trailing_observed_days: Option<u64>,
    pub timed_by: &'static str,
    pub window: Option<Window>,
    pub segments: Vec<Segment>,
    pub untimed_responses: u64,
    pub untimed_boundary_items: u64,
    pub untimed_reports: u64,
    pub billing_before_setting: &'static str,
}

struct Setting {
    id: String,
    at: i64,
    value: String,
}

/// Archive-wide plan assertions apply from their own recorded time.
fn plans(reader: &Reader<'_>, provider: &str, now: i64) -> Result<Vec<(i64, Option<Plan>)>> {
    let rows=reader.conn()?.prepare("SELECT id,set_at,value FROM source_setting WHERE key=?1 AND machine='*' AND set_at<=?2 ORDER BY set_at,id")?.query_map(params![format!("plan.{provider}"),now],|r|Ok(Setting{id:r.get(0)?,at:r.get(1)?,value:r.get(2)?}))?.collect::<rusqlite::Result<Vec<_>>>()?;
    let times: BTreeSet<_> = rows.iter().map(|r| r.at).collect();
    let mut result = vec![];
    let mut previous = None;
    for at in times {
        let row = rows
            .iter()
            .filter(|r| r.at <= at)
            .max_by(|a, b| (a.at, &a.id).cmp(&(b.at, &b.id)))
            .expect("time belongs to a setting");
        if previous.as_ref() == Some(&row.id) {
            continue;
        }
        previous = Some(row.id.clone());
        let plan = if row.value.is_empty() {
            None
        } else {
            Some(
                serde_json::from_str::<Plan>(&row.value)
                    .map_err(|e| StoreError::Invalid(format!("plan.{provider}: {e}")))?,
            )
        };
        if let Some(p) = &plan {
            validate_plan(p)?;
        }
        result.push((at, plan));
    }
    Ok(result)
}

pub fn outlook(
    reader: &Reader<'_>,
    selection: &Selection,
    now: i64,
    options: &str,
) -> Result<Outlook> {
    selection.session_condition("s.pk")?;
    let (provider, method) = options.split_once(':').unwrap_or((options, "linear"));
    validate_method(method)?;
    let mut answer = Outlook {
        provider: (!provider.is_empty()).then(|| provider.into()),
        reason: None,
        estimate: true,
        projection_method: method.into(),
        trailing_observed_days: (method == "trailing-7d-mean").then_some(7),
        timed_by: "source_record_time_utc_billing_days",
        window: None,
        segments: vec![],
        untimed_responses: 0,
        untimed_boundary_items: 0,
        untimed_reports: 0,
        billing_before_setting: "unknown",
    };
    if provider.is_empty() {
        answer.reason = Some("provider_required");
        return Ok(answer);
    }
    let history = plans(reader, provider, now)?;
    let Some((_, Some(current))) = history.last() else {
        answer.reason = Some("plan_not_configured");
        return Ok(answer);
    };
    let Some(anchor) = current.cycle_anchor_day else {
        answer.reason = Some("no_cycle_anchor");
        return Ok(answer);
    };
    let window = window(now, anchor)?;
    let mut data = vec![];
    for (i, (at, plan)) in history.iter().enumerate() {
        let Some(plan) = plan else {
            continue;
        };
        let from = window.start.max(*at);
        let until = history.get(i + 1).map_or(now, |r| r.0).min(now);
        if until < from || (until == from && *at != now) {
            continue;
        }
        let active = i + 1 == history.len();
        let price = plan
            .price_nusd
            .map(|price| {
                let stop = if active { window.end } else { until };
                Ok::<_, StoreError>(
                    ratio(price, stop - window.start, window.end - window.start)?
                        - ratio(price, from - window.start, window.end - window.start)?,
                )
            })
            .transpose()?;
        answer.segments.push(Segment {
            plan: plan.clone(),
            from,
            until,
            setting_at: *at,
            active,
            subscription_price_share_nusd: price,
            responses: BTreeMap::new(),
            recorded_reports: BTreeMap::new(),
            unknown_boundary_items: 0,
            unconfirmed_reports: 0,
            budget_pressure: None,
        });
        data.push((
            BTreeMap::<String, Amounts>::new(),
            BTreeMap::<String, Amounts>::new(),
        ));
    }
    let scopes:Vec<_>=answer.segments.iter().enumerate().map(|(i,s)|serde_json::json!({"id":i,"from":s.from,"until":s.until,"origin":s.plan.origin})).collect();
    let (condition, mut values) = selection.condition();
    values.push(serde_json::to_string(&scopes)?.into());
    let scope_param = values.len();
    values.push(window.start.into());
    let start_param = values.len();
    values.push(now.into());
    let now_param = values.len();
    // One grouped response read serves every setting span and every unit.
    let mut statement=reader.conn()?.prepare(&format!("WITH spans AS (SELECT CAST(key AS INTEGER) AS id,json_extract(value,'$.from') AS starts,json_extract(value,'$.until') AS ends,json_extract(value,'$.origin') AS origin FROM json_each(?{scope_param}))
        SELECT w.id,(m.at-?{start_param})/{DAY},({response}) AS known,count(*),
        sum(m.cost_nusd),count(m.cost_nusd),sum(m.provider_cost_nusd),count(m.provider_cost_nusd),
        sum(m.input_tokens),count(m.input_tokens),sum(m.cache_read_tokens),count(m.cache_read_tokens),sum(m.cache_write_tokens),count(m.cache_write_tokens),sum(m.output_tokens),count(m.output_tokens),sum(m.reasoning_tokens),count(m.reasoning_tokens)
        FROM message m JOIN session s ON s.pk=m.session_pk LEFT JOIN spans w ON w.origin=s.origin AND m.at>=w.starts AND m.at<w.ends
        WHERE {condition} AND m.duplicate_of IS NULL AND m.role='assistant' AND m.written_by='model' AND (m.at IS NULL OR m.at<?{now_param}) AND (w.id IS NOT NULL OR (m.at IS NULL AND s.origin IN (SELECT origin FROM spans)))
        GROUP BY w.id,2,known",response=usage::RESPONSE_ROWS))?;
    let mut rows = statement.query(rusqlite::params_from_iter(values.iter()))?;
    while let Some(row) = rows.next()? {
        let n = row.get::<_, i64>(3)? as u64;
        let Some(i) = row.get::<_, Option<i64>>(0)? else {
            if row.get::<_, bool>(2)? {
                answer.untimed_responses += n;
            } else {
                answer.untimed_boundary_items += n;
            }
            continue;
        };
        let i = i as usize;
        let day = window.start + row.get::<_, i64>(1)? * DAY;
        if !row.get::<_, bool>(2)? {
            answer.segments[i].unknown_boundary_items += n;
            continue;
        }
        for (j, &unit) in PLAN_UNITS.iter().enumerate() {
            let known = row.get::<_, i64>(5 + j * 2)? as u64;
            data[i].0.entry(unit.into()).or_default().add(
                day,
                row.get(4 + j * 2)?,
                known,
                n - known,
            )?;
        }
    }
    let origins: Vec<_> = answer
        .segments
        .iter()
        .map(|s| s.plan.origin.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let origins = origins
        .into_iter()
        .filter(|o| {
            selection
                .origins
                .as_ref()
                .is_none_or(|wanted| wanted.contains(o))
        })
        .collect();
    let scope = Selection {
        origins: Some(origins),
        ..selection.clone()
    };
    // Report totals stay in their own population, apart from response totals.
    reports::visit::<polylogue_model::UsageReport>(
        reader,
        &scope,
        polylogue_model::reports::USAGE_REPORT,
        |origin, at, report| {
            let Some(at) = at else {
                answer.untimed_reports += 1;
                return Ok(());
            };
            if at >= now {
                return Ok(());
            }
            let Some(i) = answer
                .segments
                .iter()
                .position(|s| s.plan.origin == origin && s.from <= at && at < s.until)
            else {
                return Ok(());
            };
            if report.measured_completion != Some(true) {
                answer.segments[i].unconfirmed_reports += 1;
                return Ok(());
            }
            let day = window.start + (at - window.start) / DAY * DAY;
            let cost = polylogue_model::Catalog::builtin()
                .cost_nusd(report.model.as_deref(), &report.usage);
            let amounts = [
                cost,
                None,
                as_i64(report.usage.input)?,
                as_i64(report.usage.cached_input)?,
                as_i64(report.usage.cache_write)?,
                as_i64(report.usage.output)?,
                as_i64(report.usage.reasoning)?,
            ];
            for (unit, value) in PLAN_UNITS.iter().zip(amounts) {
                data[i].1.entry((*unit).into()).or_default().add(
                    day,
                    value,
                    u64::from(value.is_some()),
                    u64::from(value.is_none()),
                )?;
            }
            Ok(())
        },
    )?;
    for (segment, (responses, recorded)) in answer.segments.iter_mut().zip(data) {
        for (population, amounts) in [
            (Population::Responses, responses),
            (Population::RecordedReports, recorded),
        ] {
            let mut projections = BTreeMap::new();
            for (unit, amounts) in &amounts {
                let mut projection = project(
                    amounts,
                    segment.from,
                    segment.until,
                    if segment.active {
                        window.end
                    } else {
                        segment.until
                    },
                    method,
                )?;
                let excluded = match population {
                    Population::Responses => segment.unknown_boundary_items,
                    Population::RecordedReports => segment.unconfirmed_reports,
                };
                projection.lower_bound |= projection.used.is_some() && excluded > 0;
                projections.insert(unit.clone(), projection);
            }
            if let Some(budget) = segment
                .plan
                .budget
                .as_ref()
                .filter(|b| b.population == population)
            {
                segment.budget_pressure = Some(pressure(
                    budget,
                    projections.get(&budget.unit),
                    amounts.get(&budget.unit),
                    segment.until,
                    window.end,
                )?);
            }
            match population {
                Population::Responses => segment.responses = projections,
                Population::RecordedReports => segment.recorded_reports = projections,
            }
        }
    }
    answer.window = Some(window);
    Ok(answer)
}

fn as_i64(value: Option<u64>) -> Result<Option<i64>> {
    value
        .map(|v| {
            i64::try_from(v)
                .map_err(|_| StoreError::Invalid("usage exceeds SQLite's integer range".into()))
        })
        .transpose()
}
fn add(a: i64, b: i64) -> Result<i64> {
    a.checked_add(b)
        .ok_or_else(|| StoreError::Invalid("billing sum exceeds SQLite's integer range".into()))
}
fn ratio(amount: i64, numerator: i64, denominator: i64) -> Result<i64> {
    if denominator <= 0 {
        return Ok(amount);
    }
    i64::try_from(i128::from(amount) * i128::from(numerator) / i128::from(denominator)).map_err(
        |_| StoreError::Invalid("billing projection exceeds SQLite's integer range".into()),
    )
}

#[derive(Default)]
struct Amounts {
    days: BTreeMap<i64, i64>,
    known: u64,
    unknown: u64,
}
impl Amounts {
    fn add(&mut self, day: i64, amount: Option<i64>, known: u64, unknown: u64) -> Result<()> {
        self.known += known;
        self.unknown += unknown;
        if let Some(amount) = amount {
            let value = self.days.entry(day).or_default();
            *value = add(*value, amount)?;
        }
        Ok(())
    }
}
fn project(amounts: &Amounts, from: i64, now: i64, end: i64, method: &str) -> Result<Projection> {
    project_with_tail(amounts, from, now, end, method, 7)
}
fn project_with_tail(
    amounts: &Amounts,
    from: i64,
    now: i64,
    end: i64,
    method: &str,
    tail_days: usize,
) -> Result<Projection> {
    let used = amounts.days.values().try_fold(None, |sum, n| {
        Ok::<_, StoreError>(Some(add(sum.unwrap_or(0), *n)?))
    })?;
    let elapsed = now - from;
    let burn = used
        .filter(|_| elapsed > 0)
        .map(|n| n as f64 * DAY as f64 / elapsed as f64);
    let projected = used
        .map(|used| {
            if elapsed <= 0 {
                return Ok(used);
            }
            if method == "trailing-7d-mean" {
                let tail: Vec<_> = amounts
                    .days
                    .values()
                    .rev()
                    .take(tail_days)
                    .copied()
                    .collect();
                let sum = tail.iter().try_fold(0, |a, b| add(a, *b))?;
                add(
                    used,
                    ratio(sum, (end - now).max(0), tail.len() as i64 * DAY)?,
                )
            } else {
                ratio(used, end - from, elapsed)
            }
        })
        .transpose()?;
    let first_day = from.div_euclid(DAY) * DAY;
    let expected_days: Vec<_> = (first_day..now)
        .step_by(DAY as usize)
        .filter(|d| *d >= from && *d + DAY <= now)
        .collect();
    let missing_days: Vec<_> = expected_days
        .iter()
        .copied()
        .filter(|d| !amounts.days.contains_key(d))
        .collect();
    let coverage_ratio = (!expected_days.is_empty())
        .then(|| 1.0 - missing_days.len() as f64 / expected_days.len() as f64);
    Ok(Projection {
        used,
        burn_per_day: burn,
        projected,
        known_items: amounts.known,
        unknown_items: amounts.unknown,
        lower_bound: amounts.unknown > 0 && amounts.known > 0,
        observed_days: amounts.days.len() as u64,
        expected_days: expected_days.len() as u64,
        coverage_ratio,
        missing_days,
    })
}
fn pressure(
    budget: &Budget,
    projection: Option<&Projection>,
    amounts: Option<&Amounts>,
    now: i64,
    end: i64,
) -> Result<Pressure> {
    let used = projection.and_then(|p| p.used);
    let projected = projection.and_then(|p| p.projected);
    let mut breach = None;
    let mut running = 0;
    if budget.amount > 0 {
        if let Some(amounts) = amounts {
            for (&day, &amount) in &amounts.days {
                running = add(running, amount)?;
                if running >= budget.amount && breach.is_none() {
                    breach = Some(day);
                }
            }
        }
        if breach.is_none()
            && projected.is_some_and(|v| v >= budget.amount)
            && let Some(burn) = projection.and_then(|p| p.burn_per_day).filter(|v| *v > 0.0)
        {
            let at = now
                + (((budget.amount - used.unwrap_or(0)).max(0) as f64 / burn) * DAY as f64) as i64;
            if at < end {
                breach = Some(at.div_euclid(DAY) * DAY);
            }
        }
    }
    let over = projected.map(|p| (p - budget.amount).max(0));
    let over_cost = over
        .zip(budget.overage_rate_nusd_per_unit)
        .map(|(a, b)| {
            a.checked_mul(b).ok_or_else(|| {
                StoreError::Invalid("budget overage exceeds SQLite's integer range".into())
            })
        })
        .transpose()?;
    Ok(Pressure {
        label: "your budget",
        lower_bound: projection.is_some_and(|p| p.lower_bound),
        population: budget.population,
        unit: budget.unit.clone(),
        budget: budget.amount,
        used,
        projected,
        used_ratio: used
            .filter(|_| budget.amount > 0)
            .map(|n| n as f64 / budget.amount as f64),
        projected_ratio: projected
            .filter(|_| budget.amount > 0)
            .map(|n| n as f64 / budget.amount as f64),
        breach_day: breach,
        actual_overage: used.map(|u| (u - budget.amount).max(0)),
        projected_overage: over,
        projected_overage_cost_nusd: over_cost,
        reason: used.is_none().then_some("budget_unit_unobserved"),
    })
}
fn validate_method(method: &str) -> Result<()> {
    if matches!(method, "linear" | "trailing-7d-mean") {
        Ok(())
    } else {
        Err(StoreError::Invalid(
            "outlook method is linear or trailing-7d-mean".into(),
        ))
    }
}
fn validate_plan(plan: &Plan) -> Result<()> {
    plan.validate()
        .map_err(|error| StoreError::Invalid(error.to_string()))
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
    Ok(Window {
        start: anchor_in(start_month)?,
        end: anchor_in(
            start_month
                .checked_add(jiff::Span::new().months(1))
                .map_err(invalid)?,
        )?,
        now,
    })
}

#[cfg(test)]
mod tests;

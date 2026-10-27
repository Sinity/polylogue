//! Comparisons of explicit profile metrics and a named metadata similarity rule.
use super::{
    Selection,
    profiles::{self, Facts},
};
use crate::{Reader, Result, StoreError};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct Correlation {
    pub metric_x: String,
    pub metric_y: String,
    pub total_sessions: usize,
    pub sample_count: usize,
    pub missing_pairs: usize,
    pub underived_sessions: usize,
    pub pearson_r: Option<f64>,
    pub reason: Option<&'static str>,
}

pub fn correlation(
    reader: &Reader<'_>,
    selection: &Selection,
    options: &str,
) -> Result<Correlation> {
    let (x, y) = options
        .split_once(':')
        .ok_or_else(|| StoreError::Invalid("correlation takes metric_x:metric_y".into()))?;
    // Validate even an empty population.
    if !profiles::METRICS.contains(&x) || !profiles::METRICS.contains(&y) {
        return Err(StoreError::Invalid(format!(
            "profile metrics: {}",
            profiles::METRICS.join(", ")
        )));
    }
    let rows = metrics(reader, selection, &[x, y])?;
    let mut pairs = vec![];
    for row in &rows {
        if let (Some(x), Some(y)) = (row.values[0], row.values[1]) {
            pairs.push((x, y));
        }
    }
    let (pearson_r, reason) = pearson(&pairs);
    Ok(Correlation {
        metric_x: x.into(),
        metric_y: y.into(),
        total_sessions: rows.len(),
        sample_count: pairs.len(),
        missing_pairs: rows.len() - pairs.len(),
        underived_sessions: rows.iter().filter(|r| !r.derived).count(),
        pearson_r,
        reason,
    })
}

pub fn pearson(pairs: &[(f64, f64)]) -> (Option<f64>, Option<&'static str>) {
    if pairs.len() < 3 {
        return (None, Some("insufficient_pairs"));
    }
    let center = |values: Vec<f64>| {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let midpoint = min / 2.0 + max / 2.0;
        let shifted: Vec<_> = values.iter().map(|v| v - midpoint).collect();
        let scale = shifted.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let scaled: Vec<_> = shifted
            .iter()
            .map(|v| if scale > 0.0 { v / scale } else { *v })
            .collect();
        let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
        scaled.into_iter().map(|v| v - mean).collect::<Vec<_>>()
    };
    let x = center(pairs.iter().map(|p| p.0).collect());
    let y = center(pairs.iter().map(|p| p.1).collect());
    let denominator =
        x.iter().map(|v| v * v).sum::<f64>().sqrt() * y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if denominator == 0.0 {
        return (None, Some("zero_variance"));
    }
    (
        Some(
            ((x.iter().zip(&y).map(|(x, y)| x * y).sum::<f64>() / denominator).clamp(-1.0, 1.0)
                * 10_000.0)
                .round()
                / 10_000.0,
        ),
        None,
    )
}

/// Metadata similarity reads no message bodies, usage or parent histories.
/// The application pages this statement using Program's existing order keys.
pub fn similarity_query(reference_sql: &str) -> String {
    let tags = super::facets::tags_sql();
    let first = polylogue_model::session_sql::first("s");
    format!("WITH reference AS (SELECT _pk FROM ({reference_sql})),
        tags AS MATERIALIZED ({tags}),
        metadata AS MATERIALIZED (
            SELECT s.pk,s.id,s.origin,p.workflow_shape,
              {first} AS first_at,
              p.session_pk IS NULL AS underived,
              sum(p.session_pk IS NULL) OVER() AS underived_sessions
            FROM session s LEFT JOIN session_profile p ON p.session_pk=s.pk
            ),
        scores AS (
            SELECT m.pk AS _pk,m.id AS session,r.id AS reference,m.underived_sessions AS _underived,
              CASE WHEN r.workflow_shape<>'unknown' AND r.workflow_shape=m.workflow_shape THEN 3 ELSE 0 END AS shape_points,
              (r.origin=m.origin) AS origin_points,
              CASE WHEN abs((r.first_at/86400000-(r.first_at<0 AND r.first_at%86400000<>0))-(m.first_at/86400000-(m.first_at<0 AND m.first_at%86400000<>0)))<=3 THEN 2
                   WHEN abs((r.first_at/86400000-(r.first_at<0 AND r.first_at%86400000<>0))-(m.first_at/86400000-(m.first_at<0 AND m.first_at%86400000<>0)))<=14 THEN 1 ELSE 0 END AS time_points,
              (SELECT count(*) FROM tags a JOIN tags b ON a.value=b.value WHERE a.pk=r.pk AND b.pk=m.pk) AS tag_points
            FROM metadata m CROSS JOIN metadata r WHERE r.pk IN (SELECT _pk FROM reference) AND m.pk<>r.pk)
        SELECT *,shape_points+origin_points+time_points+tag_points AS score,
          (SELECT json_group_array(value) FROM json_each(json_array(
            CASE WHEN shape_points>0 THEN 'same workflow shape (+3)' END,
            CASE WHEN origin_points>0 THEN 'same origin (+1)' END,
            CASE WHEN time_points=2 THEN 'within 3 UTC days (+2)' WHEN time_points=1 THEN 'within 14 UTC days (+1)' END,
            CASE WHEN tag_points>0 THEN tag_points || ' shared tags (+' || tag_points || ')' END)) WHERE value IS NOT NULL) AS reasons
        FROM scores WHERE score>0")
}

#[derive(Debug, Serialize)]
pub struct MetricValue {
    pub name: String,
    pub value: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ComparedSession {
    pub session: String,
    pub origin: String,
    pub facts: Option<Facts>,
    pub metric: Option<MetricValue>,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub profiles: Vec<ComparedSession>,
    pub differences: BTreeMap<String, Vec<Value>>,
    pub underived_sessions: usize,
}

pub fn compare(reader: &Reader<'_>, selection: &Selection, name: &str) -> Result<Comparison> {
    if !name.is_empty() && !profiles::METRICS.contains(&name) {
        return Err(StoreError::Invalid(format!(
            "unknown profile metric `{name}`"
        )));
    }
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let mut statement = conn.prepare(&format!(
        "SELECT s.pk,s.id,s.origin FROM session s WHERE {condition} ORDER BY s.id"
    ))?;
    let rows = statement.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    let mut profiles = vec![];
    for row in rows {
        let (pk, session, origin) = row?;
        let facts = profiles::facts(conn, pk)?;
        let metric = if name.is_empty() {
            None
        } else {
            let cost = if name == "cost_usd" {
                own_cost(reader, pk)?
            } else {
                None
            };
            Some(MetricValue {
                name: name.into(),
                value: profiles::metric(facts.as_ref(), cost, name)?,
            })
        };
        profiles.push(ComparedSession {
            session,
            origin,
            facts,
            metric,
        });
    }
    let mut differences = BTreeMap::new();
    // Costs are read only when explicitly selected as a metric. A comparison
    // of structural facts needs neither pricing nor parent message history.
    for field in ["origin", "workflow_shape", "final_observed_state"]
        .into_iter()
        .chain(profiles::METRICS.iter().copied().filter(|m| {
            if name.is_empty() {
                *m != "cost_usd"
            } else {
                *m == name
            }
        }))
    {
        let mut values = BTreeMap::new();
        for profile in &profiles {
            let value = match field {
                "origin" => Value::String(profile.origin.clone()),
                "workflow_shape" => profile
                    .facts
                    .as_ref()
                    .map(|p| Value::String(p.workflow_shape.clone()))
                    .unwrap_or(Value::Null),
                "final_observed_state" => profile
                    .facts
                    .as_ref()
                    .map(|p| Value::String(p.final_observed_state.clone()))
                    .unwrap_or(Value::Null),
                _ => serde_json::to_value(match &profile.metric {
                    Some(metric) if metric.name == field => metric.value,
                    _ => profiles::metric(profile.facts.as_ref(), None, field)?,
                })?,
            };
            values.entry(value.to_string()).or_insert(value);
        }
        if values.len() > 1 {
            differences.insert(field.into(), values.into_values().collect());
        }
    }
    let underived_sessions = profiles.iter().filter(|p| p.facts.is_none()).count();
    Ok(Comparison {
        profiles,
        differences,
        underived_sessions,
    })
}

struct Metrics {
    derived: bool,
    values: Vec<Option<f64>>,
}

fn metrics(reader: &Reader<'_>, selection: &Selection, names: &[&str]) -> Result<Vec<Metrics>> {
    let (condition, params) = selection.session_condition("s.pk")?;
    let columns: Vec<_> = names
        .iter()
        .map(|name| metric_sql(name))
        .collect::<Result<_>>()?;
    let sql = format!(
        "SELECT s.pk,p.session_pk IS NOT NULL,{} FROM session s LEFT JOIN session_profile p ON p.session_pk=s.pk LEFT JOIN session_survival ss ON ss.session_pk=s.pk WHERE {condition}",
        columns.join(",")
    );
    let conn = reader.conn()?;
    let mut statement = conn.prepare(&sql)?;
    let mut rows = statement.query(rusqlite::params_from_iter(params.iter()))?;
    let mut result = vec![];
    while let Some(row) = rows.next()? {
        let mut values = vec![];
        for (i, name) in names.iter().enumerate() {
            values.push(if *name == "cost_usd" {
                own_cost(reader, row.get(0)?)?
            } else {
                row.get(i + 2)?
            });
        }
        result.push(Metrics {
            derived: row.get(1)?,
            values,
        });
    }
    Ok(result)
}

fn own_cost(reader: &Reader<'_>, pk: i64) -> Result<Option<f64>> {
    let own = super::usage::own_session(reader, pk)?;
    Ok(
        (own.api_cost.unpriced_messages == 0 && !own.api_cost.response_total_unknown)
            .then_some(own.api_cost.known_usd)
            .flatten(),
    )
}

fn metric_sql(name: &str) -> Result<String> {
    if !profiles::METRICS.contains(&name) {
        return Err(StoreError::Invalid(format!(
            "unknown profile metric `{name}`"
        )));
    }
    Ok(match name {
        "cost_usd" => "NULL".into(),
        "median_tool_call_ms" => "p.tool_latency_median_ms".into(),
        "median_agent_response_ms" => "p.agent_response_latency_median_ms".into(),
        "median_user_response_ms" => "p.user_response_latency_median_ms".into(),
        "wall_duration_ms" => format!(
            "CASE WHEN p.session_pk IS NOT NULL THEN {} END",
            polylogue_model::session_sql::wall_duration("s")
        ),
        _ => format!("p.{name}"),
    })
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub metric: String,
    pub total_sessions: usize,
    pub measured_sessions: usize,
    pub missing_sessions: usize,
    pub underived_sessions: usize,
    pub median: Option<f64>,
    pub p90: Option<f64>,
    pub max: Option<f64>,
}

pub fn distribution(
    reader: &Reader<'_>,
    selection: &Selection,
    name: &str,
) -> Result<Distribution> {
    if !profiles::METRICS.contains(&name) {
        return Err(StoreError::Invalid(format!(
            "unknown profile metric `{name}`"
        )));
    }
    let rows = metrics(reader, selection, &[name])?;
    let mut values = vec![];
    for row in &rows {
        if let Some(value) = row.values[0] {
            values.push(value);
        }
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = if n == 0 {
        None
    } else if n % 2 == 1 {
        Some(values[n / 2])
    } else {
        Some(values[n / 2 - 1] / 2.0 + values[n / 2] / 2.0)
    };
    Ok(Distribution {
        metric: name.into(),
        total_sessions: rows.len(),
        measured_sessions: n,
        missing_sessions: rows.len() - n,
        underived_sessions: rows.iter().filter(|r| !r.derived).count(),
        median,
        p90: if n == 0 {
            None
        } else {
            Some(values[((n as f64 * 0.9).ceil() as usize).saturating_sub(1)])
        },
        max: values.last().copied(),
    })
}

#[cfg(test)]
mod tests;

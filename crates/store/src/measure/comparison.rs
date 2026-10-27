//! Comparisons of explicit profile metrics and a named metadata similarity rule.
use super::{
    Selection,
    profiles::{self, Profile},
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
    let rows = reader.session_profiles(selection)?;
    let mut pairs = vec![];
    for row in &rows {
        if let (Some(x), Some(y)) = (profiles::metric(row, x)?, profiles::metric(row, y)?) {
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
        Some((x.iter().zip(&y).map(|(x, y)| x * y).sum::<f64>() / denominator).clamp(-1.0, 1.0)),
        None,
    )
}

#[derive(Debug, Serialize)]
pub struct Similarity {
    pub reference: String,
    pub session: String,
    pub score: u64,
    pub reasons: Vec<String>,
}

pub fn similarity(reference: &Profile, other: &Profile) -> Option<Similarity> {
    if reference.session == other.session {
        return None;
    }
    let mut score = 0;
    let mut reasons = vec![];
    if reference.facts.workflow_shape != "unknown"
        && reference.facts.workflow_shape == other.facts.workflow_shape
    {
        score += 3;
        reasons.push("same workflow shape (+3)".into());
    }
    if reference.origin == other.origin {
        score += 1;
        reasons.push("same origin (+1)".into());
    }
    if let (Some(a), Some(b)) = (reference.facts.first_at, other.facts.first_at) {
        let days = a.div_euclid(86_400_000).abs_diff(b.div_euclid(86_400_000));
        if days <= 14 {
            let points = if days <= 3 { 2 } else { 1 };
            score += points;
            reasons.push(format!(
                "within {} UTC days (+{points})",
                if days <= 3 { 3 } else { 14 }
            ));
        }
    }
    let common: std::collections::BTreeSet<_> = reference
        .tags
        .iter()
        .filter(|tag| other.tags.contains(tag))
        .collect();
    if !common.is_empty() {
        score += common.len() as u64;
        reasons.push(format!("{} shared tags (+{})", common.len(), common.len()));
    }
    (score > 0).then(|| Similarity {
        reference: reference.session.clone(),
        session: other.session.clone(),
        score,
        reasons,
    })
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub profiles: Vec<Profile>,
    pub differences: BTreeMap<String, Vec<Value>>,
    pub similarities: Vec<Similarity>,
}

pub fn compare(reader: &Reader<'_>, selection: &Selection) -> Result<Comparison> {
    let profiles = reader.session_profiles(selection)?;
    let mut differences = BTreeMap::new();
    for field in ["origin", "workflow_shape", "terminal_state"]
        .into_iter()
        .chain(profiles::METRICS.iter().copied())
    {
        let mut values = vec![];
        for profile in &profiles {
            let value = match field {
                "origin" => Value::String(profile.origin.clone()),
                "workflow_shape" => Value::String(profile.facts.workflow_shape.clone()),
                "terminal_state" => Value::String(profile.facts.terminal_state.clone()),
                _ => serde_json::to_value(profiles::metric(profile, field)?)?,
            };
            if !values.contains(&value) {
                values.push(value);
            }
        }
        values.sort_by_key(Value::to_string);
        if values.len() > 1 {
            differences.insert(field.into(), values);
        }
    }
    let mut similarities = vec![];
    for (i, a) in profiles.iter().enumerate() {
        for b in &profiles[i + 1..] {
            if let Some(s) = similarity(a, b) {
                similarities.push(s);
            }
        }
    }
    similarities.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then(a.reference.cmp(&b.reference))
            .then(a.session.cmp(&b.session))
    });
    Ok(Comparison {
        profiles,
        differences,
        similarities,
    })
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub metric: String,
    pub total_sessions: usize,
    pub measured_sessions: usize,
    pub missing_sessions: usize,
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
    let rows = reader.session_profiles(selection)?;
    let mut values = vec![];
    for row in &rows {
        if let Some(value) = profiles::metric(row, name)? {
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

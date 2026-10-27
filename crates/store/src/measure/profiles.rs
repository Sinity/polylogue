//! Session profile reads and calculations, with costs read from the usage owner.
use super::{Selection, SessionUsage};
use crate::{Reader, Result, StoreError};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CLUSTER_GAP_MS: i64 = 300_000;
pub const RULE_VERSION: i64 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Latency {
    pub measured_pairs: usize,
    pub median_ms: Option<i64>,
    pub p90_ms: Option<i64>,
    pub max_ms: Option<i64>,
}

pub(crate) fn latency(mut values: Vec<i64>) -> Latency {
    values.sort_unstable();
    let n = values.len();
    if n == 0 {
        return Latency::default();
    }
    let median = if n % 2 == 1 {
        values[n / 2]
    } else {
        values[n / 2 - 1] + (values[n / 2] - values[n / 2 - 1]) / 2
    };
    Latency {
        measured_pairs: n,
        median_ms: Some(median),
        p90_ms: Some(values[((n as f64 * 0.9).ceil() as usize).saturating_sub(1)]),
        max_ms: values.last().copied(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Facts {
    pub rule_version: i64,
    pub cluster_gap_ms: i64,
    pub topic: Option<String>,
    pub topic_source: String,
    pub first_at: Option<i64>,
    pub last_at: Option<i64>,
    pub timestamp_source: String,
    pub timestamped_messages: u64,
    pub message_count: u64,
    pub word_count: u64,
    pub substantive_count: u64,
    pub compaction_count: u64,
    pub total_duration_ms: Option<i64>,
    pub thinking_count: u64,
    pub wall_duration_ms: Option<i64>,
    pub engaged_duration_ms: i64,
    pub tool_active_duration_ms: i64,
    pub tool_latency: Latency,
    pub agent_response_latency: Latency,
    pub user_response_latency: Latency,
    pub stuck_tool_count: u64,
    pub tools: Vec<String>,
    pub tool_call_count: u64,
    pub actions: BTreeMap<String, u64>,
    pub workflow_shape: String,
    pub workflow_features: BTreeMap<String, f64>,
    pub terminal_state: String,
    pub terminal_state_method: String,
    pub terminal_message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Repository {
    pub identity: Option<String>,
    pub label: Option<String>,
    pub paths: Vec<String>,
}

/// Repository facts already observed for every recorded working directory.
pub(crate) fn repositories(
    conn: &rusqlite::Connection,
    pk: i64,
) -> Result<(Vec<Repository>, Vec<String>)> {
    let (cwd, label, identity): (Option<String>, Option<String>, Option<String>) = conn.query_row(
        "SELECT cwd_json,repo,repo_root FROM session WHERE pk=?1",
        [pk],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    let directories: Vec<String> = cwd
        .map(|s| serde_json::from_str(&s))
        .transpose()?
        .unwrap_or_default();
    let mut result = Vec::<Repository>::new();
    let mut add = |identity: Option<String>, label: Option<String>, path: Option<String>| {
        if identity.is_none() && label.is_none() {
            return;
        }
        let same = |r: &Repository| match (&r.identity, &identity) {
            (Some(a), Some(b)) => a == b,
            _ => r
                .label
                .as_ref()
                .zip(label.as_ref())
                .is_some_and(|(a, b)| a == b),
        };
        if let Some(existing) = result.iter_mut().find(|r| same(r)) {
            if identity.is_some() {
                existing.identity = identity;
            }
            if label.is_some() {
                existing.label = label;
            }
            if let Some(path) = path
                && !existing.paths.contains(&path)
            {
                existing.paths.push(path);
            }
        } else {
            result.push(Repository {
                identity,
                label,
                paths: path.into_iter().collect(),
            });
        }
    };
    let mut lookup=conn.prepare_cached("SELECT root_commit,remote,path FROM repo_lookup
        WHERE path=?1 OR substr(?1,1,length(path)+1)=path || '/' ORDER BY length(path) DESC LIMIT 1")?;
    for directory in &directories {
        let found: Option<(Option<String>, Option<String>, String)> = lookup
            .query_row([directory], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .optional()?;
        if let Some((identity, label, path)) = found {
            add(identity, label, Some(path));
        }
    }
    add(identity, label, None);
    Ok((result, directories))
}

#[derive(Debug, Serialize)]
pub struct Profile {
    pub session: String,
    pub title: Option<String>,
    pub origin: String,
    pub primary_model: Option<String>,
    pub harness: Option<String>,
    pub repositories: Vec<Repository>,
    pub working_directories: Vec<String>,
    pub tags: Vec<String>,
    pub facts: Facts,
    pub usage: Option<SessionUsage>,
}

impl Reader<'_> {
    pub fn session_profiles(&self, selection: &Selection) -> Result<Vec<Profile>> {
        let (condition, params) = selection.session_condition("s.pk")?;
        let conn = self.conn()?;
        let mut usage: BTreeMap<_, _> = super::sessions(self, selection)?
            .into_iter()
            .map(|u| (u.session.clone(), u))
            .collect();
        let mut statement = conn.prepare(&format!(
            "SELECT s.pk,s.id,s.title,s.origin,s.harness,p.facts FROM session s
            LEFT JOIN session_profile p ON p.session_pk=s.pk WHERE {condition} ORDER BY s.id"
        ))?;
        let mut rows = statement.query(rusqlite::params_from_iter(params.iter()))?;
        let mut result = vec![];
        while let Some(row) = rows.next()? {
            let session: String = row.get(1)?;
            let encoded: Option<String> = row.get(5)?;
            let encoded = encoded.ok_or_else(|| {
                StoreError::Invalid(format!("session profile not derived for {session}"))
            })?;
            let mut tags = conn.prepare(&format!(
                "SELECT value FROM ({}) WHERE pk=?1 AND value IS NOT NULL ORDER BY value",
                super::facets::TAGS
            ))?;
            let (repositories, working_directories) = repositories(conn, row.get(0)?)?;
            result.push(Profile {session:session.clone(),title:row.get(2)?,origin:row.get(3)?,harness:row.get(4)?,repositories,working_directories,
                tags:tags.query_map([row.get::<_,i64>(0)?],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?,
                primary_model:conn.query_row("SELECT model FROM message WHERE session_pk=?1 AND role='assistant' AND written_by='model' AND model IS NOT NULL GROUP BY model ORDER BY sum(output_tokens) DESC NULLS LAST,count(*) DESC,model LIMIT 1",[row.get::<_,i64>(0)?],|r|r.get(0)).optional()?,
                facts:serde_json::from_str(&encoded)?,usage:usage.remove(&session)});
        }
        Ok(result)
    }
}

pub fn profiles(reader: &Reader<'_>, selection: &Selection) -> Result<Vec<Profile>> {
    reader.session_profiles(selection)
}

/// Every numeric metric the comparison calculations accept.
pub const METRICS: &[&str] = &[
    "message_count",
    "word_count",
    "thinking_count",
    "substantive_count",
    "total_duration_ms",
    "tool_call_count",
    "wall_duration_ms",
    "engaged_duration_ms",
    "tool_active_duration_ms",
    "median_tool_call_ms",
    "median_agent_response_ms",
    "median_user_response_ms",
    "cost_usd",
];

pub fn metric(profile: &Profile, name: &str) -> Result<Option<f64>> {
    let p = &profile.facts;
    Ok(match name {
        "message_count" => Some(p.message_count as f64),
        "word_count" => Some(p.word_count as f64),
        "substantive_count" => Some(p.substantive_count as f64),
        "total_duration_ms" => p.total_duration_ms.map(|n| n as f64),
        "thinking_count" => Some(p.thinking_count as f64),
        "tool_call_count" => Some(p.tool_call_count as f64),
        "wall_duration_ms" => p.wall_duration_ms.map(|x| x as f64),
        "engaged_duration_ms" => Some(p.engaged_duration_ms as f64),
        "tool_active_duration_ms" => Some(p.tool_active_duration_ms as f64),
        "median_tool_call_ms" => p.tool_latency.median_ms.map(|x| x as f64),
        "median_agent_response_ms" => p.agent_response_latency.median_ms.map(|x| x as f64),
        "median_user_response_ms" => p.user_response_latency.median_ms.map(|x| x as f64),
        "cost_usd" => profile.usage.as_ref().and_then(|u| {
            let m = &u.own.api_cost;
            (m.unpriced_messages == 0 && !m.response_total_unknown)
                .then_some(m.known_usd)
                .flatten()
        }),
        _ => {
            return Err(StoreError::Invalid(format!(
                "unknown profile metric `{name}`: {}",
                METRICS.join(", ")
            )));
        }
    })
}

#[cfg(test)]
mod tests;

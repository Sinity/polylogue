//! Session profile reads and calculations, with costs read from the usage owner.
use super::{Selection, Totals};
use crate::{Reader, Result, StoreError};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CLUSTER_GAP_MS: i64 = 300_000;
pub const RULE_VERSION: i64 = 2;

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
    pub message_clustered_duration_ms: i64,
    pub tool_active_duration_ms: i64,
    pub tool_latency: Latency,
    pub agent_response_latency: Latency,
    pub user_response_latency: Latency,
    pub unpaired_tools_past_gap: u64,
    pub tools: Vec<String>,
    pub tool_call_count: u64,
    pub actions: BTreeMap<String, u64>,
    pub workflow_shape: String,
    pub workflow_features: BTreeMap<String, f64>,
    pub final_observed_state: String,
    pub final_observed_state_method: String,
    pub final_message: Option<String>,
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
    pub facts: Option<Facts>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub own: Totals,
    pub with_parents: Option<Totals>,
    pub parents: crate::read::Parents,
}

fn usage(reader: &Reader<'_>, pk: i64, session: &str) -> Result<Usage> {
    let id = polylogue_model::SessionId::parse(session)
        .map_err(|e| StoreError::Invalid(e.to_string()))?;
    let own = super::usage::own_session(reader, pk)?;
    let selected = reader.select_messages(&id, true)?;
    let parents = selected.parents().clone();
    let with_parents = if parents.complete {
        Some(super::usage::with_keys(reader, &selected.keys()?, &own)?)
    } else {
        None
    };
    Ok(Usage {
        own,
        with_parents,
        parents,
    })
}

impl Reader<'_> {
    pub fn underived_profiles(&self, selection: &Selection) -> Result<u64> {
        let (condition, params) = selection.session_condition("s.pk")?;
        Ok(self.conn()?.query_row(&format!("SELECT count(*) FROM session s LEFT JOIN session_profile p ON p.session_pk=s.pk WHERE {condition} AND p.session_pk IS NULL"),rusqlite::params_from_iter(params.iter()),|r|r.get::<_,i64>(0).map(|n|n as u64))?)
    }

    pub fn session_profiles(&self, selection: &Selection) -> Result<Vec<Profile>> {
        let (condition, params) = selection.session_condition("s.pk")?;
        let conn = self.conn()?;
        let mut statement = conn.prepare(&format!(
            "SELECT s.pk,s.id,coalesce(s.title,s.derived_title),s.origin,s.harness FROM session s
            WHERE {condition} ORDER BY s.id"
        ))?;
        let mut rows = statement.query(rusqlite::params_from_iter(params.iter()))?;
        let mut result = vec![];
        while let Some(row) = rows.next()? {
            let session: String = row.get(1)?;
            let mut tags = conn.prepare_cached(&format!(
                "SELECT value FROM ({}) WHERE pk=?1 AND value IS NOT NULL ORDER BY value",
                super::facets::tags_sql()
            ))?;
            let (repositories, working_directories) = repositories(conn, row.get(0)?)?;
            result.push(Profile {session:session.clone(),title:row.get(2)?,origin:row.get(3)?,harness:row.get(4)?,repositories,working_directories,
                tags:tags.query_map([row.get::<_,i64>(0)?],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?,
                primary_model:conn.query_row("SELECT model FROM message WHERE session_pk=?1 AND role='assistant' AND written_by='model' AND model IS NOT NULL GROUP BY model ORDER BY sum(output_tokens) DESC NULLS LAST,count(*) DESC,model LIMIT 1",[row.get::<_,i64>(0)?],|r|r.get(0)).optional()?,
                facts:facts(conn,row.get(0)?)?,usage:usage(self, row.get(0)?, &session)?});
        }
        Ok(result)
    }
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
    "message_clustered_duration_ms",
    "tool_active_duration_ms",
    "median_tool_call_ms",
    "median_agent_response_ms",
    "median_user_response_ms",
    "cost_usd",
];

pub fn metric(p: Option<&Facts>, cost: Option<f64>, name: &str) -> Result<Option<f64>> {
    if !METRICS.contains(&name) {
        return Err(StoreError::Invalid(format!(
            "unknown profile metric `{name}`: {}",
            METRICS.join(", ")
        )));
    }
    if name == "cost_usd" {
        return Ok(cost);
    }
    let Some(p) = p else {
        return Ok(None);
    };
    Ok(match name {
        "message_count" => Some(p.message_count as f64),
        "word_count" => Some(p.word_count as f64),
        "substantive_count" => Some(p.substantive_count as f64),
        "total_duration_ms" => p.total_duration_ms.map(|n| n as f64),
        "thinking_count" => Some(p.thinking_count as f64),
        "tool_call_count" => Some(p.tool_call_count as f64),
        "wall_duration_ms" => p.wall_duration_ms.map(|x| x as f64),
        "message_clustered_duration_ms" => Some(p.message_clustered_duration_ms as f64),
        "tool_active_duration_ms" => Some(p.tool_active_duration_ms as f64),
        "median_tool_call_ms" => p.tool_latency.median_ms.map(|x| x as f64),
        "median_agent_response_ms" => p.agent_response_latency.median_ms.map(|x| x as f64),
        "median_user_response_ms" => p.user_response_latency.median_ms.map(|x| x as f64),
        _ => {
            return Err(StoreError::Invalid(format!(
                "unknown profile metric `{name}`: {}",
                METRICS.join(", ")
            )));
        }
    })
}

pub(crate) fn bounds(
    conn: &rusqlite::Connection,
    pk: i64,
) -> Result<(Option<i64>, Option<i64>, String)> {
    Ok(conn.query_row(
        &format!(
            "SELECT {},{},{} FROM session s WHERE s.pk=?1",
            polylogue_model::session_sql::first("s"),
            polylogue_model::session_sql::last("s"),
            polylogue_model::session_sql::source("s")
        ),
        [pk],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?)
}

fn stored_count(n: impl TryInto<i64>) -> Result<i64> {
    n.try_into()
        .map_err(|_| StoreError::Invalid("profile count exceeds SQLite's integer range".into()))
}

/// Preserve the selected session identity even when display fields omit its ID.
pub fn page_query(selection_sql: &str) -> String {
    format!("SELECT q.*,s.id AS _session_id FROM ({selection_sql}) q JOIN session s ON s.pk=q._pk")
}

pub(crate) fn write_facts(conn: &rusqlite::Connection, pk: i64, facts: &Facts) -> Result<()> {
    conn.prepare_cached("INSERT OR REPLACE INTO session_profile(session_pk,rule_version,cluster_gap_ms,timestamped_messages,message_count,word_count,substantive_count,compaction_count,total_duration_ms,thinking_count,message_clustered_duration_ms,tool_active_duration_ms,tool_latency_measured_pairs,tool_latency_median_ms,tool_latency_p90_ms,tool_latency_max_ms,agent_response_latency_measured_pairs,agent_response_latency_median_ms,agent_response_latency_p90_ms,agent_response_latency_max_ms,user_response_latency_measured_pairs,user_response_latency_median_ms,user_response_latency_p90_ms,user_response_latency_max_ms,unpaired_tools_past_gap,tools,tool_call_count,actions,workflow_shape,workflow_features,final_observed_state,final_observed_state_method,final_message) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28,?29,?30,?31,?32,?33)")?.execute(rusqlite::params![pk,
        facts.rule_version,
        facts.cluster_gap_ms,
        stored_count(facts.timestamped_messages)?,
        stored_count(facts.message_count)?,
        stored_count(facts.word_count)?,
        stored_count(facts.substantive_count)?,
        stored_count(facts.compaction_count)?,
        facts.total_duration_ms,
        stored_count(facts.thinking_count)?,
        facts.message_clustered_duration_ms,
        facts.tool_active_duration_ms,
        stored_count(facts.tool_latency.measured_pairs)?,
        facts.tool_latency.median_ms,
        facts.tool_latency.p90_ms,
        facts.tool_latency.max_ms,
        stored_count(facts.agent_response_latency.measured_pairs)?,
        facts.agent_response_latency.median_ms,
        facts.agent_response_latency.p90_ms,
        facts.agent_response_latency.max_ms,
        stored_count(facts.user_response_latency.measured_pairs)?,
        facts.user_response_latency.median_ms,
        facts.user_response_latency.p90_ms,
        facts.user_response_latency.max_ms,
        stored_count(facts.unpaired_tools_past_gap)?,
        serde_json::to_string(&facts.tools)?,
        stored_count(facts.tool_call_count)?,
        serde_json::to_string(&facts.actions)?,
        facts.workflow_shape,
        serde_json::to_string(&facts.workflow_features)?,
        facts.final_observed_state,
        facts.final_observed_state_method,
        facts.final_message,
    ])?;
    Ok(())
}

pub(crate) fn facts(conn: &rusqlite::Connection, pk: i64) -> Result<Option<Facts>> {
    let raw = conn.prepare_cached("SELECT rule_version,cluster_gap_ms,timestamped_messages,message_count,word_count,substantive_count,compaction_count,total_duration_ms,thinking_count,message_clustered_duration_ms,tool_active_duration_ms,tool_latency_measured_pairs,tool_latency_median_ms,tool_latency_p90_ms,tool_latency_max_ms,agent_response_latency_measured_pairs,agent_response_latency_median_ms,agent_response_latency_p90_ms,agent_response_latency_max_ms,user_response_latency_measured_pairs,user_response_latency_median_ms,user_response_latency_p90_ms,user_response_latency_max_ms,unpaired_tools_past_gap,tools,tool_call_count,actions,workflow_shape,workflow_features,final_observed_state,final_observed_state_method,final_message FROM session_profile WHERE session_pk=?1")?
        .query_row([pk], |r| Ok((Facts {
            rule_version: r.get(0)?,
            cluster_gap_ms: r.get(1)?,
            first_at: None,
            last_at: None,
            timestamp_source: String::new(),
            timestamped_messages: r.get::<_,i64>(2)? as u64,
            message_count: r.get::<_,i64>(3)? as u64,
            word_count: r.get::<_,i64>(4)? as u64,
            substantive_count: r.get::<_,i64>(5)? as u64,
            compaction_count: r.get::<_,i64>(6)? as u64,
            total_duration_ms: r.get(7)?,
            thinking_count: r.get::<_,i64>(8)? as u64,
            wall_duration_ms: None,
            message_clustered_duration_ms: r.get(9)?,
            tool_active_duration_ms: r.get(10)?,
            tool_latency: Latency {
                measured_pairs: r.get::<_,i64>(11)? as usize,
                median_ms: r.get(12)?,
                p90_ms: r.get(13)?,
                max_ms: r.get(14)?,
            },
            agent_response_latency: Latency {
                measured_pairs: r.get::<_,i64>(15)? as usize,
                median_ms: r.get(16)?,
                p90_ms: r.get(17)?,
                max_ms: r.get(18)?,
            },
            user_response_latency: Latency {
                measured_pairs: r.get::<_,i64>(19)? as usize,
                median_ms: r.get(20)?,
                p90_ms: r.get(21)?,
                max_ms: r.get(22)?,
            },
            unpaired_tools_past_gap: r.get::<_,i64>(23)? as u64,
            tools: Default::default(),
            tool_call_count: r.get::<_,i64>(25)? as u64,
            actions: Default::default(),
            workflow_shape: r.get(27)?,
            workflow_features: Default::default(),
            final_observed_state: r.get(29)?,
            final_observed_state_method: r.get(30)?,
            final_message: r.get(31)?,
        }, r.get::<_,String>(24)?,r.get::<_,String>(26)?,r.get::<_,String>(28)?))).optional()?;
    let Some((mut facts, tools, actions, features)) = raw else {
        return Ok(None);
    };
    facts.tools = serde_json::from_str(&tools)?;
    facts.actions = serde_json::from_str(&actions)?;
    facts.workflow_features = serde_json::from_str(&features)?;
    let (first, last, source) = bounds(conn, pk)?;
    facts.first_at = first;
    facts.last_at = last;
    facts.timestamp_source = source;
    facts.wall_duration_ms = first.zip(last).map(|(a, b)| (b - a).max(0));
    Ok(Some(facts))
}

#[cfg(test)]
mod tests;

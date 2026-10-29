//! Read-time grouped session counts. No separate facet rows are stored.
use super::Selection;
use crate::session_sql;
use crate::{Reader, Result, StoreError};
use serde::Serialize;
use serde_json::Value;

pub const BASIC: &[&str] = &[
    "origins",
    "models",
    "repositories",
    "tags",
    "months",
    "ends_with",
];
pub const DEEP: &[&str] = &["roles", "written_by", "block_kinds", "actions", "flags"];

pub(super) fn tags_sql() -> String {
    format!("SELECT DISTINCT s.pk,lower(CASE WHEN n.key IS NULL THEN n.value WHEN n.value IS NULL THEN n.key ELSE n.key||':'||n.value END) AS value
        FROM session s JOIN source_note_target nt ON nt.ref='session:'||s.id JOIN source_note n ON n.id=nt.note_id
        WHERE n.kind='tag' AND {}",crate::note_sql::live_note("n"))
}

pub fn families(options: &str) -> Result<Vec<&'static str>> {
    match options {
        "" | "basic" => Ok(BASIC.to_vec()),
        "deep" => Ok(BASIC.iter().chain(DEEP).copied().collect()),
        name if BASIC.contains(&name) || DEEP.contains(&name) => Ok(BASIC
            .iter()
            .chain(DEEP)
            .copied()
            .filter(|f| *f == name)
            .collect()),
        _ => Err(StoreError::Invalid(format!(
            "facets takes basic, deep or a family: {}",
            BASIC
                .iter()
                .chain(DEEP)
                .copied()
                .collect::<Vec<_>>()
                .join(",")
        ))),
    }
}

/// Group the selection and the unit's defaults in one read transaction.
pub fn query(selection_sql: &str, population_sql: &str, options: &str) -> Result<String> {
    let own = session_sql::own_messages("m", None);
    let mut parts = vec![];
    for family in families(options)? {
        let sql=match family {
            "origins"=>"SELECT pk,origin AS value FROM session".into(),
            "models"=>format!("SELECT DISTINCT m.session_pk AS pk,m.model AS value FROM message m WHERE {own} AND m.written_by='model'"),
            "repositories"=>"SELECT pk,coalesce(repo_root,repo) AS value FROM session
                UNION SELECT s.pk,coalesce(rl.root_commit,rl.remote) FROM session s JOIN json_each(s.cwd_json) cwd
                JOIN repo_lookup rl ON rl.path=(SELECT r.path FROM repo_lookup r WHERE r.path=cwd.value OR substr(cwd.value,1,length(r.path)+1)=r.path||'/' ORDER BY length(r.path) DESC LIMIT 1)".into(),
            "tags"=>tags_sql(),
            "months"=>"SELECT pk,pl_month(started_at) AS value FROM session".into(),
            "ends_with"=>format!("SELECT s.pk,{} AS value FROM session s",session_sql::span("s","ends_with")),
            "roles"=>format!("SELECT DISTINCT m.session_pk AS pk,m.role AS value FROM message m WHERE {own}"),
            "written_by"=>format!("SELECT DISTINCT m.session_pk AS pk,m.written_by AS value FROM message m WHERE {own}"),
            "block_kinds"=>format!("SELECT DISTINCT m.session_pk AS pk,b.kind AS value FROM message m JOIN block b ON b.message_pk=m.pk WHERE {own}"),
            "actions"=>format!("SELECT DISTINCT m.session_pk AS pk,t.action AS value FROM message m JOIN block b ON b.message_pk=m.pk JOIN tool_call t ON t.call_block_pk=b.pk WHERE {own} AND {}",session_sql::counts_as_action("t")),
            "flags"=>format!("SELECT DISTINCT m.session_pk AS pk,CASE WHEN b.kind='thinking' THEN 'thinking' WHEN b.kind='tool_call' THEN 'tool_call' ELSE 'attachment' END AS value FROM message m JOIN block b ON b.message_pk=m.pk WHERE {own} AND b.kind IN ('thinking','tool_call','image','file')"),
            _=>unreachable!("checked family"),
        };
        parts.push(format!(
            "SELECT '{family}' AS family,f.pk,CAST(f.value AS TEXT) AS value FROM ({sql}) f"
        ));
    }
    Ok(format!("WITH selected AS MATERIALIZED (SELECT _pk FROM ({selection_sql})),
        population AS MATERIALIZED ({population_sql}),
        members AS MATERIALIZED ({}),
        eligible AS (SELECT _pk FROM selected UNION SELECT _pk FROM population)
        SELECT f.family,f.value,
          count(DISTINCT CASE WHEN f.pk IN (SELECT _pk FROM selected) THEN f.pk END) AS selected_count,
          count(DISTINCT CASE WHEN f.pk IN (SELECT _pk FROM population) THEN f.pk END) AS global_count
        FROM members f JOIN eligible e ON e._pk=f.pk
        GROUP BY f.family,f.value",parts.join(" UNION ALL ")))
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub outcome: &'static str,
    pub selected_sessions: u64,
    pub global_sessions: u64,
    pub pending: u64,
}

pub fn summary(reader: &Reader<'_>, selection: &Selection) -> Result<Summary> {
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let (selected_sessions, pending) = conn.query_row(
        &format!(
            "SELECT count(*),coalesce(sum(NOT ({})),0) FROM session s WHERE {condition}",
            session_sql::current("s", "session_span")
        ),
        rusqlite::params_from_iter(params.iter()),
        |r| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64)),
    )?;
    let population = Selection {
        sessions: selection.population.clone(),
        ..Default::default()
    };
    let (condition, params) = population.session_condition("s.pk")?;
    let global_sessions = conn.query_row(
        &format!("SELECT count(*) FROM session s WHERE {condition}"),
        rusqlite::params_from_iter(params.iter()),
        |r| Ok(r.get::<_, i64>(0)? as u64),
    )?;
    Ok(Summary {
        outcome: if selected_sessions == 0 {
            "empty"
        } else {
            "ok"
        },
        selected_sessions,
        global_sessions,
        pending,
    })
}

/// Convert grouped counts into weights after paging; every value uses full counts.
pub fn weights(rows: &mut [Value], summary: &Summary) {
    for row in rows {
        let selected = row["selected_count"].as_u64().unwrap_or(0);
        let global = row["global_count"].as_u64().unwrap_or(0);
        row["idf"] = if global > 0 && summary.global_sessions > 0 {
            Value::from((summary.global_sessions as f64 / global as f64).ln())
        } else {
            Value::Null
        };
        row["lift"] = if global > 0 && summary.global_sessions > 0 && summary.selected_sessions > 0
        {
            Value::from(
                (selected as f64 / summary.selected_sessions as f64)
                    / (global as f64 / summary.global_sessions as f64),
            )
        } else {
            Value::Null
        };
        row.as_object_mut()
            .unwrap()
            .retain(|k, _| !k.starts_with('_'));
    }
}

#[cfg(test)]
mod tests;

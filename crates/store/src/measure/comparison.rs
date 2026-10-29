//! Side-by-side recorded fields, with unknown and varying values kept visible.
use super::Selection;
use crate::{Reader, Result, StoreError};
use crate::session_sql;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const FIELDS: &[&str] = &[
    "origin",
    "messages",
    "human_prompts",
    "human_commands",
    "words",
    "human_words",
    "model_words",
    "thinking_messages",
    "compactions",
    "tool_calls",
    "errors",
    "wall_span",
    "clustered_time",
    "tool_time",
    "tool_time_union",
    "tool_untimed",
    "ends_with",
    "last_stop_reason",
    "waiting_model",
    "waiting_human",
    "cost_nusd",
];
const DEFAULT_FIELDS: &str = "origin,messages,human_prompts,words,tool_calls,errors,wall_span,clustered_time,tool_time,tool_untimed,ends_with";

#[derive(Debug, Serialize)]
pub struct ComparedSession {
    pub session: String,
    pub values: BTreeMap<String, Value>,
    pub pending: bool,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub sessions: Vec<ComparedSession>,
    pub varies: Vec<String>,
    pub unknown: BTreeMap<String, u64>,
    pub pending: u64,
    pub parameters: &'static [polylogue_model::derived::Parameter],
}

pub fn fields(options: &str) -> Result<Vec<&str>> {
    let names = if options.is_empty() {
        DEFAULT_FIELDS
    } else {
        options
    };
    let fields: Vec<_> = names.split(',').map(str::trim).collect();
    if fields.iter().any(|f| !FIELDS.contains(f)) {
        return Err(StoreError::Invalid(format!(
            "compare fields: {}",
            FIELDS.join(",")
        )));
    }
    Ok(fields)
}

pub fn compare(reader: &Reader<'_>, selection: &Selection, options: &str) -> Result<Comparison> {
    let fields = fields(options)?;
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let keys = conn
        .prepare(&format!(
            "SELECT s.pk,s.id,s.origin FROM session s WHERE {condition} ORDER BY s.id"
        ))?
        .query_map(rusqlite::params_from_iter(params.iter()), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let single = keys.len() == 1;
    let needs_time = fields.iter().any(|f| {
        matches!(
            *f,
            "wall_span"
                | "clustered_time"
                | "tool_time"
                | "tool_time_union"
                | "tool_untimed"
                | "ends_with"
                | "last_stop_reason"
                | "waiting_model"
                | "waiting_human"
        )
    });
    let needs_counts = fields.iter().any(|f| {
        !matches!(
            *f,
            "origin"
                | "cost_nusd"
                | "wall_span"
                | "clustered_time"
                | "tool_time"
                | "tool_time_union"
                | "tool_untimed"
                | "ends_with"
                | "last_stop_reason"
                | "waiting_model"
                | "waiting_human"
        )
    });
    let mut sessions = vec![];
    for (pk, session, origin) in keys {
        let time = if needs_time {
            Some(super::time::read(conn, pk, single)?)
        } else {
            None
        };
        let counts_current = crate::derive::is_current(conn, "counts", pk)?;
        let counts = if needs_counts && (counts_current || single) {
            // Counts use the same function at derivation and read boundaries.
            Some(crate::derive::counts::read(conn, pk, counts_current)?)
        } else {
            None
        };
        let pending = (needs_time && time.as_ref().is_none_or(|t| t.0.is_none()))
            || (needs_counts && counts.is_none());
        let count_values = counts.as_ref().map(serde_json::to_value).transpose()?;
        let mut values = BTreeMap::new();
        for &field in &fields {
            let span = time.as_ref().and_then(|t| t.0.as_ref());
            let lanes = time.as_ref().map(|t| t.1.as_slice()).unwrap_or_default();
            let value = match field {
                "origin" => json!(origin),
                "wall_span" => json!(span.and_then(|s| s.wall_ms)),
                "clustered_time" => json!(span.and_then(|s| s.clustered_ms)),
                "tool_time" => json!(span.and_then(|s| s.tool_sum_ms)),
                "tool_time_union" => json!(span.and_then(|s| s.tool_union_ms)),
                "tool_untimed" => json!(span.map(|s| s.tool_untimed)),
                "ends_with" => json!(span.map(|s| s.ends_with.as_str())),
                "last_stop_reason" => json!(span.and_then(|s| s.last_stop_reason.as_deref())),
                "waiting_model" | "waiting_human" => json!(
                    lanes
                        .iter()
                        .find(|l| l.lane
                            == if field == "waiting_model" {
                                "model"
                            } else {
                                "human"
                            })
                        .map(|l| l.ms)
                ),
                "cost_nusd" => {
                    let sql = cost_sql();
                    let value = conn.query_row(
                        &format!("SELECT {sql} FROM session s WHERE s.pk=?1"),
                        [pk],
                        |r| r.get::<_, Option<i64>>(0),
                    )?;
                    json!(value)
                }
                _ => count_values
                    .as_ref()
                    .map(|c| c[field].clone())
                    .unwrap_or(Value::Null),
            };
            values.insert(field.into(), value);
        }
        sessions.push(ComparedSession {
            session,
            values,
            pending,
        });
    }
    let mut varies = vec![];
    let mut unknown = BTreeMap::new();
    for field in fields {
        let values: std::collections::BTreeSet<_> = sessions
            .iter()
            .map(|s| s.values[field].to_string())
            .collect();
        if values.len() > 1 {
            varies.push(field.into());
        }
        unknown.insert(
            field.into(),
            sessions
                .iter()
                .filter(|s| s.values[field].is_null())
                .count() as u64,
        );
    }
    Ok(Comparison {
        pending: sessions.iter().filter(|s| s.pending).count() as u64,
        sessions,
        varies,
        unknown,
        parameters: polylogue_model::derived::TIME_PARAMETERS,
    })
}

/// Preserve identity when display fields omit the session ID.
pub fn page_query(selection_sql: &str) -> String {
    format!("SELECT q.*,s.id AS _session_id FROM ({selection_sql}) q JOIN session s ON s.pk=q._pk")
}

/// Current stored fields used to compare the full selection without loading it.
fn field_sql(name: &str) -> Result<String> {
    fields(name)?;
    Ok(match name {
        "origin" => "s.origin".into(),
        "messages" => stored_count("s", "message_count"),
        "human_prompts" => stored_count("s", "human_prompt_count"),
        "human_commands" => stored_count("s", "human_command_count"),
        "words" => stored_count("s", "word_count"),
        "human_words" => stored_count("s", "human_word_count"),
        "model_words" => stored_count("s", "model_word_count"),
        "thinking_messages" => stored_count("s", "thinking_count"),
        "compactions" => stored_count("s", "compaction_count"),
        "tool_calls" => stored_count("s", "tool_call_count"),
        "errors" => stored_count("s", "error_count"),
        "wall_span" => session_sql::span("s", "wall_ms"),
        "clustered_time" => session_sql::span("s", "clustered_ms"),
        "tool_time" => session_sql::span("s", "tool_sum_ms"),
        "tool_time_union" => session_sql::span("s", "tool_union_ms"),
        "tool_untimed" => session_sql::span("s", "tool_untimed"),
        "ends_with" | "last_stop_reason" => session_sql::span("s", name),
        "waiting_model" | "waiting_human" => format!(
            "(SELECT ms FROM session_time st WHERE st.session_pk=s.pk AND st.lane='{}' AND st.key='' AND {})",
            if name == "waiting_model" {
                "model"
            } else {
                "human"
            },
            session_sql::current("s", "session_span")
        ),
        "cost_nusd" => cost_sql(),
        _ => unreachable!("checked field"),
    })
}

fn stored_count(s: &str, column: &str) -> String {
    format!(
        "CASE WHEN {} THEN {s}.{column} END",
        session_sql::current(s, "counts")
    )
}

fn cost_sql() -> String {
    format!(
        "(SELECT CASE WHEN count(*)=count(m.cost_nusd) AND NOT EXISTS(SELECT 1 FROM event e WHERE e.session_pk=s.pk AND e.kind='response_boundaries_unknown') AND NOT EXISTS(SELECT 1 FROM message x WHERE x.session_pk=s.pk AND x.duplicate_of IS NULL AND json_extract(x.attrs_json,'$.production.response_boundary_unknown')=1) THEN sum(m.cost_nusd) END FROM message m WHERE m.session_pk=s.pk AND m.duplicate_of IS NULL AND {})",
        super::usage::RESPONSE_ROWS
    )
}

/// Full-selection differences and unknown counts are grouped in SQL, never paged.
pub fn summary(
    reader: &Reader<'_>,
    selection: &Selection,
    options: &str,
) -> Result<(Vec<String>, BTreeMap<String, u64>, u64)> {
    if selection.sessions.as_ref().is_some_and(|s| s.len() == 1) {
        let c = compare(reader, selection, options)?;
        return Ok((c.varies, c.unknown, c.pending));
    }
    let fields = fields(options)?;
    let mut columns = vec![];
    for field in &fields {
        let sql = field_sql(field)?;
        columns.push(format!(
            "count(DISTINCT {sql}) + (count(*)>count({sql})),count(*)-count({sql})"
        ));
    }
    let needs_time = fields.iter().any(|f| {
        matches!(
            *f,
            "wall_span"
                | "clustered_time"
                | "tool_time"
                | "tool_time_union"
                | "tool_untimed"
                | "ends_with"
                | "last_stop_reason"
                | "waiting_model"
                | "waiting_human"
        )
    });
    let needs_counts = fields.iter().any(|f| {
        !matches!(
            *f,
            "origin"
                | "cost_nusd"
                | "wall_span"
                | "clustered_time"
                | "tool_time"
                | "tool_time_union"
                | "tool_untimed"
                | "ends_with"
                | "last_stop_reason"
                | "waiting_model"
                | "waiting_human"
        )
    });
    let current = format!(
        "({} OR {}) AND ({} OR {})",
        !needs_time,
        session_sql::current("s", "session_span"),
        !needs_counts,
        session_sql::current("s", "counts")
    );
    let (condition, params) = selection.session_condition("s.pk")?;
    reader
        .conn()?
        .query_row(
            &format!(
                "SELECT {},coalesce(sum(NOT ({current})),0) FROM session s WHERE {condition}",
                columns.join(",")
            ),
            rusqlite::params_from_iter(params.iter()),
            |row| {
                let mut varies = vec![];
                let mut unknown = BTreeMap::new();
                for (i, &name) in fields.iter().enumerate() {
                    if row.get::<_, i64>(i * 2)? > 1 {
                        varies.push(name.into());
                    }
                    unknown.insert(name.into(), row.get::<_, i64>(i * 2 + 1)? as u64);
                }
                Ok((varies, unknown, row.get::<_, i64>(fields.len() * 2)? as u64))
            },
        )
        .map_err(Into::into)
}

#[cfg(test)]
mod tests;

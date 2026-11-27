//! What each session contributes to completion (`session_value`,
//! `session_first_prompt`): the models its messages name and the tools its
//! calls use, with counts, and its first typed prompt, folded over its rows.
//! A row whose count changes takes the summary transaction's write counter,
//! and the completion catalog copies the rows changed since it was
//! published; a value the session no longer has keeps its row with count 0
//! so the catalog sees the removal.
use super::Output;
use super::fold::{Context, Flow, Fold, SessionRow};
use crate::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) struct Values;

/// The session's completion values with their counts, and its first typed
/// prompt with text.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Value {
    /// By field (`model` or `tool`), then value.
    counts: BTreeMap<String, BTreeMap<String, i64>>,
    first_prompt: Option<i64>,
}

impl Fold for Values {
    type State = Value;
    const NAME: &'static str = "values";
    const VERSION: i64 = 1;

    fn step(state: &mut Value, row: &SessionRow, _: &Context<'_>, _: &mut Output) -> Result<Flow> {
        let SessionRow::Message(m) = row else {
            return Ok(Flow::Continue);
        };
        if let Some(model) = m.model.as_deref().filter(|m| !m.is_empty()) {
            *state
                .counts
                .entry("model".into())
                .or_default()
                .entry(model.into())
                .or_default() += 1;
        }
        for call in &m.calls {
            *state
                .counts
                .entry("tool".into())
                .or_default()
                .entry(call.tool.clone())
                .or_default() += 1;
        }
        if state.first_prompt.is_none()
            && m.written_by == "human"
            && m.prompt_kind.as_deref() == Some("prose")
            && m.blocks.iter().any(|b| b.kind == "text" && b.has_text)
        {
            state.first_prompt = Some(m.pk);
        }
        Ok(Flow::Continue)
    }

    fn end(state: &Value, _: &Context<'_>, out: &mut Output) -> Result<()> {
        out.values = Some(state.clone());
        Ok(())
    }
}

/// Writes the counts that differ from the stored ones, and the first prompt
/// when it moved.
pub(super) fn apply(conn: &Connection, session_pk: i64, value: &Value, counter: i64) -> Result<()> {
    let stored: Vec<((String, String), i64)> = conn
        .prepare_cached("SELECT field, value, count FROM session_value WHERE session_pk = ?1")?
        .query_map([session_pk], |row| {
            Ok(((row.get(0)?, row.get(1)?), row.get(2)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut set = conn.prepare_cached(
        "INSERT INTO session_value(session_pk, field, value, count, changed) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(session_pk, field, value) DO UPDATE SET count = excluded.count, changed = excluded.changed",
    )?;
    let mut wanted: BTreeMap<(String, String), i64> = value
        .counts
        .iter()
        .flat_map(|(field, values)| {
            values
                .iter()
                .map(move |(value, n)| ((field.clone(), value.clone()), *n))
        })
        .collect();
    for (key, n) in stored {
        let now = wanted.remove(&key).unwrap_or(0);
        if now != n {
            set.execute(params![session_pk, key.0, key.1, now, counter])?;
        }
    }
    for ((field, value), n) in wanted {
        set.execute(params![session_pk, field, value, n, counter])?;
    }
    let first: Option<Option<i64>> = conn
        .prepare_cached("SELECT message_pk FROM session_first_prompt WHERE session_pk = ?1")?
        .query_row([session_pk], |row| row.get(0))
        .optional()?;
    if first != Some(value.first_prompt) {
        conn.prepare_cached(
            "INSERT INTO session_first_prompt(session_pk, message_pk, changed) VALUES (?1, ?2, ?3)
             ON CONFLICT(session_pk) DO UPDATE SET message_pk = excluded.message_pk,
               changed = excluded.changed",
        )?
        .execute(params![session_pk, value.first_prompt, counter])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

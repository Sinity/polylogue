//! The session count columns, folded over the same shown population the
//! query language reads ([`crate::session_sql::own_messages`]).
//!
//! A script call counts through the commands it recorded, so a code call
//! counts only while the session holds no call run by a script; the fold
//! keeps code calls apart and decides at the end, so a script call
//! arriving late changes the count without the earlier rows being read
//! again.
use super::Output;
use super::fold::{Context, Flow, Fold, SessionRow};
use crate::Result;
use crate::session_sql::{counts_as_action, own_messages};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

pub(crate) struct Counts;

/// The ten count columns of a session.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Value {
    pub messages: i64,
    pub human_prompts: i64,
    pub human_commands: i64,
    pub words: i64,
    pub human_words: i64,
    pub model_words: i64,
    pub thinking_messages: i64,
    pub compactions: i64,
    pub tool_calls: i64,
    pub errors: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct State {
    value: Value,
    /// Calls and failed calls of the shown messages that are not code
    /// calls, and those that are.
    calls: (i64, i64),
    code_calls: (i64, i64),
    /// A call of the session, shown or not, was run by a script.
    script: bool,
}

impl Fold for Counts {
    type State = State;
    const NAME: &'static str = "counts";
    const VERSION: i64 = 2;

    fn step(state: &mut State, row: &SessionRow, _: &Context<'_>, _: &mut Output) -> Result<Flow> {
        match row {
            SessionRow::Event(e) => {
                state.value.compactions += i64::from(e.kind == "compaction");
            }
            SessionRow::Message(m) => {
                state.script |= m.calls.iter().any(|c| c.run_by_script);
                if m.own {
                    let v = &mut state.value;
                    let human = m.written_by == "human";
                    v.messages += 1;
                    v.human_prompts +=
                        i64::from(human && m.prompt_kind.as_deref() == Some("prose"));
                    v.human_commands +=
                        i64::from(human && m.prompt_kind.as_deref() == Some("command"));
                    let human_words = if human { m.words } else { 0 };
                    let model_words = if m.written_by == "model" { m.words } else { 0 };
                    v.human_words += human_words;
                    v.model_words += model_words;
                    v.words += human_words + model_words;
                    v.thinking_messages += i64::from(m.blocks.iter().any(|b| b.kind == "thinking"));
                    for call in &m.calls {
                        let tally = if call.is_code() {
                            &mut state.code_calls
                        } else {
                            &mut state.calls
                        };
                        tally.0 += 1;
                        tally.1 += i64::from(call.outcome == "error");
                    }
                }
            }
        }
        Ok(Flow::Continue)
    }

    fn end(state: &State, _: &Context<'_>, out: &mut Output) -> Result<()> {
        let mut value = state.value.clone();
        let (calls, errors) = if state.script {
            state.calls
        } else {
            (
                state.calls.0 + state.code_calls.0,
                state.calls.1 + state.code_calls.1,
            )
        };
        value.tool_calls = calls;
        value.errors = errors;
        out.counts = Some(value);
        Ok(())
    }
}

/// The counts computed from all the session's rows in SQL: what a read of a
/// session whose summaries are due shows.
pub(crate) fn compute(conn: &Connection, pk: i64) -> Result<Value> {
    let own = own_messages("m", Some("=?1"));
    let aggregates = crate::session_sql::MESSAGE_COUNTS
        .iter()
        .map(|(_, expr)| *expr)
        .collect::<Vec<_>>()
        .join(",");
    let mut counts = conn
        .prepare_cached(&format!("SELECT {aggregates} FROM message m WHERE {own}"))?
        .query_row([pk], |r| {
            Ok(Value {
                messages: r.get(0)?,
                human_prompts: r.get(1)?,
                human_commands: r.get(2)?,
                words: r.get(3)?,
                human_words: r.get(4)?,
                model_words: r.get(5)?,
                thinking_messages: r.get(6)?,
                compactions: 0,
                tool_calls: 0,
                errors: 0,
            })
        })?;
    counts.compactions = conn
        .prepare_cached("SELECT count(*) FROM event WHERE session_pk=?1 AND kind='compaction'")?
        .query_row([pk], |row| row.get(0))?;
    (counts.tool_calls, counts.errors) = conn
        .prepare_cached(&format!(
            "SELECT count(*),coalesce(sum(t.outcome='error'),0) FROM tool_call t
             JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk
             WHERE {own} AND {}",
            counts_as_action("t")
        ))?
        .query_row([pk], |r| <(_, _)>::try_from(r))?;
    Ok(counts)
}

/// The session's counts: the stored ones when they are current, else
/// computed from its rows.
pub(crate) fn read(conn: &Connection, pk: i64, current: bool) -> Result<Value> {
    if !current {
        return compute(conn, pk);
    }
    Ok(conn.query_row(
        "SELECT message_count,human_prompt_count,human_command_count,word_count,human_word_count,
                model_word_count,thinking_count,compaction_count,tool_call_count,error_count
         FROM session WHERE pk=?1",
        [pk],
        |r| {
            Ok(Value {
                messages: r.get(0)?,
                human_prompts: r.get(1)?,
                human_commands: r.get(2)?,
                words: r.get(3)?,
                human_words: r.get(4)?,
                model_words: r.get(5)?,
                thinking_messages: r.get(6)?,
                compactions: r.get(7)?,
                tool_calls: r.get(8)?,
                errors: r.get(9)?,
            })
        },
    )?)
}

pub(crate) fn apply(tx: &Connection, pk: i64, c: &Value) -> Result<()> {
    tx.prepare_cached(
        "UPDATE session SET message_count=?2,human_prompt_count=?3,human_command_count=?4,
             word_count=?5,human_word_count=?6,model_word_count=?7,thinking_count=?8,
             compaction_count=?9,tool_call_count=?10,error_count=?11 WHERE pk=?1",
    )?
    .execute(params![
        pk,
        c.messages,
        c.human_prompts,
        c.human_commands,
        c.words,
        c.human_words,
        c.model_words,
        c.thinking_messages,
        c.compactions,
        c.tool_calls,
        c.errors
    ])?;
    Ok(())
}

#[cfg(test)]
mod tests;

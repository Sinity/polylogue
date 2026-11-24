//! Counts over the same shown path the query language reads. The transaction's
//! change summary applies changed contributions once per session. [`compute`]
//! is the independent full recompute, also run when the derivation version moves.
use super::{Class, Derivation, Output, Version};
use crate::Result;
use crate::session_sql::{counts_as_action, own_messages};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "counts",
    class: Class::PerChange,
    products: &[polylogue_model::derived::SESSION_COUNTS],
    version: Version::Rules(1),
    after: &["duplicates", "active_path", "tool_calls"],
    dirtied_by: &[],
    compute: |conn, pk, _| compute(conn, pk).map(Output::Counts),
};

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Counts {
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

impl Counts {
    pub(crate) fn accumulate(&mut self, other: &Self, sign: i64) -> Result<()> {
        for (value, contribution) in [
            (&mut self.messages, other.messages),
            (&mut self.human_prompts, other.human_prompts),
            (&mut self.human_commands, other.human_commands),
            (&mut self.words, other.words),
            (&mut self.human_words, other.human_words),
            (&mut self.model_words, other.model_words),
            (&mut self.thinking_messages, other.thinking_messages),
            (&mut self.compactions, other.compactions),
            (&mut self.tool_calls, other.tool_calls),
            (&mut self.errors, other.errors),
        ] {
            *value = contribution
                .checked_mul(sign)
                .and_then(|n| value.checked_add(n))
                .ok_or_else(|| {
                    crate::StoreError::internal("session count exceeds SQLite's integer range")
                })?;
        }
        Ok(())
    }
}

/// One message's contribution before or after a transaction changes it.
pub(crate) fn message(conn: &Connection, pk: i64) -> Result<Option<(i64, Counts)>> {
    let row = conn
        .prepare_cached(
            "SELECT session_pk,duplicate_of IS NULL AND coalesce(on_active_path,1)=1,
         written_by,prompt_kind,words,
         EXISTS(SELECT 1 FROM block b WHERE b.message_pk=message.pk AND b.kind='thinking')
         FROM message WHERE pk=?1",
        )?
        .query_row([pk], |row| {
            <(i64, bool, String, Option<String>, i64, bool)>::try_from(row)
        })
        .optional()?;
    let Some((session, shown, author, prompt, words, thinking)) = row else {
        return Ok(None);
    };
    let mut counts = Counts::default();
    if shown {
        counts.messages = 1;
        counts.human_prompts = i64::from(author == "human" && prompt.as_deref() == Some("prose"));
        counts.human_commands =
            i64::from(author == "human" && prompt.as_deref() == Some("command"));
        counts.human_words = if author == "human" { words } else { 0 };
        counts.model_words = if author == "model" { words } else { 0 };
        counts.words = counts.human_words + counts.model_words;
        counts.thinking_messages = i64::from(thinking);
        (counts.tool_calls, counts.errors) = message_actions(conn, pk)?;
    }
    Ok(Some((session, counts)))
}

pub(crate) fn message_actions(conn: &Connection, pk: i64) -> Result<(i64, i64)> {
    Ok(conn
        .prepare_cached(&format!(
            "SELECT count(*),coalesce(sum(t.outcome='error'),0) FROM block b
         JOIN tool_call t ON t.call_block_pk=b.pk WHERE b.message_pk=?1 AND {}",
            counts_as_action("t"),
        ))?
        .query_row([pk], |row| <(_, _)>::try_from(row))?)
}

pub(crate) fn session_state(conn: &Connection, session: i64) -> Result<Option<(bool, bool)>> {
    Ok(conn
        .prepare_cached(
            "SELECT
        EXISTS(SELECT 1 FROM tool_call WHERE session_pk=session.pk AND run_by_script=1),
        message_count IS NOT NULL AND human_prompt_count IS NOT NULL
        AND human_command_count IS NOT NULL AND word_count IS NOT NULL
        AND human_word_count IS NOT NULL AND model_word_count IS NOT NULL
        AND thinking_count IS NOT NULL AND compaction_count IS NOT NULL
        AND tool_call_count IS NOT NULL AND error_count IS NOT NULL
        FROM session WHERE pk=?1",
        )?
        .query_row([session], |row| <(_, _)>::try_from(row))
        .optional()?)
}

pub(crate) fn has_script(conn: &Connection, session: i64) -> Result<bool> {
    Ok(conn
        .prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM tool_call WHERE session_pk=?1 AND run_by_script=1)",
        )?
        .query_row([session], |row| row.get(0))?)
}

/// The first/last recorded script command changes which older code calls count.
/// Only that nonlocal change needs the session's complete action population.
pub(crate) fn actions(conn: &Connection, session: i64) -> Result<(i64, i64)> {
    Ok(conn
        .prepare_cached(&format!(
            "SELECT count(*),coalesce(sum(t.outcome='error'),0) FROM tool_call t
         JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk
         WHERE {} AND {}",
            own_messages("m", Some("=?1")),
            counts_as_action("t"),
        ))?
        .query_row([session], |row| <(_, _)>::try_from(row))?)
}

pub(crate) fn apply_delta(
    conn: &Connection,
    session: i64,
    c: &Counts,
    actions: Option<(i64, i64)>,
    changed: Option<i64>,
) -> Result<()> {
    conn.prepare_cached(
        "UPDATE session SET
        message_count=coalesce(message_count,0)+?2,
        human_prompt_count=coalesce(human_prompt_count,0)+?3,
        human_command_count=coalesce(human_command_count,0)+?4,
        word_count=coalesce(word_count,0)+?5,
        human_word_count=coalesce(human_word_count,0)+?6,
        model_word_count=coalesce(model_word_count,0)+?7,
        thinking_count=coalesce(thinking_count,0)+?8,
        compaction_count=coalesce(compaction_count,0)+?9,
        tool_call_count=coalesce(?12,coalesce(tool_call_count,0)+?10),
        error_count=coalesce(?13,coalesce(error_count,0)+?11),
        changed=coalesce(?14,changed) WHERE pk=?1",
    )?
    .execute(params![
        session,
        c.messages,
        c.human_prompts,
        c.human_commands,
        c.words,
        c.human_words,
        c.model_words,
        c.thinking_messages,
        c.compactions,
        c.tool_calls,
        c.errors,
        actions.map(|v| v.0),
        actions.map(|v| v.1),
        changed
    ])?;
    Ok(())
}

pub(crate) fn compute(conn: &Connection, pk: i64) -> Result<Counts> {
    let own = own_messages("m", Some("=?1"));
    let aggregates = crate::session_sql::MESSAGE_COUNTS
        .iter()
        .map(|(_, expr)| *expr)
        .collect::<Vec<_>>()
        .join(",");
    let mut counts = conn
        .prepare_cached(&format!("SELECT {aggregates} FROM message m WHERE {own}"))?
        .query_row([pk], |r| {
            Ok(Counts {
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
    (counts.tool_calls, counts.errors) = conn.prepare_cached(&format!("SELECT count(*),coalesce(sum(t.outcome='error'),0) FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk WHERE {own} AND {}",counts_as_action("t")))?.query_row([pk],|r| <(_, _)>::try_from(r))?;
    Ok(counts)
}

pub(crate) fn read(conn: &Connection, pk: i64, current: bool) -> Result<Counts> {
    if !current {
        return compute(conn, pk);
    }
    Ok(conn.query_row("SELECT message_count,human_prompt_count,human_command_count,word_count,human_word_count,model_word_count,thinking_count,compaction_count,tool_call_count,error_count FROM session WHERE pk=?1",[pk],|r|Ok(Counts{messages:r.get(0)?,human_prompts:r.get(1)?,human_commands:r.get(2)?,words:r.get(3)?,human_words:r.get(4)?,model_words:r.get(5)?,thinking_messages:r.get(6)?,compactions:r.get(7)?,tool_calls:r.get(8)?,errors:r.get(9)?}))?)
}

pub(crate) fn apply_values(
    tx: &Connection,
    pk: i64,
    c: &Counts,
    changed: Option<i64>,
) -> Result<()> {
    tx.prepare_cached("UPDATE session SET message_count=?2,human_prompt_count=?3,human_command_count=?4,word_count=?5,human_word_count=?6,model_word_count=?7,thinking_count=?8,compaction_count=?9,tool_call_count=?10,error_count=?11,changed=coalesce(?12,changed) WHERE pk=?1")?
        .execute(params![pk,c.messages,c.human_prompts,c.human_commands,c.words,c.human_words,c.model_words,c.thinking_messages,c.compactions,c.tool_calls,c.errors,changed])?;
    Ok(())
}

#[cfg(test)]
mod tests;

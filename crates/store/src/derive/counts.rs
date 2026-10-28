//! Counts over the same shown path the query language reads.
use super::{Derivation, Version};
use crate::Result;
use polylogue_model::session_sql::{counts_as_action, own_messages};
use rusqlite::{Connection, Transaction, params};
use serde::Serialize;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "counts",
    products: &[polylogue_model::derived::SESSION_COUNTS],
    version: Version::Rules(1),
    after: &["duplicates", "active_path", "tool_calls"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

#[derive(Debug, Serialize)]
pub(crate) struct Counts {
    pub messages: i64,
    pub human_prompts: i64,
    pub human_commands: i64,
    pub words: i64,
    pub human_words: i64,
    pub model_words: i64,
    #[serde(rename = "thinking_messages")]
    pub thinking: i64,
    pub compactions: i64,
    pub tool_calls: i64,
    pub errors: i64,
}

pub(crate) fn compute(conn: &Connection, pk: i64) -> Result<Counts> {
    let own = own_messages("m", Some("=?1"));
    let aggregates = polylogue_model::session_sql::MESSAGE_COUNTS
        .iter()
        .map(|(_, expr)| *expr)
        .collect::<Vec<_>>()
        .join(",");
    let mut counts = conn.query_row(
        &format!("SELECT {aggregates} FROM message m WHERE {own}"),
        [pk],
        |r| {
            Ok(Counts {
                messages: r.get(0)?,
                human_prompts: r.get(1)?,
                human_commands: r.get(2)?,
                words: r.get(3)?,
                human_words: r.get(4)?,
                model_words: r.get(5)?,
                thinking: r.get(6)?,
                compactions: 0,
                tool_calls: 0,
                errors: 0,
            })
        },
    )?;
    counts.compactions = conn.query_row(
        "SELECT count(*) FROM event WHERE session_pk=?1 AND kind='compaction'",
        [pk],
        |r| r.get(0),
    )?;
    (counts.tool_calls, counts.errors) = conn.query_row(&format!("SELECT count(*),coalesce(sum(t.outcome='error'),0) FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk WHERE {own} AND {}",counts_as_action("t")),[pk],|r|Ok((r.get(0)?,r.get(1)?)))?;
    Ok(counts)
}

pub(crate) fn read(conn: &Connection, pk: i64, current: bool) -> Result<Counts> {
    if !current {
        return compute(conn, pk);
    }
    Ok(conn.query_row("SELECT message_count,human_prompt_count,human_command_count,word_count,human_word_count,model_word_count,thinking_count,compaction_count,tool_call_count,error_count FROM session WHERE pk=?1",[pk],|r|Ok(Counts{messages:r.get(0)?,human_prompts:r.get(1)?,human_commands:r.get(2)?,words:r.get(3)?,human_words:r.get(4)?,model_words:r.get(5)?,thinking:r.get(6)?,compactions:r.get(7)?,tool_calls:r.get(8)?,errors:r.get(9)?})))
}

fn run(tx: &Transaction<'_>, pk: i64) -> Result<()> {
    let c = compute(tx, pk)?;
    tx.prepare_cached("UPDATE session SET message_count=?2,human_prompt_count=?3,human_command_count=?4,word_count=?5,human_word_count=?6,model_word_count=?7,thinking_count=?8,compaction_count=?9,tool_call_count=?10,error_count=?11 WHERE pk=?1")?
        .execute(params![pk,c.messages,c.human_prompts,c.human_commands,c.words,c.human_words,c.model_words,c.thinking,c.compactions,c.tool_calls,c.errors])?;
    Ok(())
}

#[cfg(test)]
mod tests;

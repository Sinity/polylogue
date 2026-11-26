//! The rows an older chain gives up to a newer chain of its input that
//! begins with the same bytes (docs/archive.md, readings). What a parser
//! makes of a record depends only on its chain's bytes up to the record's
//! end, so a record of the older chain that ends within the bytes the two
//! are known to share yields the same rows in the newer chain's parse; the
//! older copy says nothing and leaves. How far two chains agree is read only
//! from the agreement the archive recorded when the newer chain was stored
//! ([`crate::archive::writer::agreement`]); the index never compares bytes.
//!
//! Rows with a byte range leave by their range; the rest of the older
//! chain's parse (what it states about sessions, links, call outcomes,
//! copied prefixes, its unused counts) stays until the chain is repeated
//! whole, when its parse and state leave together. A rebuild parses each
//! chain that is not repeated whole and trims it by the same rule, so both
//! paths hold the same rows.
use super::readings::{self, Touched};
use super::*;
use crate::archive::writer::agreement::{Chains, recorded};
use crate::change_summary::ChangeSummary;

/// The parse tables whose rows have a byte range.
const RANGED: &[&str] = &[
    "message_parse",
    "message_alias",
    "event_parse",
    "document_parse",
];

/// Where a part of chain `chain`'s parse left it, for [`follow`].
pub(super) struct Seen {
    /// The stored length its parser had seen before the part (0 for a part
    /// that replaces the parse).
    pub from: u64,
    /// The stored length its parser has seen after it.
    pub to: u64,
}

/// After a part of chain `chain` of `input` is written: once its parser has
/// seen through the bytes an older chain is known to share with it, the
/// older chain's rows for those bytes leave, and the whole older parse when
/// that is all of it. Once the chain itself has been parsed to its stored
/// length, it gives up its own rows to the newer chains the same way. Each
/// happens in the transaction in which the bound is first reached.
pub(super) fn follow(
    tx: &Connection,
    changes: &ChangeSummary,
    input: &str,
    input_pk: i64,
    chain: i64,
    seen: Seen,
    touched: &mut Touched,
) -> Result<()> {
    let source = crate::source::Source::reading(tx)?;
    let stored_length = |chain: i64| -> Result<u64> {
        Ok(tx
            .scalar::<Option<u64>>(
                &source.sql("SELECT max(length) FROM @source.revision WHERE input=?1 AND chain=?2"),
                params![input, chain],
            )?
            .unwrap_or(0))
    };
    let reached = |bound: u64| seen.from < bound && bound <= seen.to;
    let chains: Chains = recorded(tx, input)?;
    // The chain's own part, once its parser has seen all of it.
    let stored = stored_length(chain)?;
    let (repeated_by, own_from) = chains.repeats(chain, stored);
    if own_from > 0 && reached(stored) {
        give_up(
            tx,
            changes,
            input_pk,
            chain,
            own_from,
            repeated_by.is_some(),
            touched,
        )?;
    }
    // What the older chains share with it. A chain that begins like none
    // agrees with none.
    if chains.0.contains_key(&chain) {
        let older: Vec<i64> = tx
            .prepare_cached("SELECT chain FROM parse_state WHERE input_pk=?1 AND chain<?2")?
            .query_map(params![input_pk, chain], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        for older in older {
            let Some(known) = chains.known(older, chain).filter(|known| *known > 0) else {
                continue;
            };
            if reached(known) {
                let whole = known >= stored_length(older)?;
                give_up(tx, changes, input_pk, older, known, whole, touched)?;
            }
        }
    }
    Ok(())
}

/// Chain `chain` gives up its rows of records that end at or before
/// `through`; with `whole`, its parse and state leave.
fn give_up(
    tx: &Connection,
    changes: &ChangeSummary,
    input_pk: i64,
    chain: i64,
    through: u64,
    whole: bool,
    touched: &mut Touched,
) -> Result<()> {
    if whole {
        readings::delete(tx, changes, input_pk, chain, touched)?;
        for table in [
            "wire_type",
            "wire_shape",
            "wire_occurrence",
            "wire_reading",
            "parse_state",
        ] {
            tx.prepare_cached(&format!(
                "DELETE FROM {table} WHERE input_pk=?1 AND chain=?2"
            ))?
            .execute(params![input_pk, chain])?;
        }
        return Ok(());
    }
    trim(tx, input_pk, chain, through, touched)
}

/// Deletes the rows with a byte range of chain `chain` whose records end at
/// or before `through`, and what belongs to them: the texts their bodies
/// hold, their injection fields and their content, released when nothing
/// else names it. What they stated is recorded in `touched`.
fn trim(
    tx: &Connection,
    input_pk: i64,
    chain: i64,
    through: u64,
    touched: &mut Touched,
) -> Result<()> {
    let through = i64::try_from(through)
        .map_err(|_| StoreError::internal("an agreement exceeds SQLite's integer range"))?;
    let key = params![input_pk, chain, through];
    let ends = "input_pk=?1 AND chain=?2 AND source_start IS NOT NULL
        AND source_start+source_length<=?3";
    let pairs = |sql: &str| -> Result<Vec<(Option<String>, String)>> {
        Ok(tx
            .prepare_cached(sql)?
            .query_map(key, |row| <(Option<String>, String)>::try_from(row))?
            .collect::<rusqlite::Result<_>>()?)
    };
    let messages = pairs(&format!(
        "SELECT session_id,message_id FROM message_parse WHERE {ends}"
    ))?;
    for (session, message) in &messages {
        touched
            .messages
            .insert((session.clone().unwrap_or_default(), message.clone()));
    }
    for (session, event) in pairs(&format!(
        "SELECT session_id,event_id FROM event_parse WHERE {ends}"
    ))? {
        if let Some(session) = &session {
            touched.whole.insert(session.clone());
        }
        touched.events.insert((session, event));
    }
    let documents = pairs(&format!(
        "SELECT NULL,document_id FROM document_parse WHERE {ends}"
    ))?;
    touched
        .documents
        .extend(documents.iter().map(|(_, id)| id.clone()));
    // An alias or a message leaving can change how the chain's other
    // records resolve, so every session the chain states is assembled again.
    for (session,) in tx
        .prepare_cached(
            "SELECT session_id FROM session_parse WHERE input_pk=?1 AND chain=?2
             UNION SELECT session_id FROM message_parse WHERE input_pk=?1 AND chain=?2",
        )?
        .query_map(params![input_pk, chain], |row| <(String,)>::try_from(row))?
        .collect::<rusqlite::Result<Vec<_>>>()?
    {
        touched.whole.insert(session);
    }
    for group in tx
        .prepare_cached(&format!(
            "SELECT content_group FROM message_parse WHERE {ends}
             UNION SELECT content_group FROM event_parse WHERE {ends}"
        ))?
        .query_map(key, |row| row.get::<_, i64>(0))?
    {
        touched.released_groups.insert(group?);
    }
    let items = messages
        .iter()
        .map(|(_, id)| (false, id))
        .chain(documents.iter().map(|(_, id)| (true, id)));
    for (document, item) in items {
        let item_key = params![input_pk, chain, document, item];
        for id in tx
            .prepare_cached(
                "SELECT text_id FROM parse_text WHERE input_pk=?1 AND chain=?2 AND document=?3 AND item=?4",
            )?
            .query_map(item_key, |row| row.get::<_, i64>(0))?
        {
            touched.released_texts.insert(id?);
        }
        tx.prepare_cached(
            "DELETE FROM parse_text WHERE input_pk=?1 AND chain=?2 AND document=?3 AND item=?4",
        )?
        .execute(item_key)?;
        if !document {
            tx.prepare_cached(
                "DELETE FROM message_injection_field WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
            )?
            .execute(params![input_pk, chain, item])?;
        }
    }
    tx.prepare_cached(
        "DELETE FROM text_fragment_source WHERE input_pk=?1 AND chain=?2 AND end<=?3",
    )?
    .execute(key)?;
    for table in RANGED {
        tx.prepare_cached(&format!("DELETE FROM {table} WHERE {ends}"))?
            .execute(key)?;
    }
    Ok(())
}

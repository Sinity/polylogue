//! Turns a session holds twice under different message IDs, because two
//! routes describe the same conversation without a shared turn ID: a
//! browser capture read from the page text (the Gemini app) and an export
//! (Gemini's Takeout activity). Both stay stored; the session shows each
//! such turn once, from the export, and the capture's messages name the
//! export message they repeat (`duplicate_of`) and why (`duplicate_reason`).
//!
//! A turn is a prompt and the replies after it, in one route's order. Two
//! turns are the same only on exact evidence: the prompt's text, with
//! whitespace runs read as one space, occurs once in each route; the
//! paired prompts keep the same order in both routes; and the turns have
//! the same roles after the prompt. A repeated prompt ("continue") or a
//! turn whose shape differs stays shown from both routes. The result
//! depends only on the stored rows, never on which route arrived first.

use super::{Derivation, Version};
use crate::Result;
use crate::encoding::{restore_text, unpack_block, unpack_text};
use polylogue_model::Block;
use rusqlite::{Transaction, params};
use std::collections::HashMap;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "duplicates",
    version: Version::Rules(1),
    after: &[],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

/// The format of browser captures (`input_ref.format`).
const CAPTURE: &str = "browser-capture";

/// Why the export's message is the one shown.
const REASON: &str =
    "the export records the turn's time and line breaks; the page capture has neither";

#[derive(Clone, Copy, PartialEq)]
enum Route {
    Capture,
    Export,
}

struct Turn {
    prompt: (i64, String),
    key: String,
    replies: Vec<(i64, String, String)>,
}

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "UPDATE message SET duplicate_of = NULL, duplicate_reason = NULL
         WHERE session_pk = ?1 AND duplicate_of IS NOT NULL",
    )?
    .execute([session_pk])?;
    let has_capture: bool = tx
        .prepare_cached(
            "SELECT EXISTS (SELECT 1 FROM message m JOIN message_source s ON s.message_pk = m.pk
                            JOIN input_ref r ON r.pk = s.input_pk
                            WHERE m.session_pk = ?1 AND r.format = ?2)",
        )?
        .query_row(params![session_pk, CAPTURE], |r| r.get(0))?;
    if !has_capture {
        return Ok(());
    }
    let mut statement = tx.prepare_cached(
        "SELECT m.pk, m.id, m.role, count(*), sum(r.format = ?2)
         FROM message m JOIN message_source s ON s.message_pk = m.pk
         JOIN input_ref r ON r.pk = s.input_pk
         WHERE m.session_pk = ?1
         GROUP BY m.pk
         ORDER BY m.order_input_id, m.seq, m.record_position, m.id",
    )?;
    let rows = statement
        .query_map(params![session_pk, CAPTURE], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut captured = Vec::new();
    let mut exported = Vec::new();
    // A message both routes already share ends the turn it would extend.
    let mut open: [Option<usize>; 2] = [None, None];
    for (pk, id, role, sources, from_capture) in rows {
        let route = match from_capture {
            0 => Route::Export,
            n if n == sources => Route::Capture,
            _ => {
                open = [None, None];
                continue;
            }
        };
        let (turns, slot) = match route {
            Route::Capture => (&mut captured, 0),
            Route::Export => (&mut exported, 1),
        };
        if role == "user" {
            turns.push(Turn {
                key: prompt_key(tx, pk)?,
                prompt: (pk, id),
                replies: Vec::new(),
            });
            open[slot] = Some(turns.len() - 1);
        } else if let Some(turn) = open[slot] {
            turns[turn].replies.push((pk, id, role));
        }
    }
    let mut mark = tx.prepare_cached(
        "UPDATE message SET duplicate_of = ?2, duplicate_reason = ?3 WHERE pk = ?1",
    )?;
    for (c, e) in paired(&captured, &exported) {
        let (c, e) = (&captured[c], &exported[e]);
        let same_shape = c.replies.len() == e.replies.len()
            && c.replies.iter().zip(&e.replies).all(|(a, b)| a.2 == b.2);
        if !same_shape {
            continue;
        }
        mark.execute(params![c.prompt.0, e.prompt.1, REASON])?;
        for (a, b) in c.replies.iter().zip(&e.replies) {
            mark.execute(params![a.0, b.1, REASON])?;
        }
    }
    Ok(())
}

/// The prompt's text blocks, whitespace runs read as one space.
fn prompt_key(tx: &Transaction<'_>, message_pk: i64) -> Result<String> {
    let mut statement = tx.prepare_cached(
        "SELECT b.body, t.body FROM block b LEFT JOIN text t ON t.id = b.text_id
         WHERE b.message_pk = ?1 ORDER BY b.idx",
    )?;
    let mut rows = statement.query([message_pk])?;
    let mut text = String::new();
    while let Some(row) = rows.next()? {
        let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
        if let Some(body) = row.get::<_, Option<Vec<u8>>>(1)? {
            restore_text(&mut block, unpack_text(&body)?);
        }
        if let Block::Text { text: part } = &block {
            text.push(' ');
            text.push_str(part);
        }
    }
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Capture and export turns whose prompt text occurs once in each route,
/// keeping the longest set of pairs in the same order in both.
fn paired(captured: &[Turn], exported: &[Turn]) -> Vec<(usize, usize)> {
    fn once(turns: &[Turn]) -> HashMap<&str, Option<usize>> {
        let mut seen: HashMap<&str, Option<usize>> = HashMap::new();
        for (i, turn) in turns.iter().enumerate() {
            if turn.key.is_empty() {
                continue;
            }
            seen.entry(&turn.key)
                .and_modify(|at| *at = None)
                .or_insert(Some(i));
        }
        seen
    }
    let in_export = once(exported);
    let matches: Vec<(usize, usize)> = once(captured)
        .into_iter()
        .filter_map(|(key, c)| Some((c?, in_export.get(key).copied().flatten()?)))
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_iter()
        .collect();
    longest_increasing(&matches)
}

/// The longest run of `pairs` (sorted by the first index) whose second
/// index also increases; where runs tie, the earlier export turn.
fn longest_increasing(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    // tails[k]: the index in `pairs` ending the best run of length k + 1.
    let mut tails: Vec<usize> = Vec::new();
    let mut before: Vec<Option<usize>> = vec![None; pairs.len()];
    for (i, pair) in pairs.iter().enumerate() {
        let k = tails.partition_point(|&t| pairs[t].1 < pair.1);
        before[i] = k.checked_sub(1).map(|k| tails[k]);
        if k == tails.len() {
            tails.push(i);
        } else {
            tails[k] = i;
        }
    }
    let mut run = Vec::new();
    let mut at = tails.last().copied();
    while let Some(i) = at {
        run.push(pairs[i]);
        at = before[i];
    }
    run.reverse();
    run
}

#[cfg(test)]
mod tests;

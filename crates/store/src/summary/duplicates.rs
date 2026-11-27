//! Turns a session holds twice under different message IDs, because two
//! inputs describe the same conversation without a shared turn ID: a
//! browser capture read from the page text (the Gemini app) and an export
//! (Gemini's Takeout activity). Both stay stored; the session shows each
//! such turn once, from the export, and the capture's messages name the
//! export message they repeat (`duplicate_of`) and why (`duplicate_reason`).
//!
//! A turn is a prompt and the replies after it, in one input's order. Two
//! turns are the same only when all of these hold: the prompt's text, with
//! whitespace runs read as one space, occurs once in each input; the
//! paired prompts keep the same order in both inputs; and the turns have
//! the same roles after the prompt. A repeated prompt ("continue") or a
//! turn whose shape differs stays shown from both inputs. The result
//! depends only on the stored rows, never on which input arrived first.

use super::{Kind, Summary, Version, When};
use crate::Result;
use polylogue_model::Block;
use rusqlite::{Connection, Transaction, params};
use std::collections::HashMap;

pub(super) const SUMMARY: Summary = Summary {
    name: "duplicates",
    products: &[],
    version: Version::Rules(1),
    kind: Kind::Marks {
        compute: |conn, pk, out| {
            out.duplicates = compute(conn, pk)?;
            Ok(())
        },
        when: When::RowsReached,
    },
};

/// The format of browser captures (`input_ref.format`).
const CAPTURE: &str = "browser-capture";

/// Why the export's message is the one shown.
const REASON: &str =
    "the export records the turn's time and line breaks; the page capture has neither";

#[derive(Clone, Copy, PartialEq)]
enum Arrival {
    Capture,
    Export,
}

struct Turn {
    prompt_pk: i64,
    prompt_id: String,
    key: String,
    replies: Vec<Reply>,
}

struct Reply {
    pk: i64,
    id: String,
    role: String,
}

/// The session's messages whose repeat mark changes, each with the message
/// it now repeats (`None`: shown).
fn compute(conn: &Connection, session_pk: i64) -> Result<Vec<(i64, Option<String>)>> {
    let marked: HashMap<i64, String> = conn
        .prepare_cached(
            "SELECT pk, duplicate_of FROM message INDEXED BY message_marked WHERE session_pk = ?1 AND duplicate_of IS NOT NULL",
        )?
        .query_map([session_pk], |r| <(_, _)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?;
    let wanted = repeated(conn, session_pk)?;
    let mut changed: Vec<(i64, Option<String>)> = marked
        .keys()
        .filter(|pk| !wanted.contains_key(pk))
        .map(|pk| (*pk, None))
        .collect();
    changed.extend(
        wanted
            .into_iter()
            .filter(|(pk, of)| marked.get(pk) != Some(of))
            .map(|(pk, of)| (pk, Some(of))),
    );
    changed.sort_unstable();
    Ok(changed)
}

pub(super) fn apply(tx: &Transaction<'_>, marks: &[(i64, Option<String>)]) -> Result<()> {
    let mut mark = tx.prepare_cached(
        "UPDATE message SET duplicate_of = ?2, duplicate_reason = ?3 WHERE pk = ?1",
    )?;
    for (pk, of) in marks {
        mark.execute(params![pk, of, of.as_ref().map(|_| REASON)])?;
    }
    Ok(())
}

/// Each repeated message of the session, with the message it repeats.
fn repeated(conn: &Connection, session_pk: i64) -> Result<HashMap<i64, String>> {
    let mut wanted = HashMap::new();
    // Through the inputs parsed into the session, not its messages: a
    // session without a capture input costs one look at its few inputs.
    // CROSS JOIN keeps the session's inputs the outer loop; the planner
    // would otherwise scan every input of the format.
    let has_capture: bool = conn
        .prepare_cached(
            "SELECT EXISTS (SELECT 1 FROM session s CROSS JOIN session_parse p ON p.session_id = s.id
                            CROSS JOIN input_ref r ON r.pk = p.input_pk
                            WHERE s.pk = ?1 AND r.origin = ?2)",
        )?
        .query_row(params![session_pk, CAPTURE], |r| r.get(0))?;
    if !has_capture {
        return Ok(wanted);
    }
    let mut statement = conn.prepare_cached(
        "SELECT m.pk, m.id, m.role, count(*), sum(r.origin = ?2)
         FROM message m JOIN message_parse s ON s.message_id = m.id
         JOIN input_ref r ON r.pk = s.input_pk
         WHERE m.session_pk = ?1
         GROUP BY m.pk
         ORDER BY m.session_order, m.id",
    )?;
    let rows = statement
        .query_map(params![session_pk, CAPTURE], |r| {
            <(i64, String, String, i64, i64)>::try_from(r)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut captured = Vec::new();
    let mut exported = Vec::new();
    // A message both arrivals already share ends the turn it would extend.
    let mut open: [Option<usize>; 2] = [None, None];
    for (pk, id, role, sources, from_capture) in rows {
        let arrival = match from_capture {
            0 => Arrival::Export,
            n if n == sources => Arrival::Capture,
            _ => {
                open = [None, None];
                continue;
            }
        };
        let (turns, slot) = match arrival {
            Arrival::Capture => (&mut captured, 0),
            Arrival::Export => (&mut exported, 1),
        };
        if role == "user" {
            turns.push(Turn {
                key: prompt_key(conn, pk)?,
                prompt_pk: pk,
                prompt_id: id,
                replies: Vec::new(),
            });
            open[slot] = Some(turns.len() - 1);
        } else if let Some(turn) = open[slot] {
            turns[turn].replies.push(Reply { pk, id, role });
        }
    }
    for (c, e) in paired(&captured, &exported) {
        let (c, e) = (&captured[c], &exported[e]);
        let same_shape = c.replies.len() == e.replies.len()
            && c.replies
                .iter()
                .zip(&e.replies)
                .all(|(a, b)| a.role == b.role);
        if !same_shape {
            continue;
        }
        wanted.insert(c.prompt_pk, e.prompt_id.clone());
        for (a, b) in c.replies.iter().zip(&e.replies) {
            wanted.insert(a.pk, b.id.clone());
        }
    }
    Ok(wanted)
}

/// The prompt's text blocks, whitespace runs read as one space.
fn prompt_key(conn: &Connection, message_pk: i64) -> Result<String> {
    let mut text = String::new();
    for block in crate::ordered_blocks::of_message(conn, message_pk, false)? {
        if let Block::Text { text: part } = &block {
            text.push(' ');
            text.push_str(part);
        }
    }
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Capture and export turns whose prompt text occurs once in each input,
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
    let mut matches: Vec<(usize, usize)> = once(captured)
        .into_iter()
        .filter_map(|(key, c)| Some((c?, in_export.get(key).copied().flatten()?)))
        .collect();
    matches.sort_unstable();
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

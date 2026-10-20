//! A session's own columns (kind, title, working directories, branch,
//! tool version, start, repository named by its records) from the facts
//! every input states about it (`session_record`), by rules that depend on
//! the records and their times, never on which input arrived first:
//!
//! - an input's time is the span of the messages it holds for the session
//!   (`message_source`); an input with none has no time and sorts after
//!   every timed one;
//! - kind: the earliest input's (a resumed or copied session's later files
//!   restate the kind the first one recorded);
//! - title: the highest-precedence source; among equals, the latest
//!   input's; then the greater text;
//! - working directories: each input's in its own order, inputs by their
//!   earliest time, each directory once where first used; `cwd` is the
//!   last;
//! - branch, tool version, mode and the record's repository: the latest
//!   input that states one; then the greater value;
//! - start: the earliest.
//!
//! Remaining ties between inputs with equal times go to the greater
//! content, so the result is the same in every arrival order (law 4).

use super::{Derivation, Version};
use crate::Result;
use polylogue_model::{Session, TitleSource};
use rusqlite::Transaction;
use std::cmp::Ordering;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "session",
    version: Version::Rules(2),
    after: &[],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

struct Input {
    facts: Session,
    first: Option<i64>,
    last: Option<i64>,
}

/// Earlier first; an input without a time after every timed one.
fn by_first(a: &Input, b: &Input) -> Ordering {
    match (a.first, b.first) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// The later of two optional times; `None` is the earliest.
fn later(a: Option<i64>, b: Option<i64>) -> Ordering {
    a.cmp(&b)
}

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let rows: Vec<(String, Option<i64>, Option<i64>)> = tx
        .prepare_cached(
            "SELECT r.facts,
                    (SELECT MIN(m.at) FROM message_source ms JOIN message m ON m.pk = ms.message_pk
                     WHERE ms.input_pk = r.input_pk AND m.session_pk = r.session_pk),
                    (SELECT MAX(m.at) FROM message_source ms JOIN message m ON m.pk = ms.message_pk
                     WHERE ms.input_pk = r.input_pk AND m.session_pk = r.session_pk)
             FROM session_record r WHERE r.session_pk = ?1",
        )?
        .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut inputs: Vec<Input> = rows
        .into_iter()
        .filter_map(|(facts, first, last)| {
            serde_json::from_str(&facts)
                .ok()
                .map(|facts| Input { facts, first, last })
        })
        .collect();
    if inputs.is_empty() {
        // Named only by events so far: a stub keeps its placeholder row.
        return Ok(());
    }
    // A fixed order for every later choice: time, then content.
    inputs.sort_by(|a, b| {
        by_first(a, b)
            .then_with(|| a.last.cmp(&b.last))
            .then_with(|| content(&a.facts).cmp(&content(&b.facts)))
    });
    let merged = merge(&inputs);
    crate::session_row::write(tx, session_pk, &merged)?;
    tx.execute("UPDATE session SET stub = 0 WHERE pk = ?1", [session_pk])?;
    Ok(())
}

/// The session's facts from every input, `inputs` sorted by time then
/// content.
fn merge(inputs: &[Input]) -> Session {
    let first = &inputs[0].facts;
    let mut s = Session::new(first.id.clone(), first.kind);
    // Title: precedence, then the latest input, then the greater text.
    s.title = inputs
        .iter()
        .filter_map(|i| i.facts.title.as_ref().map(|t| (t, i.last)))
        .max_by(|(a, at), (b, bt)| {
            precedence(b.source)
                .cmp(&precedence(a.source))
                .then_with(|| later(*at, *bt))
                .then_with(|| a.text.cmp(&b.text))
        })
        .map(|(t, _)| t.clone());
    for i in inputs {
        for dir in &i.facts.cwd {
            if !s.cwd.contains(dir) {
                s.cwd.push(dir.clone());
            }
        }
    }
    let latest = |pick: fn(&Session) -> &Option<String>| -> Option<String> {
        inputs
            .iter()
            .filter_map(|i| pick(&i.facts).as_ref().map(|v| (v, i.last)))
            .max_by(|(a, at), (b, bt)| later(*at, *bt).then_with(|| a.cmp(b)))
            .map(|(v, _)| v.clone())
    };
    s.git_branch = latest(|f| &f.git_branch);
    s.tool_version = latest(|f| &f.tool_version);
    s.mode = latest(|f| &f.mode);
    s.repo = latest(|f| &f.repo);
    s.started = inputs.iter().filter_map(|i| i.facts.started).min();
    s
}

/// Lower is stronger: the operator's rename beats the tool's names.
fn precedence(source: TitleSource) -> u8 {
    match source {
        TitleSource::Custom => 0,
        TitleSource::Generated => 1,
        TitleSource::Summary => 2,
        TitleSource::AgentName => 3,
    }
}

fn content(s: &Session) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

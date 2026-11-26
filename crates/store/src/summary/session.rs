//! A session's own columns (kind, title, working directories, branch,
//! tool version, start, repository named by its records) from the facts
//! every input states about it (`session_record`), by rules that depend on
//! the records and their times, never on which input arrived first:
//!
//! - an input's time is the span of the messages it holds for the session
//!   (`message_parse`); an input with none has no time and sorts after
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
//! content, so the result is the same in every arrival order.

use super::fold::{Context, Flow, Fold, SessionRow};
use super::{Kind, Output, Summary, Version};
use crate::Result;
use polylogue_model::Session;
use rusqlite::{Connection, Transaction, params};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub(super) const SUMMARY: Summary = Summary {
    name: "session",
    products: &[],
    version: Version::Rules(7),
    kind: Kind::AfterFolds,
};

/// The fold the session's columns read: each input's time, the span of the
/// messages it holds for the session, and the session's latest message
/// time.
pub(crate) struct Inputs;

/// What [`Inputs`] folds: each input's earliest and latest message time,
/// by input key, and the latest time of any message.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Spans {
    inputs: BTreeMap<i64, (i64, i64)>,
    last_activity_at: Option<i64>,
}

impl Fold for Inputs {
    type State = Spans;
    const NAME: &'static str = "inputs";
    const VERSION: i64 = 1;

    fn step(state: &mut Spans, row: &SessionRow, _: &Context<'_>, _: &mut Output) -> Result<Flow> {
        if let SessionRow::Message(m) = row
            && let Some(at) = m.at
        {
            state.last_activity_at = Some(state.last_activity_at.map_or(at, |l| l.max(at)));
            for input in &m.inputs {
                let span = state.inputs.entry(*input).or_insert((at, at));
                span.0 = span.0.min(at);
                span.1 = span.1.max(at);
            }
        }
        Ok(Flow::Continue)
    }

    fn end(state: &Spans, _: &Context<'_>, out: &mut Output) -> Result<()> {
        out.inputs = Some(state.clone());
        Ok(())
    }
}

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

/// What the session summary stores for a session.
#[derive(Debug)]
pub(crate) struct Facts {
    last_activity_at: Option<i64>,
    /// Its columns from its inputs' facts; `None` when no input states any
    /// and the session keeps the columns it has.
    session: Option<Session>,
}

/// The session's columns from the facts its inputs state and the folded
/// input times, after the folds ran.
pub(super) fn compute(conn: &Connection, session_pk: i64, out: &mut Output) -> Result<()> {
    let spans = out.inputs.clone().unwrap_or_default();
    let mut statement = conn.prepare_cached(
        "SELECT r.input_pk, r.facts
         FROM session_stated x JOIN session_record r ON r.session_id = x.stated_id
         WHERE x.session_pk = ?1 ORDER BY r.input_pk, r.chain",
    )?;
    let mut rows = statement.query([session_pk])?;
    let mut inputs: Vec<Input> = Vec::new();
    let mut previous = None;
    while let Some(row) = rows.next()? {
        let input: i64 = row.get(0)?;
        let facts: Session = serde_json::from_str(&row.get::<_, String>(1)?)?;
        if previous == Some(input) {
            inputs
                .last_mut()
                .expect("previous input exists")
                .facts
                .merge(&facts);
        } else {
            let span = spans.inputs.get(&input);
            inputs.push(Input {
                facts,
                first: span.map(|s| s.0),
                last: span.map(|s| s.1),
            });
            previous = Some(input);
        }
    }
    drop(rows);
    drop(statement);
    let session = if inputs.is_empty() {
        // No input states anything of the session. One the assembly left
        // as a stub, named only by events, has no facts of its own; any
        // other keeps the columns it was written with.
        let (id, stub): (String, bool) = conn
            .prepare_cached("SELECT id, stub FROM session WHERE pk=?1")?
            .query_row([session_pk], |r| <(_, _)>::try_from(r))?;
        if !stub {
            out.session = Some(Box::new(Facts {
                last_activity_at: spans.last_activity_at,
                session: None,
            }));
            return Ok(());
        }
        let id = polylogue_model::SessionId::parse(&id)
            .map_err(|e| crate::StoreError::stored_data(format!("session id: {e}")))?;
        Session::new(id, polylogue_model::SessionKind::Main)
    } else {
        // A fixed order for every later choice: time, then content.
        inputs.sort_by(|a, b| {
            by_first(a, b)
                .then_with(|| a.last.cmp(&b.last))
                .then_with(|| content(&a.facts).cmp(&content(&b.facts)))
        });
        merge(&inputs)
    };
    out.session = Some(Box::new(Facts {
        last_activity_at: spans.last_activity_at,
        session: Some(session),
    }));
    Ok(())
}

/// The session's working directory and the repository its records name,
/// as computed in this run, or as stored when its inputs state none: what
/// the repository summary reads.
pub(super) fn place(
    conn: &Connection,
    session_pk: i64,
    facts: &Facts,
) -> Result<(Option<String>, Option<String>)> {
    Ok(match &facts.session {
        Some(session) => (session.cwd.last().cloned(), session.repo.clone()),
        None => conn
            .prepare_cached("SELECT cwd, record_repo FROM session WHERE pk = ?1")?
            .query_row([session_pk], |r| <(_, _)>::try_from(r))?,
    })
}

pub(super) fn apply(tx: &Transaction<'_>, session_pk: i64, facts: &Facts) -> Result<()> {
    tx.prepare_cached(
        "UPDATE session SET last_activity_at=?2 WHERE pk=?1 AND last_activity_at IS NOT ?2",
    )?
    .execute(params![session_pk, facts.last_activity_at])?;
    if let Some(session) = &facts.session {
        crate::session_row::write(tx, session_pk, session)?;
    }
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
            b.source
                .cmp(&a.source)
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
    s.git_commit = latest(|f| &f.git_commit);
    s.model_provider = latest(|f| &f.model_provider);
    s.instructions = latest(|f| &f.instructions);
    s.tool_version = latest(|f| &f.tool_version);
    s.mode = latest(|f| &f.mode);
    s.repo = latest(|f| &f.repo);
    if let Some(input) = inputs
        .iter()
        .filter(|i| i.facts.started.is_some())
        .min_by_key(|i| i.facts.started)
    {
        s.started = input.facts.started;
        s.started_quality = input.facts.started_quality;
        s.started_precision = input.facts.started_precision;
        s.started_until = input.facts.started_until;
    }
    s
}

fn content(s: &Session) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

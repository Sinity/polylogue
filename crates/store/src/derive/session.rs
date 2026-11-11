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
//! content, so the result is the same in every arrival order.

use super::{Class, Derivation, Output, Version};
use crate::Result;
use polylogue_model::Session;
use rusqlite::{Connection, Transaction, params};
use std::cmp::Ordering;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "session",
    class: Class::Session,
    products: &[],
    version: Version::Rules(6),
    after: &[],
    dirtied_by: &[],
    compute: |conn, pk, _| compute(conn, pk).map(|facts| Output::Session(Box::new(facts))),
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

/// What the session derivation stores for a session.
#[derive(Debug)]
pub(crate) struct Facts {
    /// Its live facts folded again, when evidence arrived out of order.
    live: Option<crate::live::Refold>,
    last_activity_at: Option<i64>,
    columns: Columns,
}

#[derive(Debug)]
enum Columns {
    /// No input states anything of the session: its columns stay.
    Kept,
    /// Every input that stated something of it is gone: it is a stub.
    Stub(Box<Session>),
    Merged(Box<Session>),
}

fn compute(conn: &Connection, session_pk: i64) -> Result<Facts> {
    let reset: bool = conn
        .prepare_cached("SELECT live_reparse FROM session WHERE pk=?1")?
        .query_row([session_pk], |r| r.get(0))?;
    let live = reset
        .then(|| crate::live::refold(conn, session_pk))
        .transpose()?;
    let last_activity_at: Option<i64> = conn
        .prepare_cached("SELECT max(at) FROM message WHERE session_pk=?1")?
        .query_row([session_pk], |r| r.get(0))?;
    let mut statement = conn.prepare_cached(
        "SELECT r.input_pk,r.facts,
                (SELECT MIN(m.at) FROM message_source ms JOIN message m ON m.pk=ms.message_pk
                 WHERE ms.input_pk=r.input_pk AND m.session_pk=r.session_pk),
                (SELECT MAX(m.at) FROM message_source ms JOIN message m ON m.pk=ms.message_pk
                 WHERE ms.input_pk=r.input_pk AND m.session_pk=r.session_pk)
         FROM session_record r WHERE r.session_pk=?1
         ORDER BY r.input_pk,r.chain",
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
            inputs.push(Input {
                facts,
                first: row.get(2)?,
                last: row.get(3)?,
            });
            previous = Some(input);
        }
    }
    drop(rows);
    drop(statement);
    let columns = if inputs.is_empty() {
        if reset {
            let id: String = conn
                .prepare_cached("SELECT id FROM session WHERE pk=?1")?
                .query_row([session_pk], |r| r.get(0))?;
            let id = polylogue_model::SessionId::parse(&id)
                .map_err(|e| crate::StoreError::Invalid(format!("session id: {e}")))?;
            Columns::Stub(Box::new(Session::new(
                id,
                polylogue_model::SessionKind::Main,
            )))
        } else {
            Columns::Kept
        }
    } else {
        // A fixed order for every later choice: time, then content.
        inputs.sort_by(|a, b| {
            by_first(a, b)
                .then_with(|| a.last.cmp(&b.last))
                .then_with(|| content(&a.facts).cmp(&content(&b.facts)))
        });
        Columns::Merged(Box::new(merge(&inputs)))
    };
    Ok(Facts {
        live,
        last_activity_at,
        columns,
    })
}

pub(super) fn apply(tx: &Transaction<'_>, session_pk: i64, facts: &Facts) -> Result<()> {
    if let Some(refold) = &facts.live {
        crate::live::write_refold(tx, session_pk, refold)?;
    }
    tx.prepare_cached("UPDATE session SET last_activity_at=?2 WHERE pk=?1")?
        .execute(params![session_pk, facts.last_activity_at])?;
    let (session, stub) = match &facts.columns {
        Columns::Kept => return Ok(()),
        Columns::Stub(session) => (session, true),
        Columns::Merged(session) => (session, false),
    };
    crate::session_row::write(tx, session_pk, session)?;
    tx.prepare_cached("UPDATE session SET stub=?2 WHERE pk=?1")?
        .execute(params![session_pk, stub])?;
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

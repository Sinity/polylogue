//! A session's own columns (kind, title, working directories, branch,
//! tool version, start, repository named by its records) from the facts
//! every input states about it (`session_parse`), by rules that depend on
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
    latest: Latest,
}

/// The latest of the messages' times (docs/query.md, times): its start is
/// the latest start, and its period reaches the latest end of any period,
/// so a period `[0,10)` and an instant at 5 give `[5,10)`.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Latest {
    /// The latest start.
    start: Option<i64>,
    /// The latest last millisecond of a period with a known end.
    last: Option<i64>,
    /// The latest start of a time whose period has no known end (a
    /// calendar grain without a bound), and its precision: the latest
    /// time then has no known end either.
    open: Option<(i64, String)>,
    /// The time that starts latest, with its last millisecond and
    /// precision; the latest time is that time when nothing ends later.
    first_latest: Option<(i64, i64, String)>,
}

impl Latest {
    pub(crate) fn add(&mut self, at: i64, until: Option<i64>, precision: &str) {
        self.start = Some(self.start.map_or(at, |s| s.max(at)));
        let last = match (until, precision) {
            (Some(until), _) => until - 1,
            (None, "instant") => at,
            (None, _) => {
                if self.open.as_ref().is_none_or(|(start, _)| at > *start) {
                    self.open = Some((at, precision.to_owned()));
                }
                return;
            }
        };
        self.last = Some(self.last.map_or(last, |l| l.max(last)));
        let later = self
            .first_latest
            .as_ref()
            .is_none_or(|(start, end, _)| (at, last) > (*start, *end));
        if later {
            self.first_latest = Some((at, last, precision.to_owned()));
        }
    }

    /// The start, the exclusive end and the precision to store.
    pub(crate) fn columns(&self) -> (Option<i64>, Option<i64>, Option<String>) {
        let Some(start) = self.start else {
            return (None, None, None);
        };
        if let Some((_, precision)) = &self.open {
            return (Some(start), None, Some(precision.clone()));
        }
        let last = self.last.unwrap_or(start).max(start);
        match &self.first_latest {
            Some((at, end, precision)) if *at == start && *end == last => (
                Some(start),
                (precision != "instant").then_some(last + 1),
                Some(precision.clone()),
            ),
            _ if last == start => (Some(start), None, Some("instant".into())),
            _ => (Some(start), Some(last + 1), Some("bounded".into())),
        }
    }
}

impl Fold for Inputs {
    type State = Spans;
    const NAME: &'static str = "inputs";
    const VERSION: i64 = 2;

    fn step(state: &mut Spans, row: &SessionRow, _: &Context<'_>, _: &mut Output) -> Result<Flow> {
        if let SessionRow::Message(m) = row
            && let Some(at) = m.at
        {
            state.latest.add(
                at,
                m.at_until,
                m.at_precision.as_deref().unwrap_or("instant"),
            );
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
    /// The latest message time's start, exclusive end and precision.
    last_active: (Option<i64>, Option<i64>, Option<String>),
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
         FROM session_stated x JOIN session_parse r ON r.session_id = x.stated_id
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
        let (id, has_transcript): (String, bool) = conn
            .prepare_cached("SELECT id, has_transcript FROM session WHERE pk=?1")?
            .query_row([session_pk], |r| <(_, _)>::try_from(r))?;
        if has_transcript {
            out.session = Some(Box::new(Facts {
                last_active: spans.latest.columns(),
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
        last_active: spans.latest.columns(),
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
    let (at, until, precision) = &facts.last_active;
    tx.prepare_cached(
        "UPDATE session SET last_active_at=?2, last_active_until=?3, last_active_precision=?4
         WHERE pk=?1 AND NOT (last_active_at IS ?2 AND last_active_until IS ?3 AND last_active_precision IS ?4)",
    )?
    .execute(params![session_pk, at, until, precision])?;
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

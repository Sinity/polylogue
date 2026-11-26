//! Folds: session summaries whose value depends on the session's rows in
//! order, computed by one driver ([`Run`]) from a stored state.
//!
//! A session's rows are its messages and events, in the order of their
//! [`Key`]: the row's place (`session_order`), then messages before events,
//! then the row's ID. One ordered read of the rows after a stored state
//! serves every fold. The driver keeps the folds' state after the last row
//! it read, and at every [`EVERY`]th row, so a change before the end starts
//! again from the nearest state before it, not from the first row; it
//! never adds or subtracts what a change contributed. Folding the rows
//! after a state gives what folding every row from the start gives, which
//! `summaries_after_any_appends_equal_those_from_the_final_rows` checks.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::cell::RefCell;

/// A state is kept after every this many rows of a session.
pub(crate) const EVERY: u64 = 512;

/// How many of the latest states a session keeps besides those at every
/// [`EVERY`]th row: a change near the end (a call's result pairing its
/// call) starts from one of them.
const LATEST_KEPT: i64 = 4;

/// A row's place in its session's order, as stored bytes: the row's
/// `session_order`, 0 for a message or 1 for an event, and its ID. No
/// order is a prefix of another, so nothing separates them. The empty key comes before every row and `FF` after every row.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key(pub(crate) Vec<u8>);

impl Key {
    pub(crate) fn start() -> Self {
        Self(Vec::new())
    }

    /// After every row: what a cause outside the session's rows marks.
    pub(crate) fn past_every_row() -> Self {
        Self(vec![0xFF])
    }

    pub(crate) fn message(order: &[u8], id: &str) -> Self {
        Self::of(order, 0, id)
    }

    pub(crate) fn event(order: &[u8], id: &str) -> Self {
        Self::of(order, 1, id)
    }

    fn of(order: &[u8], kind: u8, id: &str) -> Self {
        let mut bytes = Vec::with_capacity(order.len() + id.len() + 1);
        bytes.extend_from_slice(order);
        bytes.push(kind);
        bytes.extend_from_slice(id.as_bytes());
        Self(bytes)
    }

    /// The `session_order` the key starts with.
    pub(crate) fn order(&self) -> &[u8] {
        &self.0[..crate::encoding::ORDER_LENGTH.min(self.0.len())]
    }

    pub(crate) fn is_start(&self) -> bool {
        self.0.is_empty()
    }

    /// The row the key names, as a log line says it.
    pub(crate) fn row(&self) -> String {
        if self.is_start() {
            return "the first row".into();
        }
        if *self >= Self::past_every_row() {
            return "no row".into();
        }
        let order = self.order();
        let rest = &self.0[order.len()..];
        let order = polylogue_model::hex(order);
        match rest.split_first() {
            Some((kind, id)) => format!(
                "{} {} at {order}",
                if *kind == 0 { "message" } else { "event" },
                String::from_utf8_lossy(id)
            ),
            None => format!("place {order}"),
        }
    }
}

/// One assembled row of a session, as the folds read it.
pub(crate) enum SessionRow {
    Message(Box<MessageRow>),
    Event(EventRow),
}

impl SessionRow {
    pub(crate) fn key(&self) -> &Key {
        match self {
            Self::Message(m) => &m.key,
            Self::Event(e) => &e.key,
        }
    }
}

/// A message with what the folds read of its blocks and calls.
pub(crate) struct MessageRow {
    pub key: Key,
    pub pk: i64,
    pub id: String,
    pub order: Vec<u8>,
    pub role: String,
    pub written_by: String,
    pub prompt_kind: Option<String>,
    pub model: Option<String>,
    pub at: Option<i64>,
    /// The exclusive end of its time's period, when the time is a period.
    pub at_until: Option<i64>,
    pub at_precision: Option<String>,
    pub words: i64,
    pub stop_reason: Option<String>,
    pub duplicate: bool,
    /// Shown in the session's own population: not a duplicate, and not off
    /// the active path.
    pub own: bool,
    /// The provider did not mark where its response ends.
    pub boundary_unknown: bool,
    /// Its blocks in order: key, kind and whether a text row holds its text.
    pub blocks: Vec<BlockRow>,
    pub calls: Vec<CallRow>,
    /// The inputs whose readings state the message.
    pub inputs: Vec<i64>,
}

pub(crate) struct BlockRow {
    pub pk: i64,
    pub kind: String,
    pub has_text: bool,
}

/// A tool call of a message, with its result's place.
pub(crate) struct CallRow {
    pub pk: i64,
    pub block_pk: i64,
    pub tool: String,
    pub action: Option<String>,
    pub command: Option<String>,
    pub command_norm: Option<String>,
    pub run_by_script: bool,
    /// The call's own time, else its message's.
    pub at: Option<i64>,
    pub outcome: String,
    pub outcome_reason: Option<String>,
    /// A result names the call.
    pub paired: bool,
    /// The result's message, when the call has a result and it is stored.
    pub result: Option<ResultRow>,
}

pub(crate) struct ResultRow {
    pub order: Vec<u8>,
    pub id: String,
    /// The result message's time, when that message is in its session's own
    /// population.
    pub own_at: Option<i64>,
}

impl CallRow {
    /// Whether the call counts as an action of its session: a script call
    /// counts through the commands it recorded, so a code call counts only
    /// while the session holds no recorded command.
    pub(crate) fn is_code(&self) -> bool {
        self.action.as_deref() == Some("code")
    }
}

pub(crate) struct EventRow {
    pub key: Key,
    pub kind: String,
    pub at: Option<i64>,
    /// Where its data is stored.
    pub data: crate::content::Stretch,
}

/// What a fold may read besides the row: stored content, through the run's
/// connection.
pub(crate) struct Context<'c> {
    pub conn: &'c Connection,
    reader: RefCell<crate::content::Reader>,
}

impl Context<'_> {
    pub(crate) fn json<T: DeserializeOwned>(&self, at: crate::content::Stretch) -> Result<T> {
        self.reader.borrow_mut().json(self.conn, at)
    }
}

/// Whether a fold can go on from its state.
pub(crate) enum Flow {
    Continue,
    /// The row comes before what the state already settled: every fold
    /// starts again from the session's first row.
    Restart,
}

/// A summary computed by folding a session's rows in order.
pub(crate) trait Fold {
    type State: Serialize + DeserializeOwned + Default;
    /// Its key in `summary_state`.
    const NAME: &'static str;
    /// Bumped when its rules change; every session is folded again.
    const VERSION: i64;
    fn step(
        state: &mut Self::State,
        row: &SessionRow,
        cx: &Context<'_>,
        out: &mut super::Output,
    ) -> Result<Flow>;
    /// Writes the fold's value of the whole session into `out`.
    fn end(state: &Self::State, cx: &Context<'_>, out: &mut super::Output) -> Result<()>;
    /// Prepares the state to be stored after a row: a fold that holds rows
    /// it has not yet added settles what it can.
    fn rest(_state: &mut Self::State) {}
}

/// Every fold's state, stored together: one read serves them all.
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct States {
    counts: <super::counts::Counts as Fold>::State,
    inputs: <super::session::Inputs as Fold>::State,
    span: <super::span::SessionSpan as Fold>::State,
    values: <super::values::Values as Fold>::State,
    title: <super::title::Title as Fold>::State,
    calls: <super::tool_calls::Reasons as Fold>::State,
    links: <super::links::Links as Fold>::State,
    live: <super::live::Live as Fold>::State,
}

/// Each fold, applied to its part of [`States`].
macro_rules! each_fold {
    ($states:ident, $f:ident, $body:expr) => {{
        {
            type F = super::counts::Counts;
            let $f = &mut $states.counts;
            $body
        }
        {
            type F = super::session::Inputs;
            let $f = &mut $states.inputs;
            $body
        }
        {
            type F = super::span::SessionSpan;
            let $f = &mut $states.span;
            $body
        }
        {
            type F = super::values::Values;
            let $f = &mut $states.values;
            $body
        }
        {
            type F = super::title::Title;
            let $f = &mut $states.title;
            $body
        }
        {
            type F = super::tool_calls::Reasons;
            let $f = &mut $states.calls;
            $body
        }
        {
            type F = super::links::Links;
            let $f = &mut $states.links;
            $body
        }
        {
            type F = super::live::Live;
            let $f = &mut $states.live;
            $body
        }
    }};
}

/// A state to store: the folds after the row `after`, the `rows`th of the
/// session.
pub(crate) struct Kept {
    pub after: Key,
    pub rows: u64,
    pub state: Vec<u8>,
}

/// Where a run started, as its log line says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Began {
    /// From the state kept after the session's `rows`th row.
    Kept { rows: u64 },
    /// From the first row, and why.
    First(Why),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Why {
    /// The change reached the first row, or the folds' version moved.
    Change,
    /// No state was kept before the earliest change.
    NoState,
    /// The named fold could not continue from the kept state.
    Restart(&'static str),
}

impl std::fmt::Display for Began {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kept { rows } => write!(f, "from the state kept after row {rows}"),
            Self::First(Why::Change) => write!(f, "from the first row, which the change reached"),
            Self::First(Why::NoState) => {
                write!(f, "from the first row: no state kept before the change")
            }
            Self::First(Why::Restart(fold)) => {
                write!(f, "from the first row: the {fold} fold could not continue")
            }
        }
    }
}

/// What a run folded: from where, every fold's output and the states to
/// keep.
pub(crate) struct Folded {
    /// The state the run started from stands after this row; the run read
    /// every row after it.
    pub after: Key,
    pub output: super::Output,
    pub kept: Vec<Kept>,
    /// The rows it read.
    pub read: u64,
    /// Where it started.
    pub began: Began,
    /// The session's rows after the run.
    pub rows: u64,
}

/// The driver: folds a session's rows on a connection that holds them.
pub(crate) struct Run<'c> {
    pub conn: &'c Connection,
}

impl Run<'_> {
    /// Folds every fold over the session from the nearest stored state
    /// before `from` (from its first row when `from` is the start).
    pub(crate) fn session(&self, session_pk: i64, from: &Key) -> Result<Folded> {
        match self.fold(session_pk, from, true)? {
            Ok(folded) => Ok(folded),
            // A fold could not go on from a stored state: from the start,
            // keeping no state until the end, so nothing settles early.
            Err(fold) => match self.fold(session_pk, &Key::start(), false)? {
                Ok(folded) => Ok(Folded {
                    began: Began::First(Why::Restart(fold)),
                    ..folded
                }),
                Err(_) => Err(StoreError::internal(
                    "a fold restarted a run from the start",
                )),
            },
        }
    }

    /// Folds from the nearest state before `from`; `Err` with the name of a
    /// fold that could not continue from it.
    fn fold(
        &self,
        session_pk: i64,
        from: &Key,
        sparse: bool,
    ) -> Result<std::result::Result<Folded, &'static str>> {
        let resumed: Option<(Key, u64, Vec<u8>)> = if from.is_start() {
            None
        } else {
            self.conn
                .prepare_cached(
                    "SELECT after, rows, state FROM summary_resume
                     WHERE session_pk = ?1 AND after < ?2 ORDER BY after DESC LIMIT 1",
                )?
                .query_row(params![session_pk, from.0], |r| {
                    Ok((Key(r.get(0)?), r.get::<_, i64>(1)? as u64, r.get(2)?))
                })
                .optional()?
        };
        let began = match &resumed {
            Some((_, rows, _)) => Began::Kept { rows: *rows },
            None if from.is_start() => Began::First(Why::Change),
            None => Began::First(Why::NoState),
        };
        let (mut states, after, mut rows) = match resumed {
            Some((after, rows, state)) => (serde_json::from_slice::<States>(&state)?, after, rows),
            None => (States::default(), Key::start(), 0),
        };
        let cx = Context {
            conn: self.conn,
            reader: RefCell::default(),
        };
        let mut output = super::Output::default();
        let mut kept = Vec::new();
        let mut last = None;
        let mut read = 0;
        let mut reader = Rows::after(self.conn, session_pk, &after)?;
        while let Some(row) = reader.next(self.conn)? {
            let mut restart = None;
            each_fold!(states, state, {
                if let Flow::Restart = F::step(state, &row, &cx, &mut output)? {
                    restart = restart.or(Some(F::NAME));
                }
            });
            if let Some(fold) = restart {
                return Ok(Err(fold));
            }
            rows += 1;
            read += 1;
            if sparse && rows % EVERY == 0 {
                kept.push(Kept {
                    after: row.key().clone(),
                    rows,
                    state: rest(&mut states)?,
                });
            }
            last = Some(row.key().clone());
        }
        if let Some(last) = last
            && kept.last().is_none_or(|k| k.after != last)
        {
            kept.push(Kept {
                after: last,
                rows,
                state: rest(&mut states)?,
            });
        }
        each_fold!(states, state, F::end(state, &cx, &mut output)?);
        Ok(Ok(Folded {
            after,
            output,
            kept,
            read,
            began,
            rows,
        }))
    }
}

fn rest(states: &mut States) -> Result<Vec<u8>> {
    each_fold!(states, state, F::rest(state));
    Ok(serde_json::to_vec(states)?)
}

/// Stores a run's states: those after its start are replaced by the ones
/// it kept; of the rest, the latest few and those at every [`EVERY`]th
/// row stay.
pub(crate) fn keep(tx: &Connection, session_pk: i64, after: &Key, kept: &[Kept]) -> Result<()> {
    tx.prepare_cached("DELETE FROM summary_resume WHERE session_pk = ?1 AND after > ?2")?
        .execute(params![session_pk, after.0])?;
    let mut insert = tx.prepare_cached(
        "INSERT INTO summary_resume(session_pk, after, rows, state) VALUES (?1, ?2, ?3, ?4)",
    )?;
    for kept in kept {
        insert.execute(params![
            session_pk,
            kept.after.0,
            kept.rows as i64,
            kept.state
        ])?;
    }
    tx.prepare_cached(
        "DELETE FROM summary_resume WHERE session_pk = ?1 AND rows % ?2 <> 0
           AND after NOT IN (SELECT after FROM summary_resume WHERE session_pk = ?1
                             ORDER BY after DESC LIMIT ?3)",
    )?
    .execute(params![session_pk, EVERY as i64, LATEST_KEPT])?;
    Ok(())
}

/// A session's rows after a key, messages and events merged in key order.
/// The narrow columns of every row are read at once; a message's blocks,
/// calls and readings when it is reached.
struct Rows {
    session_pk: i64,
    messages: std::iter::Peekable<std::vec::IntoIter<MessageRow>>,
    events: std::iter::Peekable<std::vec::IntoIter<EventRow>>,
}

impl Rows {
    fn after(conn: &Connection, session_pk: i64, after: &Key) -> Result<Self> {
        let order = after.order();
        let messages: Vec<MessageRow> = conn
            .prepare_cached(
                "SELECT m.pk, m.id, m.session_order, m.role, m.written_by, m.prompt_kind, m.model,
                        m.at, m.words, m.stop_reason, m.duplicate_of IS NOT NULL,
                        m.duplicate_of IS NULL AND coalesce(m.on_active_path, 1) = 1,
                        coalesce(json_extract(m.attrs_json, '$.production.response_boundary_unknown'), 0) <> 0,
                        m.at_until, m.at_precision
                 FROM message m INDEXED BY message_session
                 WHERE m.session_pk = ?1 AND m.session_order >= ?2
                 ORDER BY m.session_order, m.id",
            )?
            .query_map(params![session_pk, order], |r| {
                let id: String = r.get(1)?;
                let order: Vec<u8> = r.get(2)?;
                Ok(MessageRow {
                    key: Key::message(&order, &id),
                    pk: r.get(0)?,
                    id,
                    order,
                    role: r.get(3)?,
                    written_by: r.get(4)?,
                    prompt_kind: r.get(5)?,
                    model: r.get(6)?,
                    at: r.get(7)?,
                    words: r.get(8)?,
                    stop_reason: r.get(9)?,
                    duplicate: r.get(10)?,
                    own: r.get(11)?,
                    boundary_unknown: r.get(12)?,
                    at_until: r.get(13)?,
                    at_precision: r.get(14)?,
                    blocks: Vec::new(),
                    calls: Vec::new(),
                    inputs: Vec::new(),
                })
            })?
            .filter(|m| !matches!(m, Ok(m) if m.key <= *after))
            .collect::<rusqlite::Result<_>>()?;
        let events: Vec<EventRow> = conn
            .prepare_cached(
                "SELECT e.id, e.session_order, e.kind, e.at, e.content_group,
                        e.content_start + e.content_length, e.data_length
                 FROM event e INDEXED BY event_session_order
                 WHERE e.session_pk = ?1 AND e.session_order >= ?2
                 ORDER BY e.session_order, e.id",
            )?
            .query_map(params![session_pk, order], |r| {
                let id: String = r.get(0)?;
                let order: Vec<u8> = r.get(1)?;
                Ok(EventRow {
                    key: Key::event(&order, &id),
                    kind: r.get(2)?,
                    at: r.get(3)?,
                    data: crate::content::Stretch {
                        group: r.get(4)?,
                        start: r.get(5)?,
                        length: r.get(6)?,
                    },
                })
            })?
            .filter(|e| !matches!(e, Ok(e) if e.key <= *after))
            .collect::<rusqlite::Result<_>>()?;
        Ok(Self {
            session_pk,
            messages: messages.into_iter().peekable(),
            events: events.into_iter().peekable(),
        })
    }

    fn next(&mut self, conn: &Connection) -> Result<Option<SessionRow>> {
        let message_first = match (self.messages.peek(), self.events.peek()) {
            (None, None) => return Ok(None),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(m), Some(e)) => m.key < e.key,
        };
        if !message_first {
            return Ok(self.events.next().map(SessionRow::Event));
        }
        let mut m = self.messages.next().expect("peeked");
        m.blocks = conn
            .prepare_cached(
                "SELECT pk, kind, text_id IS NOT NULL FROM block WHERE message_pk = ?1 ORDER BY idx",
            )?
            .query_map([m.pk], |r| {
                Ok(BlockRow {
                    pk: r.get(0)?,
                    kind: r.get(1)?,
                    has_text: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        m.calls = conn
            .prepare_cached(
                "SELECT t.pk, t.call_block_pk, t.tool, t.action, t.command, t.command_norm,
                        t.run_by_script, coalesce(t.at, ?2), t.outcome, t.outcome_reason,
                        rm.session_order, rm.id,
                        CASE WHEN rm.duplicate_of IS NULL AND coalesce(rm.on_active_path, 1) = 1
                             THEN rm.at END,
                        t.result_message_id IS NOT NULL
                 FROM block b CROSS JOIN tool_call t ON t.call_block_pk = b.pk
                 LEFT JOIN message rm ON rm.id = t.result_message_id
                 WHERE b.message_pk = ?1 AND t.session_pk = ?3
                 ORDER BY b.idx, t.call_id, t.command_norm, t.tool",
            )?
            .query_map(params![m.pk, m.at, self.session_pk], |r| {
                let order: Option<Vec<u8>> = r.get(10)?;
                Ok(CallRow {
                    pk: r.get(0)?,
                    block_pk: r.get(1)?,
                    tool: r.get(2)?,
                    action: r.get(3)?,
                    command: r.get(4)?,
                    command_norm: r.get(5)?,
                    run_by_script: r.get(6)?,
                    at: r.get(7)?,
                    outcome: r.get(8)?,
                    outcome_reason: r.get(9)?,
                    paired: r.get(13)?,
                    result: match order {
                        Some(order) => Some(ResultRow {
                            order,
                            id: r.get(11)?,
                            own_at: r.get(12)?,
                        }),
                        None => None,
                    },
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        m.inputs = conn
            .prepare_cached(
                "SELECT DISTINCT input_pk FROM message_parse WHERE message_id = ?1 ORDER BY input_pk",
            )?
            .query_map([&m.id], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Some(SessionRow::Message(Box::new(m))))
    }
}

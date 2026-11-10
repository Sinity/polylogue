//! Recorded clocks and interval calculations. No notes, settings or clock reads.
use super::Selection;
use crate::change_summary::OrderKey;
use crate::session_sql;
use crate::{Reader, Result};
use polylogue_model::derived::CLUSTER_GAP_MS;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const RULE_VERSION: i64 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub first_at: Option<i64>,
    pub last_at: Option<i64>,
    pub time_source: String,
    pub wall_ms: Option<i64>,
    pub clustered_ms: Option<i64>,
    pub tool_sum_ms: Option<i64>,
    pub tool_union_ms: Option<i64>,
    pub tool_overlap_ms: Option<i64>,
    pub tool_untimed: i64,
    pub tool_no_result: i64,
    pub tool_same_timestamp: i64,
    pub tool_reversed: i64,
    pub timed_messages: i64,
    pub untimed_messages: i64,
    pub unordered_gaps: i64,
    pub first_failure_at: Option<i64>,
    pub prompts_to_failure: i64,
    pub prompted: bool,
    pub ends_with: String,
    pub ends_with_message: Option<String>,
    pub last_stop_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lane {
    pub lane: String,
    pub key: String,
    pub ms: i64,
}

#[derive(Debug, Serialize)]
pub struct SessionTime {
    pub session: String,
    pub span: Option<Span>,
    pub lanes: Vec<Lane>,
    pub computed_at_read: bool,
    pub parameters: &'static [polylogue_model::derived::Parameter],
}

struct Message {
    id: String,
    at: Option<i64>,
    author: String,
    prompt: Option<String>,
    stop: Option<String>,
    last_block: Option<String>,
    open_call: bool,
}

impl Message {
    fn human_prompt(&self) -> bool {
        self.author == "human" && self.prompt.as_deref() == Some("prose")
    }

    fn ends_with(&self) -> &'static str {
        if self.last_block.as_deref() == Some("tool_result") {
            "tool_result"
        } else if self.last_block.as_deref() == Some("tool_call") && self.open_call {
            "tool_call_open"
        } else if self.human_prompt() {
            "human_prompt"
        } else if self.author == "human" && self.prompt.as_deref() == Some("command") {
            "human_command"
        } else if self.author == "model" {
            "model_text"
        } else if self.author == "harness" {
            "harness"
        } else {
            "unknown"
        }
    }

    fn waiting_lane(&self) -> &'static str {
        if self.human_prompt() || self.last_block.as_deref() == Some("tool_result") {
            "model"
        } else if self.author == "model" && self.stop.as_deref() == Some("end_turn") {
            "human"
        } else {
            "unattributed"
        }
    }
}

struct Tool {
    message_index: usize,
    key: String,
    start: Option<i64>,
    end: Option<i64>,
    paired: bool,
    failed: bool,
}

/// What the calculation reads of a session, in session order: its own
/// messages and the calls they make, with each one's place.
struct Rows {
    messages: Vec<Message>,
    keys: Vec<OrderKey>,
    tools: Vec<Tool>,
    /// Where each call's result is, when it has one.
    result_keys: Vec<Option<OrderKey>>,
}

/// The rows after `after` in session order, or all of them.
fn rows(conn: &Connection, pk: i64, after: Option<&OrderKey>) -> Result<Rows> {
    let own = session_sql::own_messages("m", Some("=?1"));
    let mut params: Vec<rusqlite::types::Value> = vec![pk.into()];
    let tail = match after {
        Some((order, id)) => {
            params.extend([order.clone().into(), id.clone().into()]);
            " AND (m.session_order,m.id)>(?2,?3)"
        }
        None => "",
    };
    let mut statement=conn.prepare_cached(&format!("SELECT m.id,m.at,m.written_by,m.prompt_kind,m.stop_reason,
        (SELECT b.kind FROM block b WHERE b.message_pk=m.pk ORDER BY b.idx DESC LIMIT 1),
        EXISTS(SELECT 1 FROM tool_call t WHERE t.call_block_pk=(SELECT b.pk FROM block b WHERE b.message_pk=m.pk ORDER BY b.idx DESC LIMIT 1) AND t.result_block_pk IS NULL),
        m.session_order
        FROM message m WHERE {own}{tail} ORDER BY m.session_order,m.id"))?;
    let mut messages = Vec::new();
    let mut keys = Vec::new();
    let mut found = statement.query(rusqlite::params_from_iter(params.iter()))?;
    while let Some(r) = found.next()? {
        let message = Message {
            id: r.get(0)?,
            at: r.get(1)?,
            author: r.get(2)?,
            prompt: r.get(3)?,
            stop: r.get(4)?,
            last_block: r.get(5)?,
            open_call: r.get(6)?,
        };
        keys.push((r.get(7)?, message.id.clone()));
        messages.push(message);
    }
    drop(found);
    let positions: BTreeMap<_, _> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.as_str(), i))
        .collect();
    // From the messages: a tail reads only the calls of the messages in it.
    let mut statement=conn.prepare_cached(&format!("SELECT CASE WHEN t.action='shell' THEN coalesce(t.command_norm,t.tool) ELSE coalesce(t.action,t.tool) END,
        coalesce(t.at,m.at),CASE WHEN rm.duplicate_of IS NULL AND coalesce(rm.on_active_path,1)=1 THEN rm.at END,t.result_block_pk IS NOT NULL,t.outcome='error',m.id,
        rm.session_order,rm.id
        FROM message m CROSS JOIN block b ON b.message_pk=m.pk CROSS JOIN tool_call t ON t.call_block_pk=b.pk
        LEFT JOIN block rb ON rb.pk=t.result_block_pk LEFT JOIN message rm ON rm.pk=rb.message_pk
        WHERE {own}{tail} AND {}
        ORDER BY m.session_order,m.id,b.idx,t.call_id,t.command_norm,t.tool",session_sql::counts_as_action("t")))?;
    let mut tools = Vec::new();
    let mut result_keys = Vec::new();
    let mut found = statement.query(rusqlite::params_from_iter(params.iter()))?;
    while let Some(r) = found.next()? {
        tools.push(Tool {
            message_index: *positions
                .get(r.get::<_, String>(5)?.as_str())
                .ok_or(rusqlite::Error::InvalidQuery)?,
            key: r.get(0)?,
            start: r.get(1)?,
            end: r.get(2)?,
            paired: r.get(3)?,
            failed: r.get(4)?,
        });
        let result: Option<String> = r.get(7)?;
        result_keys.push(match result {
            Some(id) => Some((r.get(6)?, id)),
            None => None,
        });
    }
    Ok(Rows {
        messages,
        keys,
        tools,
        result_keys,
    })
}

/// The same calculation serves the derivation and a read of a behind session.
pub(crate) fn compute(conn: &Connection, pk: i64, gap_ms: i64) -> Result<(Span, Vec<Lane>)> {
    let rows = rows(conn, pk, None)?;
    if rows.messages.is_empty() {
        return Ok(from_events(conn, pk)?.finish());
    }
    Ok(calculate(&rows.messages, &rows.tools, gap_ms))
}

/// A session without messages takes its clocks from its events.
fn from_events(conn: &Connection, pk: i64) -> Result<Prefix> {
    let (first, last): (Option<i64>, Option<i64>) = conn.query_row(
        "SELECT min(at),max(at) FROM event WHERE session_pk=?1",
        [pk],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let mut prefix = Prefix {
        first,
        last,
        ..Prefix::default()
    };
    if let Some((a, b)) = first.zip(last) {
        prefix.add_lanes(split(a, b, &[], &[]));
    }
    Ok(prefix)
}

fn calculate(messages: &[Message], tools: &[Tool], gap_ms: i64) -> (Span, Vec<Lane>) {
    let mut prefix = Prefix::default();
    prefix.extend(messages, tools, 0, gap_ms);
    prefix.finish()
}

/// The calculation's state after a run of a session's messages in order.
/// Every part of it adds across a settled message (one no interval crosses),
/// so it is stored with the span at the last settled message
/// (`session_span.prefix`) and the next derivation reads only the messages
/// after it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Prefix {
    /// The settled message the prefix ends at, and its time.
    end: Option<(OrderKey, i64)>,
    /// Whether the session had a script call: it decides which code calls count.
    script: bool,
    first: Option<i64>,
    last: Option<i64>,
    timed: i64,
    untimed: i64,
    unordered_gaps: i64,
    /// The last message's time and the lane the gap after it waits in.
    previous: Option<(Option<i64>, String)>,
    tools: i64,
    intervals: i64,
    tool_untimed: i64,
    tool_no_result: i64,
    tool_same_timestamp: i64,
    tool_reversed: i64,
    tool_sum_ms: i64,
    tool_union_ms: i64,
    clustered_ms: i64,
    prompts: i64,
    /// The earliest failed call's start, and the prompts up to its message.
    failure: Option<(i64, i64)>,
    stop_reported: bool,
    /// The last model message's stop reason, once a model message exists.
    last_model_stop: Option<Option<String>>,
    /// What the last message ends with, and its ID, when it is timed.
    ends: Option<(String, String)>,
    lanes: Vec<Lane>,
}

impl Prefix {
    /// Adds the messages that follow, and their calls; a call's
    /// `message_index` counts from `offset`.
    fn extend(&mut self, messages: &[Message], tools: &[Tool], offset: usize, gap_ms: i64) {
        let lower = self.last;
        let mut prompts_through = Vec::with_capacity(messages.len());
        let mut clustered = Vec::new();
        let mut waiting = Vec::new();
        for m in messages {
            match m.at {
                Some(at) => {
                    self.timed += 1;
                    self.first = Some(self.first.map_or(at, |f| f.min(at)));
                    self.last = Some(self.last.map_or(at, |l| l.max(at)));
                }
                None => self.untimed += 1,
            }
            if m.human_prompt() {
                self.prompts += 1;
            }
            prompts_through.push(self.prompts);
            if m.author == "model" {
                self.stop_reported |= m.stop.is_some();
                self.last_model_stop = Some(m.stop.clone());
            }
            if let (Some((Some(a), lane)), Some(b)) = (&self.previous, m.at) {
                if b < *a {
                    self.unordered_gaps += 1;
                } else {
                    if b - a <= gap_ms {
                        clustered.push((*a, b));
                    }
                    waiting.push((*a, b, lane.clone()));
                }
            }
            self.previous = Some((m.at, m.waiting_lane().to_owned()));
        }
        if let Some(m) = messages.last() {
            self.ends = m.at.map(|_| (m.ends_with().to_owned(), m.id.clone()));
        }
        let mut intervals = Vec::new();
        for tool in tools {
            self.tools += 1;
            if tool.failed
                && let Some(start) = tool.start
                && self.failure.is_none_or(|(at, _)| start < at)
            {
                self.failure = Some((start, prompts_through[tool.message_index - offset]));
            }
            if !tool.paired {
                self.tool_no_result += 1;
                self.tool_untimed += 1;
            } else {
                match tool.start.zip(tool.end) {
                    Some((a, b)) if b > a => intervals.push((a, b, tool.key.clone())),
                    Some((a, b)) if a == b => {
                        self.tool_same_timestamp += 1;
                        self.tool_untimed += 1;
                    }
                    Some(_) => {
                        self.tool_reversed += 1;
                        self.tool_untimed += 1;
                    }
                    None => self.tool_untimed += 1,
                }
            }
        }
        let spans: Vec<(i64, i64)> = intervals.iter().map(|(a, b, _)| (*a, *b)).collect();
        self.intervals += intervals.len() as i64;
        self.tool_sum_ms += spans.iter().map(|(a, b)| b - a).sum::<i64>();
        self.tool_union_ms += union_ms(spans.clone());
        clustered.extend(spans);
        self.clustered_ms += union_ms(clustered);
        if let Some((a, b)) = lower.or(self.first).zip(self.last) {
            let waiting: Vec<_> = waiting
                .iter()
                .map(|(a, b, lane)| (*a, *b, lane.as_str()))
                .collect();
            self.add_lanes(split(a, b, &intervals, &waiting));
        }
    }

    fn add_lanes(&mut self, lanes: Vec<Lane>) {
        let mut totals: BTreeMap<(String, String), i64> = std::mem::take(&mut self.lanes)
            .into_iter()
            .map(|l| ((l.lane, l.key), l.ms))
            .collect();
        for l in lanes {
            *totals.entry((l.lane, l.key)).or_default() += l.ms;
        }
        self.lanes = totals
            .into_iter()
            .map(|((lane, key), ms)| Lane { lane, key, ms })
            .collect();
    }

    fn finish(&self) -> (Span, Vec<Lane>) {
        let time_source = if self.timed > 0 {
            "messages"
        } else if self.untimed == 0 && self.first.is_some() {
            "events"
        } else {
            "absent"
        };
        let timed_tools = self.tools == 0 || self.intervals > 0;
        let span = Span {
            first_at: self.first,
            last_at: self.last,
            time_source: time_source.into(),
            wall_ms: self.first.zip(self.last).map(|(a, b)| b - a),
            clustered_ms: (self.timed >= 2).then_some(self.clustered_ms),
            tool_sum_ms: timed_tools.then_some(self.tool_sum_ms),
            tool_union_ms: timed_tools.then_some(self.tool_union_ms),
            tool_overlap_ms: timed_tools.then_some(self.tool_sum_ms - self.tool_union_ms),
            tool_untimed: self.tool_untimed,
            tool_no_result: self.tool_no_result,
            tool_same_timestamp: self.tool_same_timestamp,
            tool_reversed: self.tool_reversed,
            timed_messages: self.timed,
            untimed_messages: self.untimed,
            unordered_gaps: self.unordered_gaps,
            first_failure_at: self.failure.map(|(at, _)| at),
            prompts_to_failure: self.failure.map_or(self.prompts, |(_, n)| n),
            prompted: self.prompts > 0,
            ends_with: self
                .ends
                .as_ref()
                .map_or("unknown", |(ends, _)| ends.as_str())
                .into(),
            ends_with_message: self.ends.as_ref().map(|(_, id)| id.clone()),
            last_stop_reason: self.last_model_stop.clone().flatten(),
        };
        let mut lanes = if self.first.is_some() && self.last.is_some() {
            self.lanes.clone()
        } else {
            Vec::new()
        };
        // Waiting for the human starts where a model message ends its turn. A
        // session none of whose model messages states a stop reason comes from
        // a provider that reports no turn end: its human lane is unknown, not 0,
        // and that time stays unattributed.
        if !self.stop_reported {
            lanes.retain(|l| l.lane != "human");
        }
        (span, lanes)
    }
}

/// The earliest time among `rows`: a message's, or the start of a call.
fn earliest(rows: &Rows) -> Option<i64> {
    let messages = rows.messages.iter().filter_map(|m| m.at);
    let calls = rows.tools.iter().filter_map(|t| t.start);
    messages.chain(calls).min()
}

/// The last message the calculation can stop at and later resume after,
/// given the latest time before `rows`: it is timed with the latest time so
/// far, no later message or call starts before it, every call up to it has
/// its result (if any) up to it, and no call up to it still waits for its
/// result (it has no model message after it), whose arrival would make the
/// next derivation read the session again.
fn settled(before: Option<i64>, rows: &Rows) -> Option<usize> {
    let n = rows.messages.len();
    let mut starts = vec![None::<i64>; n];
    for t in &rows.tools {
        if let Some(s) = t.start {
            let slot = &mut starts[t.message_index];
            *slot = Some(slot.map_or(s, |x| x.min(s)));
        }
    }
    let mut later = vec![None::<i64>; n + 1];
    for i in (0..n).rev() {
        later[i] = [later[i + 1], rows.messages[i].at, starts[i]]
            .into_iter()
            .flatten()
            .min();
    }
    let last_model = rows.messages.iter().rposition(|m| m.author == "model");
    let waiting_from = rows
        .tools
        .iter()
        .filter(|t| !t.paired && last_model.is_none_or(|lm| t.message_index >= lm))
        .map(|t| t.message_index)
        .min()
        .unwrap_or(n);
    let mut found = None;
    let mut latest = before;
    let mut furthest: Option<&OrderKey> = None;
    let mut calls = rows.tools.iter().zip(&rows.result_keys).peekable();
    for i in 0..waiting_from {
        let at = rows.messages[i].at;
        if let Some(at) = at {
            latest = Some(latest.map_or(at, |l| l.max(at)));
        }
        while let Some((_, result)) = calls.next_if(|(t, _)| t.message_index == i) {
            if let Some(result) = result {
                furthest = furthest.max(Some(result));
            }
        }
        if let Some(at) = at
            && latest == Some(at)
            && later[i + 1].is_none_or(|l| l >= at)
            && furthest.is_none_or(|k| k <= &rows.keys[i])
        {
            found = Some(i);
        }
    }
    found
}

/// Derives the session's span and lanes: from the stored prefix when
/// nothing up to its settled message changed, reading only the messages
/// after it, else from the whole session. Returns the messages read.
pub(crate) fn derive(
    tx: &Connection,
    pk: i64,
    changes: &crate::change_summary::ChangeSummary,
    from_start: bool,
) -> Result<usize> {
    let script: bool = tx
        .prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM tool_call WHERE session_pk=?1 AND run_by_script=1)",
        )?
        .query_row([pk], |r| r.get(0))?;
    let mut resumed = None;
    if !from_start && let Some(prefix) = resumable(tx, pk, changes, script)? {
        let (key, at) = prefix
            .end
            .clone()
            .expect("a stored prefix ends at a message");
        let rows = rows(tx, pk, Some(&key))?;
        if earliest(&rows).is_none_or(|e| e >= at) {
            resumed = Some((prefix, rows));
        }
    }
    let (mut prefix, rows) = match resumed {
        Some(found) => found,
        None => {
            let rows = rows(tx, pk, None)?;
            if rows.messages.is_empty() {
                let (span, lanes) = from_events(tx, pk)?.finish();
                write(tx, pk, &span, &lanes, None)?;
                return Ok(0);
            }
            (Prefix::default(), rows)
        }
    };
    let gap_ms = CLUSTER_GAP_MS;
    let (stored, whole) = match settled(prefix.last, &rows) {
        Some(c) => {
            let split = rows.tools.partition_point(|t| t.message_index <= c);
            prefix.extend(&rows.messages[..=c], &rows.tools[..split], 0, gap_ms);
            let at = rows.messages[c].at.expect("a settled message is timed");
            prefix.end = Some((rows.keys[c].clone(), at));
            prefix.script = script;
            let mut whole = prefix.clone();
            whole.extend(&rows.messages[c + 1..], &rows.tools[split..], c + 1, gap_ms);
            (Some(prefix), whole)
        }
        None => {
            let mut whole = prefix.clone();
            whole.extend(&rows.messages, &rows.tools, 0, gap_ms);
            (prefix.end.is_some().then_some(prefix), whole)
        }
    };
    let (span, lanes) = whole.finish();
    write(tx, pk, &span, &lanes, stored.as_ref())?;
    Ok(rows.messages.len())
}

/// The stored prefix of the session, when the calculation can resume from
/// it: no message up to its settled message changed in this transaction
/// (the writer forgets the prefix when one of its own transactions changes
/// one, [`forget_settled`]), no call up to it got its result after it, and
/// whether the session has a script call is unchanged.
fn resumable(
    tx: &Connection,
    pk: i64,
    changes: &crate::change_summary::ChangeSummary,
    script: bool,
) -> Result<Option<Prefix>> {
    let stored: Option<String> = tx
        .prepare_cached("SELECT prefix FROM session_span WHERE session_pk=?1")?
        .query_row([pk], |r| r.get(0))
        .optional()?
        .flatten();
    let Some(stored) = stored else {
        return Ok(None);
    };
    let prefix: Prefix = serde_json::from_str(&stored)?;
    let Some((end, _)) = &prefix.end else {
        return Ok(None);
    };
    if prefix.script != script {
        return Ok(None);
    }
    let changed = changes.messages_before(pk);
    if changes
        .places(tx, pk, &changed)?
        .iter()
        .any(|place| place <= end)
    {
        return Ok(None);
    }
    let (order, id) = end;
    let answered_later: bool = tx
        .prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM message m CROSS JOIN block b ON b.message_pk=m.pk
               CROSS JOIN tool_call t ON t.result_block_pk=b.pk
               JOIN block cb ON cb.pk=t.call_block_pk JOIN message cm ON cm.pk=cb.message_pk
             WHERE m.session_pk=?1 AND (m.session_order,m.id)>(?2,?3)
               AND (cm.session_order,cm.id)<=(?2,?3))",
        )?
        .query_row(params![pk, order, id], |r| r.get(0))?;
    Ok((!answered_later).then_some(prefix))
}

/// Forgets the stored prefix of each session whose messages up to its
/// settled message the transaction changed, or that it changed as a whole;
/// the next derivation reads those sessions from the start. (A call's
/// outcome reported outside its messages forgets it too, where the writer
/// applies it.)
pub(crate) fn forget_settled(
    tx: &Connection,
    changes: &crate::change_summary::ChangeSummary,
    sessions: &BTreeMap<i64, crate::change_summary::SessionChange>,
) -> Result<()> {
    #[derive(Deserialize)]
    struct End {
        end: Option<(OrderKey, i64)>,
    }
    for (&pk, change) in sessions {
        let stored: Option<String> = tx
            .prepare_cached("SELECT prefix FROM session_span WHERE session_pk=?1")?
            .query_row([pk], |r| r.get(0))
            .optional()?
            .flatten();
        let Some(stored) = stored else {
            continue;
        };
        let touched = match serde_json::from_str::<End>(&stored)?.end {
            Some((end, _)) if !change.whole => changes
                .places(tx, pk, &change.messages)?
                .iter()
                .any(|place| place <= &end),
            _ => true,
        };
        if touched {
            tx.prepare_cached("UPDATE session_span SET prefix=NULL WHERE session_pk=?1")?
                .execute([pk])?;
        }
    }
    Ok(())
}

fn union_ms(mut intervals: Vec<(i64, i64)>) -> i64 {
    intervals.sort_unstable();
    let mut total = 0;
    let mut end = None;
    for (a, b) in intervals {
        if b <= a {
            continue;
        }
        total += (b - a.max(end.unwrap_or(a))).max(0);
        end = Some(end.map_or(b, |e: i64| e.max(b)));
    }
    total
}

#[derive(Default)]
struct Changes<'a> {
    tool_start: Vec<usize>,
    tool_end: Vec<usize>,
    wait_start: Vec<&'a str>,
    wait_end: Vec<&'a str>,
}

/// Sweep interval boundaries once; tools take priority over waiting intervals.
fn split(
    first: i64,
    last: i64,
    tools: &[(i64, i64, String)],
    waiting: &[(i64, i64, &str)],
) -> Vec<Lane> {
    let mut points = BTreeMap::<i64, Changes<'_>>::new();
    points.entry(first).or_default();
    points.entry(last).or_default();
    for (i, (a, b, _)) in tools.iter().enumerate() {
        let (a, b) = ((*a).max(first), (*b).min(last));
        if b > a {
            points.entry(a).or_default().tool_start.push(i);
            points.entry(b).or_default().tool_end.push(i);
        }
    }
    for (a, b, lane) in waiting {
        let (a, b) = ((*a).max(first), (*b).min(last));
        if b > a {
            points.entry(a).or_default().wait_start.push(*lane);
            points.entry(b).or_default().wait_end.push(*lane);
        }
    }
    let mut totals = BTreeMap::<(String, String), i64>::new();
    for lane in ["model", "human", "unattributed"] {
        totals.insert((lane.into(), String::new()), 0);
    }
    let mut active_tools = BTreeSet::<usize>::new();
    let mut active_wait = BTreeMap::<&str, usize>::new();
    let mut previous = first;
    for (at, change) in points {
        let ms = at - previous;
        if !active_tools.is_empty() {
            let n = active_tools.len() as i64;
            for (rank, &i) in active_tools.iter().enumerate() {
                *totals
                    .entry(("tool".into(), tools[i].2.clone()))
                    .or_default() += ms / n + i64::from((rank as i64) < ms % n);
            }
        } else {
            let lane = if active_wait.len() == 1 && active_wait.values().sum::<usize>() == 1 {
                *active_wait.keys().next().unwrap()
            } else {
                "unattributed"
            };
            *totals.entry((lane.into(), String::new())).or_default() += ms;
        }
        for i in change.tool_end {
            active_tools.remove(&i);
        }
        for i in change.tool_start {
            active_tools.insert(i);
        }
        for lane in change.wait_end {
            if let Some(n) = active_wait.get_mut(lane) {
                *n -= 1;
                if *n == 0 {
                    active_wait.remove(lane);
                }
            }
        }
        for lane in change.wait_start {
            *active_wait.entry(lane).or_default() += 1;
        }
        previous = at;
    }
    totals
        .into_iter()
        .map(|((lane, key), ms)| Lane { lane, key, ms })
        .collect()
}

fn write(
    conn: &Connection,
    pk: i64,
    span: &Span,
    lanes: &[Lane],
    prefix: Option<&Prefix>,
) -> Result<()> {
    let prefix = prefix.map(serde_json::to_string).transpose()?;
    conn.prepare_cached("INSERT OR REPLACE INTO session_span(session_pk,first_at,last_at,time_source,wall_ms,clustered_ms,tool_sum_ms,tool_union_ms,tool_overlap_ms,tool_untimed,tool_no_result,tool_same_timestamp,tool_reversed,timed_messages,untimed_messages,unordered_gaps,first_failure_at,prompts_to_failure,prompted,ends_with,ends_with_message,last_stop_reason,prefix) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23)")?
        .execute(params![pk,span.first_at,span.last_at,span.time_source,span.wall_ms,span.clustered_ms,span.tool_sum_ms,span.tool_union_ms,span.tool_overlap_ms,span.tool_untimed,span.tool_no_result,span.tool_same_timestamp,span.tool_reversed,span.timed_messages,span.untimed_messages,span.unordered_gaps,span.first_failure_at,span.prompts_to_failure,span.prompted,span.ends_with,span.ends_with_message,span.last_stop_reason,prefix])?;
    conn.prepare_cached("DELETE FROM session_time WHERE session_pk=?1")?
        .execute([pk])?;
    let mut statement = conn
        .prepare_cached("INSERT INTO session_time(session_pk,lane,key,ms) VALUES(?1,?2,?3,?4)")?;
    for l in lanes {
        statement.execute(params![pk, l.lane, l.key, l.ms])?;
    }
    Ok(())
}

const SPAN_COLUMNS: &str = "first_at,last_at,time_source,wall_ms,clustered_ms,tool_sum_ms,tool_union_ms,tool_overlap_ms,tool_untimed,tool_no_result,tool_same_timestamp,tool_reversed,timed_messages,untimed_messages,unordered_gaps,first_failure_at,prompts_to_failure,prompted,ends_with,ends_with_message,last_stop_reason";

fn span_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Span> {
    Ok(Span {
        first_at: r.get(0)?,
        last_at: r.get(1)?,
        time_source: r.get(2)?,
        wall_ms: r.get(3)?,
        clustered_ms: r.get(4)?,
        tool_sum_ms: r.get(5)?,
        tool_union_ms: r.get(6)?,
        tool_overlap_ms: r.get(7)?,
        tool_untimed: r.get(8)?,
        tool_no_result: r.get(9)?,
        tool_same_timestamp: r.get(10)?,
        tool_reversed: r.get(11)?,
        timed_messages: r.get(12)?,
        untimed_messages: r.get(13)?,
        unordered_gaps: r.get(14)?,
        first_failure_at: r.get(15)?,
        prompts_to_failure: r.get(16)?,
        prompted: r.get(17)?,
        ends_with: r.get(18)?,
        ends_with_message: r.get(19)?,
        last_stop_reason: r.get(20)?,
    })
}
fn stored(conn: &Connection, pk: i64) -> Result<Option<Span>> {
    Ok(conn
        .prepare_cached(&format!(
            "SELECT {SPAN_COLUMNS} FROM session_span WHERE session_pk=?1"
        ))?
        .query_row([pk], span_row)
        .optional()?)
}

/// Survival reads all current spans in one statement, without loading lane rows.
pub(crate) fn selected_spans(
    reader: &Reader<'_>,
    selection: &Selection,
) -> Result<(Vec<Span>, u64)> {
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let (count, pk): (i64, Option<i64>) = conn.query_row(
        &format!("SELECT count(*),min(s.pk) FROM session s WHERE {condition}"),
        rusqlite::params_from_iter(params.iter()),
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if count == 1
        && let Some(pk) = pk
    {
        let span = if crate::derive::is_current(conn, "session_span", pk)? {
            stored(conn, pk)?
        } else {
            Some(compute(conn, pk, CLUSTER_GAP_MS)?.0)
        };
        return Ok((span.into_iter().collect(), 0));
    }
    let columns = SPAN_COLUMNS
        .split(',')
        .map(|c| format!("f.{c}"))
        .collect::<Vec<_>>()
        .join(",");
    let spans=conn.prepare(&format!("SELECT {columns} FROM session s JOIN session_span f ON f.session_pk=s.pk WHERE {condition} AND {}",session_sql::current("s","session_span")))?.query_map(rusqlite::params_from_iter(params.iter()),span_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
    let pending = count as u64 - spans.len() as u64;
    Ok((spans, pending))
}

pub(crate) fn read(
    conn: &Connection,
    pk: i64,
    compute_behind: bool,
) -> Result<(Option<Span>, Vec<Lane>, bool)> {
    if !crate::derive::is_current(conn, "session_span", pk)? {
        return if compute_behind {
            let (s, l) = compute(conn, pk, CLUSTER_GAP_MS)?;
            Ok((Some(s), l, true))
        } else {
            Ok((None, vec![], false))
        };
    }
    let span = stored(conn, pk)?;
    let lanes = conn
        .prepare_cached(
            "SELECT lane,key,ms FROM session_time WHERE session_pk=?1 ORDER BY lane,key",
        )?
        .query_map([pk], |r| {
            Ok(Lane {
                lane: r.get(0)?,
                key: r.get(1)?,
                ms: r.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok((span, lanes, false))
}

pub fn times(reader: &Reader<'_>, selection: &Selection) -> Result<Vec<SessionTime>> {
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let rows = conn
        .prepare(&format!(
            "SELECT s.pk,s.id FROM session s WHERE {condition} ORDER BY s.id"
        ))?
        .query_map(rusqlite::params_from_iter(params.iter()), |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let single = rows.len() == 1;
    rows.into_iter()
        .map(|(pk, session)| {
            let (span, lanes, computed_at_read) = read(conn, pk, single)?;
            Ok(SessionTime {
                session,
                span,
                lanes,
                computed_at_read,
                parameters: polylogue_model::derived::TIME_PARAMETERS,
            })
        })
        .collect()
}

pub fn pending(reader: &Reader<'_>, selection: &Selection) -> Result<u64> {
    let (condition, params) = selection.session_condition("s.pk")?;
    Ok(reader.conn()?.query_row(
        &format!(
            "SELECT count(*) FROM session s WHERE {condition} AND NOT ({})",
            session_sql::current("s", "session_span")
        ),
        rusqlite::params_from_iter(params.iter()),
        |r| Ok(r.get::<_, i64>(0)? as u64),
    )?)
}

#[cfg(test)]
mod tests;

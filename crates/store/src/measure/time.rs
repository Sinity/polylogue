//! Recorded clocks and interval calculations. No notes, settings or clock reads.
use super::Selection;
use crate::{Reader, Result};
use polylogue_model::{derived::CLUSTER_GAP_MS, session_sql};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const RULE_VERSION: i64 = 1;

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

/// The same calculation serves the derivation and a read of a behind session.
pub(crate) fn compute(conn: &Connection, pk: i64, gap_ms: i64) -> Result<(Span, Vec<Lane>)> {
    let own = session_sql::own_messages("m", Some("=?1"));
    let mut statement=conn.prepare_cached(&format!("SELECT m.id,m.at,m.written_by,m.prompt_kind,m.stop_reason,
        (SELECT b.kind FROM block b WHERE b.message_pk=m.pk ORDER BY b.idx DESC LIMIT 1),
        EXISTS(SELECT 1 FROM tool_call t WHERE t.call_block_pk=(SELECT b.pk FROM block b WHERE b.message_pk=m.pk ORDER BY b.idx DESC LIMIT 1) AND t.result_block_pk IS NULL)
        FROM message m WHERE {own} ORDER BY m.order_input_id,m.seq,m.record_position,m.id"))?;
    let messages = statement
        .query_map([pk], |r| {
            Ok(Message {
                id: r.get(0)?,
                at: r.get(1)?,
                author: r.get(2)?,
                prompt: r.get(3)?,
                stop: r.get(4)?,
                last_block: r.get(5)?,
                open_call: r.get(6)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let message_times: Vec<_> = messages.iter().filter_map(|m| m.at).collect();
    let (first, last, source) = if messages.is_empty() {
        let (a, b): (Option<i64>, Option<i64>) = conn.query_row(
            "SELECT min(at),max(at) FROM event WHERE session_pk=?1",
            [pk],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        (a, b, if a.is_some() { "events" } else { "absent" })
    } else {
        (
            message_times.iter().min().copied(),
            message_times.iter().max().copied(),
            if message_times.is_empty() {
                "absent"
            } else {
                "messages"
            },
        )
    };
    let positions: BTreeMap<_, _> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.as_str(), i))
        .collect();
    let mut statement=conn.prepare_cached(&format!("SELECT CASE WHEN t.action='shell' THEN coalesce(t.command_norm,t.tool) ELSE coalesce(t.action,t.tool) END,
        coalesce(t.at,m.at),CASE WHEN rm.duplicate_of IS NULL AND coalesce(rm.on_active_path,1)=1 THEN rm.at END,t.result_block_pk IS NOT NULL,t.outcome='error',m.id
        FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk
        LEFT JOIN block rb ON rb.pk=t.result_block_pk LEFT JOIN message rm ON rm.pk=rb.message_pk
        WHERE {own} AND {}
        ORDER BY m.order_input_id,m.seq,m.record_position,m.id,b.idx,t.call_id,t.command_norm,t.tool",session_sql::counts_as_action("t")))?;
    let tools = statement
        .query_map([pk], |r| {
            Ok(Tool {
                message_index: *positions
                    .get(r.get::<_, String>(5)?.as_str())
                    .ok_or(rusqlite::Error::InvalidQuery)?,
                key: r.get(0)?,
                start: r.get(1)?,
                end: r.get(2)?,
                paired: r.get(3)?,
                failed: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(calculate(&messages, &tools, first, last, source, gap_ms))
}

fn calculate(
    messages: &[Message],
    tools: &[Tool],
    first: Option<i64>,
    last: Option<i64>,
    source: &str,
    gap_ms: i64,
) -> (Span, Vec<Lane>) {
    let mut span = Span {
        first_at: first,
        last_at: last,
        time_source: source.into(),
        wall_ms: first.zip(last).map(|(a, b)| b - a),
        clustered_ms: None,
        tool_sum_ms: None,
        tool_union_ms: None,
        tool_overlap_ms: None,
        tool_untimed: 0,
        tool_no_result: 0,
        tool_same_timestamp: 0,
        tool_reversed: 0,
        timed_messages: messages.iter().filter(|m| m.at.is_some()).count() as i64,
        untimed_messages: messages.iter().filter(|m| m.at.is_none()).count() as i64,
        unordered_gaps: 0,
        first_failure_at: tools
            .iter()
            .filter(|t| t.failed)
            .filter_map(|t| t.start)
            .min(),
        prompts_to_failure: 0,
        prompted: messages.iter().any(Message::human_prompt),
        ends_with: "unknown".into(),
        ends_with_message: None,
        last_stop_reason: messages
            .iter()
            .rev()
            .find(|m| m.author == "model")
            .and_then(|m| m.stop.clone()),
    };
    let failure_index = tools
        .iter()
        .filter(|t| t.failed)
        .filter_map(|t| t.start.map(|at| (at, t.message_index)))
        .min()
        .map(|(_, i)| i);
    span.prompts_to_failure = messages
        .iter()
        .enumerate()
        .filter(|(i, m)| m.human_prompt() && failure_index.is_none_or(|end| *i <= end))
        .count() as i64;
    if let Some(m) = messages.last().filter(|m| m.at.is_some()) {
        span.ends_with = m.ends_with().into();
        span.ends_with_message = Some(m.id.clone());
    }
    let mut tool_intervals = Vec::new();
    for tool in tools {
        if !tool.paired {
            span.tool_no_result += 1;
            span.tool_untimed += 1;
        } else {
            match tool.start.zip(tool.end) {
                Some((a, b)) if b > a => tool_intervals.push((a, b, tool.key.clone())),
                Some((a, b)) if a == b => {
                    span.tool_same_timestamp += 1;
                    span.tool_untimed += 1;
                }
                Some(_) => {
                    span.tool_reversed += 1;
                    span.tool_untimed += 1;
                }
                None => span.tool_untimed += 1,
            }
        }
    }
    if tools.is_empty() || !tool_intervals.is_empty() {
        let sum = tool_intervals.iter().map(|(a, b, _)| b - a).sum();
        let union = union_ms(tool_intervals.iter().map(|(a, b, _)| (*a, *b)).collect());
        span.tool_sum_ms = Some(sum);
        span.tool_union_ms = Some(union);
        span.tool_overlap_ms = Some(sum - union);
    }
    let mut clustered = Vec::new();
    let mut waiting = Vec::new();
    for pair in messages.windows(2) {
        if let Some((a, b)) = pair[0].at.zip(pair[1].at) {
            if b < a {
                span.unordered_gaps += 1;
            } else {
                if b - a <= gap_ms {
                    clustered.push((a, b));
                }
                waiting.push((a, b, pair[0].waiting_lane()));
            }
        }
    }
    if span.timed_messages >= 2 {
        clustered.extend(tool_intervals.iter().map(|(a, b, _)| (*a, *b)));
        span.clustered_ms = Some(union_ms(clustered));
    }
    let lanes = first
        .zip(last)
        .map_or_else(Vec::new, |(a, b)| split(a, b, &tool_intervals, &waiting));
    (span, lanes)
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
struct Changes {
    tool_start: Vec<usize>,
    tool_end: Vec<usize>,
    wait_start: Vec<&'static str>,
    wait_end: Vec<&'static str>,
}

/// Sweep interval boundaries once; tools take priority over waiting intervals.
fn split(
    first: i64,
    last: i64,
    tools: &[(i64, i64, String)],
    waiting: &[(i64, i64, &'static str)],
) -> Vec<Lane> {
    let mut points = BTreeMap::<i64, Changes>::new();
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
            points.entry(a).or_default().wait_start.push(lane);
            points.entry(b).or_default().wait_end.push(lane);
        }
    }
    let mut totals = BTreeMap::<(String, String), i64>::new();
    for lane in ["model", "human", "unattributed"] {
        totals.insert((lane.into(), String::new()), 0);
    }
    let mut active_tools = BTreeSet::new();
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

pub(crate) fn write(conn: &Connection, pk: i64, span: &Span, lanes: &[Lane]) -> Result<()> {
    conn.prepare_cached("INSERT OR REPLACE INTO session_span(session_pk,first_at,last_at,time_source,wall_ms,clustered_ms,tool_sum_ms,tool_union_ms,tool_overlap_ms,tool_untimed,tool_no_result,tool_same_timestamp,tool_reversed,timed_messages,untimed_messages,unordered_gaps,first_failure_at,prompts_to_failure,prompted,ends_with,ends_with_message,last_stop_reason) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22)")?
        .execute(params![pk,span.first_at,span.last_at,span.time_source,span.wall_ms,span.clustered_ms,span.tool_sum_ms,span.tool_union_ms,span.tool_overlap_ms,span.tool_untimed,span.tool_no_result,span.tool_same_timestamp,span.tool_reversed,span.timed_messages,span.untimed_messages,span.unordered_gaps,span.first_failure_at,span.prompts_to_failure,span.prompted,span.ends_with,span.ends_with_message,span.last_stop_reason])?;
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
    let (count, pk): (u64, Option<i64>) = conn.query_row(
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
    let pending = count - spans.len() as u64;
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
        |r| r.get(0),
    )?)
}

#[cfg(test)]
mod tests;

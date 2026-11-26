//! The live summary: each session's live events (hook, process, registry
//! and git lines with a time) folded in row order, which is time, then the
//! input's stable identity, then the line's place in it, so lines at one
//! instant fold alike in every arrival order and a rebuild orders them as
//! upkeep does. The fold gives the session's `live_facts`
//! ([`polylogue_model::live::apply_live_event`]); the intervals each event
//! closes go to `live_interval` and process lines to `process`, each row
//! with the place of the event that wrote it, so a run from a stored state
//! replaces exactly the rows the events after that state wrote.

use super::Output;
use super::fold::{Context, Flow, Fold, Key, SessionRow};
use crate::Result;
use polylogue_model::Json;
use polylogue_model::live::{self, LiveFacts, LiveInterval};
use rusqlite::{Connection, OptionalExtension, params};

pub(crate) struct Live;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct State {
    facts: LiveFacts,
}

/// What a run of the live summary writes: the intervals and process lines
/// of the events it folded, each with the event's place, and the facts
/// after every event.
#[derive(Default)]
pub(crate) struct Emitted {
    intervals: Vec<(LiveInterval, Key)>,
    lines: Vec<(String, i64, Json, Key)>,
    facts: Option<LiveFacts>,
}

impl Fold for Live {
    type State = State;
    const NAME: &'static str = "live";
    const VERSION: i64 = 1;

    fn step(
        state: &mut State,
        row: &SessionRow,
        cx: &Context<'_>,
        out: &mut Output,
    ) -> Result<Flow> {
        let SessionRow::Event(e) = row else {
            return Ok(Flow::Continue);
        };
        // A line without a time cannot be ordered against the others.
        let Some(at) = e.at.filter(|_| live::is_live_event(&e.kind)) else {
            return Ok(Flow::Continue);
        };
        let data: Json = cx.json(e.data)?;
        let closed = live::apply_live_event(&mut state.facts, &e.kind, at, &data);
        out.live
            .intervals
            .extend(closed.into_iter().map(|i| (i, e.key.clone())));
        if e.kind == SESSION_END || e.kind.starts_with(live::PROCESS_PREFIX) {
            out.live
                .lines
                .push((e.kind.clone(), at, data, e.key.clone()));
        }
        Ok(Flow::Continue)
    }

    fn end(state: &State, _: &Context<'_>, out: &mut Output) -> Result<()> {
        out.live.facts = Some(state.facts.clone());
        Ok(())
    }
}

const SESSION_END: &str = "hook_session_end";

/// Writes a run's live rows: those the events after `after` wrote before
/// are replaced by the run's, and the facts are stored when they changed.
pub(super) fn apply(
    tx: &Connection,
    session_pk: i64,
    after: &Key,
    emitted: &Emitted,
    counter: i64,
) -> Result<()> {
    tx.prepare_cached("DELETE FROM live_interval WHERE session_pk = ?1 AND closed > ?2")?
        .execute(params![session_pk, after.0])?;
    tx.prepare_cached("DELETE FROM process WHERE session_pk = ?1 AND opened > ?2")?
        .execute(params![session_pk, after.0])?;
    tx.prepare_cached(
        "UPDATE process SET ended_at = NULL, ended = NULL WHERE session_pk = ?1 AND ended > ?2",
    )?
    .execute(params![session_pk, after.0])?;
    let mut interval = tx.prepare_cached(
        "INSERT OR REPLACE INTO live_interval(session_pk, aspect, value, start_at, end_at, closed_by, closed)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    for (i, key) in &emitted.intervals {
        interval.execute(params![
            session_pk,
            i.aspect.as_str(),
            i.value,
            i.start,
            i.end,
            i.closed_by,
            key.0
        ])?;
    }
    for (kind, at, data, key) in &emitted.lines {
        record_process(tx, session_pk, kind, *at, data, key)?;
    }
    let Some(facts) = &emitted.facts else {
        return Ok(());
    };
    let text = (facts != &LiveFacts::default())
        .then(|| serde_json::to_string(facts))
        .transpose()?;
    let stored: Option<(Option<String>, Option<i64>)> = tx
        .prepare_cached("SELECT live_facts, live_last_at FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| <(_, _)>::try_from(r))
        .optional()?;
    if stored != Some((text.clone(), facts.last_at)) {
        tx.prepare_cached(
            "UPDATE session SET live_facts = ?2, live_last_at = ?3, live_changed = ?4 WHERE pk = ?1",
        )?
        .execute(params![session_pk, text, facts.last_at, counter])?;
    }
    Ok(())
}

/// Keeps `process` rows from process lines: one row per process identity,
/// with its exit when seen, or the session's end.
fn record_process(
    tx: &Connection,
    session_pk: i64,
    kind: &str,
    at: i64,
    data: &Json,
    key: &Key,
) -> Result<()> {
    if kind == SESSION_END {
        // The session's end closes the rows of its processes: the watcher
        // stops watching an ended session, so nothing else would.
        tx.prepare_cached(
            "UPDATE process SET ended_at = ?2, ended = ?3 WHERE session_pk = ?1 AND ended_at IS NULL",
        )?
        .execute(params![session_pk, at, key.0])?;
        return Ok(());
    }
    let Some(event) = kind.strip_prefix(live::PROCESS_PREFIX) else {
        return Ok(());
    };
    let (Some(boot), Some(pid), Some(start)) = (
        data.get("boot_id").and_then(Json::as_str),
        data.get("pid").and_then(Json::as_i64),
        data.get("start").and_then(Json::as_i64),
    ) else {
        return Ok(());
    };
    match event {
        "started" => {
            let argv = data.get("argv").map(Json::to_string);
            tx.prepare_cached(
                "INSERT INTO process(session_pk, boot_id, pid, start, parent_pid, comm, argv, cwd,
                                     started_at, role, opened)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                 ON CONFLICT DO UPDATE SET parent_pid = excluded.parent_pid, comm = excluded.comm,
                     argv = excluded.argv, cwd = excluded.cwd, role = excluded.role,
                     started_at = MIN(started_at, excluded.started_at)",
            )?
            .execute(params![
                session_pk,
                boot,
                pid,
                start,
                data.get("ppid").and_then(Json::as_i64),
                data.get("comm").and_then(Json::as_str),
                argv,
                data.get("cwd").and_then(Json::as_str),
                at,
                data.get("role").and_then(Json::as_str).unwrap_or("child"),
                key.0,
            ])?;
        }
        "ended" | "root_gone" => {
            tx.prepare_cached(
                "UPDATE process SET ended_at = ?5, ended = ?6 WHERE session_pk = ?1 AND boot_id = ?2
                     AND pid = ?3 AND start = ?4 AND ended_at IS NULL",
            )?
            .execute(params![session_pk, boot, pid, start, at, key.0])?;
        }
        _ => {}
    }
    Ok(())
}

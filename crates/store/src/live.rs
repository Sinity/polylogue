//! Live facts and their history. Each stored piece of live evidence (an
//! event of route `hook`, `process` or `registry`) is folded into its
//! session's `live_facts` in the transaction that stores it
//! ([`polylogue_model::live::fold`]); the intervals it closes go to
//! `live_interval` and process lines to `process`. The live state itself
//! is computed when asked, from the facts and the time (`pl_live`); nothing
//! here looks at the clock.
//!
//! Evidence that arrives no later than what a session has already folded
//! marks the session for a refold, which the session derivation runs over
//! every stored line in (time, input, byte offset) order: lines at one
//! instant may arrive in any order, and a line stored again must not fold
//! twice. Both paths give the same rows.

use crate::Result;
use polylogue_model::Json;
use polylogue_model::live::{self, LiveFacts, LiveInterval};
use rusqlite::{OptionalExtension, Transaction, params};

/// The kinds that are live evidence, as a SQL condition on `kind`.
pub(crate) const EVIDENCE_KINDS: &str = "(kind LIKE 'hook~_%' ESCAPE '~' \
     OR kind LIKE 'process~_%' ESCAPE '~' OR kind LIKE 'registry~_%' ESCAPE '~')";

fn load(tx: &Transaction<'_>, session_pk: i64) -> Result<(LiveFacts, Option<i64>)> {
    let (facts, last): (Option<String>, Option<i64>) = tx
        .prepare_cached("SELECT live_facts, live_last_at FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?
        .unwrap_or((None, None));
    let facts = match facts {
        Some(text) => serde_json::from_str(&text)?,
        None => LiveFacts::default(),
    };
    Ok((facts, last))
}

fn store(tx: &Transaction<'_>, session_pk: i64, facts: &LiveFacts, counter: i64) -> Result<()> {
    let text = (facts != &LiveFacts::default())
        .then(|| serde_json::to_string(facts))
        .transpose()?;
    tx.prepare_cached(
        "UPDATE session SET live_facts = ?2, live_last_at = ?3, live_changed = ?4 WHERE pk = ?1",
    )?
    .execute(params![session_pk, text, facts.last_at, counter])?;
    Ok(())
}

fn record_intervals(tx: &Transaction<'_>, session_pk: i64, closed: &[LiveInterval]) -> Result<()> {
    for i in closed {
        tx.prepare_cached(
            "INSERT OR REPLACE INTO live_interval(session_pk, aspect, value, start_at, end_at, closed_by)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?
        .execute(params![
            session_pk,
            i.aspect.as_str(),
            i.value,
            i.start,
            i.end,
            i.closed_by
        ])?;
    }
    Ok(())
}

/// Keeps `process` rows from process lines: one row per process identity,
/// with its exit when seen.
fn record_process(tx: &Transaction<'_>, session_pk: i64, kind: &str, at: i64, data: &Json) -> Result<()> {
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
                "INSERT INTO process(session_pk, boot_id, pid, start, parent_pid, comm, argv, cwd, started_at, role)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
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
            ])?;
        }
        "ended" | "root_gone" => {
            tx.prepare_cached(
                "UPDATE process SET ended_at = ?5 WHERE session_pk = ?1 AND boot_id = ?2 AND pid = ?3
                     AND start = ?4 AND ended_at IS NULL",
            )?
            .execute(params![session_pk, boot, pid, start, at])?;
        }
        _ => {}
    }
    Ok(())
}

/// Folds one stored live evidence event into `session_pk`'s facts. An
/// event no later than the facts already folded marks the session for a
/// refold instead, so the result never depends on arrival order.
pub(crate) fn apply(
    tx: &Transaction<'_>,
    session_pk: i64,
    kind: &str,
    at: Option<i64>,
    data: &Json,
    counter: i64,
) -> Result<()> {
    if !live::is_evidence(kind) {
        return Ok(());
    }
    // A line without a time cannot be ordered against the others.
    let Some(at) = at else {
        return Ok(());
    };
    let (mut facts, last) = load(tx, session_pk)?;
    if last.is_some_and(|last| at <= last) {
        tx.prepare_cached("UPDATE session SET live_reparse = 1 WHERE pk = ?1")?
            .execute([session_pk])?;
        return Ok(());
    }
    let closed = live::fold(&mut facts, kind, at, data);
    record_intervals(tx, session_pk, &closed)?;
    record_process(tx, session_pk, kind, at, data)?;
    store(tx, session_pk, &facts, counter)
}

/// Folds every stored live evidence event of the session again, in
/// (time, line) order, replacing its facts, intervals and process rows.
pub(crate) fn refold(tx: &Transaction<'_>, session_pk: i64, counter: i64) -> Result<()> {
    tx.prepare_cached("DELETE FROM live_interval WHERE session_pk = ?1")?
        .execute([session_pk])?;
    tx.prepare_cached("DELETE FROM process WHERE session_pk = ?1")?
        .execute([session_pk])?;
    let mut facts = LiveFacts::default();
    let mut statement = tx.prepare(&format!(
        "SELECT kind, at, data FROM event WHERE session_pk = ?1 AND at IS NOT NULL AND {EVIDENCE_KINDS}
         ORDER BY at, input_pk, source_start, id"
    ))?;
    let events = statement.query_map([session_pk], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, Vec<u8>>(2)?,
        ))
    })?;
    for event in events {
        let (kind, at, data) = event?;
        let data: Json = crate::encoding::unpack_json(&data)?;
        let closed = live::fold(&mut facts, &kind, at, &data);
        record_intervals(tx, session_pk, &closed)?;
        record_process(tx, session_pk, &kind, at, &data)?;
    }
    store(tx, session_pk, &facts, counter)?;
    tx.prepare_cached("UPDATE session SET live_reparse = 0 WHERE pk = ?1")?
        .execute([session_pk])?;
    Ok(())
}

/// A session whose root process is alive by its evidence: what the
/// process watcher watches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRoot {
    pub session: String,
    pub root: live::ProcessIdentity,
    /// A tool call is open: descendants are sampled often.
    pub open_call: bool,
}

impl crate::Reader<'_> {
    /// Every session whose root process is alive by its evidence.
    pub fn live_roots(&self) -> Result<Vec<LiveRoot>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, live_facts FROM session
             WHERE live_last_at IS NOT NULL AND live_facts IS NOT NULL
               AND json_extract(live_facts, '$.root') IS NOT NULL
               AND json_extract(live_facts, '$.root_gone_at') IS NULL
               AND json_extract(live_facts, '$.ended_at') IS NULL",
        )?;
        let rows =
            statement.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        let mut out = Vec::new();
        for row in rows {
            let (session, facts) = row?;
            let facts: LiveFacts = serde_json::from_str(&facts)?;
            if let Some(root) = facts.root {
                out.push(LiveRoot {
                    session,
                    root,
                    open_call: facts.open_call.is_some(),
                });
            }
        }
        Ok(out)
    }
}

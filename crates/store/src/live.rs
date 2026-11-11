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
//! every stored line in (time, input identity, byte offset) order (the
//! input's stable identity, not its row key, so a rebuild orders alike): lines at one
//! instant may arrive in any order, and a line stored again must not fold
//! twice. Both paths give the same rows.

use crate::Result;
use polylogue_model::Json;
use polylogue_model::live::{self, LiveFacts, LiveInterval};
use rusqlite::{OptionalExtension, Transaction, params};
use std::collections::BTreeSet;

/// The kinds that are live evidence, as a SQL condition on `kind`.
pub(crate) const EVIDENCE_KINDS: &str = "(kind LIKE 'hook~_%' ESCAPE '~' \
     OR kind LIKE 'process~_%' ESCAPE '~' OR kind LIKE 'registry~_%' ESCAPE '~' \
     OR kind LIKE 'git~_%' ESCAPE '~')";

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

const SESSION_END: &str = "hook_session_end";

/// Keeps `process` rows from process lines: one row per process identity,
/// with its exit when seen, or the session's end.
fn record_process(
    tx: &Transaction<'_>,
    session_pk: i64,
    kind: &str,
    at: i64,
    data: &Json,
) -> Result<()> {
    if kind == SESSION_END {
        // The session's end closes the rows of its processes: the watcher
        // stops watching an ended session, so nothing else would.
        tx.prepare_cached(
            "UPDATE process SET ended_at = ?2 WHERE session_pk = ?1 AND ended_at IS NULL",
        )?
        .execute(params![session_pk, at])?;
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
        "SELECT e.kind, e.at, e.data FROM event e JOIN input_ref i ON i.pk = e.input_pk
         WHERE e.session_pk = ?1 AND e.at IS NOT NULL AND {EVIDENCE_KINDS}
         ORDER BY e.at, i.id, e.source_start, e.id"
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

/// One session's live facts and view, with where it last worked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveSession {
    pub view: live::LiveView,
    pub facts: LiveFacts,
    pub cwd: Option<String>,
}

/// One session's live state in full, for `read --view live`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveDetail {
    pub view: live::LiveView,
    pub facts: LiveFacts,
    /// Peer session IDs.
    pub peers: Vec<String>,
    pub processes: Vec<ProcessRow>,
}

/// A process row: the harness or one of its descendants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessRow {
    pub pid: i64,
    pub parent_pid: Option<i64>,
    pub start: i64,
    pub comm: Option<String>,
    pub argv: Vec<String>,
    pub cwd: Option<String>,
    pub started_at: Option<i64>,
    pub ended_at: Option<i64>,
    /// `harness` or `child`.
    pub role: String,
    /// The tool call it started during, by time interval.
    pub tool_call: Option<String>,
}

/// One session as `follow` sees it at an instant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowSnapshot {
    pub view: live::LiveView,
    pub open_tool: Option<String>,
    pub messages: Vec<FollowedMessage>,
    pub tool_calls: Vec<FollowedCall>,
}

/// A message stored after the follower's last one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowedMessage {
    /// The row key, which only grows as rows are stored.
    pub key: i64,
    pub id: String,
    pub role: String,
    pub at: Option<i64>,
}

/// A tool call stored after the follower's last one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowedCall {
    pub key: i64,
    pub call_id: Option<String>,
    pub tool: String,
    pub at: Option<i64>,
}

/// A running session's uncommitted files, for the commit watcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveFootprint {
    pub session: String,
    pub cwd: String,
    pub files: Vec<String>,
}

/// One running session as the running list shows it.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct RunningSession {
    pub session: String,
    #[schemars(with = "String")]
    pub state: live::LiveState,
    /// When it entered its state, as far as the evidence says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub waiting_on: Option<live::WaitingOn>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub stall: Option<live::Stall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    /// How many files it changed and has not committed.
    pub footprint: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
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
    /// The sessions whose live state or stall would change at an instant
    /// with no new evidence, with that instant (`live_until`): those with
    /// live evidence within a day, and those whose transcript alone grew
    /// within the live window.
    pub fn live_untils(&self, now: i64) -> Result<Vec<(String, i64)>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, pl_live(live_facts, last_activity_at, ?1, 'until') AS until FROM session
             WHERE (live_last_at IS NOT NULL AND live_last_at >= ?2)
                OR (live_last_at IS NULL AND last_activity_at >= ?3)",
        )?;
        let rows = statement.query_map(
            params![
                now,
                now - live::STALE_FOR_MS - live::LIVE_WINDOW_MS,
                now - live::LIVE_WINDOW_MS - 1
            ],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<i64>>(1)?)),
        )?;
        let mut out = Vec::new();
        for row in rows {
            if let (session, Some(until)) = row? {
                out.push((session, until));
            }
        }
        Ok(out)
    }

    /// The running sessions at `now` (working, waiting, idle, unknown):
    /// waiting first, then by how long each has been in its state.
    pub fn running(&self, now: i64) -> Result<Vec<RunningSession>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, live_facts, last_activity_at, cwd FROM session
             WHERE (live_last_at IS NOT NULL AND live_last_at >= ?1)
                OR (live_last_at IS NULL AND last_activity_at >= ?2)",
        )?;
        let rows = statement.query_map(
            params![
                now - live::STALE_FOR_MS - live::LIVE_WINDOW_MS,
                now - live::LIVE_WINDOW_MS - 1
            ],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            },
        )?;
        let mut out = Vec::new();
        for row in rows {
            let (session, facts, grew_at, cwd) = row?;
            let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
            let v = live::view(facts.as_ref(), grew_at, now);
            if !v.state.running() {
                continue;
            }
            out.push(RunningSession {
                session,
                state: v.state,
                since: v.since,
                waiting_on: v.waiting_on,
                stall: v.stall,
                open_tool: facts
                    .as_ref()
                    .and_then(|f| f.open_call.as_ref().map(|c| c.tool.clone())),
                task: facts.as_ref().and_then(|f| f.task.clone()),
                footprint: facts.as_ref().map_or(0, |f| f.footprint.len()),
                cwd,
            });
        }
        out.sort_by(|a, b| {
            (a.state.rank(), a.since.unwrap_or(i64::MAX), &a.session).cmp(&(
                b.state.rank(),
                b.since.unwrap_or(i64::MAX),
                &b.session,
            ))
        });
        Ok(out)
    }

    /// One session's live facts, its view at `now` and its last working
    /// directory; `None` for a session not in the archive.
    pub fn live_state(&self, session: &str, now: i64) -> Result<Option<LiveSession>> {
        let conn = self.conn()?;
        let found = conn
            .prepare_cached("SELECT live_facts, last_activity_at, cwd FROM session WHERE id = ?1")?
            .query_row([session], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<i64>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                ))
            })
            .optional()?;
        let Some((facts, grew_at, cwd)) = found else {
            return Ok(None);
        };
        let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
        Ok(Some(LiveSession {
            view: live::view(facts.as_ref(), grew_at, now),
            facts: facts.unwrap_or_default(),
            cwd,
        }))
    }

    /// One session's live facts, its view at `now`, its peers and its
    /// processes (the harness first, then children by start), for
    /// `read --view live`; `None` for a session not in the archive.
    pub fn live_detail(&self, session: &str, now: i64) -> Result<Option<LiveDetail>> {
        use crate::query::fields::{Aliases, processes_tool_call, sessions_peers};
        let conn = self.conn()?;
        let aliases = Aliases::plain();
        let found = conn
            .prepare(&format!(
                "SELECT s.pk, s.live_facts, s.last_activity_at, {peers} FROM session s WHERE s.id = ?1",
                peers = sessions_peers(&aliases, now)
            ))?
            .query_row([session], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            })
            .optional()?;
        let Some((pk, facts, grew_at, peers)) = found else {
            return Ok(None);
        };
        let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
        let view = live::view(facts.as_ref(), grew_at, now);
        let mut statement = conn.prepare(&format!(
            "SELECT r.pid, r.parent_pid, r.start, r.comm, r.argv, r.cwd, r.started_at, r.ended_at,
                    r.role, {tool_call}
             FROM process r WHERE r.session_pk = ?1
             ORDER BY r.role <> 'harness', r.started_at, r.pid",
            tool_call = processes_tool_call(&aliases)
        ))?;
        let processes = statement
            .query_map([pk], |r| {
                Ok(ProcessRow {
                    pid: r.get(0)?,
                    parent_pid: r.get(1)?,
                    start: r.get(2)?,
                    comm: r.get(3)?,
                    argv: r
                        .get::<_, Option<String>>(4)?
                        .and_then(|a| serde_json::from_str(&a).ok())
                        .unwrap_or_default(),
                    cwd: r.get(5)?,
                    started_at: r.get(6)?,
                    ended_at: r.get(7)?,
                    role: r.get(8)?,
                    tool_call: r.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(Some(LiveDetail {
            view,
            facts: facts.unwrap_or_default(),
            peers: peers
                .map(|p| p.split(' ').map(str::to_owned).collect())
                .unwrap_or_default(),
            processes,
        }))
    }

    /// The index write counter this read sees, and the sessions whose own
    /// rows or live columns changed after write counter `after`: what a
    /// follower that last looked at `after` needs to look at again.
    pub fn sessions_changed_after(&self, after: i64) -> Result<(i64, BTreeSet<String>)> {
        let conn = self.conn()?;
        let now = conn.query_row(
            "SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM main.meta WHERE key='write_counter'),0)",
            [],
            |r| r.get(0),
        )?;
        let changed = conn
            .prepare_cached(
                "SELECT id FROM session WHERE changed > ?1
                 UNION SELECT id FROM session WHERE live_changed > ?1",
            )?
            .query_map([after], |r| r.get(0))?
            .collect::<rusqlite::Result<BTreeSet<String>>>()?;
        Ok((now, changed))
    }

    /// What `follow` shows of one session at `now`: its live view, its open
    /// tool, and the messages and tool calls stored after the given row
    /// keys, oldest first.
    pub fn follow_snapshot(
        &self,
        session: &str,
        now: i64,
        after_message: i64,
        after_tool_call: i64,
    ) -> Result<Option<FollowSnapshot>> {
        let conn = self.conn()?;
        let found = conn
            .prepare_cached("SELECT pk, live_facts, last_activity_at FROM session WHERE id = ?1")?
            .query_row([session], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                ))
            })
            .optional()?;
        let Some((pk, facts, grew_at)) = found else {
            return Ok(None);
        };
        let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
        let messages = conn
            .prepare_cached(
                "SELECT pk, id, role, at FROM message WHERE session_pk = ?1 AND pk > ?2 ORDER BY pk",
            )?
            .query_map(params![pk, after_message], |r| {
                Ok(FollowedMessage {
                    key: r.get(0)?,
                    id: r.get(1)?,
                    role: r.get(2)?,
                    at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let tool_calls = conn
            .prepare_cached(
                "SELECT pk, call_id, tool, at FROM tool_call WHERE session_pk = ?1 AND pk > ?2 ORDER BY pk",
            )?
            .query_map(params![pk, after_tool_call], |r| {
                Ok(FollowedCall {
                    key: r.get(0)?,
                    call_id: r.get(1)?,
                    tool: r.get(2)?,
                    at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(Some(FollowSnapshot {
            view: live::view(facts.as_ref(), grew_at, now),
            open_tool: facts.and_then(|f| f.open_call.map(|c| c.tool)),
            messages,
            tool_calls,
        }))
    }

    /// The sessions not ended with a footprint: each with its working
    /// directory and the files it changed and has not committed.
    pub fn live_footprints(&self, now: i64) -> Result<Vec<LiveFootprint>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, live_facts, last_activity_at, cwd FROM session
             WHERE live_last_at IS NOT NULL AND live_last_at >= ?1 AND cwd IS NOT NULL
               AND json_extract(live_facts, '$.footprint') IS NOT NULL",
        )?;
        let rows = statement.query_map(
            params![now - live::STALE_FOR_MS - live::LIVE_WINDOW_MS],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, String>(3)?,
                ))
            },
        )?;
        let mut out = Vec::new();
        for row in rows {
            let (session, facts, grew_at, cwd) = row?;
            let facts: LiveFacts = serde_json::from_str(&facts)?;
            // A stale session keeps its footprint until a commit or its end.
            if facts.footprint.is_empty()
                || live::view(Some(&facts), grew_at, now).state == live::LiveState::Ended
            {
                continue;
            }
            out.push(LiveFootprint {
                session,
                cwd,
                files: facts.footprint.into_iter().collect(),
            });
        }
        Ok(out)
    }

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

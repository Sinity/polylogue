//! Live facts and their history, as read. The live summary
//! (summary/live.rs) folds each session's live events (events of route
//! `hook`, `process`, `registry` or `git`) into its `live_facts`, the
//! intervals they closed (`live_interval`) and its processes (`process`);
//! the live state itself is computed when asked, from the facts and the
//! time (`pl_live`); nothing here looks at the clock.

use crate::ConnExt as _;
use crate::Result;
use polylogue_model::live::{self, LiveFacts};
use rusqlite::{OptionalExtension, params};
use std::collections::BTreeSet;

/// One session's live facts and view, with where it last worked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveSession {
    pub view: live::LiveView,
    pub facts: LiveFacts,
    pub cwd: Option<String>,
    /// When its transcript last grew, Unix ms.
    pub last_active_at: Option<i64>,
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
pub struct LiveEditedFiles {
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
    /// When it entered its state, as far as its live events say.
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
    pub edited_files: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// A session's live facts as read, before its state is computed.
#[derive(Debug, Clone)]
struct RunningCandidate {
    session: String,
    facts: Option<LiveFacts>,
    grew_at: Option<i64>,
    cwd: Option<String>,
}

/// The sessions [`crate::Reader::running_candidates`] read; their states
/// depend only on the time asked, so they can be computed again later
/// without reading the archive.
#[derive(Debug, Clone, Default)]
pub struct RunningCandidates(Vec<RunningCandidate>);

impl RunningCandidates {
    /// The running sessions at `now`: waiting first, then by how long each
    /// has been in its state.
    pub fn at(&self, now: i64) -> Vec<RunningSession> {
        let mut out = Vec::new();
        for candidate in &self.0 {
            let facts = candidate.facts.as_ref();
            let v = live::view(facts, candidate.grew_at, now);
            if !v.state.running() {
                continue;
            }
            out.push(RunningSession {
                session: candidate.session.clone(),
                state: v.state,
                since: v.since,
                waiting_on: v.waiting_on,
                stall: v.stall,
                open_tool: facts.and_then(|f| f.open_call.as_ref().map(|c| c.tool.clone())),
                task: facts.and_then(|f| f.task.clone()),
                edited_files: facts.map_or(0, |f| f.edited_files.len()),
                cwd: candidate.cwd.clone(),
            });
        }
        out.sort_by(|a, b| {
            (a.state.rank(), a.since.unwrap_or(i64::MAX), &a.session).cmp(&(
                b.state.rank(),
                b.since.unwrap_or(i64::MAX),
                &b.session,
            ))
        });
        out
    }
}

/// A session whose root process is alive by its live events: what the
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
    /// with no new live event, with that instant (`live_until`): those with
    /// live events within a day, and those whose transcript alone grew
    /// within the live window.
    pub fn live_untils(&self, now: i64) -> Result<Vec<(String, i64)>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, pl_live(live_facts, last_active_at, ?1, 'until') AS until FROM session
             WHERE (live_last_at IS NOT NULL AND live_last_at >= ?2)
                OR (live_last_at IS NULL AND last_active_at >= ?3)",
        )?;
        let rows = statement.query_map(
            params![
                now,
                now - live::STALE_FOR_MS - live::LIVE_WINDOW_MS,
                now - live::LIVE_WINDOW_MS - 1
            ],
            |r| <(String, Option<i64>)>::try_from(r),
        )?;
        let mut out = Vec::new();
        for row in rows {
            if let (session, Some(until)) = row? {
                out.push((session, until));
            }
        }
        Ok(out)
    }

    /// The sessions that may be running at `now`: those with live events or
    /// activity inside the live window, or an open root. Their states are
    /// computed at a later time by [`RunningCandidates::at`]; a session that
    /// enters the window later does so through a new commit.
    pub fn running_candidates(&self, now: i64) -> Result<RunningCandidates> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, live_facts, last_active_at, cwd FROM session
             WHERE (live_last_at IS NOT NULL AND live_last_at >= ?1)
                OR (live_last_at IS NULL AND last_active_at >= ?2)
                OR (json_extract(live_facts, '$.root') IS NOT NULL
                    AND json_extract(live_facts, '$.root_gone_at') IS NULL
                    AND json_extract(live_facts, '$.ended_at') IS NULL)",
        )?;
        let rows = statement.query_map(
            params![
                now - live::STALE_FOR_MS - live::LIVE_WINDOW_MS,
                now - live::LIVE_WINDOW_MS - 1
            ],
            |r| <(String, Option<String>, Option<i64>, Option<String>)>::try_from(r),
        )?;
        let mut sessions = Vec::new();
        for row in rows {
            let (session, facts, grew_at, cwd) = row?;
            let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
            sessions.push(RunningCandidate {
                session,
                facts,
                grew_at,
                cwd,
            });
        }
        Ok(RunningCandidates(sessions))
    }

    /// One session's live facts, its view at `now` and its last working
    /// directory; `None` for a session not in the archive.
    pub fn live_state(&self, session: &str, now: i64) -> Result<Option<LiveSession>> {
        let conn = self.conn()?;
        let found = conn
            .prepare_cached("SELECT live_facts, last_active_at, cwd FROM session WHERE id = ?1")?
            .query_row([session], |r| {
                <(Option<String>, Option<i64>, Option<String>)>::try_from(r)
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
            last_active_at: grew_at,
        }))
    }

    /// One session's live facts, its view at `now`, its peers and its
    /// processes (the harness first, then children by first sighting), for
    /// `read --view live`; `None` for a session not in the archive.
    pub fn live_detail(&self, session: &str, now: i64) -> Result<Option<LiveDetail>> {
        use crate::query::fields::{Aliases, processes_tool_call, sessions_peers};
        let conn = self.conn()?;
        let aliases = Aliases::plain();
        let found = conn
            .prepare(&format!(
                "SELECT s.pk, s.live_facts, s.last_active_at, {peers} FROM session s WHERE s.id = ?1",
                peers = sessions_peers(&aliases, now)
            ))?
            .query_row([session], |r| <(i64, Option<String>, Option<i64>, Option<String>)>::try_from(r))
            .optional()?;
        let Some((pk, facts, grew_at, peers)) = found else {
            return Ok(None);
        };
        let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
        let view = live::view(facts.as_ref(), grew_at, now);
        let mut statement = conn.prepare(&format!(
            "SELECT r.pid, r.parent_pid, r.comm, r.argv, r.cwd, r.started_at, r.ended_at,
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
                    comm: r.get(2)?,
                    argv: r
                        .get::<_, Option<String>>(3)?
                        .and_then(|a| serde_json::from_str(&a).ok())
                        .unwrap_or_default(),
                    cwd: r.get(4)?,
                    started_at: r.get(5)?,
                    ended_at: r.get(6)?,
                    role: r.get(7)?,
                    tool_call: r.get(8)?,
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
        let now = conn.scalar(
            "SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM main.meta WHERE key='write_counter'),0)",
            [])?;
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
    /// keys, oldest first. Without keys (the session is first looked at)
    /// only the latest message and tool call come back: the caller needs
    /// their keys, not the history.
    pub fn follow_snapshot(
        &self,
        session: &str,
        now: i64,
        after: Option<(i64, i64)>,
    ) -> Result<Option<FollowSnapshot>> {
        let conn = self.conn()?;
        let found = conn
            .prepare_cached("SELECT pk, live_facts, last_active_at FROM session WHERE id = ?1")?
            .query_row([session], |r| {
                <(i64, Option<String>, Option<i64>)>::try_from(r)
            })
            .optional()?;
        let Some((pk, facts, grew_at)) = found else {
            return Ok(None);
        };
        let facts: Option<LiveFacts> = facts.map(|f| serde_json::from_str(&f)).transpose()?;
        let (after_message, after_tool_call) = match after {
            Some(keys) => keys,
            None => {
                let before_latest = |table: &str| -> rusqlite::Result<i64> {
                    conn.prepare_cached(&format!(
                        "SELECT COALESCE(MAX(pk), 0) - 1 FROM {table} WHERE session_pk = ?1"
                    ))?
                    .query_row([pk], |r| r.get(0))
                };
                (before_latest("message")?, before_latest("tool_call")?)
            }
        };
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

    /// The sessions not ended with edited files: each with its working
    /// directory and the files it changed and has not committed.
    pub fn live_edited_files(&self, now: i64) -> Result<Vec<LiveEditedFiles>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, live_facts, last_active_at, cwd FROM session
             WHERE live_last_at IS NOT NULL AND live_last_at >= ?1 AND cwd IS NOT NULL
               AND json_extract(live_facts, '$.edited_files') IS NOT NULL",
        )?;
        let rows = statement.query_map(
            params![now - live::STALE_FOR_MS - live::LIVE_WINDOW_MS],
            |r| <(String, String, Option<i64>, String)>::try_from(r),
        )?;
        let mut out = Vec::new();
        for row in rows {
            let (session, facts, grew_at, cwd) = row?;
            let facts: LiveFacts = serde_json::from_str(&facts)?;
            // A stale session keeps its edited files until a commit or its end.
            if facts.edited_files.is_empty()
                || live::view(Some(&facts), grew_at, now).state == live::LiveState::Ended
            {
                continue;
            }
            out.push(LiveEditedFiles {
                session,
                cwd,
                files: facts.edited_files.into_iter().collect(),
            });
        }
        Ok(out)
    }

    /// Every session whose root process is alive by its live events.
    pub fn live_roots(&self) -> Result<Vec<LiveRoot>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT id, live_facts FROM session
             WHERE live_last_at IS NOT NULL AND live_facts IS NOT NULL
               AND json_extract(live_facts, '$.root') IS NOT NULL
               AND json_extract(live_facts, '$.root_gone_at') IS NULL
               AND json_extract(live_facts, '$.ended_at') IS NULL",
        )?;
        let rows = statement.query_map([], |r| <(String, String)>::try_from(r))?;
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

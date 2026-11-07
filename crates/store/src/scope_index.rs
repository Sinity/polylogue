//! The scope index: every main session that has its transcript, by the
//! working directories it recorded, with its repository, activity, stored
//! message count and live facts. The daemon holds one in memory, so a brief
//! or a resume list finds the sessions under a directory, the repositories
//! recorded there, the sessions working there now and the Beads they hold
//! without reading the index, at any directory, `/` included.
//!
//! It is read in full once ([`Reader::scope_changes`] with no position) and
//! then follows the change feed: sessions whose rows or live columns changed
//! after its position, and sessions removed after it. Another index
//! generation is read in full again. Nothing here looks at the clock: live
//! states are computed for the `now` a caller passes.

use crate::read::ts;
use crate::{Reader, Result};
use polylogue_model::{ChangePosition, SessionId, SessionKind, Timestamp};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A session that worked in or below a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NearbySession {
    pub id: SessionId,
    pub kind: SessionKind,
    pub title: Option<String>,
    pub started: Option<Timestamp>,
    pub last_activity: Option<Timestamp>,
    /// Its shown messages as the derive step last counted them; unknown
    /// until the derive step has counted them.
    pub messages: Option<u64>,
    pub cwd: Option<String>,
}

/// One session as the scope index holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeSession {
    pub pk: i64,
    pub session: NearbySession,
    /// Every working directory it recorded, its last one included.
    pub directories: Vec<String>,
    pub repo: Option<String>,
    live: polylogue_model::live::LiveFacts,
}

impl ScopeSession {
    /// When its latest hook event happened, Unix ms.
    pub fn last_event_at(&self) -> Option<i64> {
        self.live.last_at
    }

    /// What it is waiting on: permission, question or notification.
    pub fn waiting_on(&self) -> Option<&str> {
        self.live
            .hook_waiting_on
            .as_ref()
            .map(|waiting| waiting.value.as_str())
    }

    /// The wait the common live model shows at this instant, including
    /// harness status and process evidence that outrank an earlier hook.
    pub fn waiting_on_at(&self, now: i64) -> Option<&'static str> {
        polylogue_model::live::view(
            Some(&self.live),
            self.session.last_activity.map(|at| at.unix_ms()),
            now,
        )
        .waiting_on
        .map(|waiting| waiting.as_str())
    }

    /// A tool call started and has not finished.
    pub fn open_tool_call(&self) -> bool {
        self.live.open_call.is_some()
    }

    /// Files structured edit tools changed and no commit has closed.
    pub fn footprint(&self) -> usize {
        self.live.footprint.len()
    }

    /// The harness's in-progress todo item.
    pub fn todo(&self) -> Option<&str> {
        self.live.todo.as_deref()
    }

    /// The Bead or job its latest open claim holds.
    pub fn task(&self) -> Option<&str> {
        self.live.task.as_deref()
    }
}

/// A session working, waiting or idle now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopedLiveSession<'a> {
    pub session: &'a ScopeSession,
    /// `waiting`, `working` or `idle`.
    pub state: &'static str,
}

/// A Bead whose latest claim under a directory is still open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenBead {
    pub id: String,
    pub session: String,
    pub since: Option<Timestamp>,
}

/// Session rows read from the index for the scope index: all of them, or
/// those that changed after a position.
#[derive(Debug)]
pub struct ScopeChanges {
    position: ChangePosition,
    /// The rows are every session: the index replaces what it holds.
    full: bool,
    rows: Vec<ScopeRow>,
    /// Sessions removed after the position, by ID.
    removed: Vec<String>,
}

#[derive(Debug)]
struct ScopeRow {
    pk: i64,
    id: String,
    /// A main session with its transcript; other rows leave the index.
    held: Option<ScopeSession>,
}

/// The scope index itself. See the module documentation.
#[derive(Debug, Default)]
pub struct ScopeIndex {
    position: Option<ChangePosition>,
    sessions: HashMap<i64, ScopeSession>,
    by_id: HashMap<String, i64>,
    /// Every recorded working directory, with the sessions that recorded it.
    by_directory: BTreeMap<String, BTreeSet<i64>>,
}

const COLUMNS: &str = "pk, id, kind, stub, coalesce(title, derived_title), started_at,
    last_activity_at, cwd, cwd_json, repo, message_count, live_facts";

fn scope_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ScopeRow> {
    let pk: i64 = r.get(0)?;
    let id: String = r.get(1)?;
    let kind: String = r.get(2)?;
    let stub: bool = r.get(3)?;
    let parsed = SessionId::parse(&id).ok();
    let held = match parsed {
        Some(session_id) if kind == "main" && !stub => {
            let cwd: Option<String> = r.get(7)?;
            let mut directories: Vec<String> = r
                .get::<_, Option<String>>(8)?
                .and_then(|json| serde_json::from_str::<Vec<String>>(&json).ok())
                .unwrap_or_default();
            if let Some(cwd) = &cwd {
                directories.push(cwd.clone());
            }
            directories.sort();
            directories.dedup();
            Some(ScopeSession {
                pk,
                session: NearbySession {
                    id: session_id,
                    kind: SessionKind::Main,
                    title: r.get(4)?,
                    started: ts(r.get(5)?),
                    last_activity: ts(r.get(6)?),
                    messages: r
                        .get::<_, Option<i64>>(10)?
                        .and_then(|n| u64::try_from(n).ok()),
                    cwd,
                },
                directories,
                repo: r.get(9)?,
                live: r
                    .get::<_, Option<String>>(11)?
                    .map(|json| {
                        serde_json::from_str(&json).map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                11,
                                rusqlite::types::Type::Text,
                                Box::new(error),
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or_default(),
            })
        }
        _ => None,
    };
    Ok(ScopeRow { pk, id, held })
}

impl Reader<'_> {
    /// The session rows the scope index needs after `after`: every session
    /// when `after` is `None` or names another index generation, otherwise
    /// the sessions whose rows or live columns changed after it and the
    /// sessions removed after it.
    pub fn scope_changes(&self, after: Option<&ChangePosition>) -> Result<ScopeChanges> {
        let position = self.change_position()?;
        let after = after
            .filter(|after| after.generation == position.generation)
            .map(|after| after.index);
        let conn = self.conn()?;
        let mut rows = Vec::new();
        let mut removed = Vec::new();
        match after {
            None => {
                let mut statement = conn.prepare(&format!("SELECT {COLUMNS} FROM session"))?;
                let mut found = statement.query([])?;
                while let Some(row) = found.next()? {
                    rows.push(scope_row(row)?);
                }
            }
            Some(after) => {
                for changed in ["changed", "live_changed"] {
                    let mut statement = conn.prepare_cached(&format!(
                        "SELECT {COLUMNS} FROM session WHERE {changed} > ?1"
                    ))?;
                    let mut found = statement.query([after])?;
                    while let Some(row) = found.next()? {
                        rows.push(scope_row(row)?);
                    }
                }
                let mut statement = conn
                    .prepare_cached("SELECT session_id FROM removed_session WHERE changed > ?1")?;
                for id in statement.query_map([after], |r| r.get::<_, String>(0))? {
                    removed.push(id?);
                }
            }
        }
        self.check()?;
        Ok(ScopeChanges {
            position,
            full: after.is_none(),
            rows,
            removed,
        })
    }
}

/// Whether `path` is `dir` or below it, by whole path components.
fn below(path: &str, dir: &str) -> bool {
    let base = dir.trim_end_matches('/');
    path == dir
        || path == base
        || (path.len() > base.len()
            && path.starts_with(base)
            && path.as_bytes()[base.len()] == b'/')
}

impl ScopeIndex {
    /// The index of `changes`, which must be a full read.
    pub fn new(changes: ScopeChanges) -> Self {
        let mut index = Self::default();
        index.apply(changes);
        index
    }

    /// Where it has followed the change feed to.
    pub fn position(&self) -> Option<&ChangePosition> {
        self.position.as_ref()
    }

    /// Whether it already follows the change feed past what `changes` read,
    /// in the same index generation.
    pub fn is_newer_than(&self, changes: &ScopeChanges) -> bool {
        self.position.as_ref().is_some_and(|held| {
            held.generation == changes.position.generation && held.index > changes.position.index
        })
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Takes in what changed. A full read replaces everything held.
    pub fn apply(&mut self, changes: ScopeChanges) {
        if changes.full {
            self.sessions.clear();
            self.by_id.clear();
            self.by_directory.clear();
        }
        for id in &changes.removed {
            if let Some(pk) = self.by_id.get(id).copied() {
                self.remove(pk);
            }
        }
        for row in changes.rows {
            self.remove(row.pk);
            if let Some(pk) = self.by_id.get(&row.id).copied() {
                self.remove(pk);
            }
            if let Some(held) = row.held {
                for directory in &held.directories {
                    self.by_directory
                        .entry(directory.clone())
                        .or_default()
                        .insert(held.pk);
                }
                self.by_id.insert(row.id, held.pk);
                self.sessions.insert(held.pk, held);
            }
        }
        self.position = Some(changes.position);
    }

    fn remove(&mut self, pk: i64) {
        let Some(old) = self.sessions.remove(&pk) else {
            return;
        };
        self.by_id.remove(&old.session.id.to_string());
        for directory in &old.directories {
            if let Some(pks) = self.by_directory.get_mut(directory) {
                pks.remove(&pk);
                if pks.is_empty() {
                    self.by_directory.remove(directory);
                }
            }
        }
    }

    /// Sessions that recorded a working directory at or below one of `dirs`.
    fn recorded_under(&self, dirs: &[String]) -> BTreeSet<i64> {
        let mut out = BTreeSet::new();
        for dir in dirs {
            let base = dir.trim_end_matches('/');
            for exact in [dir.as_str(), base] {
                if let Some(pks) = self.by_directory.get(exact) {
                    out.extend(pks);
                }
            }
            // Every path that starts with `base/`: '0' follows '/'.
            for (_, pks) in self
                .by_directory
                .range(format!("{base}/")..format!("{base}0"))
            {
                out.extend(pks);
            }
        }
        out
    }

    /// Sessions whose last working directory is one of `dirs` or below one,
    /// newest activity first (unknown activity last), leaving out `exclude`.
    pub fn sessions_under(&self, dirs: &[String], exclude: Option<&str>) -> Vec<&ScopeSession> {
        let mut out: Vec<&ScopeSession> = self
            .recorded_under(dirs)
            .into_iter()
            .filter_map(|pk| self.sessions.get(&pk))
            .filter(|s| {
                s.session
                    .cwd
                    .as_deref()
                    .is_some_and(|cwd| dirs.iter().any(|dir| below(cwd, dir)))
                    && exclude.is_none_or(|exclude| s.session.id.to_string() != exclude)
            })
            .collect();
        out.sort_by(|a, b| {
            b.session
                .last_activity
                .cmp(&a.session.last_activity)
                .then_with(|| a.session.id.cmp(&b.session.id))
        });
        out
    }

    /// The repositories recorded for sessions in these directories, the
    /// most recently active first.
    pub fn repos_under(&self, dirs: &[String]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for s in self.sessions_under(dirs, None) {
            if let Some(repo) = &s.repo
                && !out.contains(repo)
            {
                out.push(repo.clone());
            }
        }
        out
    }

    /// Sessions working, waiting or idle at `now` (Unix ms) that recorded a
    /// working directory under `dirs`, leaving out `exclude`: waiting first,
    /// then working, then idle, each by latest hook event.
    pub fn live_under(
        &self,
        dirs: &[String],
        exclude: Option<&str>,
        now: i64,
    ) -> Vec<ScopedLiveSession<'_>> {
        let mut out: Vec<ScopedLiveSession<'_>> = self
            .recorded_under(dirs)
            .into_iter()
            .filter_map(|pk| self.sessions.get(&pk))
            .filter(|s| exclude.is_none_or(|exclude| s.session.id.to_string() != exclude))
            .filter_map(|s| {
                let view = polylogue_model::live::view(
                    Some(&s.live),
                    s.session.last_activity.map(|at| at.unix_ms()),
                    now,
                );
                view.state.active().then_some(ScopedLiveSession {
                    session: s,
                    state: view.state.as_str(),
                })
            })
            .collect();
        let rank = |state: &str| match state {
            "waiting" => 0,
            "working" => 1,
            _ => 2,
        };
        out.sort_by(|a, b| {
            rank(a.state)
                .cmp(&rank(b.state))
                .then(b.session.live.last_at.cmp(&a.session.live.last_at))
                .then_with(|| a.session.session.id.cmp(&b.session.session.id))
        });
        out
    }

    /// Last-observed task claims under `dirs`, with no observed close or
    /// release, most recently active first, each once. Age or an unknown
    /// activity time never proves that a claim ended. External state is unknown.
    pub fn open_beads(&self, dirs: &[String]) -> Vec<OpenBead> {
        let mut out: Vec<OpenBead> = Vec::new();
        let mut seen = BTreeSet::new();
        for s in self.sessions_under(dirs, None) {
            let Some(task) = &s.live.task else {
                continue;
            };
            if seen.insert(task.as_str()) {
                out.push(OpenBead {
                    id: task.clone(),
                    session: s.session.id.to_string(),
                    since: s.session.last_activity,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests;

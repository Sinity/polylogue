//! Resuming a session. Two ways are
//! kept apart:
//!
//! - **native resume**: the harness's own command reopens the session with
//!   its full history (`claude --resume <id>`, `codex resume <id>`);
//! - **context-assisted continuation**: a new session starts with a
//!   takeover brief Polylogue prepares (`polylogue takeover <id>`), for
//!   harnesses without native resume or a session whose file is gone.
//!
//! Which sessions are worth resuming comes from what they left open, not
//! from how their text ended.

use polylogue_model::SessionId;
use polylogue_store::{OpenItems, Reader, StoreError};
use serde::Serialize;
use std::path::Path;

/// A command line to run in a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Command {
    pub program: String,
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// The harness's own resume command for a session, when it has one.
pub fn native_resume(session: &SessionId, cwd: Option<&str>) -> Option<Command> {
    let resume = polylogue_sources::origin::by_name(session.origin().as_str())?.resume?;
    let mut argv = resume(session.native().as_str())?.into_iter();
    Some(Command {
        program: argv.next()?,
        args: argv.collect(),
        cwd: cwd.map(str::to_owned),
    })
}

/// How to continue a session: its native resume command if any, and the
/// context-assisted way, which always exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub session: SessionId,
    pub native: Option<Command>,
    pub context_assisted: Command,
}

pub fn plan(reader: &Reader<'_>, session: &SessionId) -> Result<Plan, StoreError> {
    let s = reader
        .session(session)?
        .ok_or_else(|| StoreError::NotFound(format!("session {session}")))?;
    let cwd = s.cwd.last().cloned();
    Ok(Plan {
        session: session.clone(),
        native: native_resume(session, cwd.as_deref()),
        context_assisted: Command {
            program: "polylogue".into(),
            args: vec!["takeover".into(), session.to_string()],
            cwd,
        },
    })
}

/// A session worth resuming and why.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResumableSession {
    pub session: polylogue_store::NearbySession,
    pub open: OpenItems,
    /// Operator notes on it that ask for follow-up (handoffs, blockers).
    pub follow_ups: Vec<String>,
    pub native: Option<Command>,
}

/// Sessions under `cwd`, newest first, that left something open or carry
/// a handoff or blocker; sessions with nothing open are left out.
pub fn resumable(
    reader: &Reader<'_>,
    cwd: &Path,
    limit: u64,
    now: polylogue_model::Timestamp,
) -> Result<Vec<ResumableSession>, StoreError> {
    use crate::daemon::query::notes;
    use polylogue_model::{NoteKind, Target};
    let dirs = crate::briefing::scope_dirs(cwd);
    let mut out = Vec::new();
    // Look further back than `limit`: most sessions finish cleanly.
    for s in reader.sessions_under(&dirs, None, limit.saturating_mul(10).max(50))? {
        let id = s.id.to_string();
        let open = reader.open_items(&id)?.unwrap_or_default();
        let conditions = format!(
            "{} (kind:{} OR kind:{}) agent:false",
            notes::about(&[Target::Session(s.id.clone())]),
            NoteKind::Handoff.as_str(),
            NoteKind::Blocker.as_str(),
        );
        let follow_ups: Vec<String> = notes::list(reader, &conditions, now)?
            .into_iter()
            .map(|n| format!("{}: {}", n.kind.as_str(), n.value.unwrap_or_default()))
            .collect();
        if !open.any() && follow_ups.is_empty() {
            continue;
        }
        let native = native_resume(&s.id, s.cwd.as_deref());
        out.push(ResumableSession {
            session: s,
            open,
            follow_ups,
            native,
        });
        if out.len() as u64 >= limit {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;

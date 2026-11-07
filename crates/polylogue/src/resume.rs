//! Resuming a session. Two ways are
//! kept apart:
//!
//! - **harness resume**: the harness's own command reopens the session with
//!   its full history (`claude --resume <id>`, `codex resume <id>`);
//! - **context-assisted continuation**: a new session starts with a
//!   takeover brief Polylogue prepares (`polylogue takeover <id>`), for
//!   harnesses without harness resume or a session whose file is gone.
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
pub fn harness_resume(session: &SessionId, cwd: Option<&str>) -> Option<Command> {
    let resume = polylogue_sources::origin::by_name(session.origin().as_str())?.resume?;
    let mut argv = resume(session.native().as_str())?.into_iter();
    Some(Command {
        program: argv.next()?,
        args: argv.collect(),
        cwd: cwd.map(str::to_owned),
    })
}

/// How to continue a session: its harness resume command if any, and the
/// context-assisted way, which always exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub session: SessionId,
    pub harness_command: Option<Command>,
    pub context_assisted: Command,
}

pub fn plan(reader: &Reader<'_>, session: &SessionId) -> Result<Plan, StoreError> {
    let s = reader
        .session(session)?
        .ok_or_else(|| StoreError::NotFound(format!("session {session}")))?;
    let cwd = s.cwd.last().cloned();
    Ok(Plan {
        session: session.clone(),
        harness_command: harness_resume(session, cwd.as_deref()),
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
    /// Operator notes on it that ask for follow-up (handoffs, problems).
    pub follow_ups: Vec<String>,
    pub harness_command: Option<Command>,
}

/// Sessions under `cwd`, newest first, that left something open or carry
/// a handoff or problem; sessions with nothing open are left out. Which
/// sessions are under `cwd` comes from the daemon's scope index.
pub fn resumable(
    reader: &Reader<'_>,
    scope: &crate::daemon::scope::SharedScopeIndex,
    cwd: &Path,
    limit: u64,
    context: &std::sync::Arc<crate::operation::Context>,
) -> Result<Vec<ResumableSession>, StoreError> {
    use crate::daemon::query::assertions;
    use polylogue_model::{Reference, assertion::AssertionKind};
    let dirs = crate::briefing::scope_dirs(cwd);
    // Look further back than `limit`: most sessions finish cleanly.
    let looked = usize::try_from(limit.saturating_mul(10).max(50)).unwrap_or(usize::MAX);
    let stop = polylogue_store::read::limit::ReadLimit::current().and_then(|l| l.deadline());
    let candidates: Vec<polylogue_store::NearbySession> = scope
        .read(stop, |index| {
            index
                .sessions_under(&dirs, None)
                .into_iter()
                .take(looked)
                .map(|s| s.session.clone())
                .collect()
        })
        .ok_or(StoreError::ReadDeadlineExceeded)?;
    let mut out = Vec::new();
    for s in candidates {
        let id = s.id.to_string();
        let open = reader.open_items(&id)?.unwrap_or_default();
        let conditions = format!(
            "{} (kind:{} OR kind:{}) author:operator",
            assertions::about(&[Reference::Session(s.id.clone())]),
            AssertionKind::Handoff.as_str(),
            AssertionKind::Problem.as_str(),
        );
        let follow_ups: Vec<String> = assertions::list(reader, &conditions, context)?
            .into_iter()
            .map(|n| {
                format!(
                    "{}: {}",
                    n.fact.kind.as_str(),
                    n.fact.text().unwrap_or_default()
                )
            })
            .collect();
        if !open.any() && follow_ups.is_empty() {
            continue;
        }
        let harness_command = harness_resume(&s.id, s.cwd.as_deref());
        out.push(ResumableSession {
            session: s,
            open,
            follow_ups,
            harness_command,
        });
        if out.len() as u64 >= limit {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;

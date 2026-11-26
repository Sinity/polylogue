//! Resuming a session. Two ways are
//! kept apart:
//!
//! - **harness resume**: the harness's own command reopens the session with
//!   its full history (`claude --resume <id>`, `codex resume <id>`);
//! - **context-assisted continuation**: a new session starts with a
//!   session brief Polylogue prepares (`polylogue session:<id> brief`), for
//!   harnesses without harness resume or a session whose file is gone.
//!
//! Which sessions are worth resuming comes from what they left open, not
//! from how their text ended.

use polylogue_model::{Reference, SessionId};
use polylogue_store::{OpenItems, Reader, StoreError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A command line to run in a directory. A harness resume runs in the
/// session's resume directory ([`Reader::resume_directory`]). `cwd` is `None`
/// when none was recorded: a preview says so instead of the line, and
/// running it is refused, never done in the caller's directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub session: SessionId,
    pub harness_command: Option<Command>,
    pub context_assisted: Command,
}

pub fn plan(reader: &Reader<'_>, session: &SessionId) -> Result<Plan, StoreError> {
    reader
        .session(session)?
        .ok_or_else(|| StoreError::NotFound(format!("session {session}")))?;
    let cwd = reader.resume_directory(session)?;
    Ok(Plan {
        session: session.clone(),
        harness_command: harness_resume(session, cwd.as_deref()),
        context_assisted: Command {
            program: "polylogue".into(),
            args: vec![
                Reference::Session(session.clone()).to_string(),
                "brief".into(),
            ],
            // Reading an archived session does not require its old directory.
            cwd: None,
        },
    })
}

/// A session worth resuming and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResumableSession {
    pub session: polylogue_store::NearbySession,
    pub open: OpenItems,
    /// Operator notes on it that ask for follow-up (handoffs, problems).
    pub follow_ups: Vec<String>,
}

/// The shipped (or authored replacement) unfinished question supplies the
/// candidates. Resume adds its main-session requirement and existing details.
pub fn resumable(
    reader: &Reader<'_>,
    cwd: &Path,
    limit: u64,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Vec<ResumableSession>, crate::daemon::Failure> {
    use crate::daemon::query::annotations;
    use polylogue_model::{Reference, annotation::AnnotationKind};
    let text = "query:unfinished | where kind:main | fields id, kind, title, started, last_active as last_activity, messages, cwd";
    let ask = crate::daemon::query::Ask {
        text,
        cursor: None,
        page_size: Some(limit),
        why: false,
        session: None,
        context: std::sync::Arc::clone(context),
        embeddings: false,
        ranking_identity: None,
        near: None,
        fields: &[],
    };
    let mut options = crate::daemon::query::options(reader, &ask, 0)?;
    if context.working_directory() != Some(cwd) {
        options.here = Some(crate::daemon::here::scope(cwd, reader)?);
    }
    let prepared = reader.prepare_query_selection(text, &options)?;
    if prepared.program().unit != polylogue_query::ast::Unit::Sessions
        || prepared.program().shape != polylogue_query::Shape::Rows
    {
        return Err(crate::daemon::Failure::bad_request(
            "unfinished must select sessions to resume",
        ));
    }
    let mut candidates = reader.query_rows(prepared.program())?;
    for row in &mut candidates {
        polylogue_store::query::execute::present(prepared.program(), row)?;
    }
    let mut out = Vec::new();
    for row in candidates {
        let s: polylogue_store::NearbySession =
            serde_json::from_value(row).map_err(crate::daemon::Failure::internal)?;
        let id = s.id.to_string();
        let open = reader.open_items(&id)?.unwrap_or_default();
        let conditions = format!(
            "{} (kind:{} OR kind:{}) author:operator",
            annotations::about(&[Reference::Session(s.id.clone())]),
            AnnotationKind::Handoff.as_str(),
            AnnotationKind::Problem.as_str(),
        );
        let follow_ups: Vec<String> = annotations::list(reader, &conditions, context)?
            .into_iter()
            .map(|n| {
                format!(
                    "{}: {}",
                    n.fact.kind.as_str(),
                    n.fact.text().unwrap_or_default()
                )
            })
            .collect();
        out.push(ResumableSession {
            session: s,
            open,
            follow_ups,
        });
        if out.len() as u64 >= limit {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;

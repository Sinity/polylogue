//! The `follow` stream: lines for the sessions a query selects, as their
//! live state changes and their messages and tool calls arrive. It wakes on
//! each committed index change (the live clock commits transitions that
//! come from time alone), reads the selection again, and ends before the
//! request's deadline with the marks the client sends back.
use super::{Daemon, Failure, accepting};
use crate::protocol::follow::{Line, Marks, step};
use polylogue_model::SessionId;
use std::sync::Arc;
use std::time::Duration;

/// The longest wait between two readings with no commit in between.
const IDLE_WAIT: Duration = Duration::from_secs(10);
/// Time kept before the deadline to send the marks.
const MARKS_MARGIN: Duration = Duration::from_millis(500);

/// Produces the follow stream, or leaves another request to its operation.
pub(crate) fn produce(
    request: &crate::daemon::Request,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    output: &mut dyn std::io::Write,
) -> Option<Result<(), Failure>> {
    let crate::daemon::Request::Follow { query, marks } = request else {
        return None;
    };
    Some(accepting(d).and_then(|()| run(d, operation, query, marks.clone(), output)))
}

fn run(
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    query: &str,
    marks: Option<Marks>,
    output: &mut dyn std::io::Write,
) -> Result<(), Failure> {
    let commits = d.writes.as_ref().map(|w| w.completion_commits());
    let limit = polylogue_store::read::limit::ReadLimit::current();
    let mut marks = marks.unwrap_or_default();
    loop {
        accepting(d)?;
        let now = d.config.requests.now().unix_ms();
        {
            let reader = d.reads.reader()?;
            for session in selected(&reader, operation, query, now)? {
                let mark = marks.sessions.get(&session);
                let Some(snapshot) = reader.follow_snapshot(
                    &session,
                    now,
                    mark.map_or(0, |m| m.message),
                    mark.map_or(0, |m| m.tool_call),
                )?
                else {
                    continue;
                };
                let (lines, mark) = step(&session, mark, &snapshot, now);
                for line in &lines {
                    crate::protocol::stream::event(output, "record", line)
                        .map_err(|e| Failure::new("io", e))?;
                }
                marks.sessions.insert(session, mark);
            }
        }
        let left = limit
            .as_ref()
            .and_then(polylogue_store::read::limit::ReadLimit::remaining)
            .map(|left| left.saturating_sub(MARKS_MARGIN));
        if left.is_some_and(|left| left.is_zero()) {
            return crate::protocol::stream::event(output, "record", &Line::Marks { marks })
                .map_err(|e| Failure::new("io", e));
        }
        let wait = left.map_or(IDLE_WAIT, |left| left.min(IDLE_WAIT));
        match &commits {
            Some(commits) => {
                let _ = commits.recv_timeout(wait);
            }
            None => std::thread::sleep(wait),
        }
    }
}

/// The sessions the follow selects now: a session reference names one;
/// any other text is a sessions query, read again at every step so
/// sessions that start matching are followed too.
pub(crate) fn selected(
    reader: &polylogue_store::Reader<'_>,
    operation: &Arc<crate::operation::Context>,
    query: &str,
    now: i64,
) -> Result<Vec<String>, Failure> {
    if let Ok(session) = SessionId::parse(query.trim()) {
        return Ok(vec![session.to_string()]);
    }
    let fields = ["id".to_owned()];
    let ask = crate::daemon::query::Ask {
        text: query,
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        context: Arc::clone(operation),
        embeddings: false,
        ranking_identity: None,
        near: None,
        fields: &fields,
    };
    let mut options = crate::daemon::query::options(reader, &ask, 0)?;
    options.clock = query_language::time::Clock::new(
        jiff::Timestamp::from_millisecond(now).map_err(|e| Failure::new("internal", e))?,
        options.clock.day.clone(),
    );
    let program = polylogue_store::query::compile(query, &options)?;
    if program.unit != polylogue_query::ast::Unit::Sessions {
        return Err(Failure::new(
            "bad_request",
            "follow takes a session reference or a sessions query",
        ));
    }
    let mut out = Vec::new();
    reader.visit_query_values(&program, None, |row| -> Result<_, Failure> {
        if let Some(id) = row.get("id").and_then(serde_json::Value::as_str) {
            out.push(id.to_owned());
        }
        Ok(std::ops::ControlFlow::Continue(()))
    })?;
    Ok(out)
}

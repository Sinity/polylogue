//! The `follow` stream: lines for the sessions a query selects, as their
//! live state changes and their messages and tool calls arrive. It wakes on
//! each committed index change (the live clock commits transitions that
//! come from time alone), reads the selection again and looks again only at
//! the selected sessions whose rows or live state changed since its last
//! look, and ends before the request's deadline with the marks the client
//! sends back.
use super::{Daemon, Failure, accepting};
use crate::protocol::follow::{Line, Marks, step};
use polylogue_model::JsonExt as _;
use polylogue_model::SessionId;
use std::sync::Arc;
use std::time::Duration;

/// Time kept before the deadline to send the marks.
pub(super) const REPLY_MARGIN: Duration = Duration::from_millis(500);

pub(crate) fn run(
    d: &Daemon,
    operation: &Arc<polylogue_reading::operation::Context>,
    query: &str,
    marks: Option<Marks>,
    output: &mut dyn std::io::Write,
) -> Result<Marks, Failure> {
    let commits = d.writes.as_ref().map(|w| w.completion_commits());
    let limit = polylogue_store::read::limit::ReadLimit::current();
    // A cancelled follow (its client gone, or the daemon stopping) stops
    // waiting for the next commit at once.
    let _wake = match (&commits, &limit) {
        (Some(commits), Some(limit)) => {
            let wake = commits.wake_handle();
            Some(limit.cancellation.on_cancel(move || {
                let _ = wake.try_send(());
            }))
        }
        _ => None,
    };
    let mut marks = marks.unwrap_or_default();
    // The index write counter of the last look; before the first look every
    // selected session is looked at.
    let mut looked: Option<i64> = None;
    loop {
        accepting(d)?;
        if let Some(limit) = &limit {
            limit.check()?;
        }
        let now = d.config.requests.now().unix_ms();
        {
            let reader = d.reads.reader()?;
            let (counter, changed) = reader.sessions_changed_after(looked.unwrap_or(i64::MAX))?;
            let selected = selected(&reader, operation, query, now)?
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
            marks
                .sessions
                .retain(|session, _| selected.contains(session));
            for session in selected {
                let mark = marks.sessions.get(&session);
                if looked.is_some() && mark.is_some() && !changed.contains(&session) {
                    continue;
                }
                let Some(snapshot) = reader.follow_snapshot(
                    &session,
                    now,
                    mark.map(|m| (m.message, m.tool_call)),
                )?
                else {
                    continue;
                };
                let (lines, mark) = step(&session, mark, &snapshot, now);
                for line in &lines {
                    crate::protocol::stream::event(output, "record", line).map_err(Failure::io)?;
                }
                marks.sessions.insert(session, mark);
            }
            looked = Some(counter);
        }
        // Marks finish a batch before the client decides whether all sessions settled.
        crate::protocol::stream::event(
            output,
            "record",
            &Line::Marks {
                marks: marks.clone(),
            },
        )
        .map_err(Failure::io)?;
        let left = limit
            .as_ref()
            .and_then(polylogue_store::read::limit::ReadLimit::remaining)
            .map(|left| left.saturating_sub(REPLY_MARGIN));
        if left.is_some_and(|left| left.is_zero()) {
            return Ok(marks);
        }
        if commits.is_none() && left.is_none() {
            return Ok(marks);
        }
        match (&commits, left) {
            (Some(commits), Some(left)) => {
                let _ = commits.recv_timeout(left);
            }
            (Some(commits), None) => {
                let _ = commits.recv();
            }
            // A daemon without a writer commits nothing: only the deadline
            // can end the follow.
            (None, Some(left)) => std::thread::sleep(left),
            (None, None) => unreachable!("returned above"),
        }
    }
}

/// The sessions the follow selects now: a session reference names one;
/// any other text is a sessions query, read again at every step so
/// sessions that start matching are followed too.
pub(crate) fn selected(
    reader: &polylogue_store::Reader<'_>,
    operation: &Arc<polylogue_reading::operation::Context>,
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
        jiff::Timestamp::from_millisecond(now).map_err(Failure::internal)?,
        options.clock.day.clone(),
    );
    let program = polylogue_store::query::compile(query, &options)?;
    if program.unit != polylogue_query::ast::Unit::Sessions {
        return Err(Failure::bad_request(
            "read --follow takes a session reference or a sessions query",
        ));
    }
    let mut out = Vec::new();
    reader.visit_query_values(&program, None, |row| -> Result<_, Failure> {
        if let Some(id) = row.str("id") {
            out.push(id.to_owned());
        }
        Ok(std::ops::ControlFlow::Continue(()))
    })?;
    Ok(out)
}

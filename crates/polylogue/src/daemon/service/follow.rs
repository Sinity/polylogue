//! The `follow` stream: lines for the sessions a query selects, as their
//! live state changes and their messages and tool calls arrive. It wakes on
//! each committed index change (the live clock commits transitions that
//! come from time alone), reads the selection again and looks again only at
//! the selected sessions whose rows or live state changed since its last
//! look. A deadline that passes ends the request with a timeout.
use super::{Failure, accepting};
use crate::daemon::handler::Call;
use crate::protocol::follow::{Line, Marks, step};
use polylogue_model::JsonExt as _;
use polylogue_model::SessionId;
use std::sync::Arc;
use std::time::Duration;

/// Time kept before the deadline to send a waiting listing's reply.
pub(super) const REPLY_MARGIN: Duration = Duration::from_millis(500);

pub(crate) fn run(
    call: &Call<'_>,
    operation: &Arc<polylogue_reading::operation::Context>,
    query: &str,
    marks: Option<Marks>,
    output: &mut dyn std::io::Write,
) -> Result<Marks, Failure> {
    let d = call.daemon;
    let interests = polylogue_store::completion::Committed {
        content: true,
        authored: true,
        query_records: false,
    };
    let commits = d
        .writes
        .as_ref()
        .map(|w| w.completion_commits_with_wake(interests, || {}));
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
            let reader = call.reader()?;
            let (counter, changed) = reader.sessions_changed_after(looked.unwrap_or(i64::MAX))?;
            let selected = select(
                &reader,
                operation,
                query,
                now,
                looked.is_none().then_some(call),
            )?
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
            .and_then(polylogue_store::read::limit::ReadLimit::remaining);
        if left.is_some_and(|left| left.is_zero()) {
            return Err(Failure::store(
                &polylogue_store::StoreError::ReadDeadlineExceeded,
            ));
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
    select(reader, operation, query, now, None)
}

fn select(
    reader: &polylogue_store::Reader<'_>,
    operation: &Arc<polylogue_reading::operation::Context>,
    query: &str,
    now: i64,
    history: Option<&Call<'_>>,
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
    let prepared = reader.prepare_query_selection(query, &options)?;
    if prepared.program().unit != polylogue_query::ast::Unit::Sessions {
        return Err(Failure::bad_request(
            "read --follow takes a session reference or a sessions query",
        ));
    }
    let execute = || {
        let mut out = Vec::new();
        reader.visit_query_values(prepared.program(), None, |row| -> Result<_, Failure> {
            if let Some(id) = row.str("id") {
                out.push(id.to_owned());
            }
            Ok(std::ops::ControlFlow::Continue(()))
        })?;
        Ok(polylogue_reading::execution::Completed {
            count: Some(out.len() as u64),
            exact: prepared.program().incomplete.is_empty(),
            value: out,
            next: None,
        })
    };
    let Some(call) = history else {
        return execute().map(|completed| completed.value);
    };
    let query_options = polylogue_store::query::execute::options_snapshot(&options, false, None)?;
    let accepted = operation
        .accepted_request()
        .map(serde_json::to_value)
        .transpose()
        .map_err(Failure::internal)?;
    let binding = prepared.execution_binding(
        query_options,
        polylogue_model::query_objects::ExecutionPresentation::Query,
        accepted,
    )?;
    crate::daemon::query::history::RecordedQuery::from_binding(
        call,
        &ask,
        prepared.query_object()?,
        binding,
        reader.change_position()?,
    )
    .record(execute)
}

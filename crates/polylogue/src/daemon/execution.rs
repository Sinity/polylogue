//! Request deadlines and transport cancellation. Only read work is interrupted;
//! a committed write keeps its ordinary result.
use super::Request;
use command_surface::Answer;
use polylogue_store::read::limit::{ReadCancellation, ReadLimit};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

pub const DEFAULT_READ_TIMEOUT_MS: u64 = 30_000;

pub fn timeout() -> Duration {
    Duration::from_millis(
        std::env::var("POLYLOGUE_READ_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_READ_TIMEOUT_MS),
    )
}

/// Absolute Unix milliseconds on the wire; the daemon converts once to a
/// monotonic duration so clock changes cannot extend a running read.
#[derive(Deserialize)]
pub struct Incoming {
    #[serde(flatten)]
    pub request: Request,
    pub deadline_ms: Option<i64>,
    pub caller: polylogue_model::authorship::CallerEvidence,
}

#[derive(Serialize)]
pub struct Outgoing<'a> {
    #[serde(flatten)]
    pub request: &'a Request,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<i64>,
    pub caller: polylogue_model::authorship::CallerEvidence,
}

impl<'a> Outgoing<'a> {
    pub fn new(request: &'a Request, deadline_ms: Option<i64>) -> Self {
        Self {
            request,
            deadline_ms,
            caller: crate::protocol::caller::evidence(),
        }
    }
}

pub fn deadline(duration: Duration) -> i64 {
    super::service::now_ms().saturating_add(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
}

pub fn remaining(deadline_ms: Option<i64>) -> Duration {
    deadline_ms.map_or_else(timeout, |deadline| {
        Duration::from_millis(deadline.saturating_sub(super::service::now_ms()).max(0) as u64)
    })
}

pub fn limit(deadline_ms: Option<i64>, cancellation: ReadCancellation) -> ReadLimit {
    ReadLimit::new(remaining(deadline_ms), cancellation)
}

/// Limits cover read phases, including context preparation before its receipt.
/// Unbudgeted streams, inference and verification have cancellation without a
/// default deadline.
pub fn request_deadline(request: &Request, deadline_ms: Option<i64>) -> Option<i64> {
    if !request.has_read_phase() {
        None
    } else if matches!(
        request,
        Request::Read {
            delivery: polylogue_model::read::ReadDelivery::Stream,
            ..
        } | Request::SchemaInfer { .. }
            | Request::Verify { .. }
    ) {
        deadline_ms
    } else {
        Some(deadline_ms.unwrap_or_else(|| deadline(timeout())))
    }
}

pub fn limit_for(
    request: &Request,
    deadline_ms: Option<i64>,
    cancellation: ReadCancellation,
) -> ReadLimit {
    match request_deadline(request, deadline_ms) {
        Some(deadline_ms) => limit(Some(deadline_ms), cancellation),
        None => ReadLimit::without_deadline(cancellation),
    }
}

pub fn run(
    request: Request,
    limit: &ReadLimit,
    work: impl FnOnce(Request) -> Answer<Value>,
) -> Answer<Value> {
    if !request.has_read_phase() {
        return work(request);
    }
    // Every request with a read phase is named; the rest returned above.
    let (stage, text) = match &request {
        Request::Query { text, .. } => ("query", Some(text.clone())),
        Request::Explain { expression, .. } => ("explain", expression.clone()),
        Request::Context {
            ask: crate::briefing::Ask::Recall { text, .. },
            ..
        } => ("recall", Some(text.clone())),
        Request::Context { .. } => ("context", None),
        Request::AssertQuery { query, .. } => ("assert_query", Some(query.clone())),
        Request::BrowserSelectionNote { .. } => ("browser_selection_note", None),
        Request::Sql { .. } => ("sql", None),
        Request::Status => ("status", None),
        Request::Complete { .. } => ("completion", None),
        Request::Read { .. } => ("read", None),
        Request::Changes { .. } => ("changes", None),
        Request::Follow { query, .. } => ("follow", Some(query.clone())),
        Request::Attachment { .. } => ("attachment", None),
        Request::SchemaInfer { .. } => ("schema_infer", None),
        Request::SchemaUnread { .. } => ("schema_unread", None),
        Request::Verify { .. } => ("verify", None),
        Request::Timeline { .. } => ("timeline", None),
        Request::Cost { .. } => ("cost", None),
        Request::Resumable { .. } => ("resumable", None),
        Request::Continue { .. } => ("continue", None),
        Request::Setting { .. } => ("setting", None),
        Request::Embed { .. } => ("embed", None),
        Request::ChatActions => ("chat_actions", None),
        Request::ChatAction { .. } => ("chat_action", None),
        Request::ChatAttachment { .. } => ("chat_attachment", None),
        Request::BackfillJobs => ("backfill_jobs", None),
        Request::Conversations { .. } => ("conversations", None),
        Request::StoredFiles { .. } => ("stored_files", None),
        request => unreachable!("{request:?} has no read phase"),
    };
    limit.scope(|| {
        if let Err(error) = limit.check() {
            return refusal(&error, stage, text.as_deref(), limit);
        }
        let response = work(request);
        // A checked writer handoff ends the read phase. Its accepted result
        // must not become a timeout after the write commits.
        if ReadLimit::current().is_some()
            && let Err(error) = limit.check()
        {
            return refusal(&error, stage, text.as_deref(), limit);
        }
        response
    })
}

/// A read stopped by its deadline or a cancellation names the stage it
/// reached and, for a query, a narrower query that can finish in time.
pub fn refusal(
    error: &polylogue_store::StoreError,
    stage: &str,
    text: Option<&str>,
    limit: &ReadLimit,
) -> Answer<Value> {
    let mut failure = crate::protocol::store_failure(error);
    if let Some(query) = text.and_then(narrower) {
        failure = failure.with_suggestion(command_surface::Suggestion::Query(query));
    }
    failure.hint = Some("restrict the query to a shorter time range or one session".into());
    failure.retryable = true;
    let stage = limit
        .stage()
        .map_or_else(|| stage.to_owned(), str::to_owned);
    let steps = limit.sqlite_steps();
    let mut notes = vec![format!(
        "stopped during {stage} after {steps} SQLite steps; rows scanned are unknown"
    )];
    notes.extend(
        limit
            .stage_timings()
            .into_iter()
            .map(|(stage, spent)| format!("{stage} took {:.1} ms", spent.as_secs_f64() * 1000.0)),
    );
    command_surface::Answer {
        outcome: command_surface::Outcome::Error(failure),
        envelope: command_surface::Envelope {
            notes,
            next: vec![],
            ..Default::default()
        },
    }
}

pub(crate) fn narrower(query: &str) -> Option<String> {
    use polylogue_query::ast::Source;
    let parsed = polylogue_query::parse(query).ok()?;
    let suggestion = match parsed.source {
        Source::Unit {
            condition: Some(cond),
            ..
        } => {
            let span = cond.span;
            let condition = &query[span.start..span.end];
            let trimmed = condition.trim_end();
            format!(
                "{}({trimmed}) since:1d{}{}",
                &query[..span.start],
                &condition[trimmed.len()..],
                &query[span.end..]
            )
        }
        Source::Unit {
            condition: None, ..
        } => {
            let start = query.len() - query.trim_start().len();
            let end = query[start..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .map_or(query.len(), |end| start + end);
            format!("{} where since:1d{}", &query[..end], &query[end..])
        }
        _ => return None,
    };
    polylogue_query::parse(&suggestion).ok().map(|_| suggestion)
}

#[cfg(test)]
mod tests;

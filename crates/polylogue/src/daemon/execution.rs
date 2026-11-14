//! Request deadlines and transport cancellation. Only read work is interrupted;
//! a committed write keeps its ordinary result.
use command_surface::Answer;
use polylogue_store::read::limit::{ReadCancellation, ReadLimit};
use std::time::Duration;

pub const DEFAULT_READ_TIMEOUT_MS: u64 = 30_000;

pub fn timeout() -> Duration {
    Duration::from_millis(DEFAULT_READ_TIMEOUT_MS)
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

/// A read stopped by its deadline or a cancellation names the stage it
/// reached and, for a query, a narrower query that can finish in time.
pub fn refusal<T>(
    error: &polylogue_store::StoreError,
    stage: &str,
    text: Option<&str>,
    limit: &ReadLimit,
) -> Answer<T> {
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

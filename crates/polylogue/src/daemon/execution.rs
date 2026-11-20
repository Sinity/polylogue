//! Request deadlines and transport cancellation. Only read work is interrupted;
//! a committed write keeps its result.
use command_surface::{Answer, Deadline, LatencyClass};
use polylogue_store::{cancellation::Cancellation, read::limit::ReadLimit};
use std::time::{Duration, Instant};

pub const DEFAULT_READ_TIMEOUT_MS: u64 = 30_000;

pub fn timeout() -> Duration {
    Duration::from_millis(DEFAULT_READ_TIMEOUT_MS)
}

pub fn deadline(duration: Duration) -> i64 {
    polylogue_model::WallClock::new(jiff::Timestamp::now)
        .now_ms()
        .saturating_add(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
}

pub fn remaining(deadline_ms: Option<i64>) -> Duration {
    deadline_ms.map_or_else(timeout, |deadline| {
        Duration::from_millis(
            deadline
                .saturating_sub(polylogue_model::WallClock::new(jiff::Timestamp::now).now_ms())
                .max(0) as u64,
        )
    })
}

pub fn limit(deadline_ms: Option<i64>, cancellation: Cancellation) -> ReadLimit {
    ReadLimit::new(remaining(deadline_ms), cancellation)
}

/// Resolve the operation's deadline for transport work, within the caller's
/// already accepted deadline. A missing default gets the daemon's read limit
/// only when the operation takes a default.
pub(crate) fn request_limit(
    deadline: Deadline,
    latency: LatencyClass,
    default_deadline: bool,
    parent: Option<Instant>,
    cancellation: Cancellation,
) -> ReadLimit {
    let declared = crate::commands::deadlines().get(latency);
    let duration = match deadline {
        Deadline::Default if !default_deadline => None,
        Deadline::Default if declared.is_none() => Some(timeout()),
        deadline => deadline.duration(declared),
    };
    let request = duration.and_then(|duration| Instant::now().checked_add(duration));
    match [parent, request].into_iter().flatten().min() {
        Some(deadline) => ReadLimit::at(deadline, cancellation),
        None => ReadLimit::without_deadline(cancellation),
    }
}

/// A stopped read names its stage and any shorter query suited to the caller.
pub fn refusal<T>(
    error: &polylogue_store::StoreError,
    stage: &str,
    text: Option<&str>,
    limit: &ReadLimit,
) -> Answer<T> {
    let context = stage == "context";
    let stage = limit
        .stage()
        .map_or_else(|| stage.to_owned(), str::to_owned);
    let mut failure = crate::protocol::store_failure(error);
    let query = (!context && matches!(stage.as_str(), "page" | "similarity"))
        .then(|| text.and_then(narrower))
        .flatten();
    let hint = if let Some(query) = query {
        failure = failure.with_suggestion(command_surface::Suggestion::Query(query));
        "restrict the query to a shorter time range or one session"
    } else {
        "rerun with `--deadline none`, or narrow to one session"
    };
    failure.hint = Some(hint.into());
    failure.retryable = true;
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
    use polylogue_query::ast::{ConditionKind, Expression, Source, Unit};
    let parsed = polylogue_query::parse(query).ok()?;
    if let Source::Unit {
        unit,
        condition: Some(condition),
        ..
    } = &parsed.source
    {
        let time =
            Unit::from_name(unit).and_then(|unit| polylogue_query::units::spec(unit).main_time);
        let is_time = |expression: &Expression| matches!(expression, Expression::Field { name } if time.is_some_and(|time| name.eq_ignore_ascii_case(time)));
        if polylogue_query::selection::top_level(condition)
            .iter()
            .any(|condition| match &condition.kind {
                ConditionKind::Since { .. } | ConditionKind::Until { .. } => true,
                ConditionKind::Compare { left, right, .. } => is_time(left) || is_time(right),
                ConditionKind::Between { left, .. } | ConditionKind::TimeRange { left, .. } => {
                    is_time(left)
                }
                _ => false,
            })
        {
            return None;
        }
    }
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

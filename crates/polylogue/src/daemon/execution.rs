//! Deadlines as callers name them, and the refusal of a read that ran out.
//! Only read work is interrupted; a committed write keeps its result.
use command_surface::Answer;
use polylogue_store::read::limit::ReadLimit;
use std::time::Duration;

pub const DEFAULT_READ_DEADLINE_MS: u64 = 30_000;

pub fn default_read_deadline() -> Duration {
    Duration::from_millis(DEFAULT_READ_DEADLINE_MS)
}

pub fn deadline(duration: Duration) -> i64 {
    jiff::Timestamp::now()
        .as_millisecond()
        .saturating_add(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
}

pub fn remaining(deadline_ms: Option<i64>) -> Duration {
    deadline_ms.map_or_else(default_read_deadline, |deadline| {
        Duration::from_millis(
            deadline
                .saturating_sub(jiff::Timestamp::now().as_millisecond())
                .max(0) as u64,
        )
    })
}

/// A stopped read names its stage and any shorter query suited to the caller.
pub fn stopped_read<T>(
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

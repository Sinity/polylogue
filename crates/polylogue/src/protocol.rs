//! Product failure codes classified once into the shared transport contract.
pub mod caller;
pub mod complete;
pub mod operations;
pub mod query;
pub mod query_aliases;
pub mod schema;
pub mod status;
pub mod stream;
pub mod wire;
use command_surface::{Answer, Failure, FailureKind, Position};
use serde_json::Value;

pub const FAILURE_CLASSES: &[(FailureKind, &[&str])] = &[
    (FailureKind::Empty, &["empty"]),
    (
        FailureKind::InvalidQuery,
        &["query", "sql", "invalid_query"],
    ),
    (FailureKind::UnknownField, &["unknown_field"]),
    (FailureKind::BadValue, &["bad_value"]),
    (FailureKind::NotAQuery, &["not_a_query"]),
    (
        FailureKind::BadRequest,
        &[
            "bad_request",
            "invalid",
            "incomplete_parents",
            "unsupported_provider",
            "empty_text",
            "model_required",
            "attachment_name_not_a_basename",
            "invalid_action",
            "since_is_a_date_or_time",
            "unknown_outcome",
            "unknown_attachment",
            "unknown_report",
        ],
    ),
    (FailureKind::Ambiguous, &["ambiguous"]),
    (FailureKind::StaleCursor, &["stale_cursor"]),
    (
        FailureKind::NotYetAvailable,
        &["not_built", "not_yet_available"],
    ),
    (
        FailureKind::Refused,
        &["refused", "operator_required", "rejected"],
    ),
    (
        FailureKind::NotFound,
        &[
            "not_found",
            "unknown_chat_action",
            "unknown_chat_action_attachment",
            "attachment_missing",
            "unknown_backfill_job",
        ],
    ),
    (
        FailureKind::Conflict,
        &[
            "conflict",
            "stale",
            "scope_changed",
            "resync_required",
            "idempotency_key_reused",
            "action_finished",
            "not_the_claimant",
            "stage_only_action_cannot_submit",
            "not_awaiting_approval",
            "job_ended",
        ],
    ),
    (FailureKind::Degraded, &["degraded"]),
    (FailureKind::StoppedEarly, &["stopped_early"]),
    (
        FailureKind::NotRunning,
        &["not_running", "daemon_required", "daemon_stopping"],
    ),
    (FailureKind::Timeout, &["timeout", "deadline_exceeded"]),
    (FailureKind::Cancelled, &["cancelled"]),
    (
        FailureKind::Io,
        &[
            "io",
            "store",
            "internal",
            "import",
            "embed",
            "sinex_unavailable",
        ],
    ),
];

pub fn kind(code: &str) -> FailureKind {
    FAILURE_CLASSES
        .iter()
        .find_map(|(kind, codes)| codes.contains(&code).then_some(*kind))
        .unwrap_or(FailureKind::Io)
}

pub fn failure(code: &str, message: String, detail: Option<&Value>) -> Failure {
    let mut failure = Failure::new(kind(code), message);
    failure.code = Some(code.to_owned());
    if let Some(detail) = detail {
        failure.position = detail.get("position").and_then(|position| {
            let offset = position.as_u64().or_else(|| position["offset"].as_u64())?;
            Some(Position {
                offset: usize::try_from(offset).ok()?,
                length: position["length"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(1),
            })
        });
        failure.suggestion = detail
            .get("suggestion")
            .filter(|v| !v.is_null())
            .cloned()
            .map(Box::new);
        failure.hint = detail["hint"].as_str().map(str::to_owned);
        failure.retryable = detail["retryable"].as_bool().unwrap_or(failure.retryable);
    }
    failure
}

/// Pending writes have a durable local receipt but no confirmed remote commit.
/// Read outcomes carry their incompleteness in the typed answer payload.
pub fn answer(value: Value) -> Answer<Value> {
    if value["pending"].as_bool() == Some(true) || value["state"].as_str() == Some("pending") {
        Answer::Degraded {
            value,
            notes: vec!["accepted locally; awaiting Sinex confirmation".into()],
            next: vec![],
        }
    } else if value["incomplete"].as_bool() == Some(true)
        || value["outcome"].as_str() == Some("incomplete")
        || value["incomplete"]
            .as_array()
            .is_some_and(|parts| !parts.is_empty())
    {
        Answer::Degraded {
            value,
            notes: vec![],
            next: vec![],
        }
    } else if value.as_array().is_some_and(Vec::is_empty)
        || value["empty"].as_bool() == Some(true)
        || value["outcome"].as_str() == Some("empty")
    {
        Answer::Empty {
            value,
            notes: vec![],
            next: vec![],
        }
    } else {
        Answer::ok(value)
    }
}

#[cfg(test)]
mod tests;

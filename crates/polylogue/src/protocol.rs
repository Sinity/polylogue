//! Product failure codes classified once into the shared transport contract.
use command_surface::{Answer, Failure, FailureKind, Position};
use serde_json::Value;

pub fn kind(code: &str) -> FailureKind {
    match code {
        "empty" => FailureKind::Empty,
        "query" | "sql" | "invalid_query" => FailureKind::InvalidQuery,
        "unknown_field" => FailureKind::UnknownField,
        "bad_value" => FailureKind::BadValue,
        "not_a_query" => FailureKind::NotAQuery,
        "bad_request"
        | "invalid"
        | "incomplete_parents"
        | "unsupported_provider"
        | "empty_text"
        | "model_required"
        | "attachment_name_not_a_basename"
        | "invalid_action"
        | "since_is_a_date_or_time"
        | "unknown_outcome"
        | "unknown_attachment"
        | "unknown_report" => FailureKind::BadRequest,
        "ambiguous" => FailureKind::Ambiguous,
        "stale_cursor" => FailureKind::StaleCursor,
        "not_built" | "not_yet_available" => FailureKind::NotYetAvailable,
        "refused" | "operator_required" | "rejected" => FailureKind::Refused,
        "not_found"
        | "unknown_chat_action"
        | "unknown_chat_action_attachment"
        | "attachment_missing"
        | "unknown_backfill_job" => FailureKind::NotFound,
        "conflict"
        | "resync_required"
        | "idempotency_key_reused"
        | "action_finished"
        | "not_the_claimant"
        | "stage_only_action_cannot_submit"
        | "not_awaiting_approval"
        | "job_ended" => FailureKind::Conflict,
        "degraded" => FailureKind::Degraded,
        "stopped_early" => FailureKind::StoppedEarly,
        "not_running" | "daemon_required" | "daemon_stopping" => FailureKind::NotRunning,
        "timeout" | "deadline_exceeded" => FailureKind::Timeout,
        "cancelled" => FailureKind::Cancelled,
        _ => FailureKind::Io,
    }
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
    } else if value["incomplete"].as_bool() == Some(true) {
        Answer::Degraded {
            value,
            notes: vec![],
            next: vec![],
        }
    } else if value["empty"].as_bool() == Some(true) {
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

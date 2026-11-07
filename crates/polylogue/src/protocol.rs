//! Product failure codes classified once into the shared transport contract.
pub mod assertion;
pub mod attachment;
pub mod caller;
pub mod complete;
pub mod context;
pub mod follow;
pub mod import;
pub mod operations;
mod scope;
pub(crate) use scope::{ReadScopes, answer as scope_answer};
pub mod query;
pub mod query_aliases;
pub mod read;
pub mod schema;
pub mod setting;
pub mod status;
pub mod stream;
pub mod verify;
pub mod wire;
use command_surface::{Answer, Failure, FailureKind, InputSpan};

pub const FAILURE_CLASSES: &[(FailureKind, &[&str])] = &[
    (
        FailureKind::InvalidQuery,
        &["query", "sql", "invalid_query", "invalid_cursor"],
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
        &[
            "not_built",
            "not_yet_available",
            "primary_context_missing",
            "primary_context_pending",
            "primary_context_stale",
        ],
    ),
    (
        FailureKind::Refused,
        &[
            "refused",
            "operator_required",
            "embeddings_off",
            "confirm",
            "rejected",
            "primary_context_refused",
        ],
    ),
    (
        FailureKind::NotFound,
        &[
            "not_found",
            "unknown_chat_action",
            "unknown_chat_action_attachment",
            "attachment_missing",
            "attachment_unavailable",
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
            "browser_actions_off",
        ],
    ),
    (FailureKind::StoppedEarly, &["stopped_early"]),
    (
        FailureKind::NotRunning,
        &[
            "not_running",
            "daemon_required",
            "shutting_down",
            "daemon_protocol_mismatch",
        ],
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
            "primary_context_unavailable",
            "primary_context_invalid",
        ],
    ),
];

#[cfg(feature = "sinex")]
pub fn context_failure(problem: polylogue_sinex::ContextFailure) -> Failure {
    use polylogue_sinex::ContextFailureKind as Kind;
    let code = match problem.kind {
        Kind::Missing => "primary_context_missing",
        Kind::Pending => "primary_context_pending",
        Kind::Unavailable => "primary_context_unavailable",
        Kind::Refused => "primary_context_refused",
        Kind::Stale => "primary_context_stale",
        Kind::Invalid => "primary_context_invalid",
    };
    let mut failure = failure(code, problem.reason);
    failure.retryable = matches!(
        problem.kind,
        Kind::Missing | Kind::Pending | Kind::Unavailable | Kind::Stale
    );
    failure.hint = problem
        .last_known_identity
        .map(|identity| format!("last observed owner settings: {identity}"));
    failure
}

pub fn kind(code: &str) -> FailureKind {
    FAILURE_CLASSES
        .iter()
        .find_map(|(kind, codes)| codes.contains(&code).then_some(*kind))
        .unwrap_or(FailureKind::Io)
}

/// An answer whose value does not decode as its operation's declared type:
/// the daemon that answered was built from other sources.
pub fn protocol_mismatch(operation: &str, error: impl std::fmt::Display) -> Failure {
    failure(
        "daemon_protocol_mismatch",
        format!(
            "the daemon answered {operation} in a shape this build does not read ({error}); \
             it speaks another protocol version, so restart it with `polylogue daemon`"
        ),
    )
}

/// A product failure code, classified once into the shared kind.
pub fn failure(code: &str, message: impl Into<String>) -> Failure {
    let mut failure = Failure::new(kind(code), message);
    failure.code = Some(code.to_owned());
    failure
}

/// A query error keeps its position, correction and hint as typed fields.
pub fn query_failure(error: &polylogue_store::query::QueryError) -> Failure {
    use polylogue_store::query::QueryError;
    let mut failure = failure(error.kind(), error.to_string());
    let position = match error {
        QueryError::Language { error, .. } => {
            failure.hint = error.hint.clone();
            failure.retryable = error.retryable;
            Some(error.position)
        }
        QueryError::Syntax { position, .. }
        | QueryError::UnknownField { position, .. }
        | QueryError::BadValue { position, .. }
        | QueryError::Invalid { position, .. } => Some(*position),
        _ => None,
    };
    let mut failure = failure.with_candidates(match error {
        QueryError::Ambiguous { candidates, .. } => candidates.clone(),
        QueryError::UnknownField { similar, .. } => similar.clone(),
        _ => Vec::new(),
    });
    failure.position = position.map(|offset| InputSpan { offset, length: 1 });
    if let Some(text) = error.suggestion() {
        failure = failure.with_suggestion(command_surface::Suggestion::Query(text.to_owned()));
    }
    failure
}

/// Every store error reaches surfaces through this one classification.
pub fn store_failure(error: &polylogue_store::StoreError) -> Failure {
    match error {
        polylogue_store::StoreError::Query(error) => query_failure(error),
        polylogue_store::StoreError::Ambiguous { candidates, .. } => {
            failure(error.code(), error.to_string()).with_candidates(candidates.clone())
        }
        error => failure(error.code(), error.to_string()),
    }
}

/// Human-facing diagnostics include the missing parts beside ordinary notes.
pub fn notes<T>(answer: &Answer<T>) -> impl Iterator<Item = &String> {
    answer.envelope.notes.iter().chain(
        match &answer.outcome {
            command_surface::Outcome::Degraded { missing, .. } => missing.as_slice(),
            _ => &[],
        }
        .iter(),
    )
}

pub fn suggestion_text(failure: &Failure) -> Option<std::borrow::Cow<'_, str>> {
    match failure.suggestion()? {
        command_surface::Suggestion::Query(text) => Some(text.as_str().into()),
        command_surface::Suggestion::Request(request) => Some(
            format!(
                "polylogue --request {}",
                polylogue_render::shell_quote(
                    &serde_json::to_string(request).expect("request serializes")
                )
            )
            .into(),
        ),
    }
}

/// The value of an answer that carries one, or its failure.
pub fn into_value<T>(answer: Answer<T>) -> Result<T, Failure> {
    match answer {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(value),
            ..
        }
        | command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            ..
        }
        | command_surface::Answer {
            outcome: command_surface::Outcome::Degraded { value, .. },
            ..
        } => Ok(value),
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } => Err(error),
    }
}

/// A write accepted into the local primary spool is degraded until Sinex confirms it.
pub fn written<T>(value: T, pending: bool) -> Answer<T> {
    if pending {
        command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: vec![AWAITING_CONFIRMATION.into()],
            },
            envelope: command_surface::Envelope {
                next: vec![],
                ..Default::default()
            },
        }
    } else {
        Answer::ok(value)
    }
}

pub const AWAITING_CONFIRMATION: &str = "accepted locally; awaiting Sinex confirmation";

/// A read over an index that is not current keeps its value and says why.
pub fn read<T>(value: T, currency: &polylogue_model::currency::IndexCurrency) -> Answer<T> {
    match currency_note(currency) {
        Some(note) => command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: vec![note.into()],
            },
            envelope: command_surface::Envelope {
                next: vec![],
                ..Default::default()
            },
        },
        None => Answer::ok(value),
    }
}

pub fn currency_note(currency: &polylogue_model::currency::IndexCurrency) -> Option<&'static str> {
    use polylogue_model::currency::{CurrencyState, IndexCurrency};
    if !currency.degraded() {
        return None;
    }
    Some(match currency {
        IndexCurrency::Primary {
            state: CurrencyState::Behind,
            ..
        } => "primary index is behind the observed owner position",
        IndexCurrency::Primary {
            state: CurrencyState::Unavailable,
            ..
        } => "primary owner is unavailable",
        _ => "primary index currency is unknown",
    })
}

#[cfg(test)]
mod tests;

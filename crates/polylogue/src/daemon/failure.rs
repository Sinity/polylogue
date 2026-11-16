//! The error every daemon operation returns before it becomes an answer.

/// A failed operation inside the daemon, as every surface reports it. The
/// shared failure is wrapped only so that store, selection and query errors
/// convert into it with `?`.
#[derive(Debug)]
pub struct Failure {
    pub error: command_surface::Failure,
}

impl Failure {
    pub fn new(code: &str, message: impl ToString) -> Self {
        Self {
            error: crate::protocol::failure(code, message.to_string()),
        }
    }

    pub fn query(message: impl ToString) -> Self {
        Self::new("query", message)
    }
    pub fn sql(message: impl ToString) -> Self {
        Self::new("sql", message)
    }
    pub fn invalid_query(message: impl ToString) -> Self {
        Self::new("invalid_query", message)
    }
    pub fn invalid_cursor(message: impl ToString) -> Self {
        Self::new("invalid_cursor", message)
    }
    pub fn unknown_field(message: impl ToString) -> Self {
        Self::new("unknown_field", message)
    }
    pub fn bad_value(message: impl ToString) -> Self {
        Self::new("bad_value", message)
    }
    pub fn not_a_query(message: impl ToString) -> Self {
        Self::new("not_a_query", message)
    }
    pub fn bad_request(message: impl ToString) -> Self {
        Self::new("bad_request", message)
    }
    pub fn invalid(message: impl ToString) -> Self {
        Self::new("invalid", message)
    }
    pub fn incomplete_parents(message: impl ToString) -> Self {
        Self::new("incomplete_parents", message)
    }
    pub fn unsupported_provider(message: impl ToString) -> Self {
        Self::new("unsupported_provider", message)
    }
    pub fn empty_text(message: impl ToString) -> Self {
        Self::new("empty_text", message)
    }
    pub fn model_required(message: impl ToString) -> Self {
        Self::new("model_required", message)
    }
    pub fn attachment_name_not_a_basename(message: impl ToString) -> Self {
        Self::new("attachment_name_not_a_basename", message)
    }
    pub fn invalid_action(message: impl ToString) -> Self {
        Self::new("invalid_action", message)
    }
    pub fn submission_receipt_required(message: impl ToString) -> Self {
        Self::new("submission_receipt_required", message)
    }
    pub fn submission_conversation_mismatch(message: impl ToString) -> Self {
        Self::new("submission_conversation_mismatch", message)
    }
    pub fn since_is_a_date_or_time(message: impl ToString) -> Self {
        Self::new("since_is_a_date_or_time", message)
    }
    pub fn unknown_outcome(message: impl ToString) -> Self {
        Self::new("unknown_outcome", message)
    }
    pub fn unknown_attachment(message: impl ToString) -> Self {
        Self::new("unknown_attachment", message)
    }
    pub fn unknown_report(message: impl ToString) -> Self {
        Self::new("unknown_report", message)
    }
    pub fn ambiguous(message: impl ToString) -> Self {
        Self::new("ambiguous", message)
    }
    pub fn stale_cursor(message: impl ToString) -> Self {
        Self::new("stale_cursor", message)
    }
    pub fn not_built(message: impl ToString) -> Self {
        Self::new("not_built", message)
    }
    pub fn not_yet_available(message: impl ToString) -> Self {
        Self::new("not_yet_available", message)
    }
    pub fn index_building(message: impl ToString) -> Self {
        Self::new("index_building", message)
    }
    pub fn primary_context_missing(message: impl ToString) -> Self {
        Self::new("primary_context_missing", message)
    }
    pub fn primary_context_pending(message: impl ToString) -> Self {
        Self::new("primary_context_pending", message)
    }
    pub fn primary_context_stale(message: impl ToString) -> Self {
        Self::new("primary_context_stale", message)
    }
    pub fn refused(message: impl ToString) -> Self {
        Self::new("refused", message)
    }
    pub fn operator_required(message: impl ToString) -> Self {
        Self::new("operator_required", message)
    }
    pub fn embeddings_off(message: impl ToString) -> Self {
        Self::new("embeddings_off", message)
    }
    pub fn confirm(message: impl ToString) -> Self {
        Self::new("confirm", message)
    }
    pub fn rejected(message: impl ToString) -> Self {
        Self::new("rejected", message)
    }
    pub fn primary_context_refused(message: impl ToString) -> Self {
        Self::new("primary_context_refused", message)
    }
    pub fn not_found(message: impl ToString) -> Self {
        Self::new("not_found", message)
    }
    pub fn unknown_chat_action(message: impl ToString) -> Self {
        Self::new("unknown_chat_action", message)
    }
    pub fn unknown_chat_action_attachment(message: impl ToString) -> Self {
        Self::new("unknown_chat_action_attachment", message)
    }
    pub fn attachment_missing(message: impl ToString) -> Self {
        Self::new("attachment_missing", message)
    }
    pub fn attachment_unavailable(message: impl ToString) -> Self {
        Self::new("attachment_unavailable", message)
    }
    pub fn unknown_backfill_job(message: impl ToString) -> Self {
        Self::new("unknown_backfill_job", message)
    }
    pub fn conflict(message: impl ToString) -> Self {
        Self::new("conflict", message)
    }
    pub fn stale(message: impl ToString) -> Self {
        Self::new("stale", message)
    }
    pub fn scope_changed(message: impl ToString) -> Self {
        Self::new("scope_changed", message)
    }
    pub fn resync_required(message: impl ToString) -> Self {
        Self::new("resync_required", message)
    }
    pub fn idempotency_key_reused(message: impl ToString) -> Self {
        Self::new("idempotency_key_reused", message)
    }
    pub fn action_finished(message: impl ToString) -> Self {
        Self::new("action_finished", message)
    }
    pub fn not_the_claimant(message: impl ToString) -> Self {
        Self::new("not_the_claimant", message)
    }
    pub fn stage_only_action_cannot_submit(message: impl ToString) -> Self {
        Self::new("stage_only_action_cannot_submit", message)
    }
    pub fn not_awaiting_approval(message: impl ToString) -> Self {
        Self::new("not_awaiting_approval", message)
    }
    pub fn job_ended(message: impl ToString) -> Self {
        Self::new("job_ended", message)
    }
    pub fn browser_actions_off(message: impl ToString) -> Self {
        Self::new("browser_actions_off", message)
    }
    pub fn stopped_early(message: impl ToString) -> Self {
        Self::new("stopped_early", message)
    }
    pub fn not_running(message: impl ToString) -> Self {
        Self::new("not_running", message)
    }
    pub fn daemon_required(message: impl ToString) -> Self {
        Self::new("daemon_required", message)
    }
    pub fn shutting_down(message: impl ToString) -> Self {
        Self::new("shutting_down", message)
    }
    pub fn daemon_protocol_mismatch(message: impl ToString) -> Self {
        Self::new("daemon_protocol_mismatch", message)
    }
    pub fn timeout(message: impl ToString) -> Self {
        Self::new("timeout", message)
    }
    pub fn deadline_exceeded(message: impl ToString) -> Self {
        Self::new("deadline_exceeded", message)
    }
    pub fn cancelled(message: impl ToString) -> Self {
        Self::new("cancelled", message)
    }
    pub fn io(message: impl ToString) -> Self {
        Self::new("io", message)
    }
    pub fn internal(message: impl ToString) -> Self {
        Self::new("internal", message)
    }
    pub fn import(message: impl ToString) -> Self {
        Self::new("import", message)
    }
    pub fn embed(message: impl ToString) -> Self {
        Self::new("embed", message)
    }
    pub fn sinex_unavailable(message: impl ToString) -> Self {
        Self::new("sinex_unavailable", message)
    }
    pub fn primary_context_unavailable(message: impl ToString) -> Self {
        Self::new("primary_context_unavailable", message)
    }
    pub fn primary_context_invalid(message: impl ToString) -> Self {
        Self::new("primary_context_invalid", message)
    }

    pub fn store(error: &polylogue_store::StoreError) -> Self {
        Self {
            error: crate::protocol::store_failure(error),
        }
    }

    /// The product code that classified this failure.
    pub fn code(&self) -> &str {
        self.error.code.as_deref().unwrap_or_default()
    }

    pub fn answer<T>(self) -> command_surface::Answer<T> {
        command_surface::Answer::error(self.error)
    }
}

impl std::ops::Deref for Failure {
    type Target = command_surface::Failure;
    fn deref(&self) -> &Self::Target {
        &self.error
    }
}

impl From<command_surface::Failure> for Failure {
    fn from(error: command_surface::Failure) -> Self {
        Self { error }
    }
}

impl From<polylogue_store::StoreError> for Failure {
    fn from(error: polylogue_store::StoreError) -> Self {
        Self::store(&error)
    }
}

impl From<polylogue_store::selection::SelectionError> for Failure {
    fn from(error: polylogue_store::selection::SelectionError) -> Self {
        use polylogue_store::selection::SelectionError;
        match error {
            SelectionError::Store(error) => Self::from(error),
            SelectionError::Query(error) => Self::from(error),
            SelectionError::Language(error) => Self::from(error),
            SelectionError::Invalid { kind, message } => Self::new(kind, message),
        }
    }
}

impl From<query_language::Error> for Failure {
    fn from(error: query_language::Error) -> Self {
        Self::from(polylogue_store::query::QueryError::Language {
            diagnostic: error.to_string(),
            error: Box::new(error),
        })
    }
}

impl From<polylogue_store::query::QueryError> for Failure {
    fn from(error: polylogue_store::query::QueryError) -> Self {
        Self::from(polylogue_store::StoreError::Query(error))
    }
}

impl<T> From<Failure> for command_surface::Answer<T> {
    fn from(failure: Failure) -> Self {
        failure.answer()
    }
}

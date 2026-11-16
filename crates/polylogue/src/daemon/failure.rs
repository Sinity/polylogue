//! The error every daemon operation returns before it becomes an answer.

/// A failed operation inside the daemon, as every surface reports it. The
/// shared failure is wrapped only so that store, selection and query errors
/// convert into it with `?`.
#[derive(Debug)]
pub struct Failure {
    pub error: command_surface::Failure,
}

impl Failure {
    pub fn new(code: polylogue_model::Code, message: impl ToString) -> Self {
        Self {
            error: crate::protocol::failure(code, message.to_string()),
        }
    }

    pub fn invalid_query(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::InvalidQuery, message)
    }
    pub fn bad_value(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::BadValue, message)
    }
    pub fn bad_request(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::BadRequest, message)
    }
    pub fn invalid(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::Invalid, message)
    }
    pub fn stale_cursor(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::StaleCursor, message)
    }
    pub fn not_yet_available(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::NotYetAvailable, message)
    }
    pub fn refused(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::Refused, message)
    }
    pub fn not_found(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::NotFound, message)
    }
    pub fn conflict(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::Conflict, message)
    }
    pub fn shutting_down(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::ShuttingDown, message)
    }
    pub fn deadline_exceeded(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::DeadlineExceeded, message)
    }
    pub fn io(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::Io, message)
    }
    pub fn internal(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::Internal, message)
    }
    pub fn sinex_unavailable(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::SinexUnavailable, message)
    }
    pub fn primary_context_invalid(message: impl ToString) -> Self {
        Self::new(polylogue_model::Code::PrimaryContextInvalid, message)
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

impl From<polylogue_reading::Error> for Failure {
    fn from(error: polylogue_reading::Error) -> Self {
        use polylogue_reading::Error;
        match error {
            Error::Store(error) => Self::from(error),
            Error::Selection(error) => Self::from(error),
            Error::Operation(error) => Self::from(error),
            Error::Layout(error) => {
                let hint = match &error {
                    polylogue_model::read::ReadError::UnsupportedTarget {
                        accepted_views, ..
                    } => Some(format!(
                        "views that read it: {}",
                        accepted_views
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                    polylogue_model::read::ReadError::RequiredExpansion { option, .. } => {
                        Some(format!("set {option} to full"))
                    }
                    _ => None,
                };
                let error = Error::Layout(error);
                let code = polylogue_model::Code::from_name(error.code())
                    .expect("reading error has a product code");
                let mut failure = Self::new(code, error);
                failure.error.hint = hint;
                failure
            }

            error => {
                let code = polylogue_model::Code::from_name(error.code())
                    .expect("reading error has a product code");
                Self::new(code, error)
            }
        }
    }
}

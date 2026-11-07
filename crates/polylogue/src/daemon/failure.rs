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

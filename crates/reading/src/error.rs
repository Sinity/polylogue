//! Failures while composing, paging or streaming a reading.
use polylogue_render::pager::PageError;
use polylogue_store::{StoreError, selection::SelectionError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Selection(#[from] SelectionError),
    #[error(transparent)]
    Layout(polylogue_render::ReadError),
    #[error(transparent)]
    Page(PageError),
    #[error("{changed}")]
    StaleCursor { changed: String },
    #[error("{0}")]
    InvalidCursor(String),
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    BadValue(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    NotYetAvailable(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Io(String),
    #[error("{0}; this is a Polylogue bug; report it with the request ID")]
    Internal(String),
    #[error("{0}")]
    Operation(command_surface::Failure),
}
impl Error {
    pub fn code(&self) -> &str {
        match self {
            Self::Store(error) => error.code().as_str(),
            Self::Selection(error) => error.code().as_str(),
            Self::Layout(polylogue_render::ReadError::MissingPreparedData { .. }) => {
                "not_yet_available"
            }
            Self::Layout(_) => "bad_request",
            Self::Page(PageError::ItemChanged | PageError::InvalidPosition)
            | Self::StaleCursor { .. } => "stale_cursor",
            Self::Page(_) | Self::BadRequest(_) => "bad_request",
            Self::InvalidCursor(_) => "invalid_query",
            Self::BadValue(_) => "bad_value",
            Self::NotFound(_) => "not_found",
            Self::NotYetAvailable(_) => "not_yet_available",
            Self::Conflict(_) => "conflict",
            Self::Io(_) => "io",
            Self::Internal(_) => "internal",
            Self::Operation(error) => error.code.as_deref().unwrap_or("internal"),
        }
    }
}

impl Error {
    pub(crate) fn bad_request(message: impl ToString) -> Self {
        Self::BadRequest(message.to_string())
    }
    pub(crate) fn not_found(message: impl ToString) -> Self {
        Self::NotFound(message.to_string())
    }
    pub(crate) fn not_yet_available(message: impl ToString) -> Self {
        Self::NotYetAvailable(message.to_string())
    }
    pub(crate) fn conflict(message: impl ToString) -> Self {
        Self::Conflict(message.to_string())
    }
    pub(crate) fn io(message: impl ToString) -> Self {
        Self::Io(message.to_string())
    }
    pub(crate) fn internal(message: impl ToString) -> Self {
        Self::Internal(message.to_string())
    }
    pub(crate) fn invalid_query(message: impl ToString) -> Self {
        Self::InvalidCursor(message.to_string())
    }
    pub(crate) fn stale_cursor(changed: impl ToString) -> Self {
        Self::StaleCursor {
            changed: changed.to_string(),
        }
    }
}
impl From<polylogue_store::query::QueryError> for Error {
    fn from(error: polylogue_store::query::QueryError) -> Self {
        Self::Selection(SelectionError::Query(error))
    }
}
impl From<query_language::Error> for Error {
    fn from(error: query_language::Error) -> Self {
        Self::Selection(SelectionError::Language(error))
    }
}
impl From<command_surface::Failure> for Error {
    fn from(error: command_surface::Failure) -> Self {
        Self::Operation(error)
    }
}

impl From<PageError> for Error {
    fn from(error: PageError) -> Self {
        match error {
            PageError::AlreadyFull => Self::Internal(error.to_string()),
            error => Self::Page(error),
        }
    }
}
impl From<polylogue_render::ReadError> for Error {
    fn from(error: polylogue_render::ReadError) -> Self {
        match error {
            polylogue_render::ReadError::InvalidPreparedData(message) => Self::Internal(message),
            error => Self::Layout(error),
        }
    }
}

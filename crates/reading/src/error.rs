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
    InvalidQuery(String),
    /// A request the caller can correct, with its typed reason.
    #[error("{0}")]
    BadRequest(polylogue_model::Reason),
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
    pub fn code(&self) -> polylogue_model::Code {
        use polylogue_model::Code;
        match self {
            Self::Store(error) => error.code(),
            Self::Selection(error) => error.code(),
            Self::Layout(polylogue_render::ReadError::MissingPreparedData { .. }) => {
                Code::NotYetAvailable
            }
            Self::Layout(_) => Code::BadRequest,
            Self::Page(PageError::ItemChanged | PageError::InvalidPosition)
            | Self::StaleCursor { .. } => Code::StaleCursor,
            Self::Page(_) => Code::BadRequest,
            Self::BadRequest(reason) => reason.code(),
            Self::InvalidQuery(_) => Code::InvalidQuery,
            Self::BadValue(_) => Code::BadValue,
            Self::NotFound(_) => Code::NotFound,
            Self::NotYetAvailable(_) => Code::NotYetAvailable,
            Self::Conflict(_) => Code::Conflict,
            Self::Io(_) => Code::Io,
            Self::Internal(_) => Code::Internal,
            Self::Operation(error) => error
                .code
                .as_deref()
                .and_then(Code::from_name)
                .unwrap_or(Code::Internal),
        }
    }
}

impl Error {
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
        Self::InvalidQuery(message.to_string())
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
impl From<polylogue_model::Reason> for Error {
    fn from(reason: polylogue_model::Reason) -> Self {
        Self::BadRequest(reason)
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

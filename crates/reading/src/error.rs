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
    Layout(#[from] polylogue_model::read::ReadError),
    #[error(transparent)]
    Page(#[from] PageError),
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
            Self::Layout(polylogue_model::read::ReadError::MissingPreparedData { .. }) => {
                "not_yet_available"
            }
            Self::Layout(_) => "bad_request",
            Self::Page(PageError::ItemChanged | PageError::InvalidPosition)
            | Self::StaleCursor { .. } => "stale_cursor",
            Self::Page(_) | Self::BadRequest(_) => "bad_request",
            Self::InvalidCursor(_) => "invalid_cursor",
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

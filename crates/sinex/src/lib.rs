//! Polylogue's event kinds in Sinex: the kind declarations, and the encoders
//! and decoders between Polylogue's records and Sinex event payloads. Primary
//! mode is not built in this version (`docs/primary-mode.md`); its shipper and
//! the reader of Sinex's log are written on these.
pub(crate) mod attribution;
pub mod kinds;
pub mod products;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
}
impl Error {
    pub fn invalid(message: impl ToString) -> Self {
        Self::Invalid(message.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) const SOURCE_VERSION: u32 = 19;

/// This adapter declaration is installed by the operator, never self-registered
/// from a producer's requested permissions.
pub fn source_reference() -> sinex_protocol::SourceRef {
    sinex_protocol::SourceRef {
        name: "polylogue".into(),
        version: SOURCE_VERSION,
    }
}

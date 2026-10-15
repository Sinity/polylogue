//! Disposable source contributions and immutable Message preparation.
//! Canonical selection and actual-client qualification remain in development.
mod directory;
mod writer;
pub use writer::{Build, Identity, Preparation, Source};

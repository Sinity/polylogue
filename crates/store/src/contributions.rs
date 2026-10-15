//! Disposable source contributions and immutable Message preparation.
//! Canonical selection and actual-client qualification remain in development.
mod bodies;
pub use bodies::{BodyId, BodyRetirement, CandidateCollection, CandidateId};
mod blocks;
pub use blocks::{BlockPresence, MessageLayout, ProjectionState, ProjectionStep, TextKind};
mod completion;
pub use completion::PreparedProjection;
mod directory;
mod inheritance;
pub use inheritance::InheritedProjection;
mod origins;
pub use origins::SourceRange;
mod reads;
pub use reads::{BlockView, FieldRead, FieldView, MessageHeader, MessageLookup, MessageView};
mod retirement;
mod source;
pub use source::{SourceProjection, SourceSelection};
mod writer;
pub use retirement::{Retirement, ValueCollection};
pub use writer::{Build, Identity, Preparation, Source};

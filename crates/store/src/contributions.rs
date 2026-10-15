//! Disposable source contributions and immutable Message preparation.
//! Canonical selection and actual-client qualification remain in development.
mod bodies;
pub use bodies::{BodyId, BodyRetirement, CandidateCollection, CandidateId};
mod blocks;
pub use blocks::{
    BlockKind, BlockPresence, MessageLayout, ProjectionState, ProjectionStep, ToolAction,
};
mod completion;
pub use completion::PreparedProjection;
mod directory;
mod inheritance;
pub use inheritance::InheritedProjection;
mod origins;
pub use origins::SourceRange;
mod reads;
pub use reads::{
    BlockFieldAddress, BlockFieldEntry, BlockView, FieldCursor, FieldRead, FieldView,
    MessageFieldEntry, MessageHeader, MessageLookup, MessageView, SessionCoverage, SessionCursor,
    SessionDependencyCursor, SessionHeader, SessionInputDependency, SessionPosition, SessionStep,
    SessionView, SourceCursor, SourceSpan, SourceStep, ToolResultHeader,
};
mod retirement;
mod source;
pub use source::{SourceProjection, SourceSelection};
mod writer;
pub use retirement::{Retirement, ValueCollection};
pub use writer::{Build, Identity, Preparation, Source};

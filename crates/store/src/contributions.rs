//! Disposable source contributions and immutable Message preparation.
//! Canonical selection and actual-client qualification remain in development.
mod bodies;
pub use bodies::{BodyId, BodyRetirement, CandidateCollection, CandidateId};
mod blocks;
pub use blocks::{
    BlockKind, BlockPresence, MessageLayout, ProjectionState, ProjectionStep, ToolAction,
};
mod compatibility;
pub use compatibility::{CompatibilityCertificate, CompatibilityTurn, CompatibleSelection};
mod completion;
pub use completion::PreparedProjection;
mod directory;
mod inheritance;
pub use inheritance::InheritedProjection;
mod origins;
pub use origins::SourceRange;
pub(crate) mod reads;
pub use reads::{
    BlockFieldAddress, BlockFieldEntry, BlockView, ExtraFieldEntry, FieldCursor, FieldRead,
    FieldView, MessageFieldEntry, MessageHeader, MessageLookup, MessageView, OrderedExtras,
    SessionCatalog, SessionCatalogCursor, SessionCoverage, SessionCursor, SessionDependencyCursor,
    SessionHeader, SessionInputDependency, SessionPosition, SessionStep, SessionView, SourceCursor,
    SourceSpan, SourceStep, ToolResultHeader,
};
mod retirement;
mod source;
mod superseded;
pub use source::{SourceProjection, SourceSelection};
pub(crate) mod writer;
pub use retirement::{Retirement, ValueCollection};
pub use writer::{Build, Identity, Preparation, Source};

// Fixtures drive the same bounded production interface. No unbounded selector
// or superseded product route is exposed to consumers solely for old tests.
#[cfg(test)]
pub(super) trait SelectCompatibleSourceForTest {
    fn select_compatible_source_message_for_test(
        &mut self,
        source: &SourceProjection,
        allowance: usize,
    ) -> crate::Result<SourceSelection>;
}
#[cfg(test)]
impl SelectCompatibleSourceForTest for crate::Writer {
    fn select_compatible_source_message_for_test(
        &mut self,
        source: &SourceProjection,
        allowance: usize,
    ) -> crate::Result<SourceSelection> {
        let reader = self.read_handle()?;
        let mut work = self.begin_compatible_source_selection(source, allowance)?;
        for _ in 0..100000 {
            work = match work.advance(
                &reader,
                allowance,
                &std::sync::atomic::AtomicBool::new(false),
            )? {
                CompatibilityTurn::Progress(w) => w,
                CompatibilityTurn::Certified(w, c) => {
                    self.certify_body_compatibility(c)?;
                    w
                }
                CompatibilityTurn::Ready(w) => {
                    return self.install_compatible_source_selection(w, allowance);
                }
                CompatibilityTurn::Contested => return Ok(SourceSelection::Contested),
            };
        }
        Err(crate::Error::Invariant(
            "test selection failed to make bounded progress".into(),
        ))
    }
}

pub(crate) mod tool_associations;

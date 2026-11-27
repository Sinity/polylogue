//! The parts a parser produces for a chain and where those parts are written.

pub use crate::writer::ParsePart;

/// Where a chain's parts go as they are parsed, in byte order from where the
/// chain is read: a replacement staged on disk for one index transaction
/// ([`crate::writer::Replacement`]), or a rebuild worker's scratch index
/// ([`crate::writer::build::OpenChain`]). A part that extends a chain in
/// place goes to the index writer with the bytes it was read from
/// ([`crate::Batch::parse`]).
///
/// One sink serves one chain. A sink dropped without [`ParseSink::end`]
/// discards what it was given: nothing of the chain reaches the index.
pub trait ParseSink {
    /// The chain's next part.
    fn part(&mut self, part: ParsePart) -> crate::Result<()>;
    /// Bytes the parse decoded (inline images), given before the parts that
    /// name them.
    fn decoded(&mut self, batch: crate::staging::PreparedBatch) -> crate::Result<()>;
    /// The chain's parts are all given: a replacement is queued for its one
    /// index transaction, a rebuild's chain is closed into its worker's
    /// stream.
    fn end(self: Box<Self>) -> crate::Result<()>;
}

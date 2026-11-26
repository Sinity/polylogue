//! A chain's reading on its way to the index: the part a parser produced
//! for a byte range of one chain, where parts go, and the facts the
//! summaries take from them.

pub mod head;

pub use crate::writer::ParsePart;

/// Where a chain's parts go as they are parsed, in byte order from where the
/// chain is read: a replacement staged on disk for one index transaction
/// ([`crate::writer::Replacement`]), or a rebuild worker's scratch index
/// ([`crate::writer::build::OpenChain`]). A part that extends a chain in
/// place goes to the index writer with the bytes it was read from
/// ([`crate::Batch::parse`]).
pub trait ParseSink {
    /// The chain's next part.
    fn part(&mut self, part: &mut ParsePart) -> crate::Result<()>;
    /// Bytes the parse decoded (inline images), given before the parts that
    /// name them.
    fn decoded(&mut self, batch: crate::staging::PreparedBatch) -> crate::Result<()>;
}

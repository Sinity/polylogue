//! Derived products (architecture 5.6): each carries a version that
//! includes the data it uses, and is recomputed when that changes. Session
//! facts, model usage, git correlation, file edits, the work graph.

use polylogue_store::Reader;

pub trait Derivation {
    /// Changes when the derivation's code or its inputs' shape changes.
    const VERSION: u32;
    fn name(&self) -> &'static str;
    fn recompute(&self, reader: &Reader<'_>) -> Result<(), polylogue_store::StoreError>;
}

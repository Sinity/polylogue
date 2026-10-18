//! Publishing to Sinex in `primary` mode (architecture 7.1). Polylogue
//! merges routes before publishing; `have` queries decide what still needs
//! sending; a revision is checked whole against its manifest.

pub struct Publisher;

impl Publisher {
    pub fn publish_pending(&mut self) -> Result<u64, crate::EngineError> {
        // Architecture 7.1: placement or producer protocol, have queries.
        todo!()
    }
}

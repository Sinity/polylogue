//! Committed row counters from the single writer.
use super::*;
impl Engine {
    pub(super) fn flush_writes(&mut self) -> Result<WriteReport, EngineError> {
        Ok(self.writer.flush_rows()?)
    }
}

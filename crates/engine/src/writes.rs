//! Reconcile rejected writes with the same committed checkpoints used on startup.
use super::*;

impl Engine {
    pub(super) fn flush_writes(&mut self) -> Result<WriteReport, EngineError> {
        let report = self.writer.flush()?;
        self.failed_writes
            .extend(report.failed_inputs.iter().cloned());
        Ok(report)
    }

    /// Run after plans settle, so optimistic parser state cannot overwrite recovery.
    pub(super) fn restore_failed_writes(&mut self) -> Result<(), EngineError> {
        if self.failed_writes.is_empty() {
            return Ok(());
        }
        let reader = self.reads.reader()?;
        for (path, st) in &mut self.inputs {
            if !self.failed_writes.contains(&st.input_id) {
                continue;
            }
            let known = reader.known_input_at(&self.config.machine, &path.to_string_lossy())?;
            st.facts = known.as_ref().and_then(|k| k.facts);
            st.chain = known.as_ref().and_then(|k| k.chain);
            st.stored = known.as_ref().map_or(0, |k| k.stored);
            st.parsed_upto = known.as_ref().map_or(0, |k| k.parsed_upto);
            st.parser_version = known.as_ref().and_then(|k| k.parser_version);
            st.waiting = known.as_ref().is_some_and(|k| k.waiting);
            st.carry = None;
            st.tail = None;
            st.failed = None;
        }
        self.failed_writes.clear();
        Ok(())
    }
}

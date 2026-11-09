//! Resume only actual parser state at the confirmed source cursor.
use crate::*;
impl Engine {
    pub(crate) fn restored_parser_state(
        &self,
        reader: &polylogue_store::Reader<'_>,
        input: &str,
        chain: i64,
        cursor: u64,
        version: &str,
    ) -> Result<Option<polylogue_model::Json>, EngineError> {
        let Some(stored) = reader.parser_checkpoint(input, chain, version)? else {
            return Ok(None);
        };
        if stored.reading_version != version || stored.parsed_upto != cursor {
            return Ok(None);
        }
        let Some(checkpoint) = stored.checkpoint else {
            return Ok(None);
        };
        // Metadata validates the existing byte prefix without rereading it.
        reader.stored_chain(input, chain, cursor)?;
        let mut state = checkpoint
            .parser_state
            .unwrap_or(polylogue_model::Json::Null);
        without_context_changes(&mut state);
        Ok(Some(state))
    }
}

/// The parser state the loop keeps between an input's ranges. The keyed
/// context's changes reach the writer in the batch's checkpoint and the next
/// range reads the admitted context from the index, so the kept state leaves
/// them out; they hold every key a range wrote, many times the rest.
pub(crate) fn without_context_changes(state: &mut polylogue_model::Json) {
    if let Some(state) = state.as_object_mut() {
        state.remove(polylogue_model::parser_context::FIELD);
    }
}

#[cfg(test)]
mod tests;

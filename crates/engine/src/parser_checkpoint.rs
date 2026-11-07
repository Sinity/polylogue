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
        Ok(Some(
            checkpoint
                .parser_state
                .unwrap_or(polylogue_model::Json::Null),
        ))
    }
}

#[cfg(test)]
mod tests;

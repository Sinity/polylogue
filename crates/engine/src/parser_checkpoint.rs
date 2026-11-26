//! Resume only recorded parser state at the confirmed source cursor.
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
        let state = reader.chain_state(input, chain)?;
        if state.version() != Some(version) || state.parsed_to() != cursor {
            return Ok(None);
        }
        let Some(resume) = state.resume() else {
            return Ok(None);
        };
        // Metadata validates the existing byte prefix without rereading it.
        reader.stored_chain(input, chain, cursor)?;
        Ok(Some(
            resume.0.clone().unwrap_or(polylogue_model::Json::Null),
        ))
    }
}

#[cfg(test)]
mod tests;

//! Parser continuation belongs to the exact admitted cursor, never a source guess.
use super::*;
use polylogue_model::batch::ParserCheckpoint;

#[derive(Debug, Clone, PartialEq)]
pub struct StoredParserCheckpoint {
    pub reading_version: String,
    pub parsed_upto: u64,
    pub checkpoint: Option<ParserCheckpoint>,
}
impl Reader<'_> {
    pub fn parser_checkpoint(
        &self,
        input: &str,
        chain: i64,
        expected_version: &str,
    ) -> Result<Option<StoredParserCheckpoint>> {
        let row = self.conn()?.query_row(
            "SELECT reading_version,parsed_upto,checkpoint FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 AND p.chain=?2",
            params![input, chain], |r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<Vec<u8>>>(2)?))).optional()?;
        row.map(|(reading_version, cursor, bytes)| {
            let parsed_upto = u64::try_from(cursor)
                .map_err(|_| StoreError::Invalid("negative parser cursor".into()))?;
            let checkpoint: Option<ParserCheckpoint> = if reading_version == expected_version {
                bytes
                    .as_deref()
                    .map(crate::encoding::unpack_json)
                    .transpose()?
            } else {
                None
            };
            if let Some(checkpoint) = &checkpoint
                && !checkpoint.matches(&reading_version, parsed_upto)
            {
                return Err(StoreError::Invalid(
                    "stored parser checkpoint differs from its reading cursor".into(),
                ));
            }
            Ok(StoredParserCheckpoint {
                reading_version,
                parsed_upto,
                checkpoint,
            })
        })
        .transpose()
    }
}

#[cfg(test)]
mod tests;

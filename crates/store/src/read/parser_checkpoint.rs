//! Parser continuation belongs to the committed cursor, never a source guess.
use super::*;
use polylogue_model::batch::ParserCheckpoint;

/// The committed reading's status and cursor, without its opaque continuation.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredParserProgress {
    pub reading_version: String,
    pub parsed_upto: u64,
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub progress: polylogue_model::batch::ParseProgress,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredParserCheckpoint {
    pub reading_version: String,
    pub parsed_upto: u64,
    pub checkpoint: Option<ParserCheckpoint>,
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub progress: polylogue_model::batch::ParseProgress,
}
impl Reader<'_> {
    /// Completion checks need no parser-state bytes. Load the continuation only
    /// when the same reading will actually resume.
    pub fn parser_progress(&self, input: &str, chain: i64) -> Result<Option<StoredParserProgress>> {
        let row = self
            .conn()?
            .query_row(
                "SELECT reading_version,parsed_upto,status,at_length,reason,stopped_attempt
             FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 AND p.chain=?2",
                params![input, chain],
                |r| {
                    <(
                        String,
                        u64,
                        String,
                        Option<i64>,
                        Option<String>,
                        Option<Vec<u8>>,
                    )>::try_from(r)
                },
            )
            .optional()?;
        row.map(
            |(reading_version, parsed_upto, status, at_length, reason, attempt)| {
                Ok(StoredParserProgress {
                    reading_version,
                    parsed_upto,
                    stopped_attempt: attempt
                        .as_deref()
                        .map(crate::encoding::unpack_json)
                        .transpose()?,
                    progress: stored_parse_progress(&status, at_length, reason)?,
                })
            },
        )
        .transpose()
    }

    pub fn parser_checkpoint(
        &self,
        input: &str,
        chain: i64,
        expected_version: &str,
    ) -> Result<Option<StoredParserCheckpoint>> {
        #[cfg(feature = "faults")]
        crate::fault::check(
            "parser.checkpoint",
            &format!("{}:{input}", self.pool.archive.root.display()),
        )?;
        let row = self.conn()?.query_row(
            "SELECT reading_version,parsed_upto,checkpoint,status,at_length,reason,stopped_attempt FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 AND p.chain=?2",
            params![input, chain], |r| <(String, u64, Option<Vec<u8>>, String, Option<i64>, Option<String>, Option<Vec<u8>>)>::try_from(r)).optional()?;
        row.map(
            |(reading_version, cursor, bytes, status, at_length, reason, attempt)| {
                let parsed_upto = cursor;
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
                    return Err(StoreError::stored(
                        2,
                        "stored parser checkpoint differs from its reading cursor",
                    ));
                }
                Ok(StoredParserCheckpoint {
                    reading_version,
                    parsed_upto,
                    checkpoint,
                    stopped_attempt: attempt
                        .as_deref()
                        .map(crate::encoding::unpack_json)
                        .transpose()?,
                    progress: stored_parse_progress(&status, at_length, reason)?,
                })
            },
        )
        .transpose()
    }
}

#[cfg(test)]
mod tests;

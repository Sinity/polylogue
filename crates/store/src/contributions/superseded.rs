//! Retiring a preparation that a newer source state has replaced. Ready bodies
//! and stored inputs are never touched.
use super::writer::{Build, Source, check_source, preparation_digest};
use crate::{Error, Result, Writer, db, parser_state};
use polylogue_model::MessageId;
use rusqlite::OptionalExtension;

fn digest(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<[u8; 32]> {
    let bytes: Vec<u8> = row.get(column)?;
    bytes.try_into().map_err(|_| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Blob,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "preparation digest is not 32 bytes",
            )),
        )
    })
}

impl Writer {
    /// Does one retirement step for this Message's preparation if it was made
    /// from a different source state than `source`. Returns true if it did a
    /// step (call again), false if there is nothing to retire: no preparation,
    /// or one for this same source that can be resumed. `source` must be
    /// current, so a stale caller cannot retire newer work.
    pub fn retire_superseded_message_turn(
        &mut self,
        message: &MessageId,
        source: &Source<'_>,
        allowance: usize,
    ) -> Result<bool> {
        let required = 256usize
            .saturating_add(message.as_str().len())
            .saturating_add(source.predecessor.index_generation.len());
        if required > allowance {
            return Err(Error::ParserAdmission {
                resource: "superseded message preparation metadata bytes",
                required: required as u64,
            });
        }
        let path = self.root.join("index.db");
        check_source(self, source)?;
        let expected = preparation_digest(
            source.predecessor,
            source.revision,
            &source.logical_key,
            source.evidence_revision,
        );
        let row = db(&path, || {
            self._index.query_row(
                "SELECT w.pk,w.body_pk,c.pk,w.message_pk,w.session_pk,w.contributor_revision,w.removal_epoch,w.basis_digest,w.phase,c.input_pk,c.chain,c.source_state_generation,c.source_state_step,c.source_basis_digest FROM message_build w JOIN message m ON m.pk=w.message_pk JOIN message_body b ON b.pk=w.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE m.id=?1",
                [message.as_str()],
                |r| Ok((r.get::<_, String>(8)?, Build {
                    path: path.clone(),
                    pk: r.get(0)?, body: r.get(1)?, candidate: r.get(2)?,
                    message: r.get(3)?, session: r.get(4)?,
                    contributor_revision: r.get(5)?, removal_epoch: r.get(6)?,
                    digest: digest(r, 7)?,
                    source: parser_state::Token {
                        index_generation: source.predecessor.index_generation.clone(),
                        input_pk: r.get(9)?, chain: r.get(10)?,
                        generation: r.get(11)?, step: r.get(12)?, basis: digest(r, 13)?,
                    },
                })),
            ).optional()
        })?;
        let Some((phase, build)) = row else {
            return Ok(false);
        };
        if build.digest == expected && phase != "retiring" {
            return Ok(false);
        }
        if phase == "retiring" {
            self.retire_message_turn(&build)?;
        } else {
            self.retire_message_preparation(&build)?;
        }
        Ok(true)
    }
}

//! The source projection: the body one parser run produced for a logical
//! message key, plus a reference stored in a parser cell so the owner can find
//! it again after a restart. Choosing the message's selected body is separate.
use super::{blocks, bodies::BodyId, completion::PreparedProjection, writer};
use crate::{Error, Result, Writer, db, parser_state::Token};
use polylogue_model::{BlobHash, MessageId, SessionId};
use rusqlite::{OptionalExtension, params};

const NAMESPACE: &str = "message.source.v1";
const PREFIX: &[u8] = b"message-source-v1\0";
const FIXED: usize = PREFIX.len() + 8 * 4;
#[derive(Clone, Debug)]
pub struct SourceProjection {
    pub(super) body: BodyId,
    pub(super) logical: [u8; 32],
    pub(super) candidate: i64,
    pub(super) predecessor: Token,
    message: MessageId,
    evidence_revision: u64,
    hash: BlobHash,
    pub(super) reference: Vec<u8>,
}
impl SourceProjection {
    pub fn body(&self) -> &BodyId {
        &self.body
    }
    pub fn predecessor(&self) -> &Token {
        &self.predecessor
    }
    pub fn message(&self) -> &MessageId {
        &self.message
    }
    pub fn evidence_revision(&self) -> u64 {
        self.evidence_revision
    }
    pub fn version_hash(&self) -> BlobHash {
        self.hash
    }
    /// The caller's guard bytes stored with the reference.
    pub fn evidence_guard(&self) -> &[u8] {
        &self.reference[FIXED..]
    }
}
fn invalid(message: &str) -> Error {
    Error::Invariant(message.into())
}
fn reference(
    body: i64,
    candidate: i64,
    step: i64,
    guard: &[u8],
    allowance: usize,
) -> Result<Vec<u8>> {
    if body <= 0 || candidate <= 0 || step < 0 || guard.is_empty() {
        return Err(invalid("source projection reference identity or guard"));
    }
    let size = FIXED.checked_add(guard.len()).ok_or(Error::Range)?;
    if size > allowance {
        return Err(Error::ParserAdmission {
            resource: "source projection reference bytes",
            required: size as u64,
        });
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|e| Error::Io(std::io::Error::other(e)))?;
    bytes.extend_from_slice(PREFIX);
    for value in [body, candidate, step] {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes.extend_from_slice(
        &u64::try_from(guard.len())
            .map_err(|_| Error::Range)?
            .to_be_bytes(),
    );
    bytes.extend_from_slice(guard);
    Ok(bytes)
}
fn decode(bytes: &[u8]) -> Result<(i64, i64, i64, &[u8])> {
    if bytes.len() < FIXED || !bytes.starts_with(PREFIX) {
        return Err(invalid("source projection reference codec"));
    }
    let number = |offset: usize| -> Result<i64> {
        Ok(i64::from_be_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .map_err(|_| Error::Range)?,
        ))
    };
    let body = number(PREFIX.len())?;
    let candidate = number(PREFIX.len() + 8)?;
    let step = number(PREFIX.len() + 16)?;
    if body <= 0 || candidate <= 0 || step < 0 {
        return Err(invalid("source projection reference identity"));
    }
    let length = u64::from_be_bytes(
        bytes[PREFIX.len() + 24..FIXED]
            .try_into()
            .map_err(|_| Error::Range)?,
    );
    let length = usize::try_from(length).map_err(|_| Error::Range)?;
    if length == 0 || FIXED.checked_add(length) != Some(bytes.len()) {
        return Err(invalid("source projection reference guard length"));
    }
    Ok((body, candidate, step, &bytes[FIXED..]))
}
pub(super) fn read_current(
    connection: &rusqlite::Connection,
    path: &std::path::Path,
    token: &Token,
    logical_key: &[u8; 32],
    allowance: usize,
) -> Result<Option<SourceProjection>> {
    writer::source_head(connection, &path, token)?;
    let row: Option<(u64, Option<Vec<u8>>)> = db(&path, || {
        connection.query_row(
            "SELECT length(value),CASE WHEN length(value)<=?6 THEN value ELSE NULL END FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",
            params![token.input_pk,token.chain,token.generation,NAMESPACE,logical_key.as_slice(),writer::integer(allowance as u64).unwrap_or(i64::MAX)],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?))).optional()
    })?;
    let Some((size, bytes)) = row else {
        return Ok(None);
    };
    let bytes = bytes.ok_or(Error::ParserAdmission {
        resource: "source projection reference bytes",
        required: size,
    })?;
    let (body, candidate, step, _) = decode(&bytes)?;
    if step > token.step {
        return Err(invalid("source projection reference is ahead of parser"));
    }
    type Row = (
        i64,
        i64,
        u64,
        u64,
        Vec<u8>,
        Vec<u8>,
        u64,
        Option<String>,
        Option<String>,
        i64,
        i64,
        Vec<u8>,
    );
    let row: Option<Row> = db(&path, || {
        connection.query_row(
            "SELECT b.message_pk,b.session_pk,c.evidence_revision,c.revision,c.basis_digest,c.version_hash,length(CAST(m.id AS BLOB))+length(CAST(s.id AS BLOB)),CASE WHEN length(CAST(m.id AS BLOB))+length(CAST(s.id AS BLOB))<=?6 THEN m.id END,CASE WHEN length(CAST(m.id AS BLOB))+length(CAST(s.id AS BLOB))<=?6 THEN s.id END,c.source_state_generation,c.source_state_step,c.source_basis_digest FROM message_body b JOIN message_candidate c ON c.body_pk=b.pk AND c.pk=?2 JOIN message m ON m.pk=b.message_pk JOIN session s ON s.pk=b.session_pk WHERE b.pk=?1 AND b.ready=1 AND c.ready=1 AND c.eligible=1 AND c.input_pk=?3 AND c.chain=?4 AND c.logical_key=?5 AND b.removal_epoch=m.removal_epoch AND m.hidden=0",
            params![body,candidate,token.input_pk,token.chain,logical_key.as_slice(),writer::integer(allowance.saturating_sub(bytes.len()) as u64).unwrap_or(i64::MAX)],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?,crate::parser_state::unsigned(r,3)?,r.get(4)?,r.get(5)?,crate::parser_state::unsigned(r,6)?,r.get(7)?,r.get(8)?,r.get(9)?,r.get(10)?,r.get(11)?))).optional()
    })?;
    let (
        message,
        session,
        evidence,
        revision,
        basis,
        hash,
        identity_bytes,
        id,
        sid,
        generation,
        source_step,
        source_basis,
    ) = row.ok_or_else(|| invalid("source projection reference names no eligible candidate"))?;
    let (id, sid) = match (id, sid) {
        (Some(id), Some(sid)) => (id, sid),
        _ => {
            return Err(Error::ParserAdmission {
                resource: "source projection identity bytes",
                required: identity_bytes.saturating_add(bytes.len() as u64),
            });
        }
    };
    let parser = stored_parser_token(
        token,
        generation,
        source_step,
        source_basis,
        revision,
        logical_key,
        evidence,
        &basis,
    )?;
    if parser.generation != token.generation || parser.basis != token.basis || parser.step != step {
        return Err(Error::StaleMessage);
    }
    let message_id = MessageId::from_stored(
        id,
        SessionId::from_stored(sid).map_err(|e| invalid(&e.to_string()))?,
    )
    .map_err(|e| invalid(&e.to_string()))?;
    Ok(Some(SourceProjection {
        logical: *logical_key,
        body: BodyId {
            path: path.to_path_buf(),
            generation: token.index_generation.clone(),
            pk: body,
            message,
            session,
        },
        candidate,
        predecessor: parser,
        message: message_id,
        evidence_revision: evidence,
        hash: BlobHash::from_bytes(
            hash.try_into()
                .map_err(|_| invalid("candidate version_hash is not 32 bytes"))?,
        ),
        reference: bytes,
    }))
}

// Rebuild the parser token that produced a candidate from the candidate's own
// columns and check it against its digest. Parser cells are dropped on reparse,
// so they cannot be used for this.
fn stored_parser_token(
    scope: &Token,
    generation: i64,
    step: i64,
    source_basis: Vec<u8>,
    revision: u64,
    logical: &[u8; 32],
    evidence: u64,
    preparation_basis: &[u8],
) -> Result<Token> {
    if generation <= 0 || step < 0 {
        return Err(Error::StaleMessage);
    }
    let mut token = scope.clone();
    token.generation = generation;
    token.step = step;
    token.basis = source_basis
        .try_into()
        .map_err(|_| invalid("candidate source_basis_digest is not 32 bytes"))?;
    if preparation_basis != writer::basis(&token, revision, logical, evidence).as_slice() {
        return Err(Error::StaleMessage);
    }
    Ok(token)
}

/// Result of selecting a message body from its candidates. `Contested` means
/// the candidates differ or cannot be compared, and need a merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceSelection {
    Selected,
    AlreadySelected,
    Contested,
}
impl crate::parser_state::Snapshot {
    /// The current projection, checked against the message's selected body in
    /// this same snapshot.
    pub fn selected_source_projection(
        &self,
        logical: &[u8; 32],
        allowance: usize,
    ) -> Result<Option<SourceProjection>> {
        read_selected(&self.db, &self.path, &self.head.token, logical, allowance)
    }
}
pub(super) fn read_selected(
    connection: &rusqlite::Connection,
    path: &std::path::Path,
    token: &Token,
    logical: &[u8; 32],
    allowance: usize,
) -> Result<Option<SourceProjection>> {
    let source = read_current(connection, path, token, logical, allowance)?;
    let Some(source) = source else {
        return Ok(None);
    };
    let selected: bool = db(path, || {
        connection.query_row("SELECT EXISTS(SELECT 1 FROM message m JOIN message_body selected ON selected.pk=m.selected_body_pk JOIN message_candidate c ON c.pk=selected.preferred_candidate_pk JOIN message_body carrier ON carrier.pk=c.body_pk WHERE m.pk=?1 AND m.session_pk=?2 AND m.hidden=0 AND selected.ready=1 AND carrier.ready=1 AND c.eligible=1 AND selected.removal_epoch=m.removal_epoch AND carrier.removal_epoch=m.removal_epoch AND selected.block_root_pk IS carrier.block_root_pk AND selected.field_root_pk IS carrier.field_root_pk AND selected.basis_digest=carrier.basis_digest AND selected.version_hash=carrier.version_hash AND (carrier.pk=?3 OR EXISTS(SELECT 1 FROM message_body_compatibility own JOIN message_body_compatibility chosen ON chosen.digest=own.digest WHERE own.body_pk=?3 AND chosen.body_pk=carrier.pk)))",params![source.body.message,source.body.session,source.body.pk],|r|r.get(0))
    })?;
    if !selected {
        return Err(Error::StaleMessage);
    }
    Ok(Some(source))
}
impl Writer {
    /// Read the current projection for a logical key. The parser must be at
    /// `token`. A stored reference that no longer resolves is an error, not
    /// `None`.
    pub fn current_source_projection(
        &self,
        token: &Token,
        logical_key: &[u8; 32],
        allowance: usize,
    ) -> Result<Option<SourceProjection>> {
        read_current(
            &self._index,
            &self.root.join("index.db"),
            token,
            logical_key,
            allowance,
        )
    }
    /// Mark a prepared body ready and store its reference, in one transaction.
    /// With `previous`, the stored reference must still name that projection
    /// of the same logical key, and its candidate is withdrawn. Without it, only
    /// a candidate from an older parser generation may be withdrawn.
    pub fn publish_source_projection(
        &mut self,
        prepared: &PreparedProjection,
        previous: Option<&SourceProjection>,
        evidence_guard: &[u8],
        allowance: usize,
    ) -> Result<SourceProjection> {
        if evidence_guard.is_empty() {
            return Err(invalid("source projection requires an evidence-head guard"));
        }
        let build = &prepared.build;
        let path = self.root.join("index.db");
        let encoded = reference(
            build.body,
            build.candidate,
            build.source.step,
            evidence_guard,
            allowance,
        )?;
        if let Some(previous) = previous {
            if previous.body.path != path
                || previous.body.generation != build.source.index_generation
                || previous.body.message != build.message
                || previous.body.session != build.session
                || previous.predecessor.input_pk != build.source.input_pk
                || previous.predecessor.chain != build.source.chain
                || previous.predecessor.generation != build.source.generation
                || previous.predecessor.basis != build.source.basis
                || previous.predecessor.step > build.source.step
            {
                return Err(Error::StaleMessage);
            }
        }
        // Only an observed (growing) input may withdraw an older revision of a
        // logical key: on append with `previous`, or on reparse in a newer
        // parser generation whose old reference cell is gone. An immutable
        // input keeps each revision.
        let observed_source = {
            let id: String = db(&path, || {
                self._index.query_row(
                    "SELECT id FROM input_ref WHERE pk=?1",
                    [build.source.input_pk],
                    |r| r.get(0),
                )
            })?;
            let source =
                crate::observed::decode(&self.archive, &self.root.join("archive.db"), &id)?
                    .ok_or(Error::StaleMessage)?;
            source.observed() && source.input.chain == build.source.chain
        };
        let observed_append = observed_source && previous.is_some();
        let observed_reparse = observed_source && previous.is_none();
        let tx = db(&path, || self._index.transaction())?;
        writer::check_completed(&tx, &path, build)?;
        type Prepared = (
            Vec<u8>,
            i64,
            u64,
            Vec<u8>,
            bool,
            u64,
            i64,
            i64,
            Vec<u8>,
            Vec<u8>,
        );
        let row: Prepared = db(&path, || {
            tx.query_row("SELECT c.logical_key,c.input_pk,c.evidence_revision,c.version_hash,w.active_value_pk IS NULL AND w.active_origin_value_pk IS NULL AND w.active_block_pk IS NULL AND NOT EXISTS(SELECT 1 FROM message_build_frontier WHERE build_pk=w.pk),c.revision,c.source_state_generation,c.source_state_step,c.source_basis_digest,c.basis_digest FROM message_build w JOIN message_candidate c ON c.pk=?2 WHERE w.pk=?1 AND w.phase='rank' AND c.facts_ready=1",params![build.pk,build.candidate],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?,r.get(3)?,r.get(4)?,crate::parser_state::unsigned(r,5)?,r.get(6)?,r.get(7)?,r.get(8)?,r.get(9)?)))
        })?;
        let (
            logical,
            input,
            evidence,
            hash,
            complete,
            revision,
            generation,
            step,
            source_basis,
            preparation_basis,
        ) = row;
        if !complete || hash != prepared.hash.bytes().as_slice() {
            return Err(Error::MessagePreparationPending);
        }
        let logical: [u8; 32] = logical
            .try_into()
            .map_err(|_| invalid("source logical key length"))?;
        let prepared_token = stored_parser_token(
            &build.source,
            generation,
            step,
            source_basis,
            revision,
            &logical,
            evidence,
            &preparation_basis,
        )?;
        if prepared_token != build.source
            || preparation_basis != build.digest
            || input != build.source.input_pk
        {
            return Err(Error::StaleMessage);
        }
        type Existing = (i64, u64, u64, i64, i64, Vec<u8>, Vec<u8>);
        let existing: Option<Existing> = db(&path, || {
            tx.query_row("SELECT pk,revision,evidence_revision,source_state_generation,source_state_step,source_basis_digest,basis_digest FROM message_candidate INDEXED BY message_candidate_effective WHERE message_pk=?1 AND input_pk=?2 AND chain=?3 AND (revision=(SELECT revision FROM message_candidate WHERE pk=?4) OR (?6 AND pk=?7 AND revision<(SELECT revision FROM message_candidate WHERE pk=?4)) OR (?8 AND source_state_generation<?9 AND revision<(SELECT revision FROM message_candidate WHERE pk=?4))) AND logical_key=?5 AND eligible=1",params![build.message,input,build.source.chain,build.candidate,logical.as_slice(),observed_append,previous.map(|p|p.candidate).unwrap_or(0),observed_reparse,build.source.generation],|r|Ok((r.get(0)?,crate::parser_state::unsigned(r,1)?,crate::parser_state::unsigned(r,2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional()
        })?;
        let withdrawal = match existing {
            Some((candidate, revision, evidence, generation, step, basis, preparation)) => {
                let token = stored_parser_token(
                    &build.source,
                    generation,
                    step,
                    basis,
                    revision,
                    &logical,
                    evidence,
                    &preparation,
                )?;
                match previous {
                    Some(previous)
                        if previous.candidate == candidate && previous.predecessor == token => {}
                    None if token.generation < build.source.generation => {}
                    _ => return Err(Error::StaleMessage),
                }
                Some((candidate, token, preparation))
            }
            None if previous.is_none() => None,
            None => return Err(Error::StaleMessage),
        };
        let changed = if let Some(previous) = previous {
            db(&path, || {
                tx.execute("UPDATE parser_cell SET value=?6 WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5 AND value=?7",params![build.source.input_pk,build.source.chain,build.source.generation,NAMESPACE,logical.as_slice(),encoded,previous.reference])
            })?
        } else {
            db(&path, || {
                tx.execute("INSERT INTO parser_cell(input_pk,chain,state_generation,namespace,key,value) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT DO NOTHING",params![build.source.input_pk,build.source.chain,build.source.generation,NAMESPACE,logical.as_slice(),encoded])
            })?
        };
        if changed != 1 {
            return Err(Error::StaleMessage);
        }
        if let Some((candidate, token, preparation)) = withdrawal {
            if db(&path, || {
                tx.execute("UPDATE message_candidate SET eligible=0 WHERE pk=?1 AND eligible=1 AND source_state_generation=?2 AND source_state_step=?3 AND source_basis_digest=?4 AND basis_digest=?5",params![candidate,token.generation,token.step,token.basis.as_slice(),preparation])
            })? != 1
            {
                return Err(Error::StaleMessage);
            }
        }
        if db(&path, || {
            tx.execute(
                "UPDATE message_body SET version_hash=?2,ready=1 WHERE pk=?1 AND ready=0",
                params![build.body, prepared.hash.bytes().as_slice()],
            )
        })? != 1
        {
            return Err(Error::StaleMessage);
        }
        if db(&path, || {
            tx.execute("UPDATE message_candidate SET body_ready=1,ready=1,eligible=1 WHERE pk=?1 AND ready=0 AND eligible=0",[build.candidate])
        })? != 1
        {
            return Err(Error::StaleMessage);
        }
        blocks::discard_cursor(&tx, &path, build)?;
        if db(&path, || {
            tx.execute("DELETE FROM message_build WHERE pk=?1", [build.pk])
        })? != 1
        {
            return Err(Error::StaleMessage);
        }
        let output = read_current(&tx, &path, &build.source, &logical, allowance)?
            .ok_or_else(|| invalid("published source projection missing"))?;
        db(&path, || tx.commit())?;
        Ok(output)
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    #[test]
    fn reference_round_trips_and_rejects_truncation() {
        let bytes = reference(3, 7, 0, b"guard\0bytes", 4096).unwrap();
        assert_eq!(
            decode(&bytes).unwrap(),
            (3, 7, 0, b"guard\0bytes".as_slice())
        );
        for cut in 0..bytes.len() {
            assert!(decode(&bytes[..cut]).is_err(), "cut {cut}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(decode(&extra).is_err());
        assert!(reference(0, 7, 0, b"guard", 4096).is_err());
        assert!(reference(3, 7, -1, b"guard", 4096).is_err());
        assert!(reference(3, 7, 0, b"", 4096).is_err());
        assert!(matches!(
            reference(3, 7, 0, b"guard", FIXED),
            Err(Error::ParserAdmission { .. })
        ));
    }
}

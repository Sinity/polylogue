//! Current source projection, scoped to one committed parser incarnation.
//! This selects no canonical Message. Its fixed reference lets the owner resume
//! after certificate publication without treating a scheduler cut as a version.
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
    /// Provider-owned fixed evidence-head encoding, not evidence payload bytes.
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
    ) = row.ok_or_else(|| invalid("source projection lost its eligible certificate"))?;
    let (id, sid) = match (id, sid) {
        (Some(id), Some(sid)) => (id, sid),
        _ => {
            return Err(Error::ParserAdmission {
                resource: "source projection identity bytes",
                required: identity_bytes.saturating_add(bytes.len() as u64),
            });
        }
    };
    let predecessor = certified_predecessor(
        token,
        generation,
        source_step,
        source_basis,
        revision,
        logical_key,
        evidence,
        &basis,
    )?;
    if predecessor.generation != token.generation
        || predecessor.basis != token.basis
        || predecessor.step != step
    {
        return Err(Error::MessagePredecessorChanged);
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
        predecessor,
        message: message_id,
        evidence_revision: evidence,
        hash: BlobHash::from_bytes(
            hash.try_into()
                .map_err(|_| invalid("source certificate hash length"))?,
        ),
        reference: bytes,
    }))
}

// Recover the exact source predecessor from its immutable certificate rather
// than depending on disposable parser cells or guessing from parser_version.
fn certified_predecessor(
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
        return Err(Error::MessagePredecessorChanged);
    }
    let mut predecessor = scope.clone();
    predecessor.generation = generation;
    predecessor.step = step;
    predecessor.basis = source_basis
        .try_into()
        .map_err(|_| invalid("source certificate basis length"))?;
    if preparation_basis != writer::basis(&predecessor, revision, logical, evidence).as_slice() {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(predecessor)
}

/// A complete compatible source selection is a bounded equality slice.
/// Nonidentical or unsupported candidates retain explicit merge work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceSelection {
    Selected,
    AlreadySelected,
    Contested,
}
impl crate::parser_state::Snapshot {
    /// Source certification and selected-body membership share this pinned view.
    pub fn selected_source_projection(
        &self,
        logical: &[u8; 32],
        allowance: usize,
    ) -> Result<Option<SourceProjection>> {
        let source = read_current(&self.db, &self.path, &self.head.token, logical, allowance)?;
        let Some(source) = source else {
            return Ok(None);
        };
        let selected: bool = db(&self.path, || {
            self.db.query_row("SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND session_pk=?2 AND hidden=0 AND (selected_body_pk=?3 OR EXISTS(SELECT 1 FROM message_body_compatibility own JOIN message_body_compatibility chosen ON chosen.digest=own.digest JOIN message_body b ON b.pk=chosen.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE own.body_pk=?3 AND chosen.body_pk=message.selected_body_pk AND b.ready=1 AND c.eligible=1 AND b.removal_epoch=message.removal_epoch)))",params![source.body.message,source.body.session,source.body.pk],|r|r.get(0))
        })?;
        if !selected {
            return Err(Error::MessagePredecessorChanged);
        }
        Ok(Some(source))
    }
}
impl Writer {
    /// Read one fixed current reference and its certificate. The parser must be
    /// at the caller's exact predecessor with no partially applied work. A
    /// reference cannot silently become an absence after corruption/retirement.
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
    /// Publish a source certificate and its restart reference atomically. Only
    /// the identified previous scheduling cut of this same logical candidate is
    /// withdrawn. A reparse may recover its strictly older parser incarnation
    /// from verified certificate coordinates; absence within this generation
    /// never grants that authority.
    /// This does not choose across revisions/chains/inputs, publish a canonical
    /// head, consume Effects, index search, or claim complete history coverage.
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
                return Err(Error::MessagePredecessorChanged);
            }
        }
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
        let prepared_predecessor = certified_predecessor(
            &build.source,
            generation,
            step,
            source_basis,
            revision,
            &logical,
            evidence,
            &preparation_basis,
        )?;
        if prepared_predecessor != build.source
            || preparation_basis != build.digest
            || input != build.source.input_pk
        {
            return Err(Error::MessagePredecessorChanged);
        }
        type Existing = (i64, u64, u64, i64, i64, Vec<u8>, Vec<u8>);
        let existing: Option<Existing> = db(&path, || {
            tx.query_row("SELECT pk,revision,evidence_revision,source_state_generation,source_state_step,source_basis_digest,basis_digest FROM message_candidate INDEXED BY message_candidate_effective WHERE message_pk=?1 AND input_pk=?2 AND chain=?3 AND revision=(SELECT revision FROM message_candidate WHERE pk=?4) AND logical_key=?5 AND eligible=1",params![build.message,input,build.source.chain,build.candidate,logical.as_slice()],|r|Ok((r.get(0)?,crate::parser_state::unsigned(r,1)?,crate::parser_state::unsigned(r,2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional()
        })?;
        let withdrawal = match existing {
            Some((candidate, revision, evidence, generation, step, basis, preparation)) => {
                let predecessor = certified_predecessor(
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
                        if previous.candidate == candidate
                            && previous.predecessor == predecessor => {}
                    None if predecessor.generation < build.source.generation => {}
                    _ => return Err(Error::MessagePredecessorChanged),
                }
                Some((candidate, predecessor, preparation))
            }
            None if previous.is_none() => None,
            None => return Err(Error::MessagePredecessorChanged),
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
            return Err(Error::MessagePredecessorChanged);
        }
        if let Some((candidate, predecessor, preparation)) = withdrawal {
            if db(&path, || {
                tx.execute("UPDATE message_candidate SET eligible=0 WHERE pk=?1 AND eligible=1 AND source_state_generation=?2 AND source_state_step=?3 AND source_basis_digest=?4 AND basis_digest=?5",params![candidate,predecessor.generation,predecessor.step,predecessor.basis.as_slice(),preparation])
            })? != 1
            {
                return Err(Error::MessagePredecessorChanged);
            }
        }
        if db(&path, || {
            tx.execute(
                "UPDATE message_body SET version_hash=?2,ready=1 WHERE pk=?1 AND ready=0",
                params![build.body, prepared.hash.bytes().as_slice()],
            )
        })? != 1
        {
            return Err(Error::MessagePredecessorChanged);
        }
        if db(&path, || {
            tx.execute("UPDATE message_candidate SET body_ready=1,ready=1,eligible=1 WHERE pk=?1 AND ready=0 AND eligible=0",[build.candidate])
        })? != 1
        {
            return Err(Error::MessagePredecessorChanged);
        }
        blocks::discard_cursor(&tx, &path, build)?;
        if db(&path, || {
            tx.execute("DELETE FROM message_build WHERE pk=?1", [build.pk])
        })? != 1
        {
            return Err(Error::MessagePredecessorChanged);
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
    fn reference_frames_guard_length_and_refuses_every_truncation() {
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

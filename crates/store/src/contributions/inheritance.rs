//! Reuse a certified source body's immutable roots for the next effective state
//! of that SAME format-declared candidate. This is not donor filling or election.
use super::{
    blocks,
    bodies::BodyId,
    directory as d,
    writer::{self, Build},
};
use crate::{Error, Result, Writer, db};
use polylogue_model::BlobHash;
use rusqlite::{OptionalExtension, params};

/// The prior source certificate's digest is reusable only when the provider's
/// selected evidence revision is unchanged. The source owner must still reject
/// incomplete evidence/normalization and validate its pinned current head.
#[derive(Clone, Debug)]
pub struct InheritedProjection {
    pub evidence_revision: u64,
    pub version_hash: BlobHash,
}
impl Writer {
    /// Called once at the beginning of a new preparation. The owner selects the
    /// prior effective source body, not a filled canonical body or arbitrary
    /// historical donor. Dirty source units then replace/remove their own paths.
    /// At most three root leases and scalar metadata change; descendants remain
    /// shared. The source cursor commits in the same transaction.
    pub fn inherit_message_projection(
        &mut self,
        build: &Build,
        previous: &BodyId,
        previous_source: &crate::parser_state::Token,
        step: &blocks::ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<InheritedProjection> {
        blocks::admit(step, allowance)?;
        let path = self.root.join("index.db");
        if previous.path != path
            || previous.generation != build.source.index_generation
            || previous.message != build.message
            || previous.session != build.session
            || previous_source.index_generation != build.source.index_generation
            || previous_source.input_pk != build.source.input_pk
            || previous_source.chain != build.source.chain
            || previous_source.generation != build.source.generation
            || previous_source.basis != build.source.basis
            || previous_source.step > build.source.step
        {
            return Err(Error::StaleMessage);
        }
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let empty: bool = db(&path, || {
            tx.query_row(
            "SELECT w.source_cursor=0 AND w.active_value_pk IS NULL AND w.active_block_pk IS NULL AND b.block_root_pk IS NULL AND b.field_root_pk IS NULL AND b.operation_root_pk IS NULL FROM message_build w JOIN message_body b ON b.pk=w.body_pk WHERE w.pk=?1",
            [build.pk], |r|r.get(0))
        })?;
        if !empty || step.expected_sequence != 0 {
            return Err(Error::StaleMessage);
        }
        type Roots = (
            Option<i64>,
            Option<i64>,
            Option<i64>,
            u64,
            Vec<u8>,
            Vec<u8>,
            Vec<u8>,
            u64,
        );
        let roots: Option<Roots> = db(&path, || {
            tx.query_row(
            "SELECT old.block_root_pk,old.field_root_pk,old.operation_root_pk,c.evidence_revision,c.version_hash,c.basis_digest,c.logical_key,c.revision FROM message_body old JOIN message_candidate c ON c.pk=old.preferred_candidate_pk AND c.body_pk=old.pk JOIN message_candidate n ON n.pk=?2 JOIN message_body fresh ON fresh.pk=?3 WHERE old.pk=?1 AND old.message_pk=?4 AND old.session_pk=?5 AND old.ready=1 AND c.ready=1 AND c.eligible=1 AND c.input_pk=n.input_pk AND c.chain=n.chain AND c.revision=n.revision AND c.logical_key=n.logical_key AND c.parser_version=n.parser_version AND c.route=n.route AND c.evidence_revision<=n.evidence_revision AND old.layout=fresh.layout AND old.removal_epoch=fresh.removal_epoch",
            params![previous.pk,build.candidate,build.body,build.message,build.session],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,crate::parser_state::unsigned(r,3)?,r.get(4)?,r.get(5)?,r.get(6)?,crate::parser_state::unsigned(r,7)?))).optional()
        })?;
        let (blocks, fields, operations, evidence_revision, hash, old_basis, logical, revision) =
            roots.ok_or(Error::StaleMessage)?;
        let logical: [u8; 32] = logical
            .try_into()
            .map_err(|_| Error::Invariant("source candidate logical key length".into()))?;
        if old_basis != writer::basis(previous_source, revision, &logical, evidence_revision) {
            return Err(Error::StaleMessage);
        }
        let hash: [u8; 32] = hash
            .try_into()
            .map_err(|_| Error::Invariant("source certificate digest length".into()))?;
        let owner = d::Owner {
            message: build.message,
            session: build.session,
        };
        for (root, kind) in [
            (blocks, d::Kind::Groups),
            (fields, d::Kind::Fields),
            (operations, d::Kind::Operations),
        ] {
            d::summary(&tx, &path, owner, kind, root)?;
            if let Some(root) = root {
                d::acquire(&tx, &path, owner, root)?;
            }
        }
        db(&path, || {
            tx.execute("UPDATE message_body SET block_root_pk=?2,field_root_pk=?3,operation_root_pk=?4 WHERE pk=?1 AND ready=0",params![build.body,blocks,fields,operations])
        })?;
        blocks::commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(InheritedProjection {
            evidence_revision,
            version_hash: BlobHash::from_bytes(hash),
        })
    }
}

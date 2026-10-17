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
            return Err(Error::MessagePredecessorChanged);
        }
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let empty: bool = db(&path, || {
            tx.query_row(
            "SELECT w.source_cursor=0 AND w.active_value_pk IS NULL AND w.active_block_pk IS NULL AND b.block_root_pk IS NULL AND b.field_root_pk IS NULL AND b.operation_root_pk IS NULL FROM message_build w JOIN message_body b ON b.pk=w.body_pk WHERE w.pk=?1",
            [build.pk], |r|r.get(0))
        })?;
        if !empty || step.expected_sequence != 0 {
            return Err(Error::MessagePredecessorChanged);
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
            roots.ok_or(Error::MessagePredecessorChanged)?;
        let logical: [u8; 32] = logical
            .try_into()
            .map_err(|_| Error::Invariant("source candidate logical key length".into()))?;
        if old_basis != writer::basis(previous_source, revision, &logical, evidence_revision) {
            return Err(Error::MessagePredecessorChanged);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImportObservation, InputFormat, parser_state as p, values};
    use polylogue_model::{
        merge::{FinishState, MessageField, RouteKind, Slot},
        *,
    };
    fn prepare(
        w: &mut Writer,
        head: &p::Head,
        session: &SessionId,
        message: &MessageId,
        evidence: u64,
    ) -> super::super::Preparation {
        w.prepare_message(
            super::super::Identity {
                session,
                origin: &Origin::new("synthetic").unwrap(),
                native_session: "session",
                native_session_encoding: None,
                session_kind: SessionKind::Main,
                message,
                sequence: 0,
                role: Role::Assistant,
                written_by: WrittenBy::Model,
            },
            super::super::Source {
                predecessor: &head.token,
                revision: 1,
                evidence_revision: evidence,
                logical_key: [3; 32],
                route: RouteKind::Native,
            },
            4096,
        )
        .unwrap()
    }
    fn value(w: &mut Writer, build: &Build, data: &[u8]) {
        let head = w
            .begin_message_value(build, values::Codec::Json, data.len() as u64)
            .unwrap();
        w.append_value_chunk(&head.id, 0, data, data.len()).unwrap();
        let reader = w.read_handle().unwrap();
        let mut proof = w.validate_value(&head.id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, 64).unwrap();
        }
        w.seal_message_value(build, &proof).unwrap();
    }
    #[test]
    fn inheritance_shares_paths_without_copying_facts_and_rejects_replay_or_wrong_basis() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut w = Writer::open(&root).unwrap();
        let hash = w.put_file_blob(b"{}\n").unwrap();
        let input = w
            .retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &[],
                &ImportObservation {
                    id: "synthetic-observation",
                    machine: "synthetic",
                    host: "synthetic",
                    path: "/synthetic/log.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = w
            .begin_parser(&p::Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: 3,
                digest: [7; 32],
                acquired_upto: input.length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let origin = Origin::new("synthetic").unwrap();
        let session = SessionId::new(&origin, "session").unwrap();
        let message = MessageId::new(session.clone(), "message").unwrap();
        let old = prepare(&mut w, &head, &session, &message, 1).build;
        let slot = Slot::Message(MessageField::At);
        value(&mut w, &old, b"1");
        w.write_message_field(
            &old,
            &slot,
            None,
            true,
            &blocks::ProjectionStep {
                expected_sequence: 0,
                next_cursor: b"field",
            },
            4096,
        )
        .unwrap();
        w.finish_message_projection(
            &old,
            &blocks::ProjectionStep {
                expected_sequence: 1,
                next_cursor: b"finished",
            },
            BlobHash::of(b"synthetic-evidence"),
            FinishState::Unknown,
            4096,
        )
        .unwrap();
        // This test supplies readiness only. It does not qualify election or the
        // provider mapping, which remain the canonical owner's obligations.
        w._index.execute_batch("BEGIN; UPDATE message_body SET ready=1,version_hash=(SELECT version_hash FROM message_candidate WHERE pk=message_body.preferred_candidate_pk); UPDATE message_candidate SET body_ready=1,ready=1,eligible=1; UPDATE message SET selected_body_pk=(SELECT pk FROM message_body); DELETE FROM message_build; COMMIT;").unwrap();
        let reader = w.read_handle().unwrap();
        let pinned = reader.selected_message(&message).unwrap().unwrap();
        let path = root.join("index.db");
        let next = p::write_step(
            &mut w._index,
            &path,
            &head,
            p::Progress {
                inspected_upto: 0,
                parsed_upto: 0,
                phase: p::Phase::Parsing,
                finalization_pending: true,
                counts: Default::default(),
            },
            &[],
            p::Budget {
                cells: 0,
                encoded_bytes: 0,
            },
            |_| Ok(()),
        )
        .unwrap();
        let new = prepare(&mut w, &next, &session, &message, 2).build;
        let before: i64 = w
            ._index
            .query_row("SELECT count(*) FROM message_field_fact", [], |r| r.get(0))
            .unwrap();
        let mut wrong = head.token.clone();
        wrong.generation += 1;
        let step = blocks::ProjectionStep {
            expected_sequence: 0,
            next_cursor: b"inherited",
        };
        assert!(matches!(
            w.inherit_message_projection(&new, &old.body_id(), &wrong, &step, 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        w._index.execute_batch("CREATE TRIGGER fail_cursor BEFORE UPDATE OF source_cursor ON message_build BEGIN SELECT RAISE(ABORT,'synthetic cursor failure'); END;").unwrap();
        assert!(
            w.inherit_message_projection(&new, &old.body_id(), &head.token, &step, 4096)
                .is_err()
        );
        assert_eq!(w.message_projection(&new, 4096).unwrap().sequence, 0);
        w._index.execute_batch("DROP TRIGGER fail_cursor").unwrap();
        let inherited = w
            .inherit_message_projection(&new, &old.body_id(), &head.token, &step, 4096)
            .unwrap();
        assert_eq!(inherited.evidence_revision, 1);
        assert_eq!(inherited.version_hash, BlobHash::of(b"synthetic-evidence"));
        assert_eq!(
            w._index
                .query_row::<i64, _, _>("SELECT count(*) FROM message_field_fact", [], |r| r.get(0))
                .unwrap(),
            before
        );
        assert!(
            w.inherit_message_projection(&new, &old.body_id(), &head.token, &step, 4096)
                .is_err()
        );
        value(&mut w, &new, b"2");
        w.set_message_field(
            &new,
            &slot,
            None,
            true,
            &blocks::ProjectionStep {
                expected_sequence: 1,
                next_cursor: b"changed",
            },
            4096,
        )
        .unwrap();
        match pinned.field(&slot, 4096).unwrap().unwrap() {
            super::super::FieldRead::Value(field) => {
                assert_eq!(field.range(0, 64).unwrap().bytes, b"1")
            }
            _ => panic!("unexpected conflict"),
        }
        // Invalidating the new preparation cannot revoke the old selected graph.
        w.retire_message_preparation(&new).unwrap();
        while w.retire_message_turn(&new).unwrap() != super::super::Retirement::Complete {}
        assert!(pinned.field(&slot, 4096).unwrap().is_some());
    }
}

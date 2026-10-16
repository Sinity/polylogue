//! Durable preparation control. These APIs never select a partially built body,
//! make a candidate eligible, or certify parser completion. Projection, ranking,
//! evidence verification and atomic publication still have separate obligations.
use crate::{Error, Result, Writer, db, parser_state, values};
use polylogue_model::merge::RouteKind;
use polylogue_model::{MessageId, Origin, Role, SessionId, SessionKind, WrittenBy};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};

/// Identity metadata is admitted before any transaction. Session relations are
/// supplied explicitly; opaque Message IDs are never split to recover a parent.
pub struct Identity<'a> {
    pub session: &'a SessionId,
    pub origin: &'a Origin,
    pub native_session: &'a str,
    pub session_kind: SessionKind,
    pub message: &'a MessageId,
    pub sequence: u64,
    pub role: Role,
    pub written_by: WrittenBy,
}
/// One format-declared logical candidate in one retained archive revision.
/// Scheduling cuts do not create a logical key or a new revision.
pub struct Source<'a> {
    pub predecessor: &'a parser_state::Token,
    pub revision: u64,
    pub evidence_revision: u64,
    pub logical_key: [u8; 32],
    pub route: RouteKind,
}
/// Only committed preparation IDs escape. The source predecessor, archive
/// revision and local index path/incarnation are checked at every mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Build {
    pub(super) path: PathBuf,
    pub(super) source: parser_state::Token,
    pub(super) pk: i64,
    pub(super) body: i64,
    pub(super) candidate: i64,
    pub(super) message: i64,
    pub(super) session: i64,
    pub(super) contributor_revision: i64,
    pub(super) removal_epoch: i64,
    pub(super) digest: [u8; 32],
}
impl Build {
    pub fn body_pk(&self) -> i64 {
        self.body
    }
    pub fn candidate_pk(&self) -> i64 {
        self.candidate
    }
    pub fn predecessor(&self) -> &parser_state::Token {
        &self.source
    }
}
#[derive(Clone, Debug)]
pub struct Preparation {
    pub build: Build,
    pub active_value: Option<values::Head>,
    /// Independent descriptor producer; ready JSON is not semantic validation.
    pub active_origin_value: Option<values::Head>,
    pub projection: Option<super::completion::PreparedProjection>,
}
fn bad(message: &str) -> Error {
    Error::Invariant(message.into())
}
pub(super) fn integer(n: u64) -> Result<i64> {
    i64::try_from(n).map_err(|_| Error::Range)
}
pub(super) fn basis(
    source: &parser_state::Token,
    revision: u64,
    logical: &[u8; 32],
    evidence: u64,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"polylogue.message-preparation.v1\0");
    h.update(&(source.index_generation.len() as u64).to_be_bytes());
    h.update(source.index_generation.as_bytes());
    for n in [
        source.input_pk,
        source.chain,
        source.generation,
        source.step,
    ] {
        h.update(&n.to_be_bytes());
    }
    h.update(&source.basis);
    h.update(&revision.to_be_bytes());
    h.update(logical);
    h.update(&evidence.to_be_bytes());
    *h.finalize().as_bytes()
}
pub(super) fn source_head(
    conn: &Connection,
    path: &Path,
    source: &parser_state::Token,
) -> Result<parser_state::Head> {
    let head = parser_state::head(conn, path, source.input_pk, source.chain)?
        .ok_or(Error::MessagePredecessorChanged)?;
    if head.token != *source || head.status == "retired" {
        return Err(Error::MessagePredecessorChanged);
    }
    let pending: bool = db(path, || {
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM parse_work WHERE input_pk=?1 AND chain=?2)",
            params![source.input_pk, source.chain],
            |r| r.get(0),
        )
    })?;
    if pending {
        return Err(Error::ParserWorkPending);
    }
    let excised: Option<bool> = db(path, || {
        conn.query_row(
            "SELECT excised FROM input_ref WHERE pk=?1",
            [source.input_pk],
            |r| r.get(0),
        )
        .optional()
    })?;
    if excised != Some(false) {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(head)
}
pub(super) fn check(conn: &Connection, path: &Path, build: &Build) -> Result<()> {
    check_at(conn, path, build, true)
}
pub(super) fn check_completed(conn: &Connection, path: &Path, build: &Build) -> Result<()> {
    check_at(conn, path, build, false)
}
fn check_at(conn: &Connection, path: &Path, build: &Build, mutating: bool) -> Result<()> {
    if path != build.path {
        return Err(Error::MessagePredecessorChanged);
    }
    source_head(conn, path, &build.source)?;
    let row=db(path,||conn.query_row("SELECT b.phase,m.hidden,m.contributor_revision,m.removal_epoch FROM message_build b JOIN message m ON m.pk=b.message_pk JOIN message_body body ON body.pk=b.body_pk JOIN message_candidate c ON c.pk=body.preferred_candidate_pk WHERE body.ready=0 AND c.ready=0 AND c.eligible=0 AND b.pk=?1 AND b.body_pk=?2 AND b.message_pk=?3 AND b.session_pk=?4 AND b.basis_digest=?5 AND b.contributor_revision=?6 AND b.removal_epoch=?7",params![build.pk,build.body,build.message,build.session,build.digest.as_slice(),build.contributor_revision,build.removal_epoch],|r|Ok((r.get::<_,String>(0)?,r.get::<_,bool>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?))).optional())?.ok_or(Error::MessagePredecessorChanged)?;
    if row.1
        || row.2 != build.contributor_revision
        || row.3 != build.removal_epoch
        || !matches!(row.0.as_str(), "project" | "rank")
        || (mutating && row.0 != "project")
    {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(())
}
impl Writer {
    /// Reserve one pending body/candidate. Repeating the identical pinned
    /// request resumes it; a different request must retire the old preparation.
    /// Existing selected bodies and source eligibility are untouched.
    pub fn prepare_message(
        &mut self,
        identity: Identity<'_>,
        source: Source<'_>,
        allowance: usize,
    ) -> Result<Preparation> {
        if identity.message.session() != identity.session || identity.native_session.is_empty() {
            return Err(bad("invalid preparation identity relation"));
        }
        let mut bytes = 256usize;
        for text in [
            identity.session.as_str(),
            identity.origin.as_str(),
            identity.native_session,
            identity.message.as_str(),
            source.predecessor.index_generation.as_str(),
        ] {
            bytes = bytes.checked_add(text.len()).ok_or(Error::Range)?;
        }
        if bytes > allowance {
            return Err(Error::ParserAdmission {
                resource: "message preparation metadata bytes",
                required: bytes as u64,
            });
        }
        let sequence = integer(identity.sequence)?;
        let revision = integer(source.revision)?;
        let evidence = integer(source.evidence_revision)?;
        if revision <= 0 {
            return Err(Error::Range);
        }
        let path = self.root.join("index.db");
        let head = source_head(&self._index, &path, source.predecessor)?;
        let input: String = db(&path, || {
            self._index.query_row(
                "SELECT id FROM input_ref WHERE pk=?1",
                [source.predecessor.input_pk],
                |r| r.get(0),
            )
        })?;
        let archive = self.root.join("archive.db");
        let length: Option<u64> = db(&archive, || {
            self.archive
                .query_row(
                    "SELECT length FROM revision WHERE input=?1 AND number=?2 AND chain=?3",
                    params![input, revision, source.predecessor.chain],
                    |r| parser_state::unsigned(r, 0),
                )
                .optional()
        })?;
        let length = length.ok_or(Error::MessagePredecessorChanged)?;
        if head.acquired_upto > length {
            return Err(Error::MessagePredecessorChanged);
        }
        let length_i = integer(length)?;
        let digest = basis(
            source.predecessor,
            source.revision,
            &source.logical_key,
            source.evidence_revision,
        );
        let tx = db(&path, || self._index.transaction())?;
        source_head(&tx, &path, source.predecessor)?;
        let kind = match identity.session_kind {
            SessionKind::Main => "main",
            SessionKind::Subagent => "subagent",
            SessionKind::Chat => "chat",
        };
        db(&path, || {
            tx.execute("INSERT INTO session(id,origin,native_id,kind,changed) VALUES(?1,?2,?3,?4,0) ON CONFLICT(id) DO NOTHING",params![identity.session.as_str(),identity.origin.as_str(),identity.native_session,kind])
        })?;
        let (session, origin, native, stored_kind): (i64, String, String, String) =
            db(&path, || {
                tx.query_row(
                    "SELECT pk,origin,native_id,kind FROM session WHERE id=?1",
                    [identity.session.as_str()],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
            })?;
        if origin != identity.origin.as_str()
            || native != identity.native_session
            || stored_kind != kind
        {
            return Err(bad("stored Session identity disagrees with preparation"));
        }
        db(&path, || {
            tx.execute("INSERT INTO message(id,session_pk,seq) VALUES(?1,?2,?3) ON CONFLICT(id) DO NOTHING",params![identity.message.as_str(),session,sequence])
        })?;
        let (message, owner, contributor, removal, hidden): (i64, i64, i64, i64, bool) = db(
            &path,
            || {
                tx.query_row("SELECT pk,session_pk,contributor_revision,removal_epoch,hidden FROM message WHERE id=?1",[identity.message.as_str()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))
            },
        )?;
        if owner != session || hidden {
            return Err(Error::MessagePredecessorChanged);
        }
        let existing: Option<(i64, i64, Vec<u8>, String)> = db(&path, || {
            tx.query_row(
                "SELECT pk,body_pk,basis_digest,phase FROM message_build WHERE message_pk=?1",
                [message],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()
        })?;
        let (pk, body, candidate) = if let Some((pk, body, old, phase)) = existing {
            if old != digest || phase == "retiring" {
                return Err(Error::MessagePreparationPending);
            }
            let candidate = db(&path, || {
                tx.query_row(
                    "SELECT preferred_candidate_pk FROM message_body WHERE pk=?1",
                    [body],
                    |r| r.get(0),
                )
            })?;
            (pk, body, candidate)
        } else {
            let role = match identity.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };
            let by = match identity.written_by {
                WrittenBy::Human => "human",
                WrittenBy::Model => "model",
                WrittenBy::Harness => "harness",
                WrittenBy::Unknown => "unknown",
            };
            db(&path, || {
                tx.execute("INSERT INTO message_body(message_pk,session_pk,basis_digest,contributor_revision,removal_epoch,layout,role,written_by,input_pk,source_chain,parser_version,source_sequence) VALUES(?1,?2,?3,?4,?5,'flat',?6,?7,?8,?9,?10,?11)",params![message,session,digest.as_slice(),contributor,removal,role,by,source.predecessor.input_pk,source.predecessor.chain,head.parser_version,sequence])
            })?;
            let body = tx.last_insert_rowid();
            let route = match source.route {
                RouteKind::PageDom => 0,
                RouteKind::AccountExport => 1,
                RouteKind::Native => 2,
            };
            db(&path, || {
                tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,body_pk,finish,route) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,1,?15)",params![message,session,source.predecessor.input_pk,source.predecessor.chain,revision,length_i,evidence,source.logical_key.as_slice(),head.parser_version,source.predecessor.generation,source.predecessor.step,source.predecessor.basis.as_slice(),digest.as_slice(),body,route])
            })?;
            let candidate = tx.last_insert_rowid();
            db(&path, || {
                tx.execute(
                    "UPDATE message_body SET preferred_candidate_pk=?2 WHERE pk=?1",
                    params![body, candidate],
                )
            })?;
            db(&path, || {
                tx.execute("INSERT INTO message_build(message_pk,session_pk,body_pk,basis_digest,contributor_revision,removal_epoch,phase) VALUES(?1,?2,?3,?4,?5,?6,'project')",params![message,session,body,digest.as_slice(),contributor,removal])
            })?;
            (tx.last_insert_rowid(), body, candidate)
        };
        let build = Build {
            path: path.clone(),
            source: source.predecessor.clone(),
            pk,
            body,
            candidate,
            message,
            session,
            contributor_revision: contributor,
            removal_epoch: removal,
            digest,
        };
        let compatible: bool = db(&path, || {
            tx.query_row("SELECT b.ready=0 AND c.ready=0 AND c.eligible=0 AND b.role=?2 AND b.written_by=?3 AND c.route=?4 FROM message_body b JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE b.pk=?1",params![body,match identity.role {Role::User=>"user",Role::Assistant=>"assistant",Role::System=>"system",Role::Tool=>"tool"},match identity.written_by {WrittenBy::Human=>"human",WrittenBy::Model=>"model",WrittenBy::Harness=>"harness",WrittenBy::Unknown=>"unknown"},match source.route {RouteKind::PageDom=>0,RouteKind::AccountExport=>1,RouteKind::Native=>2}],|r|r.get(0))
        })?;
        if !compatible {
            return Err(bad("preparation metadata changed under the same basis"));
        }
        let (phase, hash): (String, Option<Vec<u8>>) = db(&path, || {
            tx.query_row("SELECT w.phase,c.version_hash FROM message_build w JOIN message_body b ON b.pk=w.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE w.pk=?1",[pk],|r|Ok((r.get(0)?,r.get(1)?)))
        })?;
        check_at(&tx, &path, &build, false)?;
        let projection = if phase == "rank" {
            let hash: [u8; 32] = hash
                .ok_or_else(|| bad("completed projection lacks digest"))?
                .try_into()
                .map_err(|_| bad("projection digest length"))?;
            Some(super::completion::PreparedProjection {
                build: build.clone(),
                hash: polylogue_model::BlobHash::from_bytes(hash),
            })
        } else {
            None
        };
        let (active, origin): (Option<i64>, Option<i64>) = db(&path, || {
            tx.query_row(
                "SELECT active_value_pk,active_origin_value_pk FROM message_build WHERE pk=?1",
                [pk],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })?;
        let active_value = active
            .map(|pk| {
                values::head(
                    &tx,
                    &path,
                    &values::ValueId {
                        pk,
                        index_generation: source.predecessor.index_generation.clone(),
                        index_path: path.clone(),
                    },
                )
            })
            .transpose()?;
        let active_origin_value = origin
            .map(|pk| {
                values::head(
                    &tx,
                    &path,
                    &values::ValueId {
                        pk,
                        index_generation: source.predecessor.index_generation.clone(),
                        index_path: path.clone(),
                    },
                )
            })
            .transpose()?;
        db(&path, || tx.commit())?;
        Ok(Preparation {
            build,
            active_value,
            active_origin_value,
            projection,
        })
    }
    /// The value and its durable producer lease are admitted in one transaction.
    /// The returned ID cannot name a rolled-back allocation.
    pub fn begin_message_value(
        &mut self,
        build: &Build,
        codec: values::Codec,
        length: u64,
    ) -> Result<values::Head> {
        self.begin_owned_message_value(build, ValuePurpose::Normalized, codec, length)
    }
    /// Stage a source-owned descriptor separately from its normalized value.
    /// JSON codec/readiness proves integrity only, not selector/span semantics.
    pub fn begin_message_origin_value(
        &mut self,
        build: &Build,
        length: u64,
    ) -> Result<values::Head> {
        self.begin_owned_message_value(build, ValuePurpose::Origin, values::Codec::Json, length)
    }
    fn begin_owned_message_value(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        codec: values::Codec,
        length: u64,
    ) -> Result<values::Head> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, build)?;
        let column = purpose.column();
        let active: Option<i64> = db(&path, || {
            tx.query_row(
                &format!("SELECT {column} FROM message_build WHERE pk=?1"),
                [build.pk],
                |r| r.get(0),
            )
        })?;
        if active.is_some() {
            return Err(Error::MessagePreparationPending);
        }
        let value = values::begin_in(&tx, &path, codec, length, 1)?;
        db(&path, || {
            tx.execute(
                &format!("UPDATE message_build SET {column}=?2 WHERE pk=?1"),
                params![build.pk, value.id.pk()],
            )
        })?;
        db(&path, || tx.commit())?;
        Ok(value)
    }
    /// Integrity sealing retains this producer's lease, including when the
    /// normalized value and descriptor deduplicate to the exact same PK.
    pub fn seal_message_value(
        &mut self,
        build: &Build,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        self.seal_owned_message_value(build, ValuePurpose::Normalized, proof)
    }
    pub fn seal_message_origin_value(
        &mut self,
        build: &Build,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        self.seal_owned_message_value(build, ValuePurpose::Origin, proof)
    }
    fn seal_owned_message_value(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, build)?;
        let column = purpose.column();
        let active: Option<i64> = db(&path, || {
            tx.query_row(
                &format!("SELECT {column} FROM message_build WHERE pk=?1"),
                [build.pk],
                |r| r.get(0),
            )
        })?;
        if active != Some(proof.value().pk()) {
            return Err(Error::MessagePredecessorChanged);
        }
        let sealed = values::seal_in(&tx, &path, proof, |tx, ready| {
            if purpose == ValuePurpose::Origin && ready.codec != values::Codec::Json {
                return Err(Error::ValueNotReady);
            }
            if ready.id != *proof.value() {
                db(&path, || {
                    tx.execute(
                        "UPDATE body_value SET incoming=incoming+1 WHERE pk=?1 AND state='ready'",
                        [ready.id.pk()],
                    )
                })?;
                if db(&path, || {
                    tx.execute(
                        "UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming>0",
                        [proof.value().pk()],
                    )
                })? != 1
                {
                    return Err(Error::MessagePredecessorChanged);
                }
                db(&path, || {
                    tx.execute(
                        &format!("UPDATE message_build SET {column}=?2 WHERE pk=?1"),
                        params![build.pk, ready.id.pk()],
                    )
                })?;
            }
            Ok(())
        })?;
        db(&path, || tx.commit())?;
        Ok(sealed)
    }

    /// Transfer the active ready value to one source-supplied field fact and
    /// copy at most one fixed-depth field-directory path. The caller establishes
    /// semantic block correspondence before naming a Slot; ordinal coincidence
    /// is insufficient. The filled body never supplies its own rank facts.
    pub fn write_message_field(
        &mut self,
        build: &Build,
        slot: &polylogue_model::merge::Slot,
        source_time: Option<i64>,
        supplied: bool,
        step: &super::blocks::ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        self.update_message_field(
            build,
            slot,
            source_time,
            supplied,
            step,
            allowance,
            FieldUpdate::Insert,
        )
    }
    /// Explicitly set one dirty Message-local leaf (insert or replace).
    /// Source suppliedness remains mapper-owned; no root inspection is needed.
    pub fn set_message_field(
        &mut self,
        build: &Build,
        slot: &polylogue_model::merge::Slot,
        source_time: Option<i64>,
        supplied: bool,
        step: &super::blocks::ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        self.update_message_field(
            build,
            slot,
            source_time,
            supplied,
            step,
            allowance,
            FieldUpdate::Set,
        )
    }
    /// Removal consumes no value. Missing is idempotent only at a fresh cursor;
    /// a stale replay rolls back the same cursor CAS as other projection work.
    pub fn remove_message_field(
        &mut self,
        build: &Build,
        slot: &polylogue_model::merge::Slot,
        step: &super::blocks::ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        self.update_message_field(
            build,
            slot,
            None,
            false,
            step,
            allowance,
            FieldUpdate::Remove,
        )
    }
    fn update_message_field(
        &mut self,
        build: &Build,
        slot: &polylogue_model::merge::Slot,
        source_time: Option<i64>,
        supplied: bool,
        step: &super::blocks::ProjectionStep<'_>,
        allowance: usize,
        intent: FieldUpdate,
    ) -> Result<()> {
        if matches!(
            slot,
            polylogue_model::merge::Slot::Block { .. }
                | polylogue_model::merge::Slot::Extra { block: Some(_), .. }
        ) {
            return Err(bad("block-local fields require active block ownership"));
        }
        super::blocks::admit(step, allowance)?;
        let key = field_key(slot, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, build)?;
        update_field_in(&tx, &path, build, &key, None, source_time, supplied, intent)?;
        super::blocks::commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Invalidation may outlive its source basis or privacy epoch. It can only
    /// retire the exact committed private build, never resume it or make output
    /// visible. Payload/node draining remains separately owed after this step.
    pub fn retire_message_preparation(&mut self, build: &Build) -> Result<()> {
        let path = self.root.join("index.db");
        if path != build.path {
            return Err(Error::MessagePredecessorChanged);
        }
        let tx = db(&path, || self._index.transaction())?;
        let generation: String = db(&path, || {
            tx.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
                r.get(0)
            })
        })?;
        if generation != build.source.index_generation {
            return Err(Error::MessagePredecessorChanged);
        }
        let active: Option<(Option<i64>, Option<i64>, Option<i64>)> = db(&path, || {
            tx.query_row("SELECT w.active_value_pk,w.active_origin_value_pk,w.active_block_pk FROM message_build w JOIN message_body b ON b.pk=w.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE w.pk=?1 AND w.body_pk=?2 AND w.basis_digest=?3 AND b.ready=0 AND c.ready=0 AND c.eligible=0",params![build.pk,build.body,build.digest.as_slice()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()
        })?;
        let (active, origin, block) = active.ok_or(Error::MessagePredecessorChanged)?;
        db(&path, || {
            tx.execute(
                "UPDATE message_build SET phase='retiring',active_value_pk=NULL,active_origin_value_pk=NULL,active_block_pk=NULL WHERE pk=?1",
                [build.pk],
            )
        })?;
        if let Some(block) = block {
            super::blocks::release_unreferenced_block(&tx, &path, block)?;
        }
        for value in [active, origin].into_iter().flatten() {
            let changed = db(&path, || {
                tx.execute(
                    "UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming>0",
                    [value],
                )
            })?;
            if changed != 1 {
                return Err(bad("missing active value producer lease"));
            }
            db(&path, || {
                tx.execute(
                    "UPDATE body_value SET state='retiring' WHERE pk=?1 AND incoming=0",
                    [value],
                )
            })?;
        }
        db(&path, || tx.commit())?;
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ValuePurpose {
    Normalized,
    Origin,
}
impl ValuePurpose {
    fn column(self) -> &'static str {
        match self {
            Self::Normalized => "active_value_pk",
            Self::Origin => "active_origin_value_pk",
        }
    }
}

/// One active ready producer lease becomes one field fact/root lease. The
/// caller has checked private block ownership (if any) in this transaction.
pub(super) fn write_field_in(
    tx: &rusqlite::Transaction<'_>,
    path: &Path,
    build: &Build,
    key: &[u8],
    block: Option<i64>,
    source_time: Option<i64>,
    supplied: bool,
) -> Result<()> {
    update_field_in(
        tx,
        path,
        build,
        key,
        block,
        source_time,
        supplied,
        FieldUpdate::Insert,
    )
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldUpdate {
    Insert,
    Set,
    Remove,
}
fn update_field_in(
    tx: &rusqlite::Transaction<'_>,
    path: &Path,
    build: &Build,
    key: &[u8],
    block: Option<i64>,
    source_time: Option<i64>,
    supplied: bool,
    intent: FieldUpdate,
) -> Result<()> {
    let (value, origin): (Option<i64>, Option<i64>) = db(path, || {
        tx.query_row(
            "SELECT active_value_pk,active_origin_value_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
    })?;
    if intent == FieldUpdate::Remove {
        if value.is_some() || origin.is_some() {
            return Err(Error::MessagePreparationPending);
        }
    } else {
        let value = value.ok_or(Error::ValueNotReady)?;
        let head = values::head(
            tx,
            path,
            &values::ValueId {
                pk: value,
                index_generation: build.source.index_generation.clone(),
                index_path: path.to_path_buf(),
            },
        )?;
        if head.state != values::State::Ready {
            return Err(Error::ValueNotReady);
        }
    }
    if let Some(origin) = origin {
        let head = values::head(
            tx,
            path,
            &values::ValueId {
                pk: origin,
                index_generation: build.source.index_generation.clone(),
                index_path: path.to_path_buf(),
            },
        )?;
        if head.state != values::State::Ready || head.codec != values::Codec::Json {
            return Err(Error::ValueNotReady);
        }
    }
    let mut slot: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT pk FROM message_field_slot WHERE message_pk=?1 AND semantic_key=?2",
            params![build.message, key],
            |r| r.get(0),
        )
        .optional()
    })?;
    if slot.is_none() && intent != FieldUpdate::Remove {
        db(path, || {
            tx.execute("INSERT INTO message_field_slot(message_pk,session_pk,semantic_key) VALUES(?1,?2,?3)",params![build.message,build.session,key])
        })?;
        slot = Some(tx.last_insert_rowid());
    }
    let Some(slot) = slot else { return Ok(()) };
    let old: Option<i64> = db(path, || match block {
        Some(pk) => tx.query_row(
            "SELECT field_root_pk FROM block WHERE pk=?1 AND message_pk=?2 AND session_pk=?3",
            params![pk, build.message, build.session],
            |r| r.get(0),
        ),
        None => tx.query_row(
            "SELECT field_root_pk FROM message_body WHERE pk=?1 AND ready=0",
            [build.body],
            |r| r.get(0),
        ),
    })?;
    let owner = super::directory::Owner {
        message: build.message,
        session: build.session,
    };
    let mut work = super::directory::Work::default();
    let key = u64::try_from(slot).map_err(|_| Error::Range)?;
    let existing = super::directory::lookup(
        tx,
        path,
        owner,
        super::directory::Kind::Fields,
        old,
        key,
        &mut work,
    )?;
    match (intent, existing.is_some()) {
        (FieldUpdate::Insert, true) => return Err(bad("duplicate field in this immutable root")),
        (FieldUpdate::Remove, false) => return Ok(()),
        _ => {}
    }
    let edge = if intent == FieldUpdate::Remove {
        None
    } else {
        let value = value.ok_or(Error::ValueNotReady)?;
        db(path, || {
            tx.execute("INSERT INTO message_field_fact(message_pk,session_pk,candidate_pk,slot_pk,value_pk,origin_value_pk,source_time,supplied) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",params![build.message,build.session,build.candidate,slot,value,origin,source_time,supplied])
        })?;
        Some(super::directory::Edge {
            link: super::directory::Link::Field(tx.last_insert_rowid()),
            counts: super::directory::Counts {
                source_fields: i64::from(supplied),
                ..Default::default()
            },
        })
    };
    let (root, _) = super::directory::replace(
        tx,
        path,
        owner,
        super::directory::Kind::Fields,
        old,
        key,
        edge,
        &mut work,
    )?;
    super::directory::acquire(tx, path, owner, root)?;
    if let Some(old) = old {
        super::directory::release(tx, path, owner, old)?;
    }
    db(path, || match block {
        Some(pk) => tx.execute(
            "UPDATE block SET field_root_pk=?2 WHERE pk=?1",
            params![pk, root],
        ),
        None => tx.execute(
            "UPDATE message_body SET field_root_pk=?2 WHERE pk=?1 AND ready=0",
            params![build.body, root],
        ),
    })?;
    if intent != FieldUpdate::Remove {
        db(path, || {
            tx.execute(
                "UPDATE message_build SET active_value_pk=NULL,active_origin_value_pk=NULL WHERE pk=?1",
                [build.pk],
            )
        })?;
    }
    Ok(())
}

pub(super) fn field_key(slot: &polylogue_model::merge::Slot, allowance: usize) -> Result<Vec<u8>> {
    use polylogue_model::{BlockSlot, merge::Slot};
    let (block, extra) = match slot {
        Slot::Block { block, .. } => (Some(block), 0),
        Slot::Extra { block, key } => (block.as_ref(), key.len()),
        _ => (None, 0),
    };
    let attachment = match block {
        Some(BlockSlot::Attachment(key)) => key.len(),
        _ => 0,
    };
    let lower = extra.checked_add(attachment).ok_or(Error::Range)?;
    encoded_key(slot, lower, allowance)
}
/// Shared block-local field address codec. The block's root supplies its
/// correspondence; an ordinal never becomes part of this semantic key.
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) enum BlockFieldKey<'a> {
    Field(polylogue_model::merge::BlockField),
    Extra(#[serde(borrow)] std::borrow::Cow<'a, str>),
}
pub(super) fn block_field_key(
    slot: &polylogue_model::merge::Slot,
    allowance: usize,
) -> Result<Vec<u8>> {
    use polylogue_model::merge::Slot;
    match slot {
        Slot::Block { field, .. } => encoded_key(&BlockFieldKey::Field(*field), 0, allowance),
        Slot::Extra {
            block: Some(_),
            key,
        } => encoded_key(
            &BlockFieldKey::Extra(std::borrow::Cow::Borrowed(key)),
            key.len(),
            allowance,
        ),
        _ => Err(bad("block field requires mapper-proved block membership")),
    }
}
fn encoded_key(value: &impl serde::Serialize, lower: usize, allowance: usize) -> Result<Vec<u8>> {
    if lower > allowance {
        return Err(Error::ParserAdmission {
            resource: "message field key bytes",
            required: lower as u64,
        });
    }
    struct Limited {
        bytes: Vec<u8>,
        limit: usize,
        refused: bool,
    }
    impl std::io::Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let end = self
                .bytes
                .len()
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("field key overflow"))?;
            if end > self.limit {
                self.refused = true;
                return Err(std::io::Error::other("field key admission"));
            }
            self.bytes
                .try_reserve_exact(bytes.len())
                .map_err(std::io::Error::other)?;
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited {
        bytes: Vec::new(),
        limit: allowance,
        refused: false,
    };
    if let Err(error) = serde_json::to_writer(&mut output, value) {
        return Err(if output.refused {
            Error::ParserAdmission {
                resource: "message field key bytes",
                required: (allowance as u64).saturating_add(1),
            }
        } else {
            Error::Io(std::io::Error::other(error))
        });
    }
    Ok(output.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contributions::Retirement;
    use crate::{ImportObservation, InputFormat};
    use polylogue_model::merge::{MessageField, Slot};
    struct Fixture {
        dir: tempfile::TempDir,
        root: PathBuf,
        writer: Writer,
        head: parser_state::Head,
        origin: Origin,
        session: SessionId,
        message: MessageId,
    }
    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("archive");
            let mut writer = Writer::open(&root).unwrap();
            let blob = writer.put_file_blob(b"{\"synthetic\":true}\n").unwrap();
            let input = writer
                .retain_immutable_input(
                    blob,
                    InputFormat::CodexRollout,
                    &[],
                    &ImportObservation {
                        id: "synthetic-observation",
                        machine: "synthetic-machine",
                        host: "synthetic-host",
                        path: "synthetic/input.jsonl",
                        seen_at: 0,
                    },
                    4096,
                )
                .unwrap();
            let head = writer
                .begin_parser(&parser_state::Basis {
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
            let session = SessionId::new(&origin, "session:with:colons").unwrap();
            let message = MessageId::new(session.clone(), "message:with:colons").unwrap();
            Self {
                dir,
                root,
                writer,
                head,
                origin,
                session,
                message,
            }
        }
        fn prepare(&mut self) -> Result<Preparation> {
            self.writer.prepare_message(
                Identity {
                    session: &self.session,
                    origin: &self.origin,
                    native_session: "session:with:colons",
                    session_kind: SessionKind::Main,
                    message: &self.message,
                    sequence: 0,
                    role: Role::Assistant,
                    written_by: WrittenBy::Model,
                },
                Source {
                    predecessor: &self.head.token,
                    revision: 1,
                    evidence_revision: 1,
                    logical_key: [3; 32],
                    route: RouteKind::Native,
                },
                4096,
            )
        }
        fn reopen(self) -> Self {
            let Self {
                dir,
                root,
                writer,
                head,
                origin,
                session,
                message,
            } = self;
            drop(writer);
            let writer = Writer::open(&root).unwrap();
            Self {
                dir,
                root,
                writer,
                head,
                origin,
                session,
                message,
            }
        }
        fn scalar(&self, sql: &str) -> i64 {
            self.writer._index.query_row(sql, [], |r| r.get(0)).unwrap()
        }
        fn value(&mut self, build: &Build, text: &[u8]) -> values::Seal {
            let h = self
                .writer
                .begin_message_value(build, values::Codec::Utf8, text.len() as u64)
                .unwrap();
            if !text.is_empty() {
                self.writer
                    .append_value_chunk(&h.id, 0, text, text.len())
                    .unwrap();
            }
            let reader = self.writer.read_handle().unwrap();
            let mut proof = self.writer.validate_value(&h.id).unwrap();
            while !proof.progress().complete {
                proof.advance(&reader, 2).unwrap();
            }
            drop(reader);
            self.writer.seal_message_value(build, &proof).unwrap()
        }
    }
    #[test]
    fn text_block_checkpoint_rolls_back_and_resumes_without_publication() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.value(&build, b"hello");
        f.writer._index.execute_batch("CREATE TRIGGER refuse_projection BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic checkpoint failure'); END;").unwrap();
        let step = ProjectionStep {
            expected_sequence: 0,
            next_cursor: b"next",
        };
        assert!(matches!(
            f.writer.write_block(
                &build,
                0,
                0,
                BlockKind::Text,
                BlockPresence {
                    source_block: true,
                    source_text: true
                },
                &step,
                4096
            ),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_body_node"), 0);
        assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            0
        );
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_projection")
            .unwrap();
        let mut f = f.reopen();
        f.writer
            .write_block(
                &build,
                0,
                0,
                BlockKind::Text,
                BlockPresence {
                    source_block: true,
                    source_text: true,
                },
                &step,
                4096,
            )
            .unwrap();
        let state = f.writer.message_projection(&build, 4096).unwrap();
        assert_eq!(state.sequence, 1);
        assert_eq!(state.cursor.as_deref(), Some(b"next".as_slice()));
        assert_eq!(
            f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
            0
        );
        assert_eq!(f.scalar("SELECT ready FROM message_body"), 0);
        assert!(matches!(
            f.writer.advance_message_projection(&build, &step, 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(f.writer.message_projection(&build, 4096).unwrap(), state);
        f.writer.retire_message_preparation(&build).unwrap();
        for turn in 0..200 {
            let nodes = f.scalar("SELECT count(*) FROM message_body_node");
            let result = f.writer.retire_message_turn(&build).unwrap();
            if result == Retirement::Node {
                assert_eq!(
                    f.scalar("SELECT count(*) FROM message_body_node"),
                    nodes - 1
                );
            }
            if result == Retirement::Complete {
                break;
            }
            assert!(turn < 199);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
        assert_eq!(
            f.scalar("SELECT count(*) FROM parser_cell WHERE namespace='message.projection.v1'"),
            0
        );
        assert_eq!(f.scalar("SELECT incoming FROM body_value"), 0);
    }
    #[test]
    fn text_directory_replacement_reclaims_only_unleased_leaves() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        for n in 0..3 {
            f.value(&build, b"shared");
            f.writer
                .write_block(
                    &build,
                    0,
                    n % 2,
                    BlockKind::Text,
                    BlockPresence {
                        source_block: true,
                        source_text: true,
                    },
                    &ProjectionStep {
                        expected_sequence: n,
                        next_cursor: b"cursor",
                    },
                    4096,
                )
                .unwrap();
        }
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
        assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 3);
        for turn in 0..200 {
            let nodes = f.scalar("SELECT count(*) FROM message_body_node");
            if !f.writer.collect_message_nodes(&f.message).unwrap() {
                break;
            }
            assert_eq!(
                f.scalar("SELECT count(*) FROM message_body_node"),
                nodes - 1
            );
            assert!(turn < 199);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
        assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
        f.value(&build, b"");
        assert!(matches!(
            f.writer.write_block(
                &build,
                1,
                0,
                BlockKind::Text,
                BlockPresence {
                    source_block: true,
                    source_text: false
                },
                &ProjectionStep {
                    expected_sequence: 3,
                    next_cursor: b"invalid"
                },
                4096
            ),
            Err(Error::Invariant(_))
        ));
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            3
        );
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
    }
    #[test]
    fn selected_read_pins_sparse_ordinals_payload_and_archive_authority() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        use polylogue_model::BlockSlot;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let reader = f.writer.read_handle().unwrap();
        assert!(reader.selected_message(&f.message).unwrap().is_none());
        for (sequence, group, local, text) in [
            (0, 4096, 900, b"later".as_slice()),
            (1, 1, 17, b"earlier".as_slice()),
        ] {
            f.value(&build, text);
            f.writer
                .write_block(
                    &build,
                    group,
                    local,
                    BlockKind::Text,
                    BlockPresence {
                        source_block: true,
                        source_text: true,
                    },
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"selected-read-test",
                    },
                    4096,
                )
                .unwrap();
        }
        let certificate = f
            .writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"projection-complete",
                },
                polylogue_model::BlobHash::of(b"synthetic-source-evidence"),
                polylogue_model::merge::FinishState::Unknown,
                4096,
            )
            .unwrap();
        assert_eq!(certificate.body_pk(), build.body_pk());
        drop(reader);
        let mut f = f.reopen();
        let reader = f.writer.read_handle().unwrap();
        assert_eq!(
            f.prepare().unwrap().projection.unwrap().version_hash(),
            certificate.version_hash()
        );
        assert_eq!(f.scalar("SELECT eligible FROM message_candidate"), 0);
        // Selection alone is fixture setup. The canonical election/consumer
        // runtime is not qualified by this reader test.
        f.writer._index.execute_batch("BEGIN; UPDATE message_body SET ready=1,version_hash=(SELECT version_hash FROM message_candidate WHERE pk=message_body.preferred_candidate_pk); UPDATE message_candidate SET body_ready=1,ready=1,eligible=1; UPDATE message SET selected_body_pk=(SELECT pk FROM message_body); COMMIT;").unwrap();
        let view = reader.selected_message(&f.message).unwrap().unwrap();
        assert_eq!(view.id(), &f.message);
        let first = view.block(BlockSlot::Item(0)).unwrap().unwrap();
        assert_eq!((first.group, first.local), (1, 17));
        assert_eq!(first.text_range(0, 3).unwrap().unwrap().bytes, b"ear");
        let second = view.block(BlockSlot::Item(1)).unwrap().unwrap();
        assert_eq!((second.group, second.local), (4096, 900));
        assert_eq!(second.text_range(0, 100).unwrap().unwrap().bytes, b"later");
        assert!(view.block(BlockSlot::Item(2)).unwrap().is_none());
        assert!(view.block(BlockSlot::Reasoning(0)).unwrap().is_none());
        // Test-only body retirement setup releases its root; this isolates the
        // real node/value collector and read snapshot. Canonical body election
        // and ready-body retirement remain separate runtime obligations.
        let root = f.scalar("SELECT block_root_pk FROM message_body");
        let path = f.root.join("index.db");
        let tx = f.writer._index.transaction().unwrap();
        tx.execute_batch("PRAGMA defer_foreign_keys=ON; UPDATE message SET selected_body_pk=NULL; DELETE FROM message_build; DELETE FROM message_candidate; DELETE FROM message_body;").unwrap();
        super::super::blocks::discard_cursor(&tx, &path, &build).unwrap();
        super::super::directory::release(
            &tx,
            &path,
            super::super::directory::Owner {
                message: build.message,
                session: build.session,
            },
            root,
        )
        .unwrap();
        tx.commit().unwrap();
        for n in 0..300 {
            if !f.writer.collect_message_nodes(&f.message).unwrap() {
                break;
            }
            assert!(n < 299);
        }
        for n in 0..100 {
            if f.writer.collect_retiring_value().unwrap() == super::super::ValueCollection::Idle {
                break;
            }
            assert!(n < 99);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
        assert!(reader.selected_message(&f.message).unwrap().is_none());
        assert_eq!(first.text_range(3, 100).unwrap().unwrap().bytes, b"lier");
        f.writer
            ._index
            .execute(
                "UPDATE message SET hidden=1,removal_epoch=removal_epoch+1",
                [],
            )
            .unwrap();
        assert!(matches!(
            first.text_range(0, 100),
            Err(Error::MessagePredecessorChanged)
        ));
        assert!(matches!(
            view.block(BlockSlot::Item(0)),
            Err(Error::MessagePredecessorChanged)
        ));
        f.writer
            ._index
            .execute("UPDATE message SET hidden=0", [])
            .unwrap();
        assert!(matches!(
            second.text_range(0, 100),
            Err(Error::MessagePredecessorChanged)
        ));
        drop(reader);
        drop(f.writer);
        assert!(matches!(
            second.text_range(0, 100),
            Err(Error::MessagePredecessorChanged)
        ));
        assert!(matches!(
            crate::ReadHandle::open(&f.root),
            Err(Error::WriterHeld { .. })
        ));
        drop(first);
        drop(second);
        drop(view);
        assert!(crate::ReadHandle::open(&f.root).is_ok());
    }
    #[test]
    fn completed_projection_is_atomic_unready_restartable_and_abortable() {
        use super::super::blocks::ProjectionStep;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.writer._index.execute_batch("CREATE TRIGGER refuse_certificate BEFORE UPDATE OF facts_ready ON message_candidate WHEN NEW.facts_ready=1 BEGIN SELECT RAISE(ABORT,'synthetic certificate failure'); END;").unwrap();
        let step = ProjectionStep {
            expected_sequence: 0,
            next_cursor: b"done",
        };
        let hash = polylogue_model::BlobHash::of(b"synthetic-evidence");
        assert!(matches!(
            f.writer.finish_message_projection(
                &build,
                &step,
                hash,
                polylogue_model::merge::FinishState::Unknown,
                4096
            ),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(f.scalar("SELECT ready FROM message_body"), 0);
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            0
        );
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_certificate")
            .unwrap();
        f.writer
            .finish_message_projection(
                &build,
                &step,
                hash,
                polylogue_model::merge::FinishState::Unknown,
                4096,
            )
            .unwrap();
        assert_eq!(f.scalar("SELECT in_progress IS NULL FROM message_body"), 1);
        assert_eq!(f.scalar("SELECT finish FROM message_candidate"), 1);
        assert_eq!(f.scalar("SELECT eligible FROM message_candidate"), 0);
        assert_eq!(f.scalar("SELECT selected_body_pk IS NULL FROM message"), 1);
        assert!(matches!(
            f.writer.advance_message_projection(&build, &step, 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        let mut f = f.reopen();
        assert_eq!(
            f.prepare().unwrap().projection.unwrap().version_hash(),
            hash
        );
        assert_eq!(f.scalar("SELECT ready FROM message_body"), 0);
        assert_eq!(f.scalar("SELECT ready FROM message_candidate"), 0);
        f.writer.retire_message_preparation(&build).unwrap();
        for turn in 0..100 {
            if f.writer.retire_message_turn(&build).unwrap() == Retirement::Complete {
                break;
            }
            assert!(turn < 99);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
    }
    #[test]
    fn ready_source_retirement_preserves_shared_canonical_and_frontier_leases() {
        use super::super::{
            BodyRetirement as R, CandidateCollection as C,
            blocks::{BlockKind, BlockPresence, ProjectionStep},
            directory as d,
        };
        use polylogue_model::{BlobHash, BlockSlot, merge::FinishState};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let source = build.body_id();
        let candidate = build.candidate_id();
        f.value(&build, b"shared text");
        f.writer
            .write_block(
                &build,
                3,
                7,
                BlockKind::Text,
                BlockPresence {
                    source_block: true,
                    source_text: true,
                },
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"field",
                },
                4096,
            )
            .unwrap();
        f.value(&build, b"synthetic-model");
        f.writer
            .write_message_field(
                &build,
                &Slot::Message(MessageField::Model),
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"finish",
                },
                4096,
            )
            .unwrap();
        f.writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"done",
                },
                BlobHash::of(b"synthetic-ready-evidence"),
                FinishState::Unknown,
                4096,
            )
            .unwrap();
        let blocks = f.scalar("SELECT block_root_pk FROM message_body");
        let fields = f.scalar("SELECT field_root_pk FROM message_body");
        let path = f.root.join("index.db");
        let owner = d::Owner {
            message: build.message,
            session: build.session,
        };
        // Only readiness/election is fixture setup. Real projection, leases,
        // retirement, snapshot reads and collectors are the target APIs.
        let tx = f.writer._index.transaction().unwrap();
        tx.execute_batch("UPDATE message_body SET ready=1,version_hash=(SELECT version_hash FROM message_candidate WHERE pk=message_body.preferred_candidate_pk); UPDATE message_candidate SET ready=1,body_ready=1,eligible=1;").unwrap();
        tx.execute("INSERT INTO message_body(message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,block_root_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version) SELECT message_pk,session_pk,1,zeroblob(32),0,0,preferred_candidate_pk,block_root_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version FROM message_body WHERE pk=?1",[build.body]).unwrap();
        let canonical_pk = tx.last_insert_rowid();
        d::acquire(&tx, &path, owner, blocks).unwrap();
        d::acquire(&tx, &path, owner, fields).unwrap();
        tx.execute("UPDATE message SET selected_body_pk=?1", [build.body])
            .unwrap();
        tx.commit().unwrap();
        assert_eq!(f.writer.retire_message_body(&source).unwrap(), R::Selected);
        f.writer
            ._index
            .execute("UPDATE message SET selected_body_pk=?1", [canonical_pk])
            .unwrap();
        assert_eq!(
            f.writer.retire_message_body(&source).unwrap(),
            R::ActiveBuild
        );
        let tx = f.writer._index.transaction().unwrap();
        super::super::blocks::discard_cursor(&tx, &path, &build).unwrap();
        tx.execute("DELETE FROM message_build WHERE pk=?1", [build.pk])
            .unwrap();
        tx.commit().unwrap();
        assert_eq!(f.writer.retire_message_body(&source).unwrap(), R::Eligible);
        f.writer
            ._index
            .execute(
                "UPDATE message_candidate SET eligible=0 WHERE pk=?1",
                [build.candidate],
            )
            .unwrap();
        assert_eq!(
            f.writer.retire_message_body(&source).unwrap(),
            R::DetachedCandidate
        );
        let mut f = f.reopen(); // restart after committed pointer detach
        f.writer._index.execute_batch("CREATE TRIGGER refuse_body_release BEFORE UPDATE OF incoming ON message_body_node BEGIN SELECT RAISE(ABORT,'synthetic body release failure'); END;").unwrap();
        assert!(matches!(
            f.writer.retire_message_body(&source),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 2);
        assert_eq!(f.scalar("SELECT incoming FROM message_body_node WHERE pk=(SELECT block_root_pk FROM message_body LIMIT 1)"),2);
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_body_release")
            .unwrap();
        assert_eq!(f.writer.retire_message_body(&source).unwrap(), R::Body);
        assert_eq!(f.writer.retire_message_body(&source).unwrap(), R::Missing);
        assert!(!f.writer.collect_message_nodes(&f.message).unwrap());
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 1);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        assert_eq!(f.scalar("SELECT ready FROM message_candidate"), 1);
        assert_eq!(f.scalar("SELECT body_pk IS NULL FROM message_candidate"), 1);
        assert_eq!(
            f.writer.collect_message_candidate(&candidate).unwrap(),
            C::Retained
        );
        let mut f = f.reopen();
        let reader = f.writer.read_handle().unwrap();
        let view = reader.selected_message(&f.message).unwrap().unwrap();
        let canonical = view.body_id();
        match view
            .field(&Slot::Message(MessageField::Model), 4096)
            .unwrap()
            .unwrap()
        {
            super::super::reads::FieldRead::Value(field) => {
                assert!(field.supplied);
                assert_eq!(field.range(0, 100).unwrap().bytes, b"synthetic-model");
            }
            _ => panic!("unexpected field conflict"),
        }
        assert_eq!(
            view.block(BlockSlot::Item(0))
                .unwrap()
                .unwrap()
                .text_range(0, 100)
                .unwrap()
                .unwrap()
                .bytes,
            b"shared text"
        );
        assert_eq!(
            f.writer.retire_message_body(&canonical).unwrap(),
            R::Selected
        );
        drop(view);
        drop(reader);
        // A different active build holds one shared field-root frontier lease.
        let next = f.prepare().unwrap().build;
        assert!(next.body > canonical_pk);
        let tx = f.writer._index.transaction().unwrap();
        d::acquire(&tx, &path, owner, fields).unwrap();
        tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,?2,?3,0,?4,0)",params![next.pk,next.message,next.session,fields]).unwrap();
        tx.execute("UPDATE message SET selected_body_pk=NULL", [])
            .unwrap();
        tx.commit().unwrap();
        assert_eq!(f.writer.retire_message_body(&canonical).unwrap(), R::Body);
        for n in 0..100 {
            if !f.writer.collect_message_nodes(&f.message).unwrap() {
                break;
            }
            assert!(n < 99);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        assert_eq!(
            f.writer.collect_message_candidate(&candidate).unwrap(),
            C::Retained
        );
        f.writer.retire_message_preparation(&next).unwrap();
        f.writer._index.execute_batch("CREATE TRIGGER refuse_fact_release BEFORE DELETE ON message_field_fact BEGIN SELECT RAISE(ABORT,'synthetic fact release failure'); END;").unwrap();
        let mut failed = false;
        for n in 0..100 {
            let nodes = f.scalar("SELECT count(*) FROM message_body_node");
            let edges = f.scalar("SELECT count(*) FROM message_body_edge");
            match f.writer.retire_message_turn(&next) {
                Err(Error::Sqlite { .. }) => {
                    assert_eq!(f.scalar("SELECT count(*) FROM message_body_node"), nodes);
                    assert_eq!(f.scalar("SELECT count(*) FROM message_body_edge"), edges);
                    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
                    failed = true;
                    break;
                }
                Ok(Retirement::Complete) => panic!("fact failure was bypassed"),
                Ok(_) => {}
                Err(error) => panic!("unexpected {error}"),
            }
            assert!(n < 99);
        }
        assert!(failed);
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_fact_release")
            .unwrap();
        for n in 0..150 {
            if f.writer.retire_message_turn(&next).unwrap() == Retirement::Complete {
                break;
            }
            assert!(n < 149);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), 0);
        assert_eq!(
            f.writer.collect_message_candidate(&candidate).unwrap(),
            C::Candidate
        );
        assert_eq!(
            f.writer.collect_message_candidate(&candidate).unwrap(),
            C::Missing
        );
        for n in 0..100 {
            if f.writer.collect_retiring_value().unwrap() == super::super::ValueCollection::Idle {
                break;
            }
            assert!(n < 99);
        }
        assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
        let mut other = Fixture::new();
        assert!(matches!(
            other.writer.retire_message_body(&canonical),
            Err(Error::MessagePredecessorChanged)
        ));
        assert!(matches!(
            other.writer.collect_message_candidate(&candidate),
            Err(Error::MessagePredecessorChanged)
        ));
        let generation: String = f
            .writer
            ._index
            .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
                r.get(0)
            })
            .unwrap();
        f.writer
            ._index
            .execute(
                "UPDATE meta SET value='synthetic-new-index' WHERE key='generation'",
                [],
            )
            .unwrap();
        assert!(matches!(
            f.writer.retire_message_body(&source),
            Err(Error::MessagePredecessorChanged)
        ));
        assert!(matches!(
            f.writer.collect_message_candidate(&candidate),
            Err(Error::MessagePredecessorChanged)
        ));
        f.writer
            ._index
            .execute(
                "UPDATE meta SET value=?1 WHERE key='generation'",
                [generation],
            )
            .unwrap();
        assert_eq!(
            f.writer
                .archive
                .query_row("SELECT count(*) FROM revision", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
    #[test]
    fn preparation_value_and_field_lease_survive_restart_without_publication() {
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        assert_eq!(
            f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
            0
        );
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_candidate WHERE ready=1 OR eligible=1"),
            0
        );
        let value = f
            .writer
            .begin_message_value(&build, values::Codec::Utf8, 5)
            .unwrap();
        f.writer
            .append_value_chunk(&value.id, 0, b"hel", 3)
            .unwrap();
        let mut f = f.reopen();
        let resumed = f.prepare().unwrap();
        assert_eq!(resumed.build, build);
        let active = resumed.active_value.unwrap();
        assert_eq!(active.appended, 3);
        assert_eq!(active.id, value.id);
        assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
        f.writer
            .append_value_chunk(&active.id, 3, b"lo", 2)
            .unwrap();
        let reader = f.writer.read_handle().unwrap();
        let mut proof = f.writer.validate_value(&active.id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, 1).unwrap();
        }
        drop(reader);
        f.writer.seal_message_value(&build, &proof).unwrap();
        f.writer
            .write_message_field(
                &build,
                &Slot::Message(MessageField::Model),
                None,
                true,
                &super::super::blocks::ProjectionStep {
                    expected_sequence: f.writer.message_projection(&build, 4096).unwrap().sequence,
                    next_cursor: b"field",
                },
                4096,
            )
            .unwrap();
        assert!(f.prepare().unwrap().active_value.is_none());
        assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_body WHERE ready=1"),
            0
        );
        assert_eq!(
            f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
            0
        );
        // A new lease can now be admitted; the earlier value belongs to its fact.
        f.writer
            .begin_message_value(&build, values::Codec::Json, 2)
            .unwrap();
        assert_eq!(f.scalar("SELECT sum(incoming) FROM body_value"), 2);
    }
    #[test]
    fn owned_deduplication_transfers_only_its_own_lease_and_abort_releases_it() {
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let seal = f.value(&build, b"same");
        f.writer
            .write_message_field(
                &build,
                &Slot::Message(MessageField::Model),
                None,
                true,
                &super::super::blocks::ProjectionStep {
                    expected_sequence: f.writer.message_projection(&build, 4096).unwrap().sequence,
                    next_cursor: b"field",
                },
                4096,
            )
            .unwrap();
        let duplicate = f.value(&build, b"same");
        assert_eq!(duplicate.ready, seal.ready);
        assert!(duplicate.retired_duplicate.is_some());
        assert_eq!(
            f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
            2
        );
        assert_eq!(
            f.scalar("SELECT count(*) FROM body_value WHERE state='retiring' AND incoming=0"),
            1
        );
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
            1
        );
        assert_eq!(f.scalar("SELECT count(*) FROM message_build WHERE phase='retiring' AND active_value_pk IS NULL"),1);
        assert!(matches!(
            f.writer.begin_message_value(&build, values::Codec::Utf8, 1),
            Err(Error::MessagePredecessorChanged)
        ));
        // Retrying invalidation releases no second lease.
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
            1
        );
    }
    #[test]
    fn changed_source_removal_epoch_and_index_identity_reject_mutation_but_allow_cleanup() {
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let value = f
            .writer
            .begin_message_value(&build, values::Codec::Utf8, 10)
            .unwrap();
        f.writer
            ._index
            .execute("UPDATE message SET hidden=1,removal_epoch=1", [])
            .unwrap();
        assert!(matches!(
            f.writer.begin_message_value(&build, values::Codec::Utf8, 0),
            Err(Error::MessagePredecessorChanged)
        ));
        assert!(matches!(f.prepare(), Err(Error::MessagePredecessorChanged)));
        f.writer
            .begin_parser(&parser_state::Basis {
                input_pk: f.head.token.input_pk,
                chain: 1,
                parser_version: 4,
                digest: [8; 32],
                acquired_upto: f.head.acquired_upto,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        // Cleanup has no authority to resume a stale source; it only drops its lease.
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.writer.value_head(&value.id).unwrap().state,
            values::State::Retiring
        );
        let mut other = Fixture::new();
        assert!(matches!(
            other.writer.retire_message_preparation(&build),
            Err(Error::MessagePredecessorChanged)
        ));
    }
    #[test]
    fn changed_same_basis_metadata_and_failed_reservation_leave_no_uncommitted_handle() {
        let mut f = Fixture::new();
        f.writer._index.execute_batch("CREATE TRIGGER refuse_build BEFORE INSERT ON message_build BEGIN SELECT RAISE(ABORT,'synthetic reservation failure'); END;").unwrap();
        assert!(matches!(f.prepare(), Err(Error::Sqlite { .. })));
        assert_eq!(f.scalar("SELECT count(*) FROM message"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_candidate"), 0);
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_build")
            .unwrap();
        let build = f.prepare().unwrap().build;
        let error = f.writer.prepare_message(
            Identity {
                session: &f.session,
                origin: &f.origin,
                native_session: "session:with:colons",
                session_kind: SessionKind::Main,
                message: &f.message,
                sequence: 0,
                role: Role::User,
                written_by: WrittenBy::Model,
            },
            Source {
                predecessor: &f.head.token,
                revision: 1,
                evidence_revision: 1,
                logical_key: [3; 32],
                route: RouteKind::Native,
            },
            4096,
        );
        assert!(matches!(error, Err(Error::Invariant(_))));
        assert_eq!(f.prepare().unwrap().build, build);
        assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 1);
    }
    #[test]
    fn field_key_admission_is_before_mutation_and_failed_duplicate_does_not_lose_lease() {
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.value(&build, b"model");
        let slot = Slot::Extra {
            block: None,
            key: "x".repeat(1000),
        };
        assert!(matches!(
            f.writer.write_message_field(
                &build,
                &slot,
                None,
                true,
                &super::super::blocks::ProjectionStep {
                    expected_sequence: f.writer.message_projection(&build, 4096).unwrap().sequence,
                    next_cursor: b"field"
                },
                8
            ),
            Err(Error::ParserAdmission { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        assert!(f.prepare().unwrap().active_value.is_some());
        let slot = Slot::Message(MessageField::Model);
        f.writer
            .write_message_field(
                &build,
                &slot,
                None,
                true,
                &super::super::blocks::ProjectionStep {
                    expected_sequence: f.writer.message_projection(&build, 4096).unwrap().sequence,
                    next_cursor: b"field",
                },
                4096,
            )
            .unwrap();
        f.value(&build, b"other-model");
        assert!(matches!(
            f.writer.write_message_field(
                &build,
                &slot,
                None,
                true,
                &super::super::blocks::ProjectionStep {
                    expected_sequence: f.writer.message_projection(&build, 4096).unwrap().sequence,
                    next_cursor: b"field"
                },
                4096
            ),
            Err(Error::Invariant(_))
        ));
        assert!(f.prepare().unwrap().active_value.is_some());
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    }
    #[test]
    fn same_prefix_growth_preserves_preparation_but_changed_source_cannot_seal_it() {
        let mut f = Fixture::new();
        let length = f.head.acquired_upto;
        f.head = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: f.head.token.input_pk,
                chain: 1,
                parser_version: 3,
                digest: [11; 32],
                acquired_upto: 0,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let build = f.prepare().unwrap().build;
        let grown = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: f.head.token.input_pk,
                chain: 1,
                parser_version: 3,
                digest: [11; 32],
                acquired_upto: length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        assert_eq!(grown.token, build.source);
        let value = f
            .writer
            .begin_message_value(&build, values::Codec::Utf8, 0)
            .unwrap();
        let proof = f.writer.validate_value(&value.id).unwrap();
        f.writer
            .begin_parser(&parser_state::Basis {
                input_pk: f.head.token.input_pk,
                chain: 1,
                parser_version: 4,
                digest: [12; 32],
                acquired_upto: length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        assert!(matches!(
            f.writer.seal_message_value(&build, &proof),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(
            f.writer.value_head(&value.id).unwrap().state,
            values::State::Building
        );
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_body WHERE ready=1"),
            0
        );
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.writer.value_head(&value.id).unwrap().state,
            values::State::Retiring
        );
    }
    #[test]
    fn invalidated_preparation_drains_in_restartable_turns_and_does_not_block_new_basis() {
        use crate::contributions::{Retirement, ValueCollection};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        for i in 0..8 {
            f.value(&build, format!("value-{i}").as_bytes());
            f.writer
                .write_message_field(
                    &build,
                    &Slot::Extra {
                        block: None,
                        key: format!("field-{i}"),
                    },
                    None,
                    true,
                    &super::super::blocks::ProjectionStep {
                        expected_sequence: f
                            .writer
                            .message_projection(&build, 4096)
                            .unwrap()
                            .sequence,
                        next_cursor: b"field",
                    },
                    4096,
                )
                .unwrap();
        }
        assert!(matches!(
            f.writer.retire_message_turn(&build),
            Err(Error::MessagePreparationPending)
        ));
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.writer.retire_message_turn(&build).unwrap(),
            Retirement::Root
        );
        f.head = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: f.head.token.input_pk,
                chain: 1,
                parser_version: 4,
                digest: [19; 32],
                acquired_upto: f.head.acquired_upto,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let mut f = f.reopen();
        let before = (
            f.scalar("SELECT count(*) FROM message_body_node"),
            f.scalar("SELECT coalesce(sum(incoming),0) FROM message_body_node"),
        );
        f.writer._index.execute_batch("CREATE TRIGGER refuse_node_drain BEFORE DELETE ON message_body_edge BEGIN SELECT RAISE(ABORT,'synthetic node drain failure'); END;").unwrap();
        assert!(matches!(
            f.writer.retire_message_turn(&build),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(
            (
                f.scalar("SELECT count(*) FROM message_body_node"),
                f.scalar("SELECT coalesce(sum(incoming),0) FROM message_body_node")
            ),
            before
        );
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_node_drain")
            .unwrap();
        let mut turns = 0;
        loop {
            let before_nodes = f.scalar("SELECT count(*) FROM message_body_node");
            let before_facts = f.scalar("SELECT count(*) FROM message_field_fact");
            let step = f.writer.retire_message_turn(&build).unwrap();
            turns += 1;
            match step {
                Retirement::Node => assert_eq!(
                    f.scalar("SELECT count(*) FROM message_body_node"),
                    before_nodes - 1
                ),
                Retirement::Fact => assert_eq!(
                    f.scalar("SELECT count(*) FROM message_field_fact"),
                    before_facts - 1
                ),
                Retirement::Complete => break,
                _ => {}
            }
            if turns == 5 || turns == 20 {
                f = f.reopen();
            }
            assert!(turns < 200);
        }
        assert!(turns > 16);
        assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_candidate"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        assert_eq!(
            f.writer.retire_message_turn(&build).unwrap(),
            Retirement::Complete
        );
        let new = f.prepare().unwrap().build;
        assert_ne!(new, build);
        assert!(new.pk > build.pk);
        assert_eq!(
            f.writer.retire_message_turn(&build).unwrap(),
            Retirement::Complete
        );
        assert_eq!(f.prepare().unwrap().build, new);
        let mut chunks = 0;
        loop {
            let n = f.scalar("SELECT count(*) FROM body_value_chunk");
            match f.writer.collect_retiring_value().unwrap() {
                ValueCollection::Chunk => {
                    chunks += 1;
                    assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), n - 1);
                }
                ValueCollection::Value => {
                    assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), n)
                }
                ValueCollection::Idle => break,
            }
        }
        assert_eq!(chunks, 8);
        assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
        let revisions: i64 = f
            .writer
            .archive
            .query_row("SELECT count(*) FROM revision", [], |r| r.get(0))
            .unwrap();
        assert_eq!(revisions, 1);
    }
    #[test]
    fn retired_value_collection_deletes_one_chunk_and_keeps_unclaimed_generic_values() {
        use crate::contributions::{Retirement, ValueCollection};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let h = f
            .writer
            .begin_message_value(&build, values::Codec::Utf8, 65537)
            .unwrap();
        f.writer
            .append_value_chunk(&h.id, 0, &vec![b'x'; 65536], 65536)
            .unwrap();
        f.writer.append_value_chunk(&h.id, 65536, b"y", 1).unwrap();
        let generic = f.writer.begin_value(values::Codec::Utf8, 1).unwrap();
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.writer.retire_message_turn(&build).unwrap(),
            Retirement::Complete
        );
        assert_eq!(
            f.writer.collect_retiring_value().unwrap(),
            ValueCollection::Chunk
        );
        let mut f = f.reopen();
        assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), 1);
        assert_eq!(
            f.writer.collect_retiring_value().unwrap(),
            ValueCollection::Chunk
        );
        assert_eq!(
            f.writer.collect_retiring_value().unwrap(),
            ValueCollection::Value
        );
        assert_eq!(
            f.writer.collect_retiring_value().unwrap(),
            ValueCollection::Idle
        );
        assert_eq!(
            f.writer.value_head(&generic.id).unwrap().state,
            values::State::Building
        );
        assert!(matches!(
            f.writer.value_head(&h.id),
            Err(Error::ValuePredecessorChanged)
        ));
    }
    #[test]
    fn staged_thinking_restart_fields_install_and_source_counts() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        use polylogue_model::{BlockSlot, merge::BlockField};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.writer
            .begin_block(
                &build,
                4,
                91,
                BlockKind::Thinking { encrypted: None },
                BlockPresence {
                    source_block: true,
                    source_text: false,
                },
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"staged",
                },
                4096,
            )
            .unwrap();
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
        let mut f = f.reopen();
        assert_eq!(f.prepare().unwrap().build, build);
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_build WHERE active_block_pk IS NOT NULL"),
            1
        );
        assert!(!f.writer.collect_message_nodes(&f.message).unwrap());
        f.value(&build, b"false");
        let slot = Slot::Block {
            block: BlockSlot::Item(42),
            field: BlockField::ThinkingData,
        };
        assert!(matches!(
            f.writer.write_block_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"bad"
                },
                4096
            ),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        assert!(f.prepare().unwrap().active_value.is_some());
        assert!(matches!(
            f.writer.install_block(
                &build,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"bad"
                },
                4096
            ),
            Err(Error::ValueNotReady)
        ));
        // Carrier bytes only: this synthetic JSON makes no supported selector claim.
        let descriptor = b"{\"codec\":\"synthetic-unvalidated\"}";
        seal_owned_json(&mut f, &build, ValuePurpose::Origin, descriptor);
        f.writer
            .write_block_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"field",
                },
                4096,
            )
            .unwrap();
        assert!(matches!(
            f.writer.write_block_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"field"
                },
                4096
            ),
            Err(Error::ValueNotReady)
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        let signature = Slot::Block {
            block: BlockSlot::Item(0),
            field: BlockField::ThinkingSignature,
        };
        seal_owned_json(
            &mut f,
            &build,
            ValuePurpose::Normalized,
            b"\"opaque-signature\"",
        );
        let signature_step = ProjectionStep {
            expected_sequence: 2,
            next_cursor: b"signature",
        };
        assert!(matches!(
            f.writer
                .write_block_field(&build, &signature, None, true, &signature_step, 4096),
            Err(Error::Invariant(_))
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        f.writer
            .write_block_field(&build, &signature, None, false, &signature_step, 4096)
            .unwrap();
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 2);
        f.writer
            .install_block(
                &build,
                &ProjectionStep {
                    expected_sequence: 3,
                    next_cursor: b"installed",
                },
                4096,
            )
            .unwrap();
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_build WHERE active_block_pk IS NOT NULL"),
            0
        );
        f.writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 4,
                    next_cursor: b"done",
                },
                polylogue_model::BlobHash::of(b"evidence"),
                polylogue_model::merge::FinishState::Unknown,
                4096,
            )
            .unwrap();
        assert_eq!(f.scalar("SELECT source_fields FROM message_candidate"), 1);
        f.writer._index.execute_batch("BEGIN; UPDATE message_body SET ready=1,version_hash=(SELECT version_hash FROM message_candidate WHERE pk=message_body.preferred_candidate_pk); UPDATE message_candidate SET body_ready=1,ready=1,eligible=1; UPDATE message SET selected_body_pk=(SELECT pk FROM message_body); COMMIT;").unwrap();
        let reader = f.writer.read_handle().unwrap();
        let view = reader.selected_message(&f.message).unwrap().unwrap();
        let block = view.block(BlockSlot::Item(0)).unwrap().unwrap();
        assert_eq!(block.encrypted, None);
        assert!(block.text_range(0, 10).unwrap().is_none());
        let actual = Slot::Block {
            block: BlockSlot::Item(0),
            field: BlockField::ThinkingData,
        };
        match block.field(&actual, 4096).unwrap().unwrap() {
            super::super::reads::FieldRead::Value(field) => {
                assert_eq!(field.range(0, 100).unwrap().bytes, b"false");
                assert_eq!(
                    field.origin_range(0, 100).unwrap().unwrap().bytes,
                    descriptor
                );
            }
            _ => panic!("unexpected block field conflict"),
        }
        assert!(matches!(
            view.field(&actual, 4096).unwrap(),
            Some(super::super::reads::FieldRead::Value(_))
        ));
        assert!(block.field(&slot, 4096).is_err());
        match block.field(&signature, 4096).unwrap().unwrap() {
            super::super::reads::FieldRead::Value(field) => {
                assert!(!field.supplied);
                assert_eq!(field.range(0, 100).unwrap().bytes, b"\"opaque-signature\"");
            }
            _ => panic!("unexpected signature conflict"),
        }
    }
    #[test]
    fn two_blocks_share_local_semantic_keys_and_staged_abort_releases_only_private_leases() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        use polylogue_model::{BlockSlot, merge::BlockField};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        for n in 0..3 {
            f.value(&build, b"shared text");
            f.writer
                .begin_block(
                    &build,
                    0,
                    n,
                    BlockKind::Thinking {
                        encrypted: Some(false),
                    },
                    BlockPresence {
                        source_block: true,
                        source_text: true,
                    },
                    &ProjectionStep {
                        expected_sequence: n * 3,
                        next_cursor: b"begin",
                    },
                    4096,
                )
                .unwrap();
            f.value(&build, if n == 0 { b"field zero" } else { b"field one!" });
            f.writer
                .write_block_field(
                    &build,
                    &Slot::Block {
                        block: BlockSlot::Item(n as u32),
                        field: BlockField::ThinkingData,
                    },
                    None,
                    true,
                    &ProjectionStep {
                        expected_sequence: n * 3 + 1,
                        next_cursor: b"field",
                    },
                    4096,
                )
                .unwrap();
            if n < 2 {
                f.writer
                    .install_block(
                        &build,
                        &ProjectionStep {
                            expected_sequence: n * 3 + 2,
                            next_cursor: b"install",
                        },
                        4096,
                    )
                    .unwrap();
            }
        }
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 3);
        assert_eq!(
            f.scalar("SELECT count(DISTINCT value_pk) FROM message_field_fact"),
            2
        );
        assert_eq!(f.scalar("SELECT n.source_fields FROM message_body b JOIN message_body_node n ON n.pk=b.block_root_pk"),2);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), 1);
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 3);
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
        assert_eq!(
            f.scalar("SELECT count(*) FROM body_value WHERE state='ready'"),
            3
        );
        while f.writer.retire_message_turn(&build).unwrap() != Retirement::Complete {}
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    }
    #[test]
    fn empty_text_keeps_fields_without_completeness_and_active_block_blocks_finish() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        use polylogue_model::BlockSlot;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.value(&build, b"");
        f.writer
            .begin_block(
                &build,
                0,
                0,
                BlockKind::Text,
                BlockPresence::default(),
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"empty",
                },
                4096,
            )
            .unwrap();
        assert!(matches!(
            f.writer.finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"premature"
                },
                polylogue_model::BlobHash::of(b"evidence"),
                polylogue_model::merge::FinishState::Unknown,
                4096
            ),
            Err(Error::ValueNotReady)
        ));
        f.value(&build, b"false");
        let slot = Slot::Extra {
            block: Some(BlockSlot::Item(0)),
            key: String::new(),
        };
        assert!(matches!(
            f.writer.write_message_field(
                &build,
                &slot,
                None,
                false,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"wrong owner"
                },
                4096
            ),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            f.writer.write_block_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"wrong presence"
                },
                4096
            ),
            Err(Error::Invariant(_))
        ));
        f.writer
            .write_block_field(
                &build,
                &slot,
                None,
                false,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"metadata",
                },
                4096,
            )
            .unwrap();
        f.writer
            .install_block(
                &build,
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"installed",
                },
                4096,
            )
            .unwrap();
        f.writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 3,
                    next_cursor: b"done",
                },
                polylogue_model::BlobHash::of(b"evidence"),
                polylogue_model::merge::FinishState::Unknown,
                4096,
            )
            .unwrap();
        assert_eq!(
            f.scalar(
                "SELECT source_blocks+source_text_blocks+source_fields FROM message_candidate"
            ),
            0
        );
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    }
    #[test]
    fn dirty_field_set_remove_reuses_roots_and_preserves_pinned_old_value() {
        use super::super::{blocks::ProjectionStep, directory as d, reads::FieldRead};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let slot = Slot::Message(MessageField::Model);
        f.value(&build, b"old");
        f.writer
            .write_message_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"insert",
                },
                4096,
            )
            .unwrap();
        let path = f.root.join("index.db");
        let owner = d::Owner {
            message: build.message,
            session: build.session,
        };
        // Ready selection is synthetic fixture setup, not canonical election.
        // Both bodies genuinely lease the same immutable source-field root.
        let tx = f.writer._index.transaction().unwrap();
        // Separate immutable preference certificate keeps the active source
        // preparation unready and mutable while satisfying ready-body authority.
        let source_root: Option<i64> = tx
            .query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1",
                [build.body],
                |r| r.get(0),
            )
            .unwrap();
        let source_fields = d::summary(&tx, &path, owner, d::Kind::Fields, source_root)
            .unwrap()
            .source_fields;
        tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route,source_blocks,source_text_blocks,source_fields,version_hash,facts_ready,ready) SELECT message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route,0,0,?2,zeroblob(32),1,1 FROM message_candidate WHERE pk=?1",params![build.candidate,source_fields]).unwrap();
        let certificate = tx.last_insert_rowid();
        tx.execute("INSERT INTO message_body(message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version) SELECT message_pk,session_pk,1,zeroblob(32),contributor_revision,removal_epoch,?2,field_root_pk,layout,role,written_by,input_pk,zeroblob(32),parser_version FROM message_body WHERE pk=?1",params![build.body,certificate]).unwrap();
        let selected = tx.last_insert_rowid();
        let root: i64 = tx
            .query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1",
                [selected],
                |r| r.get(0),
            )
            .unwrap();
        d::acquire(&tx, &path, owner, root).unwrap();
        tx.execute("UPDATE message SET selected_body_pk=?1", [selected])
            .unwrap();
        tx.commit().unwrap();
        let reader = f.writer.read_handle().unwrap();
        let old = reader.selected_message(&f.message).unwrap().unwrap();
        f.value(&build, b"new");
        f.writer._index.execute_batch("CREATE TRIGGER refuse_field_cursor BEFORE UPDATE ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic checkpoint failure'); END;").unwrap();
        assert!(matches!(
            f.writer.set_message_field(
                &build,
                &slot,
                None,
                false,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"replace"
                },
                4096
            ),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        assert!(f.prepare().unwrap().active_value.is_some());
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_field_cursor")
            .unwrap();
        assert!(matches!(
            f.writer.set_message_field(
                &build,
                &slot,
                None,
                false,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"stale"
                },
                4096
            ),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
        f.writer
            .set_message_field(
                &build,
                &slot,
                None,
                false,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"replace",
                },
                4096,
            )
            .unwrap();
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 2);
        assert_eq!(f.scalar(&format!("SELECT n.source_fields FROM message_body b JOIN message_body_node n ON n.pk=b.field_root_pk WHERE b.pk={}",build.body)),0);
        match old.field(&slot, 4096).unwrap().unwrap() {
            FieldRead::Value(v) => assert_eq!(v.range(0, 100).unwrap().bytes, b"old"),
            _ => panic!("unexpected conflict"),
        }
        // Inspect the inherited root's replacement by the same typed snapshot.
        let tx = f.writer._index.transaction().unwrap();
        // Separate immutable preference certificate keeps the active source
        // preparation unready and mutable while satisfying ready-body authority.
        let source_root: Option<i64> = tx
            .query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1",
                [build.body],
                |r| r.get(0),
            )
            .unwrap();
        let source_fields = d::summary(&tx, &path, owner, d::Kind::Fields, source_root)
            .unwrap()
            .source_fields;
        tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route,source_blocks,source_text_blocks,source_fields,version_hash,facts_ready,ready) SELECT message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route,0,0,?2,zeroblob(32),1,1 FROM message_candidate WHERE pk=?1",params![build.candidate,source_fields]).unwrap();
        let certificate = tx.last_insert_rowid();
        tx.execute("INSERT INTO message_body(message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version) SELECT message_pk,session_pk,1,zeroblob(32),contributor_revision,removal_epoch,?2,field_root_pk,layout,role,written_by,input_pk,zeroblob(32),parser_version FROM message_body WHERE pk=?1",params![build.body,certificate]).unwrap();
        let selected = tx.last_insert_rowid();
        let root: i64 = tx
            .query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1",
                [selected],
                |r| r.get(0),
            )
            .unwrap();
        d::acquire(&tx, &path, owner, root).unwrap();
        tx.execute("UPDATE message SET selected_body_pk=?1", [selected])
            .unwrap();
        tx.commit().unwrap();
        let newer = reader.selected_message(&f.message).unwrap().unwrap();
        match newer.field(&slot, 4096).unwrap().unwrap() {
            FieldRead::Value(v) => {
                assert!(!v.supplied);
                assert_eq!(v.range(0, 100).unwrap().bytes, b"new")
            }
            _ => panic!("unexpected conflict"),
        }
        f.writer
            .remove_message_field(
                &build,
                &slot,
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"removed",
                },
                4096,
            )
            .unwrap();
        assert_eq!(f.scalar(&format!("SELECT n.source_fields FROM message_body b JOIN message_body_node n ON n.pk=b.field_root_pk WHERE b.pk={}",build.body)),0);
        // Both old pinned bodies still resolve their own retained value.
        assert!(old.field(&slot, 4096).unwrap().is_some());
        assert!(newer.field(&slot, 4096).unwrap().is_some());
        let slots = f.scalar("SELECT count(*) FROM message_field_slot");
        f.writer
            .remove_message_field(
                &build,
                &Slot::Extra {
                    block: None,
                    key: "never supplied".into(),
                },
                &ProjectionStep {
                    expected_sequence: 3,
                    next_cursor: b"absent",
                },
                4096,
            )
            .unwrap();
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), slots);
        assert!(matches!(
            f.writer.remove_message_field(
                &build,
                &slot,
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"replay"
                },
                4096
            ),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            4
        );
    }
    #[test]
    fn explicit_set_inserts_but_remove_never_discards_an_active_value() {
        use super::super::blocks::ProjectionStep;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let slot = Slot::Message(MessageField::Model);
        f.value(&build, b"model");
        assert!(matches!(
            f.writer.remove_message_field(
                &build,
                &slot,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"lost"
                },
                4096
            ),
            Err(Error::MessagePreparationPending)
        ));
        assert!(f.prepare().unwrap().active_value.is_some());
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), 0);
        f.writer
            .set_message_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"new",
                },
                4096,
            )
            .unwrap();
        f.writer
            .remove_message_field(
                &build,
                &slot,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"gone",
                },
                4096,
            )
            .unwrap();
        f.writer
            .remove_message_field(
                &build,
                &slot,
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"still gone",
                },
                4096,
            )
            .unwrap();
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            3
        );
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    }
    fn advance_source_fixture(f: &mut Fixture) {
        let path = f.root.join("index.db");
        f.head = parser_state::write_step(
            &mut f.writer._index,
            &path,
            &f.head,
            parser_state::Progress {
                inspected_upto: 0,
                parsed_upto: 0,
                phase: parser_state::Phase::Parsing,
                finalization_pending: true,
                counts: Default::default(),
            },
            &[],
            parser_state::Budget {
                cells: 0,
                encoded_bytes: 0,
            },
            |_| Ok(()),
        )
        .unwrap();
    }
    fn source_json_fixture(f: &mut Fixture, build: &Build, bytes: &[u8]) -> values::ReadyValue {
        let head = f
            .writer
            .begin_message_value(build, values::Codec::Json, bytes.len() as u64)
            .unwrap();
        f.writer
            .append_value_chunk(&head.id, 0, bytes, bytes.len())
            .unwrap();
        let reader = f.writer.read_handle().unwrap();
        let mut proof = f.writer.validate_value(&head.id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, 64).unwrap();
        }
        drop(reader);
        f.writer.seal_message_value(build, &proof).unwrap().ready
    }
    fn finish_source_fixture(f: &mut Fixture, build: &Build) -> super::super::PreparedProjection {
        let sequence = f.writer.message_projection(build, 4096).unwrap().sequence;
        f.writer
            .finish_message_projection(
                build,
                &super::super::blocks::ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"source done",
                },
                polylogue_model::BlobHash::of(b"synthetic evidence unchanged"),
                polylogue_model::merge::FinishState::Unknown,
                4096,
            )
            .unwrap()
    }
    #[test]
    fn actual_source_certificate_reopens_inherits_and_replaces_only_current_cut_atomically() {
        use super::super::blocks::ProjectionStep;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let slot = Slot::Message(MessageField::At);
        // Synthetic provider owns this fixed digest; this test qualifies store
        // control/lease atomicity, not provider evidence membership semantics.
        let old_value = source_json_fixture(
            &mut f,
            &build,
            br#"{"at":1,"precision":"instant","time_quality":"intrinsic","until":null}"#,
        );
        f.writer
            .write_message_field(
                &build,
                &slot,
                Some(1),
                true,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"at",
                },
                4096,
            )
            .unwrap();
        let prepared = finish_source_fixture(&mut f, &build);
        let old = f
            .writer
            .publish_source_projection(&prepared, None, b"old source head", 4096)
            .unwrap();
        assert_eq!(old.message(), &f.message);
        assert_eq!(old.evidence_revision(), 1);
        assert_eq!(old.evidence_guard(), b"old source head");
        assert_eq!(old.predecessor(), &f.head.token);
        assert_eq!(old.body().pk, build.body);
        assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
        assert_eq!(
            f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
            0
        );
        let mut f = f.reopen();
        let old = f
            .writer
            .current_source_projection(&f.head.token, &[3; 32], 4096)
            .unwrap()
            .unwrap();
        assert_eq!(old.version_hash(), prepared.version_hash());
        assert!(
            f.writer
                .read_handle()
                .unwrap()
                .selected_message(&f.message)
                .unwrap()
                .is_none()
        );
        advance_source_fixture(&mut f);
        let new = f.prepare().unwrap().build;
        f.writer
            .inherit_message_projection(
                &new,
                old.body(),
                old.predecessor(),
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"inherited",
                },
                4096,
            )
            .unwrap();
        let new_value = source_json_fixture(
            &mut f,
            &new,
            br#"{"at":2,"precision":"instant","time_quality":"intrinsic","until":null}"#,
        );
        f.writer
            .set_message_field(
                &new,
                &slot,
                Some(2),
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"dirty at",
                },
                4096,
            )
            .unwrap();
        let completed = finish_source_fixture(&mut f, &new);
        assert!(matches!(
            f.writer
                .publish_source_projection(&completed, Some(&old), b"new source head", 1),
            Err(Error::ParserAdmission { .. })
        ));
        assert_eq!(
            f.scalar(&format!(
                "SELECT ready FROM message_body WHERE pk={}",
                new.body
            )),
            0
        );
        f.writer._index.execute_batch("CREATE TRIGGER refuse_source_delete BEFORE DELETE ON message_build BEGIN SELECT RAISE(ABORT,'late source publish failure'); END;").unwrap();
        assert!(matches!(
            f.writer
                .publish_source_projection(&completed, Some(&old), b"new source head", 4096),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(
            f.scalar(&format!(
                "SELECT eligible FROM message_candidate WHERE pk={}",
                build.candidate
            )),
            1
        );
        assert_eq!(
            f.scalar(&format!(
                "SELECT ready FROM message_body WHERE pk={}",
                new.body
            )),
            0
        );
        let current = f
            .writer
            .current_source_projection(&f.head.token, &[3; 32], 4096)
            .unwrap()
            .unwrap();
        assert_eq!(current.body(), old.body());
        assert_eq!(current.evidence_guard(), b"old source head");
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_source_delete")
            .unwrap();
        let published = f
            .writer
            .publish_source_projection(&completed, Some(&old), b"new source head", 4096)
            .unwrap();
        assert_eq!(published.version_hash(), old.version_hash());
        assert_eq!(published.body().pk, new.body);
        assert_eq!(published.evidence_guard(), b"new source head");
        assert_eq!(
            f.scalar(&format!(
                "SELECT eligible FROM message_candidate WHERE pk={}",
                build.candidate
            )),
            0
        );
        assert_eq!(
            f.scalar(&format!(
                "SELECT eligible FROM message_candidate WHERE pk={}",
                new.candidate
            )),
            1
        );
        assert_eq!(
            f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
            0
        );
        assert!(matches!(
            f.writer
                .publish_source_projection(&completed, Some(&old), b"replay", 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        let reader = f.writer.read_handle().unwrap();
        assert_eq!(
            reader.value_range(&old_value.id, 0, 256).unwrap().bytes,
            br#"{"at":1,"precision":"instant","time_quality":"intrinsic","until":null}"#
        );
        assert_eq!(
            reader.value_range(&new_value.id, 0, 256).unwrap().bytes,
            br#"{"at":2,"precision":"instant","time_quality":"intrinsic","until":null}"#
        );
        drop(reader);
        let mut f = f.reopen();
        assert_eq!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 4096)
                .unwrap()
                .unwrap()
                .body(),
            published.body()
        );
        advance_source_fixture(&mut f);
        let third = f.prepare().unwrap().build;
        let completed = finish_source_fixture(&mut f, &third);
        assert!(matches!(
            f.writer
                .publish_source_projection(&completed, Some(&old), b"stale previous", 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 4096)
                .unwrap()
                .unwrap()
                .body(),
            published.body()
        );
    }
    #[test]
    fn current_source_reference_corruption_admission_and_wrong_handles_are_explicit() {
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let prepared = finish_source_fixture(&mut f, &build);
        let prior = f
            .writer
            .publish_source_projection(&prepared, None, b"head", 4096)
            .unwrap();
        assert!(matches!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 1),
            Err(Error::ParserAdmission { .. })
        ));
        let mut wrong = f.head.token.clone();
        wrong.basis[0] ^= 1;
        assert!(matches!(
            f.writer.current_source_projection(&wrong, &[3; 32], 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        wrong = f.head.token.clone();
        wrong.generation += 1;
        assert!(matches!(
            f.writer.current_source_projection(&wrong, &[3; 32], 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        wrong = f.head.token.clone();
        wrong.index_generation.push('x');
        assert!(matches!(
            f.writer.current_source_projection(&wrong, &[3; 32], 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        advance_source_fixture(&mut f);
        f.message = MessageId::new(f.session.clone(), "other message").unwrap();
        let other = f.prepare().unwrap().build;
        let completed = finish_source_fixture(&mut f, &other);
        assert!(matches!(
            f.writer
                .publish_source_projection(&completed, Some(&prior), b"wrong identity", 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(
            f.scalar(&format!(
                "SELECT ready FROM message_body WHERE pk={}",
                other.body
            )),
            0
        );
        f.writer
            ._index
            .execute(
                "UPDATE parser_cell SET value=?1 WHERE namespace='message.source.v1' AND key=?2",
                params![b"corrupt reference".as_slice(), [3_u8; 32].as_slice()],
            )
            .unwrap();
        assert!(matches!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 4096),
            Err(Error::Invariant(_))
        ));
        assert!(
            f.writer
                .current_source_projection(&f.head.token, &[9; 32], 4096)
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn uncontested_source_selection_is_atomic_and_contested_stays_pending() {
        use super::super::{BlockKind, BlockPresence, ProjectionStep, SourceSelection};
        use polylogue_model::{BlobHash, BlockSlot, merge::FinishState};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.value(&build, b"selected text");
        f.writer
            .write_block(
                &build,
                0,
                0,
                BlockKind::Text,
                BlockPresence {
                    source_block: true,
                    source_text: true,
                },
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"text",
                },
                4096,
            )
            .unwrap();
        let done = f
            .writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"complete",
                },
                BlobHash::of(b"producer evidence"),
                FinishState::Unknown,
                4096,
            )
            .unwrap();
        let source = f
            .writer
            .publish_source_projection(&done, None, b"evidence guard", 4096)
            .unwrap();
        let reader = f.writer.read_handle().unwrap();
        assert!(reader.selected_message(&f.message).unwrap().is_none());
        let before = f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'");
        f.writer._index.execute_batch("CREATE TRIGGER refuse_selected_session BEFORE UPDATE OF changed ON session BEGIN SELECT RAISE(ABORT,'synthetic selected session failure'); END;").unwrap();
        assert!(
            f.writer
                .select_uncontested_source_message(&source, 4096)
                .is_err()
        );
        assert!(reader.selected_message(&f.message).unwrap().is_none());
        assert_eq!(
            f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'"),
            before
        );
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_selected_session")
            .unwrap();
        assert_eq!(
            f.writer
                .select_uncontested_source_message(&source, 4096)
                .unwrap(),
            SourceSelection::Selected
        );
        assert_eq!(
            f.writer
                .select_uncontested_source_message(&source, 4096)
                .unwrap(),
            SourceSelection::AlreadySelected
        );
        assert_eq!(
            f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'"),
            before + 1
        );
        let view = reader.selected_message(&f.message).unwrap().unwrap();
        assert_eq!(
            view.block(BlockSlot::Item(0))
                .unwrap()
                .unwrap()
                .text_range(0, 64)
                .unwrap()
                .unwrap()
                .bytes,
            b"selected text"
        );
        // Another input supplies an independently eligible candidate. This
        // initial selector must not silently choose one side or fill by arrival.
        let blob = f
            .writer
            .put_file_blob(b"different immutable input\n")
            .unwrap();
        let input = f
            .writer
            .retain_immutable_input(
                blob,
                InputFormat::CodexRollout,
                &[],
                &ImportObservation {
                    id: "second observation",
                    machine: "synthetic",
                    host: "synthetic",
                    path: "/synthetic/second.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        f.head = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: 3,
                digest: [7; 32],
                acquired_upto: input.length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let other = f.prepare().unwrap().build;
        let done = f
            .writer
            .finish_message_projection(
                &other,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"complete",
                },
                BlobHash::of(b"other evidence"),
                FinishState::Unknown,
                4096,
            )
            .unwrap();
        let other = f
            .writer
            .publish_source_projection(&done, None, b"other evidence guard", 4096)
            .unwrap();
        assert_eq!(
            f.writer
                .select_uncontested_source_message(&other, 4096)
                .unwrap(),
            SourceSelection::Contested
        );
        assert_eq!(
            f.writer
                .select_uncontested_source_message(&source, 4096)
                .unwrap(),
            SourceSelection::Contested
        );
        assert_eq!(
            f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'"),
            before + 1
        );
        assert_eq!(
            view.block(BlockSlot::Item(0))
                .unwrap()
                .unwrap()
                .text_range(0, 64)
                .unwrap()
                .unwrap()
                .bytes,
            b"selected text"
        );
        let selected: i64 = f
            .writer
            ._index
            .query_row(
                "SELECT selected_body_pk FROM message WHERE id=?1",
                [f.message.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(selected, build.body);
    }
    #[test]
    fn reference_reads_distinguish_pending_and_preserve_interleaved_reasoning_source_order() {
        use super::super::{
            BlockKind, BlockPresence, MessageLayout, MessageLookup, ProjectionStep,
        };
        use polylogue_model::BlockSlot;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let reader = f.writer.read_handle().unwrap();
        assert!(matches!(
            reader.message_reference(f.message.as_str(), 4096).unwrap(),
            MessageLookup::Pending
        ));
        assert!(matches!(
            reader.message_reference("absent", 4096).unwrap(),
            MessageLookup::Absent
        ));
        assert!(matches!(
            reader.message_reference(f.message.as_str(), 1),
            Err(Error::ParserAdmission { .. })
        ));
        f.writer
            .set_message_layout(
                &build,
                MessageLayout::SeparateReasoning,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"layout",
                },
                4096,
            )
            .unwrap();
        for (group, local, kind, text) in [
            (
                7,
                100,
                BlockKind::Thinking { encrypted: None },
                "reason one",
            ),
            (7, 500, BlockKind::Text, "answer one"),
            (
                900,
                0,
                BlockKind::Thinking {
                    encrypted: Some(false),
                },
                "reason two",
            ),
            (900, 2, BlockKind::Text, "answer two"),
        ] {
            f.value(&build, text.as_bytes());
            let sequence = f.writer.message_projection(&build, 4096).unwrap().sequence;
            f.writer
                .write_block(
                    &build,
                    group,
                    local,
                    kind,
                    BlockPresence {
                        source_block: true,
                        source_text: true,
                    },
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"block",
                    },
                    4096,
                )
                .unwrap();
        }
        assert!(matches!(
            reader.message_reference(f.message.as_str(), 4096).unwrap(),
            MessageLookup::Pending
        ));
        let prepared = finish_source_fixture(&mut f, &build);
        let source = f
            .writer
            .publish_source_projection(&prepared, None, b"synthetic selected evidence", 4096)
            .unwrap();
        f.writer
            .select_uncontested_source_message(&source, 4096)
            .unwrap();
        let MessageLookup::Ready(view) =
            reader.message_reference(f.message.as_str(), 4096).unwrap()
        else {
            panic!("ready selected reference")
        };
        assert_eq!(view.id(), &f.message);
        assert_eq!(view.header().unwrap().role, Role::Assistant);
        assert_eq!(view.header().unwrap().written_by, WrittenBy::Model);
        for (index, slot, text) in [
            (0, BlockSlot::Reasoning(0), "reason one"),
            (1, BlockSlot::Item(0), "answer one"),
            (2, BlockSlot::Reasoning(1), "reason two"),
            (3, BlockSlot::Item(1), "answer two"),
        ] {
            let block = view.block_at(index).unwrap().unwrap();
            assert_eq!(block.slot, slot);
            assert_eq!(
                block.text_range(0, 100).unwrap().unwrap().bytes,
                text.as_bytes()
            );
            let same = view.block(slot).unwrap().unwrap();
            assert_eq!((same.group, same.local), (block.group, block.local));
        }
        assert!(view.block_at(4).unwrap().is_none());
    }
    #[test]
    fn exact_disjoint_source_set_retires_one_range_per_abort_or_candidate_turn() {
        use super::super::{BodyRetirement, CandidateCollection};
        fn ranges(f: &mut Fixture, build: &Build) {
            for (start, end) in [(0, 1), (3, 5), (10, 12)] {
                f.writer._index.execute("INSERT INTO message_source(candidate_pk,message_pk,session_pk,input_pk,chain,source_start,source_end) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![build.candidate,build.message,build.session,build.source.input_pk,build.source.chain,start,end]).unwrap();
            }
            assert!(f.writer._index.execute("INSERT INTO message_source(candidate_pk,message_pk,session_pk,input_pk,chain,source_start,source_end) VALUES(?1,?2,?3,?4,?5,0,1)",params![build.candidate,build.message,build.session,build.source.input_pk,build.source.chain]).is_err());
        }
        // This is a schema/retirement control; the producer supplies these
        // synthetic ranges directly, without claiming provider interpretation.
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        ranges(&mut f, &build);
        f.writer.retire_message_preparation(&build).unwrap();
        f.writer._index.execute_batch("CREATE TRIGGER refuse_locator_delete BEFORE DELETE ON message_source BEGIN SELECT RAISE(ABORT,'synthetic locator delete failure'); END;").unwrap();
        assert!(matches!(
            f.writer.retire_message_turn(&build),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_source"), 3);
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_locator_delete")
            .unwrap();
        for remaining in (0..3).rev() {
            assert_eq!(
                f.writer.retire_message_turn(&build).unwrap(),
                Retirement::Source
            );
            assert_eq!(f.scalar("SELECT count(*) FROM message_source"), remaining);
        }
        assert_eq!(
            f.writer.retire_message_turn(&build).unwrap(),
            Retirement::Complete
        );

        let mut f = Fixture::new();
        let old = f.prepare().unwrap().build;
        ranges(&mut f, &old);
        let candidate = old.candidate_id();
        let completed = finish_source_fixture(&mut f, &old);
        let previous = f
            .writer
            .publish_source_projection(&completed, None, b"old source guard", 4096)
            .unwrap();
        advance_source_fixture(&mut f);
        let new = f.prepare().unwrap().build;
        let completed = finish_source_fixture(&mut f, &new);
        f.writer
            .publish_source_projection(&completed, Some(&previous), b"new source guard", 4096)
            .unwrap();
        assert_eq!(
            f.writer.retire_message_body(previous.body()).unwrap(),
            BodyRetirement::DetachedCandidate
        );
        assert_eq!(
            f.writer.retire_message_body(previous.body()).unwrap(),
            BodyRetirement::Body
        );
        for remaining in (0..3).rev() {
            assert_eq!(
                f.writer.collect_message_candidate(&candidate).unwrap(),
                CandidateCollection::Source
            );
            assert_eq!(f.scalar("SELECT count(*) FROM message_source"), remaining);
        }
        assert_eq!(
            f.writer.collect_message_candidate(&candidate).unwrap(),
            CandidateCollection::Candidate
        );
    }
    // These fixtures qualify descriptor-byte leases only. Their JSON has no
    // supported selector/membership proof and certifies no source provenance.
    fn seal_owned_json(
        f: &mut Fixture,
        build: &Build,
        purpose: ValuePurpose,
        bytes: &[u8],
    ) -> values::ReadyValue {
        let head = match purpose {
            ValuePurpose::Normalized => {
                f.writer
                    .begin_message_value(build, values::Codec::Json, bytes.len() as u64)
            }
            ValuePurpose::Origin => f
                .writer
                .begin_message_origin_value(build, bytes.len() as u64),
        }
        .unwrap();
        if !bytes.is_empty() {
            f.writer
                .append_value_chunk(&head.id, 0, bytes, bytes.len())
                .unwrap();
        }
        finish_owned_json(f, build, purpose, &head.id)
    }
    fn finish_owned_json(
        f: &mut Fixture,
        build: &Build,
        purpose: ValuePurpose,
        id: &values::ValueId,
    ) -> values::ReadyValue {
        let reader = f.writer.read_handle().unwrap();
        let mut proof = reader.validate_value(id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, 2).unwrap();
        }
        drop(reader);
        match purpose {
            ValuePurpose::Normalized => f.writer.seal_message_value(build, &proof),
            ValuePurpose::Origin => f.writer.seal_message_origin_value(build, &proof),
        }
        .unwrap()
        .ready
    }
    #[test]
    fn normalized_and_origin_producers_resume_and_transfer_with_one_cursor_commit() {
        use super::super::blocks::ProjectionStep;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let normalized = seal_owned_json(&mut f, &build, ValuePurpose::Normalized, b"false");
        let origin = f.writer.begin_message_origin_value(&build, 2).unwrap();
        f.writer.append_value_chunk(&origin.id, 0, b"{", 1).unwrap();
        let mut f = f.reopen();
        let resumed = f.prepare().unwrap();
        assert_eq!(resumed.active_value.unwrap().id, normalized.id);
        let resumed_origin = resumed.active_origin_value.unwrap();
        assert_eq!(resumed_origin.id, origin.id);
        assert_eq!(resumed_origin.appended, 1);
        let slot = Slot::Message(MessageField::Model);
        let step = ProjectionStep {
            expected_sequence: 0,
            next_cursor: b"field",
        };
        assert!(matches!(
            f.writer
                .write_message_field(&build, &slot, None, true, &step, 4096),
            Err(Error::ValueNotReady)
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        f.writer.append_value_chunk(&origin.id, 1, b"}", 1).unwrap();
        let descriptor = finish_owned_json(&mut f, &build, ValuePurpose::Origin, &origin.id);
        f.writer._index.execute_batch("CREATE TRIGGER refuse_origin_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'origin checkpoint failure'); END;").unwrap();
        assert!(matches!(
            f.writer
                .write_message_field(&build, &slot, None, true, &step, 4096),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        let resumed = f.prepare().unwrap();
        assert_eq!(resumed.active_value.unwrap().id, normalized.id);
        assert_eq!(resumed.active_origin_value.unwrap().id, descriptor.id);
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_origin_cursor")
            .unwrap();
        assert!(matches!(
            f.writer.write_message_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"stale"
                },
                4096
            ),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        f.writer
            .write_message_field(&build, &slot, None, true, &step, 4096)
            .unwrap();
        let resumed = f.prepare().unwrap();
        assert!(resumed.active_value.is_none());
        assert!(resumed.active_origin_value.is_none());
        assert_eq!(
            f.scalar("SELECT origin_value_pk FROM message_field_fact"),
            descriptor.id.pk()
        );
        assert_eq!(
            f.scalar("SELECT value_pk FROM message_field_fact"),
            normalized.id.pk()
        );
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            1
        );
    }
    #[test]
    fn aliased_normalized_origin_values_keep_two_leases_until_fact_retirement() {
        use super::super::blocks::ProjectionStep;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let normalized = seal_owned_json(&mut f, &build, ValuePurpose::Normalized, b"null");
        let origin = seal_owned_json(&mut f, &build, ValuePurpose::Origin, b"null");
        assert_eq!(normalized.id, origin.id);
        assert_eq!(
            f.scalar(&format!(
                "SELECT incoming FROM body_value WHERE pk={}",
                origin.id.pk()
            )),
            2
        );
        f.writer
            .write_message_field(
                &build,
                &Slot::Message(MessageField::Model),
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"pair",
                },
                4096,
            )
            .unwrap();
        assert_eq!(
            f.scalar("SELECT value_pk=origin_value_pk FROM message_field_fact"),
            1
        );
        assert_eq!(
            f.scalar(&format!(
                "SELECT incoming FROM body_value WHERE pk={}",
                origin.id.pk()
            )),
            2
        );
        f.writer.retire_message_preparation(&build).unwrap();
        while f.writer.retire_message_turn(&build).unwrap() != Retirement::Complete {}
        assert_eq!(
            f.scalar(&format!(
                "SELECT incoming FROM body_value WHERE pk={}",
                origin.id.pk()
            )),
            0
        );
        assert_eq!(
            f.writer.value_head(&origin.id).unwrap().state,
            values::State::Retiring
        );
        while f.writer.collect_retiring_value().unwrap() != super::super::ValueCollection::Idle {}
        assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
        assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), 0);
    }
    #[test]
    fn pending_origin_blocks_skip_install_finish_and_abort_releases_both_aliases() {
        use super::super::blocks::{BlockKind, BlockPresence, MessageLayout, ProjectionStep};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let origin = seal_owned_json(&mut f, &build, ValuePurpose::Origin, b"null");
        let step = ProjectionStep {
            expected_sequence: 0,
            next_cursor: b"skip",
        };
        assert!(matches!(
            f.writer.advance_message_projection(&build, &step, 4096),
            Err(Error::MessagePreparationPending)
        ));
        assert!(matches!(
            f.writer
                .set_message_layout(&build, MessageLayout::Flat, &step, 4096),
            Err(Error::MessagePreparationPending)
        ));
        assert!(matches!(
            f.writer.begin_block(
                &build,
                0,
                0,
                BlockKind::Thinking { encrypted: None },
                BlockPresence::default(),
                &step,
                4096
            ),
            Err(Error::MessagePreparationPending)
        ));
        assert!(matches!(
            f.writer
                .remove_message_field(&build, &Slot::Message(MessageField::Model), &step, 4096),
            Err(Error::MessagePreparationPending)
        ));
        assert!(matches!(
            f.writer.finish_message_projection(
                &build,
                &step,
                polylogue_model::BlobHash::of(b"synthetic"),
                polylogue_model::merge::FinishState::Unknown,
                4096
            ),
            Err(Error::ValueNotReady)
        ));
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            0
        );
        let normal = seal_owned_json(&mut f, &build, ValuePurpose::Normalized, b"null");
        assert_eq!(normal.id, origin.id);
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.scalar(&format!(
                "SELECT incoming FROM body_value WHERE pk={}",
                origin.id.pk()
            )),
            0
        );
        assert_eq!(
            f.writer.value_head(&origin.id).unwrap().state,
            values::State::Retiring
        );
    }
    #[test]
    fn replacing_a_fact_retains_shared_origin_until_its_last_owner_is_drained() {
        use super::super::blocks::ProjectionStep;
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        let slot = Slot::Message(MessageField::Model);
        seal_owned_json(&mut f, &build, ValuePurpose::Normalized, b"\"first\"");
        let origin = seal_owned_json(
            &mut f,
            &build,
            ValuePurpose::Origin,
            b"{\"codec\":\"synthetic-unvalidated\"}",
        );
        f.writer
            .write_message_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"first",
                },
                4096,
            )
            .unwrap();
        seal_owned_json(&mut f, &build, ValuePurpose::Normalized, b"\"second\"");
        let again = seal_owned_json(
            &mut f,
            &build,
            ValuePurpose::Origin,
            b"{\"codec\":\"synthetic-unvalidated\"}",
        );
        assert_eq!(origin.id, again.id);
        f.writer
            .set_message_field(
                &build,
                &slot,
                None,
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"second",
                },
                4096,
            )
            .unwrap();
        assert_eq!(
            f.scalar(&format!(
                "SELECT incoming FROM body_value WHERE pk={}",
                origin.id.pk()
            )),
            2
        );
        f.writer.retire_message_preparation(&build).unwrap();
        let mut facts = 0;
        loop {
            let result = f.writer.retire_message_turn(&build).unwrap();
            if result == Retirement::Fact {
                facts += 1;
                let head = f.writer.value_head(&origin.id).unwrap();
                if facts == 1 {
                    assert_eq!(head.state, values::State::Ready);
                    assert_eq!(
                        f.scalar(&format!(
                            "SELECT incoming FROM body_value WHERE pk={}",
                            origin.id.pk()
                        )),
                        1
                    );
                } else {
                    assert_eq!(head.state, values::State::Retiring);
                }
            }
            if result == Retirement::Complete {
                break;
            }
        }
        assert_eq!(facts, 2);
    }
    #[test]
    fn pending_block_origin_cannot_be_bypassed_by_install_or_atomic_block_write() {
        use super::super::blocks::{BlockKind, BlockPresence, ProjectionStep};
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.writer
            .begin_block(
                &build,
                0,
                0,
                BlockKind::Thinking { encrypted: None },
                BlockPresence::default(),
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"block",
                },
                4096,
            )
            .unwrap();
        let origin = seal_owned_json(&mut f, &build, ValuePurpose::Origin, b"null");
        let step = ProjectionStep {
            expected_sequence: 1,
            next_cursor: b"install",
        };
        assert!(matches!(
            f.writer.install_block(&build, &step, 4096),
            Err(Error::MessagePreparationPending)
        ));
        assert!(matches!(
            f.writer.write_block(
                &build,
                1,
                0,
                BlockKind::Thinking { encrypted: None },
                BlockPresence::default(),
                &step,
                4096
            ),
            Err(Error::MessagePreparationPending)
        ));
        let resumed = f.prepare().unwrap();
        assert_eq!(
            f.writer.message_projection(&build, 4096).unwrap().sequence,
            1
        );
        assert_eq!(resumed.active_origin_value.unwrap().id, origin.id);
        assert_eq!(f.scalar("SELECT count(*) FROM block"), 1);
        f.writer.retire_message_preparation(&build).unwrap();
        assert_eq!(
            f.writer.value_head(&origin.id).unwrap().state,
            values::State::Retiring
        );
    }
    fn reparse_source_fixture(f: &mut Fixture) {
        let old = f.head.clone();
        f.head = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: old.token.input_pk,
                chain: old.token.chain,
                parser_version: old.parser_version,
                digest: [8; 32],
                acquired_upto: old.acquired_upto,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        assert!(f.head.token.generation > old.token.generation);
        assert_eq!(f.head.parser_version, old.parser_version);
        // Collect the actual old receipt through the existing indexed drain.
        loop {
            let result = parser_state::collect_retired(
                &mut f.writer._index,
                &f.root.join("index.db"),
                &f.head.token,
                parser_state::Budget {
                    cells: 1,
                    encoded_bytes: 4096,
                },
            )
            .unwrap();
            if !result.remaining {
                break;
            }
        }
        assert_eq!(f.scalar("SELECT count(*) FROM parser_cell"), 0);
    }
    #[test]
    fn reparse_certificate_survives_cell_collection_and_late_failure_restart() {
        let mut f = Fixture::new();
        let old_build = f.prepare().unwrap().build;
        let done = finish_source_fixture(&mut f, &old_build);
        let old = f
            .writer
            .publish_source_projection(&done, None, b"old", 4096)
            .unwrap();
        f.writer
            .select_uncontested_source_message(&old, 4096)
            .unwrap();
        reparse_source_fixture(&mut f);
        let new = f.prepare().unwrap().build;
        let done = finish_source_fixture(&mut f, &new);
        let mut f = f.reopen();
        f.writer._index.execute_batch("CREATE TRIGGER refuse_reparse_delete BEFORE DELETE ON message_build BEGIN SELECT RAISE(ABORT,'late reparse failure'); END;").unwrap();
        assert!(matches!(
            f.writer
                .publish_source_projection(&done, None, b"new", 4096),
            Err(Error::Sqlite { .. })
        ));
        assert_eq!(
            f.scalar(&format!(
                "SELECT eligible FROM message_candidate WHERE pk={}",
                old_build.candidate
            )),
            1
        );
        assert_eq!(
            f.scalar(&format!(
                "SELECT ready FROM message_body WHERE pk={}",
                new.body
            )),
            0
        );
        assert!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 4096)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            &f.writer
                .read_handle()
                .unwrap()
                .selected_message(&f.message)
                .unwrap()
                .unwrap()
                .body_id(),
            old.body()
        );
        f.writer
            ._index
            .execute_batch("DROP TRIGGER refuse_reparse_delete")
            .unwrap();
        let replacement = f
            .writer
            .publish_source_projection(&done, None, b"new", 4096)
            .unwrap();
        assert_eq!(
            f.scalar(&format!(
                "SELECT eligible FROM message_candidate WHERE pk={}",
                old_build.candidate
            )),
            0
        );
        assert_eq!(
            &f.writer
                .read_handle()
                .unwrap()
                .selected_message(&f.message)
                .unwrap()
                .unwrap()
                .body_id(),
            old.body()
        );
        f.writer
            .select_uncontested_source_message(&replacement, 4096)
            .unwrap();
        let f = f.reopen();
        assert_eq!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 4096)
                .unwrap()
                .unwrap()
                .body(),
            replacement.body()
        );
    }
    #[test]
    fn reparse_refuses_missing_same_generation_receipt_and_inconsistent_certificates() {
        for damage in [
            "missing same generation",
            "old generation",
            "old step",
            "old source basis",
            "old preparation basis",
            "future generation",
            "new source basis",
        ] {
            let mut f = Fixture::new();
            let old_build = f.prepare().unwrap().build;
            let done = finish_source_fixture(&mut f, &old_build);
            let old = f
                .writer
                .publish_source_projection(&done, None, b"old", 4096)
                .unwrap();
            f.writer
                .select_uncontested_source_message(&old, 4096)
                .unwrap();
            if damage == "missing same generation" {
                advance_source_fixture(&mut f);
                f.writer
                    ._index
                    .execute(
                        "DELETE FROM parser_cell WHERE namespace='message.source.v1'",
                        [],
                    )
                    .unwrap();
            } else {
                reparse_source_fixture(&mut f);
            }
            let new = f.prepare().unwrap().build;
            let done = finish_source_fixture(&mut f, &new);
            let (pk, column, value) = match damage {
                "old generation" => (
                    old_build.candidate,
                    "source_state_generation",
                    "source_state_generation+10".to_string(),
                ),
                "old step" => (
                    old_build.candidate,
                    "source_state_step",
                    "source_state_step+1".to_string(),
                ),
                "old source basis" => (
                    old_build.candidate,
                    "source_basis_digest",
                    "zeroblob(32)".to_string(),
                ),
                "old preparation basis" => (
                    old_build.candidate,
                    "basis_digest",
                    "zeroblob(32)".to_string(),
                ),
                "new source basis" => (
                    new.candidate,
                    "source_basis_digest",
                    "zeroblob(32)".to_string(),
                ),
                "future generation" => {
                    let mut token = old.predecessor().clone();
                    token.generation = f.head.token.generation + 1;
                    let digest = basis(&token, 1, &[3; 32], 1);
                    f.writer._index.execute("UPDATE message_candidate SET source_state_generation=?2,basis_digest=?3 WHERE pk=?1",params![old_build.candidate,token.generation,digest.as_slice()]).unwrap();
                    (
                        old_build.candidate,
                        "source_state_step",
                        token.step.to_string(),
                    )
                }
                _ => (
                    old_build.candidate,
                    "source_state_step",
                    old.predecessor().step.to_string(),
                ),
            };
            f.writer
                ._index
                .execute(
                    &format!("UPDATE message_candidate SET {column}={value} WHERE pk=?1"),
                    [pk],
                )
                .unwrap();
            assert!(
                matches!(
                    f.writer
                        .publish_source_projection(&done, None, b"new", 4096),
                    Err(Error::MessagePredecessorChanged)
                ),
                "{damage}"
            );
            assert_eq!(
                f.scalar(&format!(
                    "SELECT eligible FROM message_candidate WHERE pk={}",
                    old_build.candidate
                )),
                1,
                "{damage}"
            );
            assert_eq!(
                f.scalar(&format!(
                    "SELECT ready FROM message_body WHERE pk={}",
                    new.body
                )),
                0,
                "{damage}"
            );
            assert_eq!(
                &f.writer
                    .read_handle()
                    .unwrap()
                    .selected_message(&f.message)
                    .unwrap()
                    .unwrap()
                    .body_id(),
                old.body()
            );
            assert!(
                f.writer
                    .current_source_projection(&f.head.token, &[3; 32], 4096)
                    .unwrap()
                    .is_none()
            );
        }
    }
    #[test]
    fn reparse_withdrawal_never_crosses_logical_candidate_or_archive_revision() {
        for other_revision in [false, true] {
            let mut f = Fixture::new();
            let old_build = f.prepare().unwrap().build;
            let done = finish_source_fixture(&mut f, &old_build);
            f.writer
                .publish_source_projection(&done, None, b"old", 4096)
                .unwrap();
            reparse_source_fixture(&mut f);
            let preparation = f.writer.prepare_message(
                Identity {
                    session: &f.session,
                    origin: &f.origin,
                    native_session: "session:with:colons",
                    session_kind: SessionKind::Main,
                    message: &f.message,
                    sequence: 0,
                    role: Role::Assistant,
                    written_by: WrittenBy::Model,
                },
                Source {
                    predecessor: &f.head.token,
                    revision: if other_revision { 2 } else { 1 },
                    evidence_revision: 1,
                    logical_key: if other_revision { [3; 32] } else { [4; 32] },
                    route: RouteKind::Native,
                },
                4096,
            );
            if other_revision {
                // This input has only its retained immutable revision1. An
                // unproved revision must refuse before allocating a candidate.
                assert!(matches!(preparation, Err(Error::MessagePredecessorChanged)));
                assert_eq!(
                    f.scalar("SELECT count(*) FROM message_candidate WHERE eligible=1"),
                    1
                );
                assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
                continue;
            }
            let new = preparation.unwrap().build;
            let done = finish_source_fixture(&mut f, &new);
            let replacement = f
                .writer
                .publish_source_projection(&done, None, b"independent", 4096)
                .unwrap();
            assert_eq!(
                f.scalar("SELECT count(*) FROM message_candidate WHERE eligible=1"),
                2
            );
            assert_eq!(
                f.writer
                    .select_uncontested_source_message(&replacement, 4096)
                    .unwrap(),
                super::super::SourceSelection::Contested
            );
        }
    }
    fn compatibility_input(f: &mut Fixture, raw: &[u8], context: &[&str]) -> String {
        let blob = f.writer.put_file_blob(raw).unwrap();
        let format = if context.is_empty() {
            InputFormat::CodexRollout
        } else {
            InputFormat::ClaudeCodeJsonl
        };
        let input = f
            .writer
            .retain_immutable_input(
                blob,
                format,
                context,
                &ImportObservation {
                    id: std::str::from_utf8(raw).unwrap(),
                    machine: "synthetic",
                    host: "synthetic",
                    path: "synthetic/compatible",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        f.head = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: 3,
                digest: [7; 32],
                acquired_upto: input.length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        f.head = parser_state::write_step(
            &mut f.writer._index,
            &f.root.join("index.db"),
            &f.head,
            parser_state::Progress {
                inspected_upto: input.length,
                parsed_upto: input.length,
                phase: parser_state::Phase::Parsing,
                finalization_pending: true,
                counts: Default::default(),
            },
            &[],
            parser_state::Budget {
                cells: 0,
                encoded_bytes: 0,
            },
            |_| Ok(()),
        )
        .unwrap();
        input.id
    }
    fn compatible_body(
        f: &mut Fixture,
        text: &[u8],
        parent: &[u8],
        time: i64,
        origin: &[u8],
    ) -> super::super::SourceProjection {
        compatible_body_sequence(f, text, parent, time, origin, 0)
    }
    fn compatible_body_sequence(
        f: &mut Fixture,
        text: &[u8],
        parent: &[u8],
        time: i64,
        origin: &[u8],
        sequence: u64,
    ) -> super::super::SourceProjection {
        use super::super::{BlockKind, BlockPresence, ProjectionStep, SourceRange};
        use polylogue_model::{
            BlobHash,
            merge::{FinishState, MessageField, Slot},
        };
        let build = f
            .writer
            .prepare_message(
                Identity {
                    session: &f.session,
                    origin: &f.origin,
                    native_session: "session:with:colons",
                    session_kind: SessionKind::Main,
                    message: &f.message,
                    sequence,
                    role: Role::Assistant,
                    written_by: WrittenBy::Model,
                },
                Source {
                    predecessor: &f.head.token,
                    revision: 1,
                    evidence_revision: 1,
                    logical_key: [3; 32],
                    route: RouteKind::Native,
                },
                4096,
            )
            .unwrap()
            .build;
        let value = f
            .writer
            .begin_message_value(&build, values::Codec::Utf8, text.len() as u64)
            .unwrap();
        for (i, chunk) in text.chunks(values::MAX_CHUNK).enumerate() {
            f.writer
                .append_value_chunk(
                    &value.id,
                    (i * values::MAX_CHUNK) as u64,
                    chunk,
                    values::MAX_CHUNK,
                )
                .unwrap();
        }
        let reader = f.writer.read_handle().unwrap();
        let mut proof = reader.validate_value(&value.id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, values::MAX_CHUNK).unwrap();
        }
        f.writer.seal_message_value(&build, &proof).unwrap();
        f.writer
            .write_block(
                &build,
                0,
                0,
                BlockKind::Text,
                BlockPresence {
                    source_block: true,
                    source_text: !text.is_empty(),
                },
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"block",
                },
                4096,
            )
            .unwrap();
        seal_owned_json(f, &build, ValuePurpose::Normalized, parent);
        seal_owned_json(f, &build, ValuePurpose::Origin, origin);
        f.writer
            .write_message_field(
                &build,
                &Slot::Message(MessageField::Parent),
                Some(time),
                true,
                &ProjectionStep {
                    expected_sequence: 1,
                    next_cursor: b"parent",
                },
                4096,
            )
            .unwrap();
        f.writer
            .add_message_source(
                &build,
                SourceRange {
                    start: 0,
                    end: f.head.parsed_upto,
                },
                &ProjectionStep {
                    expected_sequence: 2,
                    next_cursor: b"source",
                },
                4096,
            )
            .unwrap();
        let done = f
            .writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: 3,
                    next_cursor: b"done",
                },
                BlobHash::of(b"identical raw version evidence"),
                FinishState::Unknown,
                4096,
            )
            .unwrap();
        f.writer
            .publish_source_projection(&done, None, b"same source head", 4096)
            .unwrap()
    }
    fn compatible_ready(
        f: &mut Fixture,
        source: &super::super::SourceProjection,
    ) -> Result<Option<super::super::CompatibleSelection>> {
        use super::super::CompatibilityTurn;
        let mut work = f.writer.begin_compatible_source_selection(source, 4096)?;
        let reader = f.writer.read_handle()?;
        for _ in 0..10000 {
            work = match work.advance(&reader, 4096, &std::sync::atomic::AtomicBool::new(false))? {
                CompatibilityTurn::Progress(w) => w,
                CompatibilityTurn::Certified(w, c) => {
                    f.writer.certify_body_compatibility(c)?;
                    w
                }
                CompatibilityTurn::Ready(w) => return Ok(Some(w)),
                CompatibilityTurn::Contested => return Ok(None),
            };
        }
        panic!("compatibility failed to make bounded progress")
    }
    #[test]
    fn compatible_snapshots_are_one_body_with_order_independent_origins_and_all_exact_citations() {
        use super::super::{FieldRead, SourceStep};
        use polylogue_model::{
            BlockSlot,
            merge::{MessageField, Slot},
        };
        let mut outcomes = Vec::new();
        for order in [[0, 1], [1, 0]] {
            let mut f = Fixture::new();
            let mut ids = Vec::new();
            let mut sources = Vec::new();
            for i in order {
                let id = compatibility_input(
                    &mut f,
                    if i == 0 { b"snapshot-a" } else { b"snapshot-b" },
                    &[],
                );
                let origin = if i == 0 {
                    br#"{"witness":"a"}"#.as_slice()
                } else {
                    br#"{"witness":"b"}"#.as_slice()
                };
                let source = compatible_body(&mut f, &vec![b'x'; 131073], b"null", 1, origin);
                let proof = compatible_ready(&mut f, &source).unwrap().unwrap();
                f.writer
                    .install_compatible_source_selection(proof, 4096)
                    .unwrap();
                ids.push(id);
                sources.push((f.head.clone(), source));
            }
            let reader = f.writer.read_handle().unwrap();
            let view = reader.selected_message(&f.message).unwrap().unwrap();
            let FieldRead::Value(field) = view
                .field(&Slot::Message(MessageField::Parent), 4096)
                .unwrap()
                .unwrap()
            else {
                panic!("conflict")
            };
            let origin = field.origin_range(0, 4096).unwrap().unwrap().bytes;
            assert_eq!(
                view.block(BlockSlot::Item(0))
                    .unwrap()
                    .unwrap()
                    .text_range(131072, 1)
                    .unwrap()
                    .unwrap()
                    .bytes,
                b"x"
            );
            let mut cursor = None;
            let mut citations = Vec::new();
            let mut preferred = Vec::new();
            loop {
                match view.next_source(cursor.as_ref(), 4096).unwrap() {
                    SourceStep::Continue { cursor: c } => cursor = Some(c),
                    SourceStep::Span { cursor: c, source } => {
                        if source.preferred {
                            preferred.push(source.input.clone())
                        }
                        citations.push((source.input, source.chain, source.start, source.end));
                        cursor = Some(c)
                    }
                    SourceStep::End => break,
                }
            }
            assert_eq!(preferred, vec![ids.iter().min().unwrap().clone()]);
            assert_eq!(citations.len(), 2);
            assert!(
                citations
                    .iter()
                    .all(|(_, chain, start, end)| (*chain, *start, *end) == (1, 0, 10))
            );
            for (head, source) in sources {
                let snapshot = f.writer.parser_snapshot(&head.token).unwrap();
                let own = snapshot
                    .selected_source_projection(&[3; 32], 4096)
                    .unwrap()
                    .unwrap();
                assert_eq!(own.body(), source.body());
                assert_eq!(own.evidence_guard(), source.evidence_guard());
            }
            citations.sort();
            outcomes.push((origin, citations, preferred));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
    #[test]
    fn compatibility_equal_raw_evidence_cannot_hide_normalized_text_parent_time_or_context_disagreement()
     {
        for variant in 0..5 {
            let mut f = Fixture::new();
            compatibility_input(&mut f, b"first snapshot", &[]);
            let first = compatible_body(&mut f, b"same text", b"null", 1, b"{}");
            let proof = compatible_ready(&mut f, &first).unwrap().unwrap();
            f.writer
                .install_compatible_source_selection(proof, 4096)
                .unwrap();
            compatibility_input(
                &mut f,
                b"second snapshot",
                if variant == 3 {
                    &["claude-main", "context-session"]
                } else {
                    &[]
                },
            );
            let second = compatible_body_sequence(
                &mut f,
                if variant == 0 {
                    b"different text"
                } else {
                    b"same text"
                },
                if variant == 1 {
                    b"\"other-parent\""
                } else {
                    b"null"
                },
                if variant == 2 { 2 } else { 1 },
                b"{}",
                if variant == 4 { 1 } else { 0 },
            );
            assert_eq!(first.version_hash(), second.version_hash());
            assert!(
                compatible_ready(&mut f, &second).unwrap().is_none(),
                "variant {variant}"
            );
            assert_eq!(
                f.scalar("SELECT selected_body_pk FROM message"),
                first.body().pk
            );
        }
    }
    #[test]
    fn compatibility_selection_fences_membership_aba_removal_admission_and_cancellation() {
        use super::super::CompatibilityTurn;
        let mut f = Fixture::new();
        compatibility_input(&mut f, b"snapshot-one", &[]);
        let source = compatible_body(&mut f, b"text", b"null", 1, b"{}");
        let reader = f.writer.read_handle().unwrap();
        let work = f
            .writer
            .begin_compatible_source_selection(&source, 4096)
            .unwrap();
        assert!(matches!(
            work.advance(&reader, 0, &std::sync::atomic::AtomicBool::new(false)),
            Err(Error::ParserAdmission { .. })
        ));
        let work = f
            .writer
            .begin_compatible_source_selection(&source, 4096)
            .unwrap();
        assert!(matches!(
            work.advance(&reader, 4096, &std::sync::atomic::AtomicBool::new(true)),
            Err(Error::Cancelled)
        ));
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_body_compatibility"),
            0
        );
        let proof = compatible_ready(&mut f, &source).unwrap().unwrap();
        f.writer
            ._index
            .execute(
                "UPDATE message_candidate SET eligible=0 WHERE pk=?1",
                [source.candidate],
            )
            .unwrap();
        f.writer
            ._index
            .execute(
                "UPDATE message_candidate SET eligible=1 WHERE pk=?1",
                [source.candidate],
            )
            .unwrap();
        assert!(matches!(
            f.writer.install_compatible_source_selection(proof, 4096),
            Err(Error::MessagePredecessorChanged)
        ));
        let proof = compatible_ready(&mut f, &source).unwrap().unwrap();
        f.writer
            ._index
            .execute(
                "UPDATE message SET removal_epoch=removal_epoch+1,hidden=1",
                [],
            )
            .unwrap();
        assert!(
            f.writer
                .install_compatible_source_selection(proof, 4096)
                .is_err()
        );
        assert!(matches!(
            f.writer.begin_compatible_source_selection(&source, 4096),
            Err(Error::Invariant(_))
        ));
        // A cancelled or stale attempt carries no durable partial selection.
        assert_eq!(
            f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
            0
        );
        let _ = std::mem::size_of::<CompatibilityTurn>();
    }
    #[test]
    fn compatibility_reparse_withdraws_old_membership_and_retired_body_drops_its_certificate() {
        use super::super::BodyRetirement;
        let mut f = Fixture::new();
        compatibility_input(&mut f, b"reparse snapshot", &[]);
        let old = compatible_body(&mut f, b"old interpretation", b"null", 1, b"{}");
        let proof = compatible_ready(&mut f, &old).unwrap().unwrap();
        f.writer
            .install_compatible_source_selection(proof, 4096)
            .unwrap();
        let stale = compatible_ready(&mut f, &old).unwrap().unwrap();
        let reader = f.writer.read_handle().unwrap();
        let pinned = reader.selected_message(&f.message).unwrap().unwrap();
        let length = f.head.acquired_upto;
        f.head = f
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: f.head.token.input_pk,
                chain: f.head.token.chain,
                parser_version: 4,
                digest: [8; 32],
                acquired_upto: length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        f.head = parser_state::write_step(
            &mut f.writer._index,
            &f.root.join("index.db"),
            &f.head,
            parser_state::Progress {
                inspected_upto: length,
                parsed_upto: length,
                phase: parser_state::Phase::Parsing,
                finalization_pending: true,
                counts: Default::default(),
            },
            &[],
            parser_state::Budget {
                cells: 0,
                encoded_bytes: 0,
            },
            |_| Ok(()),
        )
        .unwrap();
        let new = compatible_body(&mut f, b"new interpretation", b"null", 1, b"{}");
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_candidate WHERE eligible=1"),
            1
        );
        assert!(
            f.writer
                .install_compatible_source_selection(stale, 4096)
                .is_err()
        );
        let proof = compatible_ready(&mut f, &new).unwrap().unwrap();
        f.writer
            .install_compatible_source_selection(proof, 4096)
            .unwrap();
        assert_eq!(
            pinned
                .block(polylogue_model::BlockSlot::Item(0))
                .unwrap()
                .unwrap()
                .text_range(0, 4096)
                .unwrap()
                .unwrap()
                .bytes,
            b"old interpretation"
        );
        drop(pinned);
        drop(reader);
        assert_eq!(
            f.writer.retire_message_body(old.body()).unwrap(),
            BodyRetirement::DetachedCandidate
        );
        // A body with roots uses the existing bounded root-release path first.
        for _ in 0..10 {
            if f.writer.retire_message_body(old.body()).unwrap() == BodyRetirement::Missing {
                break;
            }
        }
        assert_eq!(
            f.scalar(&format!(
                "SELECT count(*) FROM message_body_compatibility WHERE body_pk={}",
                old.body().pk
            )),
            0
        );
        assert!(
            f.writer
                .current_source_projection(&old.predecessor, &[3; 32], 4096)
                .is_err()
        );
    }
}

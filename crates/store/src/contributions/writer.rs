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
pub(crate) fn source_head(
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
mod tests;

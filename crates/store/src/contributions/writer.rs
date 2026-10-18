//! Message preparation: reserve an unready body and candidate for one Message
//! from one source revision, then fill it with values, fields and blocks. Every
//! write checks the parser state it was prepared under. Nothing here selects a
//! body or makes a candidate eligible.
use crate::{Error, Result, Writer, db, parser_state, values};
use polylogue_model::merge::RouteKind;
use polylogue_model::{MessageId, Origin, Role, SessionId, SessionKind, WrittenBy};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};

/// The Message being prepared. Message IDs are opaque: the session is passed
/// here, never parsed out of the Message ID.
pub struct Identity<'a> {
    pub session: &'a SessionId,
    pub origin: &'a Origin,
    pub native_session: &'a str,
    pub native_session_encoding: Option<&'a str>,
    pub session_kind: SessionKind,
    pub message: &'a MessageId,
    pub sequence: u64,
    pub role: Role,
    pub written_by: WrittenBy,
}
/// Where the Message comes from: one logical candidate (`logical_key`, chosen
/// by the format) in one stored archive revision, read under the parser state
/// `token`. How the parser splits its work does not change either key.
pub struct Source<'a> {
    pub token: &'a parser_state::Token,
    pub revision: u64,
    pub evidence_revision: u64,
    pub logical_key: [u8; 32],
    pub route: RouteKind,
}
/// Handle to a committed preparation. Every write checks that its parser
/// state, archive revision and index file are still current.
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
    pub fn token(&self) -> &parser_state::Token {
        &self.source
    }
}
/// A preparation and its unfinished work: the value and source descriptor
/// being written, and the finished projection once the build reaches its rank
/// phase.
#[derive(Clone, Debug)]
pub struct Preparation {
    pub build: Build,
    pub active_value: Option<values::Head>,
    /// The source descriptor being written, kept apart from the normalized
    /// value. Sealing checks that it is JSON, not what it means.
    pub active_origin_value: Option<values::Head>,
    pub projection: Option<super::completion::PreparedProjection>,
}
fn invalid(message: &str) -> Error {
    Error::Invariant(message.into())
}
pub(super) fn to_i64(n: u64) -> Result<i64> {
    i64::try_from(n).map_err(|_| Error::Range)
}
/// Digest of everything a preparation is computed from: parser state, archive
/// revision, logical key and evidence revision.
pub(super) fn preparation_digest(
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
        .ok_or(Error::StaleMessage)?;
    if head.token != *source || head.status == crate::parser_state::Phase::Retired {
        return Err(Error::StaleMessage);
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
        return Err(Error::StaleMessage);
    }
    Ok(head)
}
/// Checks that `path` and the index generation are the ones a handle was made
/// for, so a handle from another or a rebuilt index is refused.
pub(super) fn check_index(
    conn: &Connection,
    path: &Path,
    expected_path: &Path,
    generation: &str,
) -> Result<()> {
    if path != expected_path {
        return Err(Error::StaleMessage);
    }
    let current: String = db(path, || {
        conn.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
    })?;
    if current != generation {
        return Err(Error::StaleMessage);
    }
    Ok(())
}
/// Checks that `build` is current and still in its projection phase.
pub(super) fn check_writable(conn: &Connection, path: &Path, build: &Build) -> Result<()> {
    check_build(conn, path, build, true)
}
/// Checks that `build` is current; a finished projection also passes.
pub(super) fn check_current(conn: &Connection, path: &Path, build: &Build) -> Result<()> {
    check_build(conn, path, build, false)
}
fn check_build(conn: &Connection, path: &Path, build: &Build, writable: bool) -> Result<()> {
    if path != build.path {
        return Err(Error::StaleMessage);
    }
    source_head(conn, path, &build.source)?;
    let row=db(path,||conn.query_row("SELECT b.phase,m.hidden,m.contributor_revision,m.removal_epoch FROM message_build b JOIN message m ON m.pk=b.message_pk JOIN message_body body ON body.pk=b.body_pk JOIN message_candidate c ON c.pk=body.preferred_candidate_pk WHERE body.ready=0 AND c.ready=0 AND c.eligible=0 AND b.pk=?1 AND b.body_pk=?2 AND b.message_pk=?3 AND b.session_pk=?4 AND b.basis_digest=?5 AND b.contributor_revision=?6 AND b.removal_epoch=?7",params![build.pk,build.body,build.message,build.session,build.digest.as_slice(),build.contributor_revision,build.removal_epoch],|r|Ok((r.get::<_,String>(0)?,r.get::<_,bool>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?))).optional())?.ok_or(Error::StaleMessage)?;
    if row.1
        || row.2 != build.contributor_revision
        || row.3 != build.removal_epoch
        || !matches!(row.0.as_str(), "project" | "rank")
        || (writable && row.0 != "project")
    {
        return Err(Error::StaleMessage);
    }
    Ok(())
}
pub(super) fn check_source(
    writer: &Writer,
    source: &Source<'_>,
) -> Result<(parser_state::Head, u64)> {
    let revision = to_i64(source.revision)?;
    to_i64(source.evidence_revision)?;
    if revision <= 0 {
        return Err(Error::Range);
    }
    let path = writer.root.join("index.db");
    let head = source_head(&writer._index, &path, source.token)?;
    let input: String = db(&path, || {
        writer._index.query_row(
            "SELECT id FROM input_ref WHERE pk=?1",
            [source.token.input_pk],
            |r| r.get(0),
        )
    })?;
    let archive = writer.root.join("archive.db");
    let length: Option<u64> = db(&archive, || {
        writer
            .archive
            .query_row(
                "SELECT length FROM revision WHERE input=?1 AND number=?2 AND chain=?3",
                params![input, revision, source.token.chain],
                |r| parser_state::unsigned(r, 0),
            )
            .optional()
    })?;
    let length = length.ok_or(Error::StaleMessage)?;
    if head.acquired_upto > length {
        return Err(Error::StaleMessage);
    }
    Ok((head, length))
}
fn check_request(
    identity: &Identity<'_>,
    source: &Source<'_>,
    allowance: usize,
) -> Result<(i64, i64, i64)> {
    if identity.message.session() != identity.session || identity.native_session.is_empty() {
        return Err(invalid(
            "Message ID is not in the given session, or the native session ID is empty",
        ));
    }
    let mut bytes = 256usize;
    for text in [
        identity.session.as_str(),
        identity.origin.as_str(),
        identity.native_session,
        identity.native_session_encoding.unwrap_or_default(),
        identity.message.as_str(),
        source.token.index_generation.as_str(),
    ] {
        bytes = bytes.checked_add(text.len()).ok_or(Error::Range)?;
    }
    if bytes > allowance {
        return Err(Error::ParserAdmission {
            resource: "message preparation metadata bytes",
            required: bytes as u64,
        });
    }
    polylogue_model::NativeId::from_wire(identity.native_session, identity.native_session_encoding)
        .map_err(|e| invalid(&e.to_string()))?;
    let sequence = to_i64(identity.sequence)?;
    let revision = to_i64(source.revision)?;
    let evidence = to_i64(source.evidence_revision)?;
    if revision <= 0 {
        return Err(Error::Range);
    }
    Ok((sequence, revision, evidence))
}

/// Inserts the session if it is new and returns its pk. An existing session
/// must have the same origin, native ID and kind.
fn insert_session(conn: &Connection, path: &Path, identity: &Identity<'_>) -> Result<i64> {
    let kind = kind_name(identity.session_kind);
    db(path, || {
        conn.execute("INSERT INTO session(id,origin,native_id,native_id_encoding,kind,changed) VALUES(?1,?2,?3,?4,?5,0) ON CONFLICT(id) DO NOTHING",params![identity.session.as_str(),identity.origin.as_str(),identity.native_session,identity.native_session_encoding,kind])
    })?;
    let (session, origin, native, encoding, stored_kind): (
        i64,
        String,
        String,
        Option<String>,
        String,
    ) = db(path, || {
        conn.query_row(
            "SELECT pk,origin,native_id,native_id_encoding,kind FROM session WHERE id=?1",
            [identity.session.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
    })?;
    if origin != identity.origin.as_str()
        || native != identity.native_session
        || encoding.as_deref() != identity.native_session_encoding
        || stored_kind != kind
    {
        return Err(invalid(
            "stored session differs from the one being prepared",
        ));
    }
    Ok(session)
}

fn load_preparation(
    conn: &Connection,
    path: &Path,
    build: &Build,
    identity: &Identity<'_>,
    source: &Source<'_>,
) -> Result<Preparation> {
    check_current(conn, path, build)?;
    if source.token != &build.source
        || preparation_digest(
            source.token,
            source.revision,
            &source.logical_key,
            source.evidence_revision,
        ) != build.digest
    {
        return Err(Error::StaleMessage);
    }
    let kind = kind_name(identity.session_kind);
    let session_ok: bool = db(path, || {
        conn.query_row(
        "SELECT id=?2 AND origin=?3 AND native_id=?4 AND native_id_encoding IS ?5 AND kind=?6 FROM session WHERE pk=?1",
        params![build.session,identity.session.as_str(),identity.origin.as_str(),identity.native_session,identity.native_session_encoding,kind],|r|r.get(0))
    })?;
    if !session_ok {
        return Err(invalid(
            "stored session differs from the one being prepared",
        ));
    }
    let message_ok: bool = db(path, || {
        conn.query_row(
            "SELECT id=?2 AND session_pk=?3 FROM message WHERE pk=?1",
            params![build.message, identity.message.as_str(), build.session],
            |r| r.get(0),
        )
    })?;
    if !message_ok {
        return Err(Error::StaleMessage);
    }
    let compatible: bool = db(path, || {
        conn.query_row("SELECT b.role=?2 AND b.written_by=?3 AND c.route=?4 AND c.pk=?5 FROM message_body b JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE b.pk=?1",params![build.body,role_name(identity.role),written_by_name(identity.written_by),route_code(source.route),build.candidate],|r|r.get(0))
    })?;
    if !compatible {
        return Err(invalid(
            "role, author or route differs from the existing preparation",
        ));
    }
    let (phase, hash): (String, Option<Vec<u8>>) = db(path, || {
        conn.query_row("SELECT w.phase,c.version_hash FROM message_build w JOIN message_body b ON b.pk=w.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE w.pk=?1",[build.pk],|r|Ok((r.get(0)?,r.get(1)?)))
    })?;
    let projection = if phase == "rank" {
        let hash: [u8; 32] = hash
            .ok_or_else(|| invalid("finished projection has no version hash"))?
            .try_into()
            .map_err(|_| invalid("version hash is not 32 bytes"))?;
        Some(super::completion::PreparedProjection {
            build: build.clone(),
            hash: polylogue_model::BlobHash::from_bytes(hash),
        })
    } else {
        None
    };
    let (active, origin): (Option<i64>, Option<i64>) = db(path, || {
        conn.query_row(
            "SELECT active_value_pk,active_origin_value_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
    })?;
    Ok(Preparation {
        build: build.clone(),
        active_value: active
            .map(|pk| value_head(conn, path, build, pk))
            .transpose()?,
        active_origin_value: origin
            .map(|pk| value_head(conn, path, build, pk))
            .transpose()?,
        projection,
    })
}

/// Head of a value owned by `build`, in the same index.
pub(super) fn value_head(
    conn: &Connection,
    path: &Path,
    build: &Build,
    pk: i64,
) -> Result<values::Head> {
    values::head(
        conn,
        path,
        &values::ValueId {
            pk,
            index_generation: build.source.index_generation.clone(),
            index_path: path.to_path_buf(),
        },
    )
}

fn kind_name(kind: SessionKind) -> &'static str {
    match kind {
        SessionKind::Main => "main",
        SessionKind::Subagent => "subagent",
        SessionKind::Chat => "chat",
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

fn written_by_name(written_by: WrittenBy) -> &'static str {
    match written_by {
        WrittenBy::Human => "human",
        WrittenBy::Model => "model",
        WrittenBy::Harness => "harness",
        WrittenBy::Unknown => "unknown",
    }
}

fn route_code(route: RouteKind) -> i64 {
    match route {
        RouteKind::PageDom => 0,
        RouteKind::AccountExport => 1,
        RouteKind::Native => 2,
    }
}

impl Writer {
    /// Reserves an unready body and candidate for this Message, or resumes the
    /// existing preparation when the request is identical. A different request
    /// returns `MessagePreparationPending` until the old one is retired.
    pub fn prepare_message(
        &mut self,
        identity: Identity<'_>,
        source: Source<'_>,
        allowance: usize,
    ) -> Result<Preparation> {
        let (sequence, revision, evidence) = check_request(&identity, &source, allowance)?;
        let path = self.root.join("index.db");
        let (head, length) = check_source(self, &source)?;
        let length_i = to_i64(length)?;
        let digest = preparation_digest(
            source.token,
            source.revision,
            &source.logical_key,
            source.evidence_revision,
        );
        let tx = db(&path, || self._index.transaction())?;
        source_head(&tx, &path, source.token)?;
        let session = insert_session(&tx, &path, &identity)?;
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
            return Err(Error::StaleMessage);
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
            let role = role_name(identity.role);
            let by = written_by_name(identity.written_by);
            db(&path, || {
                tx.execute("INSERT INTO message_body(message_pk,session_pk,basis_digest,contributor_revision,removal_epoch,layout,role,written_by,input_pk,source_chain,parser_version,source_sequence) VALUES(?1,?2,?3,?4,?5,'flat',?6,?7,?8,?9,?10,?11)",params![message,session,digest.as_slice(),contributor,removal,role,by,source.token.input_pk,source.token.chain,head.parser_version,sequence])
            })?;
            let body = tx.last_insert_rowid();
            let route = route_code(source.route);
            db(&path, || {
                tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,body_pk,finish,route) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,1,?15)",params![message,session,source.token.input_pk,source.token.chain,revision,length_i,evidence,source.logical_key.as_slice(),head.parser_version,source.token.generation,source.token.step,source.token.basis.as_slice(),digest.as_slice(),body,route])
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
            source: source.token.clone(),
            pk,
            body,
            candidate,
            message,
            session,
            contributor_revision: contributor,
            removal_epoch: removal,
            digest,
        };
        let preparation = load_preparation(&tx, &path, &build, &identity, &source)?;
        db(&path, || tx.commit())?;
        Ok(preparation)
    }
    /// Reads the state of a committed preparation without writing anything.
    /// The request is checked as in `prepare_message`.
    pub fn inspect_message_preparation(
        &mut self,
        build: &Build,
        identity: Identity<'_>,
        source: Source<'_>,
        allowance: usize,
    ) -> Result<Preparation> {
        check_request(&identity, &source, allowance)?;
        check_source(self, &source)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        let preparation = load_preparation(&tx, &path, build, &identity, &source)?;
        db(&path, || tx.commit())?;
        Ok(preparation)
    }
    /// Starts the normalized value for the next field or block. The value and
    /// its lease are created in one transaction.
    pub fn begin_message_value(
        &mut self,
        build: &Build,
        codec: values::Codec,
        length: u64,
    ) -> Result<values::Head> {
        self.begin_build_value(build, ValuePurpose::Normalized, codec, length)
    }
    /// Starts the JSON source descriptor for the next field, kept apart from
    /// its normalized value. Sealing checks that it is JSON, not what it means.
    pub fn begin_message_origin_value(
        &mut self,
        build: &Build,
        length: u64,
    ) -> Result<values::Head> {
        self.begin_build_value(build, ValuePurpose::Origin, values::Codec::Json, length)
    }
    fn begin_build_value(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        codec: values::Codec,
        length: u64,
    ) -> Result<values::Head> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check_writable(&tx, &path, build)?;
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
    /// Seals the active value. If identical bytes already exist, the active
    /// value moves to that copy; the build keeps exactly one lease either way,
    /// also when the normalized value and descriptor end up the same row.
    pub fn seal_message_value(
        &mut self,
        build: &Build,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        self.seal_build_value(build, ValuePurpose::Normalized, proof)
    }
    pub fn seal_message_origin_value(
        &mut self,
        build: &Build,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        self.seal_build_value(build, ValuePurpose::Origin, proof)
    }
    fn seal_build_value(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check_writable(&tx, &path, build)?;
        let column = purpose.column();
        let active: Option<i64> = db(&path, || {
            tx.query_row(
                &format!("SELECT {column} FROM message_build WHERE pk=?1"),
                [build.pk],
                |r| r.get(0),
            )
        })?;
        if active != Some(proof.value().pk()) {
            return Err(Error::StaleMessage);
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
                    return Err(Error::StaleMessage);
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

    /// Writes the sealed active value as a new Message field. Fails if the
    /// field already exists; use `set_message_field` to replace it. Block
    /// fields go through `write_block_field`.
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
    /// Writes the sealed active value as a Message field, replacing any value
    /// already there.
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
    /// Removes a Message field. Removing a missing field still advances the
    /// cursor; a replayed step fails like any other.
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
            return Err(invalid("block fields are written with write_block_field"));
        }
        super::blocks::check_step(step, allowance)?;
        let key = field_key(slot, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check_writable(&tx, &path, build)?;
        update_field_in(&tx, &path, build, &key, None, source_time, supplied, intent)?;
        super::blocks::commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Marks the preparation for retirement and releases its active value and
    /// block. Works even after the source or Message has changed. The rest is
    /// removed by `retire_message_turn`.
    pub fn retire_message_preparation(&mut self, build: &Build) -> Result<()> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check_index(&tx, &path, &build.path, &build.source.index_generation)?;
        let active: Option<(Option<i64>, Option<i64>, Option<i64>)> = db(&path, || {
            tx.query_row("SELECT w.active_value_pk,w.active_origin_value_pk,w.active_block_pk FROM message_build w JOIN message_body b ON b.pk=w.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE w.pk=?1 AND w.body_pk=?2 AND w.basis_digest=?3 AND b.ready=0 AND c.ready=0 AND c.eligible=0",params![build.pk,build.body,build.digest.as_slice()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()
        })?;
        let (active, origin, block) = active.ok_or(Error::StaleMessage)?;
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
                return Err(invalid("active value has no lease"));
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FieldUpdate {
    Insert,
    Set,
    Remove,
}
/// Applies one field change to the Message's field root, or to `block`'s.
/// Insert and Set move the active value's lease to the new field fact. The
/// caller has checked that `block`, if any, is this build's active block.
pub(super) fn update_field_in(
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
        let head = value_head(tx, path, build, value)?;
        if head.state != values::State::Ready {
            return Err(Error::ValueNotReady);
        }
    }
    if let Some(origin) = origin {
        let head = value_head(tx, path, build, origin)?;
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
        (FieldUpdate::Insert, true) => return Err(invalid("field is already written")),
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
/// Key of a field inside a block. It never contains the block's ordinal: the
/// block's own field root scopes it.
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
        _ => Err(invalid("slot is not a block field")),
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

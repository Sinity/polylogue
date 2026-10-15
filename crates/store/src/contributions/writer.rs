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
    path: PathBuf,
    source: parser_state::Token,
    pk: i64,
    body: i64,
    candidate: i64,
    message: i64,
    session: i64,
    contributor_revision: i64,
    removal_epoch: i64,
    digest: [u8; 32],
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
    pub retiring: bool,
}
fn bad(message: &str) -> Error {
    Error::Invariant(message.into())
}
fn integer(n: u64) -> Result<i64> {
    i64::try_from(n).map_err(|_| Error::Range)
}
fn basis(
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
fn source_head(
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
fn check(conn: &Connection, path: &Path, build: &Build, allow_retiring: bool) -> Result<bool> {
    if path != build.path {
        return Err(Error::MessagePredecessorChanged);
    }
    source_head(conn, path, &build.source)?;
    let row=db(path,||conn.query_row("SELECT b.phase,m.hidden,m.contributor_revision,m.removal_epoch FROM message_build b JOIN message m ON m.pk=b.message_pk JOIN message_body body ON body.pk=b.body_pk JOIN message_candidate c ON c.pk=body.preferred_candidate_pk WHERE body.ready=0 AND c.ready=0 AND c.eligible=0 AND b.pk=?1 AND b.body_pk=?2 AND b.message_pk=?3 AND b.session_pk=?4 AND b.basis_digest=?5 AND b.contributor_revision=?6 AND b.removal_epoch=?7",params![build.pk,build.body,build.message,build.session,build.digest.as_slice(),build.contributor_revision,build.removal_epoch],|r|Ok((r.get::<_,String>(0)?,r.get::<_,bool>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?))).optional())?.ok_or(Error::MessagePredecessorChanged)?;
    if row.1
        || row.2 != build.contributor_revision
        || row.3 != build.removal_epoch
        || (!allow_retiring && row.0 == "retiring")
    {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(row.0 == "retiring")
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
                tx.execute("INSERT INTO message_body(message_pk,session_pk,basis_digest,contributor_revision,removal_epoch,layout,role,written_by,input_pk,source_chain,parser_version) VALUES(?1,?2,?3,?4,?5,'flat',?6,?7,?8,?9,?10)",params![message,session,digest.as_slice(),contributor,removal,role,by,source.predecessor.input_pk,source.predecessor.chain,head.parser_version])
            })?;
            let body = tx.last_insert_rowid();
            let route = match source.route {
                RouteKind::PageDom => 0,
                RouteKind::AccountExport => 1,
                RouteKind::Native => 2,
            };
            db(&path, || {
                tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,basis_digest,body_pk,finish,route) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,1,?12)",params![message,session,source.predecessor.input_pk,source.predecessor.chain,revision,length_i,evidence,source.logical_key.as_slice(),head.parser_version,digest.as_slice(),body,route])
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
        check(&tx, &path, &build, false)?;
        let active: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT active_value_pk FROM message_build WHERE pk=?1",
                [pk],
                |r| r.get(0),
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
        db(&path, || tx.commit())?;
        Ok(Preparation {
            build,
            active_value,
            retiring: false,
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
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, build, false)?;
        let active: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT active_value_pk FROM message_build WHERE pk=?1",
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
                "UPDATE message_build SET active_value_pk=?2 WHERE pk=?1",
                params![build.pk, value.id.pk()],
            )
        })?;
        db(&path, || tx.commit())?;
        Ok(value)
    }
    /// Integrity sealing retains the build's lease, moving it atomically to an
    /// existing deduplicated value if needed. No payload hash runs in this tx.
    pub fn seal_message_value(
        &mut self,
        build: &Build,
        proof: &values::Validation,
    ) -> Result<values::Seal> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, build, false)?;
        let active: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT active_value_pk FROM message_build WHERE pk=?1",
                [build.pk],
                |r| r.get(0),
            )
        })?;
        if active != Some(proof.value().pk()) {
            return Err(Error::MessagePredecessorChanged);
        }
        let sealed = values::seal_in(&tx, &path, proof, |tx, ready| {
            if ready.id != *proof.value() {
                db(&path, || {
                    tx.execute(
                        "UPDATE body_value SET incoming=incoming+1 WHERE pk=?1 AND state='ready'",
                        [ready.id.pk()],
                    )
                })?;
                let changed = db(&path, || {
                    tx.execute(
                        "UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming>0",
                        [proof.value().pk()],
                    )
                })?;
                if changed != 1 {
                    return Err(Error::MessagePredecessorChanged);
                }
                db(&path, || {
                    tx.execute(
                        "UPDATE message_build SET active_value_pk=?2 WHERE pk=?1",
                        params![build.pk, ready.id.pk()],
                    )
                })?;
            }
            Ok(())
        })?;
        db(&path, || tx.commit())?;
        Ok(sealed)
    }
}

impl Writer {
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
        allowance: usize,
    ) -> Result<()> {
        let key = field_key(slot, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, build, false)?;
        let value: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT active_value_pk FROM message_build WHERE pk=?1",
                [build.pk],
                |r| r.get(0),
            )
        })?;
        let value = value.ok_or(Error::ValueNotReady)?;
        let head = values::head(
            &tx,
            &path,
            &values::ValueId {
                pk: value,
                index_generation: build.source.index_generation.clone(),
                index_path: path.clone(),
            },
        )?;
        if head.state != values::State::Ready {
            return Err(Error::ValueNotReady);
        }
        db(&path, || {
            tx.execute("INSERT INTO message_field_slot(message_pk,session_pk,semantic_key) VALUES(?1,?2,?3) ON CONFLICT(message_pk,semantic_key) DO NOTHING",params![build.message,build.session,key])
        })?;
        let slot: i64 = db(&path, || {
            tx.query_row(
                "SELECT pk FROM message_field_slot WHERE message_pk=?1 AND semantic_key=?2",
                params![build.message, key],
                |r| r.get(0),
            )
        })?;
        db(&path, || {
            tx.execute("INSERT INTO message_field_fact(message_pk,session_pk,candidate_pk,slot_pk,value_pk,source_time,supplied) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![build.message,build.session,build.candidate,slot,value,source_time,supplied])
        })?;
        let fact = tx.last_insert_rowid();
        let old: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1 AND ready=0",
                [build.body],
                |r| r.get(0),
            )
        })?;
        let owner = super::directory::Owner {
            message: build.message,
            session: build.session,
        };
        let mut work = super::directory::Work::default();
        let (root, _) = super::directory::replace(
            &tx,
            &path,
            owner,
            super::directory::Kind::Fields,
            old,
            u64::try_from(slot).map_err(|_| Error::Range)?,
            Some(super::directory::Edge {
                link: super::directory::Link::Field(fact),
                counts: super::directory::Counts {
                    source_fields: i64::from(supplied),
                    ..Default::default()
                },
            }),
            &mut work,
        )?;
        super::directory::acquire(&tx, &path, owner, root)?;
        if let Some(old) = old {
            super::directory::release(&tx, &path, owner, old)?;
        }
        db(&path, || {
            tx.execute(
                "UPDATE message_body SET field_root_pk=?2 WHERE pk=?1 AND ready=0",
                params![build.body, root],
            )
        })?;
        // One producer lease becomes one fact lease; its incoming count stays 1.
        db(&path, || {
            tx.execute(
                "UPDATE message_build SET active_value_pk=NULL WHERE pk=?1",
                [build.pk],
            )
        })?;
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
        let active: Option<Option<i64>> = db(&path, || {
            tx.query_row("SELECT active_value_pk FROM message_build WHERE pk=?1 AND body_pk=?2 AND basis_digest=?3",params![build.pk,build.body,build.digest.as_slice()],|r|r.get(0)).optional()
        })?;
        let active = active.ok_or(Error::MessagePredecessorChanged)?;
        db(&path, || {
            tx.execute(
                "UPDATE message_build SET phase='retiring',active_value_pk=NULL WHERE pk=?1",
                [build.pk],
            )
        })?;
        if let Some(value) = active {
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

fn field_key(slot: &polylogue_model::merge::Slot, allowance: usize) -> Result<Vec<u8>> {
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
    if let Err(error) = serde_json::to_writer(&mut output, slot) {
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
            f.writer.write_message_field(&build, &slot, None, true, 8),
            Err(Error::ParserAdmission { .. })
        ));
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
        assert!(f.prepare().unwrap().active_value.is_some());
        let slot = Slot::Message(MessageField::Model);
        f.writer
            .write_message_field(&build, &slot, None, true, 4096)
            .unwrap();
        f.value(&build, b"other-model");
        assert!(matches!(
            f.writer
                .write_message_field(&build, &slot, None, true, 4096),
            Err(Error::Sqlite { .. })
        ));
        assert!(f.prepare().unwrap().active_value.is_some());
        assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    }
}

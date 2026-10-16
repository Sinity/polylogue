//! Conservative equality of complete immutable source bodies. This is not
//! version ranking or missing-field fill. Each worker turn visits one leaf.
use super::{
    directory as d,
    source::{self, SourceProjection, SourceSelection},
    writer,
};
use crate::{Error, ReadHandle, Result, Writer, db};
use rusqlite::{Connection, OptionalExtension, params, types::ValueRef};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

fn changed() -> Error {
    Error::MessagePredecessorChanged
}
fn invalid(s: &str) -> Error {
    Error::Invariant(s.into())
}

/// Private proof producer, restartable from zero. It owns no durable leases:
/// eligible bodies retain their payload and the writer fences membership.
pub struct CompatibleSelection {
    path: PathBuf,
    source: SourceProjection,
    revision: i64,
    epoch: i64,
    after: i64,
    digest: Option<[u8; 32]>,
    carrier: Option<(String, i64)>,
    hashing: Option<BodyDigest>,
    complete: bool,
    singleton: bool,
}
pub enum CompatibilityTurn {
    Progress(CompatibleSelection),
    Certified(CompatibleSelection, CompatibilityCertificate),
    Ready(CompatibleSelection),
    Contested,
}
/// Constructible only by complete immutable-directory traversal.
pub struct CompatibilityCertificate {
    body: i64,
    candidate: i64,
    owner: d::Owner,
    basis: [u8; 32],
    epoch: i64,
    path: PathBuf,
    generation: String,
    digest: [u8; 32],
}
struct BodyDigest {
    certificate: CompatibilityCertificate,
    hasher: blake3::Hasher,
    fields: Option<i64>,
    groups: Option<i64>,
    field_after: Option<u64>,
    group_after: Option<u64>,
    blocks: Option<i64>,
    block_after: Option<u64>,
    block_fields: Option<i64>,
    block_field_after: Option<u64>,
    phase: u8,
}

/// Frame each SQLite scalar including null/type/length. Rows are borrowed;
/// metadata admission precedes hashing and no normalized payload is decoded.
pub(super) fn hash_row(
    connection: &Connection,
    path: &Path,
    sql: &str,
    parameter: i64,
    hash: &mut blake3::Hasher,
    allowance: usize,
) -> Result<()> {
    db(path, || {
        connection.query_row(sql, [parameter], |row| {
            let count = row.as_ref().column_count();
            let mut total = 0usize;
            for i in 0..count {
                let value = row.get_ref(i)?;
                total = total.saturating_add(match value {
                    ValueRef::Text(b) | ValueRef::Blob(b) => b.len(),
                    _ => 8,
                });
            }
            if total > allowance {
                return Ok(Err(Error::ParserAdmission {
                    resource: "body compatibility metadata",
                    required: total as u64,
                }));
            }
            hash.update(&(count as u64).to_be_bytes());
            for i in 0..count {
                match row.get_ref(i)? {
                    ValueRef::Null => {
                        hash.update(&[0]);
                    }
                    ValueRef::Integer(n) => {
                        hash.update(&[1]);
                        hash.update(&n.to_be_bytes());
                    }
                    ValueRef::Real(n) => {
                        hash.update(&[2]);
                        hash.update(&n.to_bits().to_be_bytes());
                    }
                    ValueRef::Text(b) | ValueRef::Blob(b) => {
                        hash.update(&[if matches!(row.get_ref(i)?, ValueRef::Text(_)) {
                            3
                        } else {
                            4
                        }]);
                        hash.update(&(b.len() as u64).to_be_bytes());
                        hash.update(b);
                    }
                }
            }
            Ok(Ok(()))
        })
    })?
}

impl BodyDigest {
    fn begin(
        reader: &ReadHandle,
        source: &SourceProjection,
        body: i64,
        candidate: i64,
        allowance: usize,
    ) -> Result<Self> {
        let path = reader.root.join("index.db");
        let owner = d::Owner {
            message: source.body.message,
            session: source.body.session,
        };
        let (basis, epoch, fields, groups, operations, sequence): (
            Vec<u8>,
            i64,
            Option<i64>,
            Option<i64>,
            Option<i64>,
            Option<i64>,
        ) = db(&path, || {
            reader.index.query_row(
            "SELECT b.basis_digest,b.removal_epoch,b.field_root_pk,b.block_root_pk,b.operation_root_pk,b.source_sequence FROM message_body b JOIN message_candidate c ON c.pk=b.preferred_candidate_pk AND c.body_pk=b.pk WHERE b.pk=?1 AND c.pk=?2 AND b.message_pk=?3 AND b.session_pk=?4 AND b.ready=1 AND c.ready=1 AND c.eligible=1 AND c.basis_digest=b.basis_digest",
            params![body,candidate,owner.message,owner.session], |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?)))
        })?;
        if operations.is_some() || sequence.is_none() {
            return Err(Error::MessagePreparationPending);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"polylogue.complete-source-body.compatibility.v1\0");
        // Public normalized scalars, never acquisition/source coordinates or PKs.
        hash_row(
            &reader.index,
            &path,
            "SELECT layout,role,written_by,prompt_kind,at,completed_at,at_until,id_kind,in_progress,on_active_path,copy_origin,usage_present,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,cache_write_5m_tokens,cache_write_1h_tokens,reasoning_tokens,web_requests,provider_cost_nusd,cost_nusd,duration_ms,words,parser_version,source_sequence FROM message_body WHERE pk=?1",
            body,
            &mut hasher,
            allowance,
        )?;
        hash_row(
            &reader.index,
            &path,
            "SELECT finish,route,source_blocks,source_text_blocks,source_fields,version_hash,parser_version FROM message_candidate WHERE pk=?1",
            candidate,
            &mut hasher,
            allowance,
        )?;
        let (length, input): (u64, Option<String>) = db(&path, || {
            reader.index.query_row("SELECT length(CAST(i.id AS BLOB)),CASE WHEN length(CAST(i.id AS BLOB))<=?2 THEN i.id END FROM message_candidate c JOIN input_ref i ON i.pk=c.input_pk WHERE c.pk=?1",params![candidate,allowance as i64],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?)))
        })?;
        let input = input.ok_or(Error::ParserAdmission {
            resource: "compatibility input identity",
            required: length,
        })?;
        let archive_path = reader.root.join("archive.db");
        let retained = crate::observed::decode(&reader.archive, &archive_path, &input)?
            .ok_or_else(|| Error::Invariant("compatibility source unavailable".into()))?;
        let format = retained.format;
        let parts = retained.context;
        let required = parts.iter().map(|part| part.len() + 8).sum::<usize>();
        if required > allowance {
            return Err(Error::ParserAdmission {
                resource: "compatibility interpretation context",
                required: required as u64,
            });
        }
        let name = match format {
            crate::InputFormat::CodexRollout => "codex-rollout",
            crate::InputFormat::CodexHistorySqlite => "codex-history-sqlite",
            crate::InputFormat::ClaudeCodeJsonl => "claude-code-jsonl",
        };
        hasher.update(&(name.len() as u64).to_be_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(parts.len() as u64).to_be_bytes());
        for part in parts {
            hasher.update(&(part.len() as u64).to_be_bytes());
            hasher.update(part.as_bytes());
        }
        Ok(Self {
            certificate: CompatibilityCertificate {
                body,
                candidate,
                owner,
                basis: basis
                    .try_into()
                    .map_err(|_| invalid("compatibility body basis"))?,
                epoch,
                path,
                generation: source.body.generation.clone(),
                digest: [0; 32],
            },
            hasher,
            fields,
            groups,
            field_after: None,
            group_after: None,
            blocks: None,
            block_after: None,
            block_fields: None,
            block_field_after: None,
            phase: 0,
        })
    }
    fn field(
        &mut self,
        reader: &ReadHandle,
        root: Option<i64>,
        after: Option<u64>,
        allowance: usize,
    ) -> Result<Option<u64>> {
        let path = &self.certificate.path;
        let Some((key, edge)) = d::next(
            &reader.index,
            path,
            self.certificate.owner,
            d::Kind::Fields,
            root,
            after,
            &mut d::Work::default(),
        )?
        else {
            return Ok(None);
        };
        let d::Link::Field(fact) = edge.link else {
            return Err(Error::MessagePreparationPending);
        };
        self.hasher.update(b"field\0");
        hash_row(
            &reader.index,
            path,
            "SELECT s.semantic_key,f.supplied,f.source_time,v.codec,v.length,v.hash FROM message_field_fact f JOIN message_field_slot s ON s.pk=f.slot_pk JOIN body_value v ON v.pk=f.value_pk AND v.state='ready' WHERE f.pk=?1",
            fact,
            &mut self.hasher,
            allowance,
        )?;
        Ok(Some(key))
    }
    fn advance(&mut self, reader: &ReadHandle, allowance: usize) -> Result<bool> {
        let path = &self.certificate.path;
        match self.phase {
            0 => match self.field(reader, self.fields, self.field_after, allowance)? {
                Some(k) => self.field_after = Some(k),
                None => {
                    self.hasher.update(b"end-message-fields\0");
                    self.phase = 1;
                }
            },
            1 => {
                let next = d::next(
                    &reader.index,
                    path,
                    self.certificate.owner,
                    d::Kind::Groups,
                    self.groups,
                    self.group_after,
                    &mut d::Work::default(),
                )?;
                match next {
                    None => {
                        self.certificate.digest = *self.hasher.finalize().as_bytes();
                        return Ok(true);
                    }
                    Some((key, edge)) => {
                        let d::Link::Node {
                            pk,
                            kind: d::Kind::Blocks,
                            ..
                        } = edge.link
                        else {
                            return Err(invalid("compatibility group shape"));
                        };
                        self.hasher.update(b"group\0");
                        self.hasher.update(&key.to_be_bytes());
                        self.group_after = Some(key);
                        self.blocks = Some(pk);
                        self.block_after = None;
                        self.phase = 2;
                    }
                }
            }
            2 => {
                match d::next(
                    &reader.index,
                    path,
                    self.certificate.owner,
                    d::Kind::Blocks,
                    self.blocks,
                    self.block_after,
                    &mut d::Work::default(),
                )? {
                    None => {
                        self.hasher.update(b"end-group\0");
                        self.phase = 1;
                    }
                    Some((key, edge)) => {
                        let d::Link::Block { pk, call } = edge.link else {
                            return Err(invalid("compatibility block shape"));
                        };
                        self.hasher.update(b"block\0");
                        self.hasher.update(&key.to_be_bytes());
                        hash_row(
                            &reader.index,
                            path,
                            "SELECT b.slot_kind,b.attach_key,b.kind,b.source_block,b.source_text,b.blob_hash,b.direction,b.size,b.encrypted,b.result_outcome,b.result_reason,b.result_exit_code,b.result_duration_ms,b.looks_like_secret,v.codec,v.length,v.hash FROM block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN body_value v ON v.pk=t.value_pk AND v.state='ready' WHERE b.pk=?1",
                            pk,
                            &mut self.hasher,
                            allowance,
                        )?;
                        let unsupported: bool = db(path, || {
                            reader.index.query_row("SELECT produced_by_pk IS NOT NULL OR input_pk IS NOT NULL FROM block WHERE pk=?1",[pk],|r|r.get(0))
                        })?;
                        if unsupported {
                            return Err(Error::MessagePreparationPending);
                        }
                        if let Some(call) = call {
                            let unsupported: bool = db(path, || {
                                reader.index.query_row("SELECT tool_item_pk IS NOT NULL OR result_block_pk IS NOT NULL FROM tool_call WHERE pk=?1",[call],|r|r.get(0))
                            })?;
                            if unsupported {
                                return Err(Error::MessagePreparationPending);
                            }
                            hash_row(
                                &reader.index,
                                path,
                                "SELECT operation_id,at,action,outcome,outcome_reason,exit_code,duration_ms,child_session_id,inherited FROM tool_call WHERE pk=?1",
                                call,
                                &mut self.hasher,
                                allowance,
                            )?;
                            let values: [Option<i64>; 4] = db(path, || {
                                reader.index.query_row("SELECT tool_value_pk,command_value_pk,command_norm_value_pk,path_value_pk FROM tool_call WHERE pk=?1",[call],|r|Ok([r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?]))
                            })?;
                            for value in values {
                                match value {
                                    None => {
                                        self.hasher.update(b"absent-call-value\0");
                                    }
                                    Some(value) => hash_row(
                                        &reader.index,
                                        path,
                                        "SELECT codec,length,hash FROM body_value WHERE pk=?1 AND state='ready'",
                                        value,
                                        &mut self.hasher,
                                        allowance,
                                    )?,
                                }
                            }
                        }
                        self.block_fields = db(path, || {
                            reader.index.query_row(
                                "SELECT field_root_pk FROM block WHERE pk=?1",
                                [pk],
                                |r| r.get(0),
                            )
                        })?;
                        self.block_after = Some(key);
                        self.block_field_after = None;
                        self.phase = 3;
                    }
                }
            }
            3 => match self.field(reader, self.block_fields, self.block_field_after, allowance)? {
                Some(k) => self.block_field_after = Some(k),
                None => {
                    self.hasher.update(b"end-block\0");
                    self.phase = 2;
                }
            },
            _ => return Err(invalid("compatibility phase")),
        }
        Ok(false)
    }
}

impl CompatibleSelection {
    pub fn advance(
        mut self,
        reader: &ReadHandle,
        allowance: usize,
        cancelled: &AtomicBool,
    ) -> Result<CompatibilityTurn> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if reader.root.join("index.db") != self.path || self.complete {
            return Err(changed());
        }
        if self.singleton {
            self.complete = true;
            return Ok(CompatibilityTurn::Ready(self));
        }
        if let Some(mut hashing) = self.hashing.take() {
            match hashing.advance(reader, allowance) {
                Err(Error::MessagePreparationPending) => return Ok(CompatibilityTurn::Contested),
                Err(e) => return Err(e),
                Ok(true) => return Ok(CompatibilityTurn::Certified(self, hashing.certificate)),
                Ok(false) => {
                    self.hashing = Some(hashing);
                    return Ok(CompatibilityTurn::Progress(self));
                }
            }
        }
        type Candidate = (i64, i64, i64, u64, Option<String>, Option<Vec<u8>>, bool);
        let row: Option<Candidate> = db(&self.path, || {
            reader.index.query_row("SELECT c.pk,c.body_pk,c.input_pk,length(CAST(i.id AS BLOB)),CASE WHEN length(CAST(i.id AS BLOB))<=?3 THEN i.id END,k.digest,EXISTS(SELECT 1 FROM message_candidate x WHERE x.message_pk=c.message_pk AND x.input_pk=c.input_pk AND x.eligible=1 AND x.pk<>c.pk) FROM message_candidate c JOIN input_ref i ON i.pk=c.input_pk LEFT JOIN message_body_compatibility k ON k.body_pk=c.body_pk WHERE c.message_pk=?1 AND c.eligible=1 AND c.pk>?2 ORDER BY c.pk LIMIT 1",params![self.source.body.message,self.after,allowance as i64],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,crate::parser_state::unsigned(r,3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional()
        })?;
        let Some((candidate, body, _, length, input, digest, ambiguous)) = row else {
            self.complete = true;
            return Ok(CompatibilityTurn::Ready(self));
        };
        if ambiguous {
            return Ok(CompatibilityTurn::Contested);
        }
        let input = input.ok_or(Error::ParserAdmission {
            resource: "compatibility carrier identity",
            required: length,
        })?;
        let Some(digest) = digest else {
            self.hashing = match BodyDigest::begin(reader, &self.source, body, candidate, allowance)
            {
                Ok(h) => Some(h),
                Err(Error::MessagePreparationPending) => return Ok(CompatibilityTurn::Contested),
                Err(e) => return Err(e),
            };
            return Ok(CompatibilityTurn::Progress(self));
        };
        let digest: [u8; 32] = digest
            .try_into()
            .map_err(|_| invalid("compatibility certificate digest"))?;
        if self.digest.is_some_and(|d| d != digest) {
            return Ok(CompatibilityTurn::Contested);
        }
        self.digest = Some(digest);
        if self
            .carrier
            .as_ref()
            .is_none_or(|(prior, _)| input < *prior)
        {
            self.carrier = Some((input, body))
        }
        self.after = candidate;
        Ok(CompatibilityTurn::Progress(self))
    }
}

impl Writer {
    pub fn begin_compatible_source_selection(
        &self,
        source: &SourceProjection,
        allowance: usize,
    ) -> Result<CompatibleSelection> {
        let path = self.root.join("index.db");
        let current = source::read_current(
            &self._index,
            &path,
            &source.predecessor,
            &source.logical,
            allowance,
        )?
        .ok_or_else(changed)?;
        if current.reference != source.reference {
            return Err(changed());
        }
        let (revision, epoch) = db(&path, || {
            self._index.query_row(
                "SELECT contributor_revision,removal_epoch FROM message WHERE pk=?1 AND hidden=0",
                [source.body.message],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })?;
        // A singleton needs no cross-input equality or import interpretation
        // witness. The immutable source certificate and membership fence suffice.
        let singleton: bool = db(&path, || {
            self._index.query_row("SELECT NOT EXISTS(SELECT 1 FROM message_candidate WHERE message_pk=?1 AND eligible=1 AND pk<>?2 LIMIT 1)",params![source.body.message,source.candidate],|r|r.get(0))
        })?;
        Ok(CompatibleSelection {
            path,
            source: source.clone(),
            revision,
            epoch,
            after: 0,
            digest: None,
            carrier: if singleton {
                Some((String::new(), source.body.pk))
            } else {
                None
            },
            hashing: None,
            complete: false,
            singleton,
        })
    }
    pub fn certify_body_compatibility(&mut self, proof: CompatibilityCertificate) -> Result<()> {
        let path = self.root.join("index.db");
        if path != proof.path {
            return Err(changed());
        }
        let tx = db(&path, || self._index.transaction())?;
        let valid: bool = db(&path, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM message_body b JOIN message_candidate c ON c.pk=b.preferred_candidate_pk JOIN message m ON m.pk=b.message_pk WHERE b.pk=?1 AND c.pk=?2 AND b.message_pk=?3 AND b.session_pk=?4 AND b.basis_digest=?5 AND c.basis_digest=?5 AND b.ready=1 AND c.ready=1 AND c.eligible=1 AND b.removal_epoch=?6 AND m.removal_epoch=?6 AND m.hidden=0) AND EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?7)",params![proof.body,proof.candidate,proof.owner.message,proof.owner.session,proof.basis.as_slice(),proof.epoch,proof.generation],|r|r.get(0))
        })?;
        if !valid {
            return Err(changed());
        }
        db(&path, || {
            tx.execute("INSERT INTO message_body_compatibility(body_pk,digest) VALUES(?1,?2) ON CONFLICT(body_pk) DO NOTHING",params![proof.body,proof.digest.as_slice()])
        })?;
        let retained: Vec<u8> = db(&path, || {
            tx.query_row(
                "SELECT digest FROM message_body_compatibility WHERE body_pk=?1",
                [proof.body],
                |r| r.get(0),
            )
        })?;
        if retained != proof.digest {
            return Err(invalid("immutable compatibility certificate changed"));
        }
        db(&path, || tx.commit())
    }
    pub fn install_compatible_source_selection(
        &mut self,
        proof: CompatibleSelection,
        allowance: usize,
    ) -> Result<SourceSelection> {
        if !proof.complete
            || (!proof.singleton && proof.digest.is_none())
            || proof.hashing.is_some()
        {
            return Err(Error::MessagePreparationPending);
        }
        let path = self.root.join("index.db");
        if proof.path != path {
            return Err(changed());
        }
        let tx = db(&path, || self._index.transaction())?;
        let current = source::read_current(
            &tx,
            &path,
            &proof.source.predecessor,
            &proof.source.logical,
            allowance,
        )?
        .ok_or_else(changed)?;
        if current.reference != proof.source.reference {
            return Err(changed());
        }
        let (revision, epoch): (i64, i64) = db(&path, || {
            tx.query_row(
                "SELECT contributor_revision,removal_epoch FROM message WHERE pk=?1 AND hidden=0",
                [proof.source.body.message],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })?;
        if revision != proof.revision || epoch != proof.epoch {
            return Err(changed());
        }
        let body = super::tool_associations::carry_selected(
            &tx,
            &path,
            d::Owner {
                message: proof.source.body.message,
                session: proof.source.body.session,
            },
            proof.carrier.ok_or_else(changed)?.1,
        )?;
        let changed = db(&path, || {
            tx.execute(
                "UPDATE message SET selected_body_pk=?2 WHERE pk=?1 AND selected_body_pk IS NOT ?2",
                params![proof.source.body.message, body],
            )
        })?;
        if changed == 0 {
            return Ok(SourceSelection::AlreadySelected);
        }
        let next = writer::integer(
            crate::counter(&tx, &path)?
                .checked_add(1)
                .ok_or(Error::Range)?,
        )?;
        db(&path, || {
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='write_counter'",
                [next.to_string()],
            )
        })?;
        db(&path, || {
            tx.execute(
                "UPDATE session SET changed=?2 WHERE pk=?1",
                params![proof.source.body.session, next],
            )
        })?;
        db(&path, || tx.commit())?;
        Ok(SourceSelection::Selected)
    }
}

//! One prepared text block and its resumable projection checkpoint commit
//! together. Payload interpretation/serialization/hash work precedes this tx.
use super::{
    directory as d,
    writer::{self, Build},
};
use crate::{Error, Result, Writer, db, values};
use rusqlite::{OptionalExtension, Transaction, params};
use std::path::Path;
const NAMESPACE: &str = "message.projection.v1";
#[derive(Clone, Copy, Debug)]
pub enum TextKind {
    Text,
    Thinking { encrypted: Option<bool> },
}
/// Provider-owned block ordinal convention, fixed before block preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageLayout {
    Flat,
    SeparateReasoning,
}
/// Mapper witnesses before fill/enrichment, independent of normalized defaults.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockPresence {
    pub source_block: bool,
    pub source_text: bool,
}
/// Cursor bytes are a bounded provider-owned resume object, never a Message or
/// growing member list. Sequence CAS prevents replay of an already committed unit.
pub struct ProjectionStep<'a> {
    pub expected_sequence: u64,
    pub next_cursor: &'a [u8],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionState {
    pub sequence: u64,
    pub cursor: Option<Vec<u8>>,
}
pub(super) fn admit(step: &ProjectionStep<'_>, allowance: usize) -> Result<()> {
    let required = step
        .next_cursor
        .len()
        .checked_add(NAMESPACE.len() + 8)
        .ok_or(Error::Range)?;
    if required > allowance {
        return Err(Error::ParserAdmission {
            resource: "message projection cursor bytes",
            required: required as u64,
        });
    }
    writer::integer(step.expected_sequence.checked_add(1).ok_or(Error::Range)?)?;
    Ok(())
}
pub(super) fn commit_cursor(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    step: &ProjectionStep<'_>,
) -> Result<()> {
    let expected = writer::integer(step.expected_sequence)?;
    let changed = db(path, || {
        tx.execute("UPDATE message_build SET source_cursor=source_cursor+1 WHERE pk=?1 AND source_cursor=?2",params![build.pk,expected])
    })?;
    if changed != 1 {
        return Err(Error::MessagePredecessorChanged);
    }
    db(path, || {
        tx.execute("INSERT INTO parser_cell(input_pk,chain,state_generation,namespace,key,value) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(input_pk,chain,state_generation,namespace,key) DO UPDATE SET value=excluded.value",params![build.source.input_pk,build.source.chain,build.source.generation,NAMESPACE,build.pk.to_be_bytes().as_slice(),step.next_cursor])
    })?;
    Ok(())
}
impl Writer {
    pub fn message_projection(&self, build: &Build, allowance: usize) -> Result<ProjectionState> {
        let overhead = NAMESPACE.len() + 8;
        let payload_allowance = allowance
            .checked_sub(overhead)
            .ok_or(Error::ParserAdmission {
                resource: "message projection cursor bytes",
                required: overhead as u64,
            })?;
        let path = self.root.join("index.db");
        // Reading the durable cursor also resumes the completed rank phase.
        // Mutating APIs retain their stricter project-only checks.
        writer::check_completed(&self._index, &path, build)?;
        let sequence = db(&path, || {
            self._index.query_row(
                "SELECT source_cursor FROM message_build WHERE pk=?1",
                [build.pk],
                |r| crate::parser_state::unsigned(r, 0),
            )
        })?;
        let row: Option<(u64, Option<Vec<u8>>)> = db(&path, || {
            self._index.query_row("SELECT length(value),CASE WHEN length(value)<=?6 THEN value ELSE NULL END FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",params![build.source.input_pk,build.source.chain,build.source.generation,NAMESPACE,build.pk.to_be_bytes().as_slice(),i64::try_from(payload_allowance).unwrap_or(i64::MAX)],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?))).optional()
        })?;
        let cursor = match row {
            Some((required, None)) => {
                return Err(Error::ParserAdmission {
                    resource: "message projection cursor bytes",
                    required: required.saturating_add(overhead as u64),
                });
            }
            Some((_, Some(bytes))) => Some(bytes),
            None if sequence == 0 => None,
            None => {
                return Err(Error::Invariant(
                    "committed Message projection lost its cursor".into(),
                ));
            }
        };
        Ok(ProjectionState { sequence, cursor })
    }
    /// A no-output source unit still advances only after its checkpoint commits.
    pub fn advance_message_projection(
        &mut self,
        build: &Build,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        admit(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        refuse_origin(&tx, &path, build)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Layout affects public block IDs. It cannot change after a block/root
    /// exists; the owning source supplies it instead of inferring from content.
    pub fn set_message_layout(
        &mut self,
        build: &Build,
        layout: MessageLayout,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        admit(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        refuse_origin(&tx, &path, build)?;
        let layout = match layout {
            MessageLayout::Flat => "flat",
            MessageLayout::SeparateReasoning => "separate_reasoning",
        };
        if db(&path, || {
            tx.execute("UPDATE message_body SET layout=?2 WHERE pk=?1 AND ready=0 AND block_root_pk IS NULL AND NOT EXISTS(SELECT 1 FROM message_build WHERE pk=?3 AND active_block_pk IS NOT NULL)",params![build.body,layout,build.pk])
        })? != 1
        {
            return Err(Error::MessagePreparationPending);
        }
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Stage one private text/thinking block. The active optional UTF-8 value
    /// transfers to text ownership; no published directory can see this block.
    pub fn begin_text_block(
        &mut self,
        build: &Build,
        group: u64,
        local: u64,
        kind: TextKind,
        presence: BlockPresence,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<i64> {
        admit(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let block = begin_in(&tx, &path, build, group, local, kind, presence)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(block)
    }
    /// Mapper-proved membership is THIS active block, not a coincident final
    /// ordinal. The root owns correspondence; keys contain only local semantics.
    pub fn write_block_field(
        &mut self,
        build: &Build,
        slot: &polylogue_model::merge::Slot,
        source_time: Option<i64>,
        supplied: bool,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        admit(step, allowance)?;
        let key = writer::block_field_key(slot, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let block = active_block(&tx, &path, build)?;
        let empty:bool=db(&path,||tx.query_row("SELECT b.kind='text' AND v.length=0 FROM block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN body_value v ON v.pk=t.value_pk WHERE b.pk=?1",[block],|r|r.get::<_,Option<bool>>(0)))?.unwrap_or(false);
        if supplied && empty {
            return Err(Error::Invariant(
                "empty Text cannot supply field rank".into(),
            ));
        }
        writer::write_field_in(&tx, &path, build, &key, Some(block), source_time, supplied)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Install only after every field value has transferred and the active
    /// value is empty. Each fixed directory path and cursor CAS commit together.
    pub fn install_text_block(
        &mut self,
        build: &Build,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<i64> {
        admit(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let block = install_in(&tx, &path, build)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(block)
    }
    /// Atomic convenience over the SAME staging and installation core.
    pub fn write_text_block(
        &mut self,
        build: &Build,
        group: u64,
        local: u64,
        kind: TextKind,
        presence: BlockPresence,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<i64> {
        admit(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        begin_in(&tx, &path, build, group, local, kind, presence)?;
        let block = install_in(&tx, &path, build)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(block)
    }
}

fn refuse_origin(tx: &Transaction<'_>, path: &Path, build: &Build) -> Result<()> {
    let pending: bool = db(path, || {
        tx.query_row(
            "SELECT active_origin_value_pk IS NOT NULL FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    if pending {
        return Err(Error::MessagePreparationPending);
    }
    Ok(())
}
fn active_block(tx: &Transaction<'_>, path: &Path, build: &Build) -> Result<i64> {
    let block: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT active_block_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    let block = block.ok_or(Error::MessagePreparationPending)?;
    let private: bool = db(path, || {
        tx.query_row("SELECT EXISTS(SELECT 1 FROM block WHERE pk=?1 AND message_pk=?2 AND session_pk=?3) AND NOT EXISTS(SELECT 1 FROM message_body_edge WHERE block_pk=?1) AND NOT EXISTS(SELECT 1 FROM tool_call WHERE call_block_pk=?1 OR result_block_pk=?1)",params![block,build.message,build.session],|r|r.get(0))
    })?;
    if !private {
        return Err(Error::Invariant(
            "active block is not private to preparation".into(),
        ));
    }
    Ok(block)
}
fn begin_in(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    group: u64,
    local: u64,
    kind: TextKind,
    presence: BlockPresence,
) -> Result<i64> {
    refuse_origin(tx, path, build)?;
    let active: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT active_block_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    if active.is_some() {
        return Err(Error::MessagePreparationPending);
    }
    let group_i = writer::integer(group)?;
    let local_i = writer::integer(local)?;
    if presence.source_text && !presence.source_block {
        return Err(Error::Invariant(
            "text witness without block witness".into(),
        ));
    }
    let value: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT active_value_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    let text = if let Some(value) = value {
        let head = values::head(
            tx,
            path,
            &values::ValueId {
                pk: value,
                index_generation: build.source.index_generation.clone(),
                index_path: path.to_path_buf(),
            },
        )?;
        if head.state != values::State::Ready || head.codec != values::Codec::Utf8 {
            return Err(Error::ValueNotReady);
        }
        if presence.source_text && head.length == 0 {
            return Err(Error::Invariant(
                "empty text payload cannot supply text-block rank".into(),
            ));
        }
        if matches!(kind, TextKind::Text)
            && head.length == 0
            && (presence.source_block || presence.source_text)
        {
            return Err(Error::Invariant(
                "empty Text cannot supply source rank".into(),
            ));
        }
        let hash = head.hash.ok_or(Error::ValueNotReady)?;
        let old_text: Option<(i64, i64)> = db(path, || {
            tx.query_row(
                "SELECT id,value_pk FROM text WHERE hash=?1",
                [hash.as_slice()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
        })?;
        let text = if let Some((text, stored)) = old_text {
            if stored != value {
                return Err(Error::Invariant(
                    "text dedupe disagrees with canonical value".into(),
                ));
            }
            if db(path, || {
                tx.execute("UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND state='ready' AND incoming>1",[value])
            })? != 1
            {
                return Err(Error::Invariant("shared text lost its value lease".into()));
            }
            text
        } else {
            db(path, || {
                tx.execute(
                    "INSERT INTO text(hash,value_pk) VALUES(?1,?2)",
                    params![hash.as_slice(), value],
                )
            })?;
            tx.last_insert_rowid()
        };
        Some(text)
    } else if matches!(kind, TextKind::Thinking { .. }) && !presence.source_text {
        None
    } else {
        return Err(Error::ValueNotReady);
    };

    let layout: String = db(path, || {
        tx.query_row(
            "SELECT layout FROM message_body WHERE pk=?1 AND ready=0",
            [build.body],
            |r| r.get(0),
        )
    })?;
    let (name, encrypted) = match kind {
        TextKind::Text => ("text", None),
        TextKind::Thinking { encrypted } => ("thinking", encrypted),
    };
    let reasoning = matches!(kind, TextKind::Thinking { .. }) && layout == "separate_reasoning";
    db(path, || {
        tx.execute("INSERT INTO block(message_pk,session_pk,group_ordinal,local_ordinal,slot_kind,kind,text_id,encrypted,source_block,source_text) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",params![build.message,build.session,group_i,local_i,if reasoning {"reasoning"}else{"ordinary"},name,text,encrypted,presence.source_block,presence.source_text])
    })?;
    let block = tx.last_insert_rowid();
    db(path, || {
        tx.execute(
            "UPDATE message_build SET active_value_pk=NULL,active_block_pk=?2 WHERE pk=?1",
            params![build.pk, block],
        )
    })?;
    Ok(block)
}
fn install_in(tx: &Transaction<'_>, path: &Path, build: &Build) -> Result<i64> {
    refuse_origin(tx, path, build)?;
    let block = active_block(tx, path, build)?;
    let active: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT active_value_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    if active.is_some() {
        return Err(Error::ValueNotReady);
    }
    let (group, local, kind, reasoning, source_block, source_text, field_root, text_length): (
        u64,
        u64,
        String,
        bool,
        bool,
        bool,
        Option<i64>,
        Option<i64>,
    ) = db(path, || {
        tx.query_row("SELECT b.group_ordinal,b.local_ordinal,b.kind,b.slot_kind='reasoning',b.source_block,b.source_text,b.field_root_pk,v.length FROM block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN body_value v ON v.pk=t.value_pk WHERE b.pk=?1",[block],|r|Ok((crate::parser_state::unsigned(r,0)?,crate::parser_state::unsigned(r,1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?)))
    })?;
    let presence = BlockPresence {
        source_block,
        source_text,
    };
    let old: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT block_root_pk FROM message_body WHERE pk=?1 AND ready=0",
            [build.body],
            |r| r.get(0),
        )
    })?;
    let fields = d::summary(
        tx,
        path,
        d::Owner {
            message: build.message,
            session: build.session,
        },
        d::Kind::Fields,
        field_root,
    )?;
    let owner = d::Owner {
        message: build.message,
        session: build.session,
    };
    let mut work = d::Work::default();
    let inner = match d::lookup(tx, path, owner, d::Kind::Groups, old, group, &mut work)? {
        Some(d::Edge {
            link:
                d::Link::Node {
                    pk,
                    kind: d::Kind::Blocks,
                    height: 15,
                },
            ..
        }) => Some(pk),
        None => None,
        _ => return Err(Error::Invariant("invalid block group root".into())),
    };
    let counts = d::Counts {
        ordinary: i64::from(!reasoning),
        reasoning: i64::from(reasoning),
        source_blocks: i64::from(presence.source_block),
        source_text_blocks: i64::from(presence.source_text),
        source_fields: if kind == "text" && text_length == Some(0) {
            0
        } else {
            fields.source_fields
        },
    };
    let (inner, summary) = d::replace(
        tx,
        path,
        owner,
        d::Kind::Blocks,
        inner,
        local,
        Some(d::Edge {
            link: d::Link::Block {
                pk: block,
                call: None,
            },
            counts,
        }),
        &mut work,
    )?;
    let (root, _) = d::replace(
        tx,
        path,
        owner,
        d::Kind::Groups,
        old,
        group,
        Some(d::Edge {
            link: d::Link::Node {
                pk: inner,
                kind: d::Kind::Blocks,
                height: 15,
            },
            counts: summary,
        }),
        &mut work,
    )?;
    d::acquire(tx, path, owner, root)?;
    if let Some(old) = old {
        d::release(tx, path, owner, old)?;
    }
    db(path, || {
        tx.execute(
            "UPDATE message_body SET block_root_pk=?2 WHERE pk=?1 AND ready=0",
            params![build.body, root],
        )
    })?;
    db(path, || {
        tx.execute(
            "UPDATE message_build SET active_block_pk=NULL WHERE pk=?1",
            [build.pk],
        )
    })?;
    Ok(block)
}
pub(super) fn discard_cursor(tx: &Transaction<'_>, path: &Path, build: &Build) -> Result<()> {
    db(path, || {
        tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",params![build.source.input_pk,build.source.chain,build.source.generation,NAMESPACE,build.pk.to_be_bytes().as_slice()])
    })?;
    Ok(())
}

/// Drop one known text/thinking leaf only after its final directory reference
/// disappears. Future operation-summary retirement must call this same helper
/// after releasing its call/result reference. Other block forms remain owed.
pub(super) fn release_unreferenced_block(tx: &Transaction<'_>, path: &Path, pk: i64) -> Result<()> {
    let referenced: bool = db(path, || {
        tx.query_row("SELECT EXISTS(SELECT 1 FROM message_build WHERE active_block_pk=?1) OR EXISTS(SELECT 1 FROM message_body_edge WHERE block_pk=?1) OR EXISTS(SELECT 1 FROM tool_call WHERE call_block_pk=?1) OR EXISTS(SELECT 1 FROM tool_call WHERE result_block_pk=?1)",[pk],|r|r.get(0))
    })?;
    if referenced {
        return Ok(());
    }
    let row: Option<(String, Option<i64>, Option<i64>, i64, i64)> = db(path, || {
        tx.query_row(
            "SELECT kind,text_id,field_root_pk,message_pk,session_pk FROM block WHERE pk=?1",
            [pk],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .optional()
    })?;
    let Some((kind, text, fields, message, session)) = row else {
        return Ok(());
    };
    if !matches!(kind.as_str(), "text" | "thinking") {
        return Err(Error::Invariant(
            "this block kind still requires its owning retirement implementation".into(),
        ));
    }
    db(path, || tx.execute("DELETE FROM block WHERE pk=?1", [pk]))?;
    if let Some(root) = fields {
        d::release(tx, path, d::Owner { message, session }, root)?;
    }
    if let Some(text) = text {
        let retained: bool = db(path, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM block WHERE text_id=?1) OR EXISTS(SELECT 1 FROM tool_item_version WHERE result_text_id=?1)",[text],|r|r.get(0))
        })?;
        if !retained {
            let value: i64 = db(path, || {
                tx.query_row("SELECT value_pk FROM text WHERE id=?1", [text], |r| {
                    r.get(0)
                })
            })?;
            db(path, || tx.execute("DELETE FROM text WHERE id=?1", [text]))?;
            super::retirement::release_ready_value(tx, path, value)?;
        }
    }
    Ok(())
}
impl Writer {
    /// Reclaim one zero-lease Message-local node, including at most16 known
    /// text leaves. Selected/history roots and shared subtrees retain leases.
    /// No scan for an unreferenced block or payload population is performed.
    pub fn collect_message_nodes(&mut self, message: &polylogue_model::MessageId) -> Result<bool> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        let owner: Option<i64> = db(&path, || {
            tx.query_row("SELECT m.pk FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.id=?1 AND s.id=?2",params![message.as_str(),message.session().as_str()],|r|r.get(0)).optional()
        })?;
        let Some(owner) = owner else { return Ok(false) };
        let node: Option<i64> = db(&path, || {
            tx.query_row("SELECT pk FROM message_body_node WHERE message_pk=?1 AND incoming=0 ORDER BY pk LIMIT 1",[owner],|r|r.get(0)).optional()
        })?;
        let collected = match node {
            Some(node) => d::collect(&tx, &path, node)?,
            None => false,
        };
        db(&path, || tx.commit())?;
        Ok(collected)
    }
}

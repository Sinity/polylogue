//! Blocks of a prepared Message, and the projection cursor. Each write commits
//! together with one cursor step, so a source unit applies exactly once. Values
//! are hashed and serialized before these transactions start.
use super::{
    directory as d,
    writer::{self, Build, FieldUpdate},
};
use crate::{Error, Result, Writer, db, values};
use rusqlite::{OptionalExtension, Transaction, params};
use std::path::Path;
const NAMESPACE: &str = "message.projection.v1";
#[derive(Clone, Copy, Debug)]
pub enum BlockKind<'a> {
    Text,
    Thinking {
        encrypted: Option<bool>,
    },
    /// `operation_id` comes from the caller; it is never derived from a block
    /// or native ID.
    ToolCall {
        operation_id: &'a str,
        at: Option<i64>,
        action: Option<ToolAction>,
    },
    ToolResult {
        outcome: &'a polylogue_model::ToolOutcome,
        exit_code: Option<i64>,
        duration_ms: Option<u64>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ToolAction {
    Shell,
    Code,
    FileRead,
    FileWrite,
    FileEdit,
    Search,
    Web,
    Subagent,
    Mcp,
    Other,
}
impl ToolAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::Shell => "shell",
            Self::Code => "code",
            Self::FileRead => "file_read",
            Self::FileWrite => "file_write",
            Self::FileEdit => "file_edit",
            Self::Search => "search",
            Self::Web => "web",
            Self::Subagent => "subagent",
            Self::Mcp => "mcp",
            Self::Other => "other",
        }
    }
}
/// How block ordinals are numbered. The source sets it before the first block;
/// it cannot change afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageLayout {
    Flat,
    SeparateReasoning,
}
/// Whether the source itself had this block and its text, before defaults or
/// filling. Both count toward the candidate's rank.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockPresence {
    pub source_block: bool,
    pub source_text: bool,
}
/// One step of a resumable projection. `expected_sequence` must equal the stored
/// sequence, so a step that already committed cannot apply twice. `next_cursor`
/// is the source's small resume token, not a growing list.
pub struct ProjectionStep<'a> {
    pub expected_sequence: u64,
    pub next_cursor: &'a [u8],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionState {
    pub sequence: u64,
    pub cursor: Option<Vec<u8>>,
}
pub(super) fn check_step(step: &ProjectionStep<'_>, allowance: usize) -> Result<()> {
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
    writer::to_i64(step.expected_sequence.checked_add(1).ok_or(Error::Range)?)?;
    Ok(())
}
pub(super) fn commit_cursor(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    step: &ProjectionStep<'_>,
) -> Result<()> {
    let expected = writer::to_i64(step.expected_sequence)?;
    let changed = db(path, || {
        tx.execute("UPDATE message_build SET source_cursor=source_cursor+1 WHERE pk=?1 AND source_cursor=?2",params![build.pk,expected])
    })?;
    if changed != 1 {
        return Err(Error::StaleMessage);
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
        // Readable after the projection is finished, unlike the writes.
        writer::check_current(&self._index, &path, build)?;
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
                return Err(Error::Invariant("projection cursor is missing".into()));
            }
        };
        Ok(ProjectionState { sequence, cursor })
    }
    /// Advances the cursor for a source unit that writes nothing.
    pub fn advance_message_projection(
        &mut self,
        build: &Build,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        check_step(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        require_no_pending_origin(&tx, &path, build)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Sets the block layout, which determines public block IDs. Fails once a
    /// block exists. The source declares it; it is never inferred from content.
    pub fn set_message_layout(
        &mut self,
        build: &Build,
        layout: MessageLayout,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        check_step(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        require_no_pending_origin(&tx, &path, build)?;
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
    /// Starts a block that later field writes attach to; `install_block` adds
    /// it to the Message. Text and thinking blocks take the active UTF-8 value
    /// as their text. Tool blocks need no active value; a tool call also gets
    /// its tool_call row here.
    pub fn begin_block(
        &mut self,
        build: &Build,
        group: u64,
        local: u64,
        kind: BlockKind<'_>,
        presence: BlockPresence,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<i64> {
        check_step(step, allowance)?;
        check_kind(kind, step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        let block = begin_in(&tx, &path, build, group, local, kind, presence)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(block)
    }
    /// Writes the sealed active value as a field of the block started by
    /// `begin_block`. The field key does not include the block's ordinal.
    pub fn write_block_field(
        &mut self,
        build: &Build,
        slot: &polylogue_model::merge::Slot,
        source_time: Option<i64>,
        supplied: bool,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        check_step(step, allowance)?;
        let key = writer::block_field_key(slot, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        write_field_in(&tx, &path, build, slot, &key, source_time, supplied)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Stores the active UTF-8 value as the started tool call's command. The
    /// caller prepares the command text; it is not decoded from a JSON field.
    /// Each tool_call column holds its own lease, even on the same value.
    pub fn write_tool_command(
        &mut self,
        build: &Build,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        check_step(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        require_no_pending_origin(&tx, &path, build)?;
        let block = active_block(&tx, &path, build)?;
        let call = active_call(&tx, &path, build, block)?;
        let value = active_value(&tx, &path, build, values::Codec::Utf8)?;
        if db(&path, || {
            tx.execute(
                "UPDATE tool_call SET command_value_pk=?2 WHERE pk=?1 AND command_value_pk IS NULL",
                params![call, value],
            )
        })? != 1
        {
            return Err(Error::Invariant("tool call already has a command".into()));
        }
        if db(&path, || {
            tx.execute(
                "UPDATE message_build SET active_value_pk=NULL WHERE pk=?1 AND active_value_pk=?2",
                params![build.pk, value],
            )
        })? != 1
        {
            return Err(Error::StaleMessage);
        }
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// Adds the started block to the Message's block directory. Requires that
    /// no active value is left.
    pub fn install_block(
        &mut self,
        build: &Build,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<i64> {
        check_step(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        let block = install_in(&tx, &path, build)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(block)
    }
    /// `begin_block` and `install_block` in one transaction.
    pub fn write_block(
        &mut self,
        build: &Build,
        group: u64,
        local: u64,
        kind: BlockKind<'_>,
        presence: BlockPresence,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<i64> {
        check_step(step, allowance)?;
        check_kind(kind, step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check_writable(&tx, &path, build)?;
        begin_in(&tx, &path, build, group, local, kind, presence)?;
        let block = install_in(&tx, &path, build)?;
        commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(block)
    }
}

fn outcome_reason(outcome: &polylogue_model::ToolOutcome) -> Option<std::borrow::Cow<'_, str>> {
    use polylogue_model::{ErrorReason, ToolOutcome};
    match outcome {
        ToolOutcome::Ok | ToolOutcome::Error { reason: None } => None,
        ToolOutcome::Unknown { reason } => Some(std::borrow::Cow::Borrowed(reason.as_str())),
        ToolOutcome::Error {
            reason: Some(ErrorReason::Denied(kind)),
        } => Some(std::borrow::Cow::Owned(format!("denied:{kind}"))),
        ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted),
        } => Some("interrupted".into()),
        ToolOutcome::Error {
            reason: Some(ErrorReason::TimedOut),
        } => Some("timed_out".into()),
        ToolOutcome::Error {
            reason: Some(ErrorReason::BlockedByHook),
        } => Some("blocked_by_hook".into()),
        ToolOutcome::Error {
            reason: Some(ErrorReason::Other(reason)),
        } => Some(reason.as_str().into()),
    }
}
/// Checks that the block's metadata and the cursor fit in `allowance`.
fn check_kind(kind: BlockKind<'_>, step: &ProjectionStep<'_>, allowance: usize) -> Result<()> {
    let bytes = match kind {
        BlockKind::ToolCall { operation_id, .. } => {
            if operation_id.is_empty() {
                return Err(Error::Invariant("tool call needs an operation ID".into()));
            }
            operation_id.len()
        }
        BlockKind::ToolResult {
            outcome,
            duration_ms,
            ..
        } => {
            duration_ms.map(writer::to_i64).transpose()?;
            outcome_reason(outcome).map_or(0, |reason| reason.len())
        }
        _ => 0,
    };
    let required = step
        .next_cursor
        .len()
        .checked_add(NAMESPACE.len() + 8)
        .and_then(|base| base.checked_add(bytes))
        .ok_or(Error::Range)?;
    if required > allowance {
        return Err(Error::ParserAdmission {
            resource: "block metadata and cursor bytes",
            required: required as u64,
        });
    }
    Ok(())
}
fn require_no_pending_origin(tx: &Transaction<'_>, path: &Path, build: &Build) -> Result<()> {
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
pub(super) fn write_field_in(
    tx: &Transaction<'_>,
    path: &std::path::Path,
    build: &Build,
    slot: &polylogue_model::merge::Slot,
    key: &[u8],
    source_time: Option<i64>,
    supplied: bool,
) -> Result<()> {
    if supplied
        && matches!(
            slot,
            polylogue_model::merge::Slot::Block {
                field: polylogue_model::merge::BlockField::ThinkingSignature,
                ..
            }
        )
    {
        return Err(Error::Invariant(
            "a thinking signature cannot be source-supplied".into(),
        ));
    }
    let block = active_block(tx, path, build)?;
    let empty:bool=db(path,||tx.query_row("SELECT b.kind='text' AND v.length=0 FROM block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN body_value v ON v.pk=t.value_pk WHERE b.pk=?1",[block],|r|r.get::<_,Option<bool>>(0)))?.unwrap_or(false);
    if supplied && empty {
        return Err(Error::Invariant(
            "fields of an empty text block cannot be source-supplied".into(),
        ));
    }
    if matches!(
        slot,
        polylogue_model::merge::Slot::Block {
            field: polylogue_model::merge::BlockField::Tool,
            ..
        }
    ) {
        let call = active_call(tx, path, build, block)?;
        let value = active_value(tx, path, build, values::Codec::Json)?;
        if db(path, || {
            tx.execute("UPDATE body_value SET incoming=incoming+1 WHERE pk=?1 AND state='ready' AND incoming>0",[value])
        })? != 1
        {
            return Err(Error::ValueNotReady);
        }
        if db(path, || {
            tx.execute(
                "UPDATE tool_call SET tool_value_pk=?2 WHERE pk=?1 AND tool_value_pk IS NULL",
                params![call, value],
            )
        })? != 1
        {
            return Err(Error::Invariant(
                "tool call already has a tool field".into(),
            ));
        }
    }
    writer::update_field_in(
        tx,
        path,
        build,
        key,
        Some(block),
        source_time,
        supplied,
        FieldUpdate::Insert,
    )
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
        tx.query_row("SELECT EXISTS(SELECT 1 FROM block WHERE pk=?1 AND message_pk=?2 AND session_pk=?3) AND NOT EXISTS(SELECT 1 FROM message_body_edge WHERE block_pk=?1) AND NOT EXISTS(SELECT 1 FROM tool_call WHERE result_block_pk=?1)",params![block,build.message,build.session],|r|r.get(0))
    })?;
    if !private {
        return Err(Error::Invariant(
            "active block is already referenced outside this preparation".into(),
        ));
    }
    let kind: String = db(path, || {
        tx.query_row("SELECT kind FROM block WHERE pk=?1", [block], |r| r.get(0))
    })?;
    let call = super::retirement::tool_call_of(tx, path, block)?;
    match (kind.as_str(), call) {
        ("tool_call", Some(call)) => {
            if super::retirement::tool_call_referenced(tx, path, call)? {
                return Err(Error::Invariant(
                    "active tool call is already referenced".into(),
                ));
            }
        }
        ("tool_call", None) => {
            return Err(Error::Invariant(
                "tool call block has no tool_call row".into(),
            ));
        }
        (_, Some(_)) => {
            return Err(Error::Invariant(
                "non-call block has a tool_call row".into(),
            ));
        }
        _ => {}
    }
    Ok(block)
}
fn active_call(tx: &Transaction<'_>, path: &Path, build: &Build, block: i64) -> Result<i64> {
    let call = super::retirement::tool_call_of(tx, path, block)?
        .ok_or(Error::Invariant("active block is not a tool call".into()))?;
    let owned: bool = db(path, || {
        tx.query_row("SELECT call_block_pk=?2 AND message_pk=?3 AND session_pk=?4 FROM tool_call WHERE pk=?1",params![call,block,build.message,build.session],|r|r.get(0))
    })?;
    if !owned {
        return Err(Error::Invariant(
            "tool_call row belongs to another block or message".into(),
        ));
    }
    Ok(call)
}
fn active_value(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    codec: values::Codec,
) -> Result<i64> {
    let value: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT active_value_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    let value = value.ok_or(Error::ValueNotReady)?;
    let head = writer::value_head(tx, path, build, value)?;
    if head.state != values::State::Ready || head.codec != codec {
        return Err(Error::ValueNotReady);
    }
    Ok(value)
}
fn begin_in(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    group: u64,
    local: u64,
    kind: BlockKind<'_>,
    presence: BlockPresence,
) -> Result<i64> {
    require_no_pending_origin(tx, path, build)?;
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
    let group_i = writer::to_i64(group)?;
    let local_i = writer::to_i64(local)?;
    if presence.source_text && !presence.source_block {
        return Err(Error::Invariant(
            "source text without a source block".into(),
        ));
    }
    let value: Option<i64> = db(path, || {
        tx.query_row(
            "SELECT active_value_pk FROM message_build WHERE pk=?1",
            [build.pk],
            |r| r.get(0),
        )
    })?;
    let tool = matches!(
        kind,
        BlockKind::ToolCall { .. } | BlockKind::ToolResult { .. }
    );
    if tool && presence.source_text {
        return Err(Error::Invariant(
            "tool blocks cannot have source text".into(),
        ));
    }
    if tool && value.is_some() {
        return Err(Error::MessagePreparationPending);
    }
    let text = if tool {
        None
    } else if let Some(value) = value {
        let head = writer::value_head(tx, path, build, value)?;
        if head.state != values::State::Ready || head.codec != values::Codec::Utf8 {
            return Err(Error::ValueNotReady);
        }
        if presence.source_text && head.length == 0 {
            return Err(Error::Invariant(
                "empty text cannot count as source text".into(),
            ));
        }
        if matches!(kind, BlockKind::Text)
            && head.length == 0
            && (presence.source_block || presence.source_text)
        {
            return Err(Error::Invariant(
                "an empty text block cannot count as a source block".into(),
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
                    "stored text points at a different value".into(),
                ));
            }
            if db(path, || {
                tx.execute("UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND state='ready' AND incoming>1",[value])
            })? != 1
            {
                return Err(Error::Invariant("shared text value has no lease".into()));
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
    } else if matches!(kind, BlockKind::Thinking { .. }) && !presence.source_text {
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
        BlockKind::Text => ("text", None),
        BlockKind::Thinking { encrypted } => ("thinking", encrypted),
        BlockKind::ToolCall { .. } => ("tool_call", None),
        BlockKind::ToolResult { .. } => ("tool_result", None),
    };
    let reasoning = matches!(kind, BlockKind::Thinking { .. }) && layout == "separate_reasoning";
    db(path, || {
        tx.execute("INSERT INTO block(message_pk,session_pk,group_ordinal,local_ordinal,slot_kind,kind,text_id,encrypted,source_block,source_text) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",params![build.message,build.session,group_i,local_i,if reasoning {"reasoning"}else{"ordinary"},name,text,encrypted,presence.source_block,presence.source_text])
    })?;
    let block = tx.last_insert_rowid();
    match kind {
        BlockKind::ToolCall {
            operation_id,
            at,
            action,
        } => {
            db(path, || {
                tx.execute("INSERT INTO tool_call(session_pk,message_pk,operation_id,call_block_pk,at,action,outcome,outcome_reason) VALUES(?1,?2,?3,?4,?5,?6,'unknown','no_result')",params![build.session,build.message,operation_id,block,at,action.map(ToolAction::as_str)])
            })?;
        }
        BlockKind::ToolResult {
            outcome,
            exit_code,
            duration_ms,
        } => {
            let reason = outcome_reason(outcome);
            let duration = duration_ms.map(writer::to_i64).transpose()?;
            db(path, || {
                tx.execute("UPDATE block SET result_outcome=?2,result_reason=?3,result_exit_code=?4,result_duration_ms=?5 WHERE pk=?1",params![block,outcome.as_str(),reason.as_deref(),exit_code,duration])
            })?;
        }
        _ => {}
    }
    db(path, || {
        tx.execute(
            "UPDATE message_build SET active_value_pk=NULL,active_block_pk=?2 WHERE pk=?1",
            params![build.pk, block],
        )
    })?;
    Ok(block)
}
fn install_in(tx: &Transaction<'_>, path: &Path, build: &Build) -> Result<i64> {
    require_no_pending_origin(tx, path, build)?;
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
        _ => return Err(Error::Invariant("malformed block group entry".into())),
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
                call: super::retirement::tool_call_of(tx, path, block)?,
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

/// Deletes a block once nothing references it, together with its tool_call
/// row, field root and text. If the tool_call row is still referenced
/// elsewhere, the block stays; that owner calls this again after releasing it.
pub(super) fn release_unreferenced_block(tx: &Transaction<'_>, path: &Path, pk: i64) -> Result<()> {
    let referenced: bool = db(path, || {
        tx.query_row("SELECT EXISTS(SELECT 1 FROM message_build WHERE active_block_pk=?1) OR EXISTS(SELECT 1 FROM message_body_edge WHERE block_pk=?1) OR EXISTS(SELECT 1 FROM tool_call WHERE result_block_pk=?1)",[pk],|r|r.get(0))
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
    if !matches!(
        kind.as_str(),
        "text" | "thinking" | "tool_call" | "tool_result"
    ) {
        return Err(Error::Invariant(format!(
            "cannot release a block of kind {kind}"
        )));
    }
    if let Some(call) = super::retirement::tool_call_of(tx, path, pk)? {
        if kind != "tool_call" {
            return Err(Error::Invariant(
                "non-call block has a tool_call row".into(),
            ));
        }
        if !super::retirement::release_tool_call(tx, path, call, pk, session)? {
            return Ok(());
        }
    } else if kind == "tool_call" {
        return Err(Error::Invariant(
            "tool call block has no tool_call row".into(),
        ));
    }
    db(path, || tx.execute("DELETE FROM block WHERE pk=?1", [pk]))?;
    if let Some(root) = fields {
        d::release(tx, path, d::Owner { message, session }, root)?;
    }
    if let Some(text) = text {
        let in_use: bool = db(path, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM block WHERE text_id=?1) OR EXISTS(SELECT 1 FROM tool_item_version WHERE result_text_id=?1)",[text],|r|r.get(0))
        })?;
        if !in_use {
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
    /// Deletes one unreferenced directory node of this Message, and the blocks
    /// only it held (at most 16). Returns false when none is left; call until
    /// then.
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

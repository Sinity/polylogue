//! One selected call and its recorded counterpart, inside the Reader snapshot.
//! The indexed foreign-key relation decides pairing; neither arrival order nor
//! a matching name elsewhere in the archive supplies a result.
use crate::row::{BlockPk, Rows as _, row};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    Block, ErrorReason, MessageId, SessionId, ToolOutcome,
    bundle::ShownBlocks,
    read::{Source, SourceBlock, ToolPair, ValueState},
};
use rusqlite::params;

/// The line beneath a failed call: the first output line that names the
/// failure (error, failed, failure, fatal, panicked, exception or denied, in
/// any case), else the first non-blank line.
fn failure_line(output: &str) -> Option<String> {
    const NAMES: [&str; 7] = [
        "error",
        "failed",
        "failure",
        "fatal",
        "panicked",
        "exception",
        "denied",
    ];
    let mut lines = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    let names = |line: &str| {
        let line = line.to_lowercase();
        NAMES.iter().any(|name| line.contains(name))
    };
    Some(
        std::iter::once(first)
            .chain(lines)
            .find(|line| names(line))
            .unwrap_or(first)
            .to_owned(),
    )
}
/// The columns of one shown block of a pair, before `matched`.
macro_rules! block_columns {
    ($block:literal, $message:literal, $session:literal, $as:literal) => {
        concat!(
            $block,
            ".pk AS ",
            $as,
            "pk, ",
            $block,
            ".position AS ",
            $as,
            "position, ",
            "coalesce(",
            $block,
            ".source_position, cast(",
            $block,
            ".position AS TEXT)) AS ",
            $as,
            "ordinal, ",
            $message,
            ".id AS ",
            $as,
            "message, ",
            $session,
            ".id AS ",
            $as,
            "session, ",
            $block,
            ".call_id AS ",
            $as,
            "call_id, ",
            $message,
            ".version_hash AS ",
            $as,
            "version, ",
            $message,
            ".layout AS ",
            $as,
            "layout, ",
            "(SELECT count(*) FROM block WHERE message_pk=",
            $message,
            ".pk) AS ",
            $as,
            "blocks"
        )
    };
}

row! {
    /// A shown block at an original position.
    struct BlockAt {
        pk: BlockPk,
        position: usize,
        ordinal: String,
        message: MessageId,
        session: SessionId,
        call_id: Option<String>,
        version: Option<Vec<u8>>,
        layout: Option<Vec<u8>>,
        blocks: usize,
    }
    statements {
        const OF_MESSAGE = concat!(
            "SELECT ",
            block_columns!("b", "m", "s", ""),
            " FROM message m JOIN session s ON s.pk=m.session_pk
            JOIN block b ON b.message_pk=m.pk AND b.session_pk=m.session_pk
            WHERE m.id=?1 AND m.duplicate_of IS NULL AND coalesce(b.source_position,cast(b.position AS TEXT))=?2"
        );
    }
}

row! {
    /// A recorded call, the block its result names, and whether the two
    /// belong together.
    struct CallAndResult {
        pk: BlockPk,
        position: usize,
        ordinal: String,
        message: MessageId,
        session: SessionId,
        call_id: Option<String>,
        version: Option<Vec<u8>>,
        layout: Option<Vec<u8>>,
        blocks: usize,
        result_pk: Option<BlockPk>,
        result_position: Option<usize>,
        result_ordinal: Option<String>,
        result_message: Option<MessageId>,
        result_session: Option<SessionId>,
        result_call_id: Option<String>,
        result_version: Option<Vec<u8>>,
        result_layout: Option<Vec<u8>>,
        result_blocks: usize,
        matched: Option<bool>,
    }
    statements {
        const OF_BLOCK = concat!(
            "SELECT ",
            block_columns!("cb", "cm", "cs", ""),
            ", ",
            block_columns!("rb", "rm", "rs", "result_"),
            ", rb.kind='tool_result' AND rb.session_pk=t.session_pk AND rm.session_pk=t.session_pk
              AND rm.duplicate_of IS NULL AND cb.call_id IS NOT NULL AND rb.call_id=cb.call_id AS matched
            FROM tool_call t
            JOIN block cb ON cb.pk=t.call_block_pk AND cb.session_pk=t.session_pk AND cb.kind='tool_call'
            JOIN message cm ON cm.pk=cb.message_pk AND cm.session_pk=t.session_pk
            JOIN session cs ON cs.pk=cm.session_pk
            LEFT JOIN message rm ON rm.id=t.result_message_id
            LEFT JOIN block rb ON rb.message_pk=rm.pk AND rb.position=t.result_position
            LEFT JOIN session rs ON rs.pk=rm.session_pk
            WHERE (t.call_block_pk=?1 OR t.pk IN (SELECT t2.pk FROM block b2 JOIN message m2 ON m2.pk=b2.message_pk
                JOIN tool_call t2 ON t2.result_message_id=m2.id AND t2.result_position=b2.position WHERE b2.pk=?1))
              AND cm.duplicate_of IS NULL AND t.call_id IS cb.call_id"
        );
    }
}

struct BlockFact {
    pk: BlockPk,
    index: usize,
    ordinal: u64,
    message: MessageId,
    session: SessionId,
    call_id: Option<String>,
    version: Option<Vec<u8>>,
    layout: Option<Vec<u8>>,
    blocks: usize,
}
impl BlockFact {
    fn of(block: BlockAt) -> Result<Self> {
        let ordinal = &block.ordinal;
        Ok(Self {
            pk: block.pk,
            index: block.position,
            ordinal: ordinal.parse().map_err(|_| {
                StoreError::stored_data(format!("block ordinal {ordinal:?} is not a number"))
            })?,
            message: block.message,
            session: block.session,
            call_id: block.call_id,
            version: block.version,
            layout: block.layout,
            blocks: block.blocks,
        })
    }
    fn block_selection(&self) -> Result<Option<ShownBlocks>> {
        self.version
            .as_ref()
            .map(|version| {
                crate::encoding::message_block_selection(
                    version,
                    self.layout.as_deref(),
                    self.blocks,
                )
            })
            .transpose()
    }
    fn visible(&self) -> Result<bool> {
        Ok(self.block_selection()?.is_some_and(|block_selection| {
            block_selection.block_ordinals.get(self.index) == Some(&self.ordinal)
        }))
    }
    fn source(&self) -> Source {
        Source::block(&self.message, &self.session, self.ordinal, false)
    }
}
struct PairFact {
    call: BlockFact,
    result: Option<BlockFact>,
}
impl Reader<'_> {
    /// One selected original call or result position. An unavailable
    /// or orphan result stays absent; only current shown blocks are read.
    /// The caller releases this pair before preparing another call's payload.
    pub fn message_tool_pair(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<ToolPair>> {
        self.message_tool_pair_detail(message, original_block_ordinal, true)
    }

    /// A collapsed view restores the recorded block header only. Remote
    /// output text, structured data and attachment bodies remain unread.
    pub fn message_tool_pair_header(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<ToolPair>> {
        self.message_tool_pair_detail(message, original_block_ordinal, false)
    }

    /// The result's emitted shape, without restoring output or structured
    /// payloads. Data presence checks SQL nullness; attachments come only from
    /// the packed stored header before a collapsed view clears them.
    pub fn message_tool_pair_result_shape(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<(bool, u64)>> {
        self.check()?;
        let Some(fact) = self.tool_pair_fact(message, original_block_ordinal)? else {
            return Ok(None);
        };
        let Some(result) = fact.result else {
            return Ok(None);
        };
        self.check()?;
        let conn = self.conn()?;
        let piece = block_piece(conn, result.pk)?;
        let data = piece.data_length.is_some();
        let Block::ToolResult {
            call_id,
            attachments,
            ..
        } = crate::ordered_blocks::body(
            conn,
            &mut crate::content::Reader::default(),
            piece,
            false,
        )?
        else {
            return Err(crate::StoreError::stored(
                0,
                "indexed result is not a tool result",
            ));
        };
        if call_id != fact.call.call_id {
            return Err(crate::StoreError::stored(
                0,
                "indexed result identity differs from its stored block",
            ));
        }
        let count = u64::try_from(attachments.len()).map_err(|_| {
            crate::StoreError::stored(0, "attachment count exceeds its address space")
        })?;
        self.check()?;
        Ok(Some((data, count)))
    }

    /// The indexed sources only, for the caller's existing selection-membership
    /// check before any off-selected counterpart payload is read.
    pub fn message_tool_pair_sources(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<(Source, Option<Source>)>> {
        self.check()?;
        let pair = self
            .tool_pair_fact(message, original_block_ordinal)?
            .map(|fact| {
                (
                    fact.call.source(),
                    fact.result.map(|result| result.source()),
                )
            });
        self.check()?;
        Ok(pair)
    }

    fn message_tool_pair_detail(
        &self,
        message: &MessageId,
        ordinal: u64,
        full: bool,
    ) -> Result<Option<ToolPair>> {
        self.check()?;
        let Some(fact) = self.tool_pair_fact(message, ordinal)? else {
            return Ok(None);
        };
        let call = self.tool_pair_block(&fact.call, full)?;
        let Block::ToolCall { call_id, .. } = &call.block else {
            return Err(crate::StoreError::stored(
                0,
                "indexed call is not a tool call",
            ));
        };
        if call_id != &fact.call.call_id {
            return Err(crate::StoreError::stored(
                0,
                "indexed call identity differs from its stored block",
            ));
        }
        let result = fact
            .result
            .as_ref()
            .map(|result| self.tool_pair_block(result, full))
            .transpose()?;
        if let Some(result) = &result {
            let Block::ToolResult { call_id, .. } = &result.block else {
                return Err(crate::StoreError::stored(
                    0,
                    "indexed result is not a tool result",
                ));
            };
            if call_id != &fact.call.call_id {
                return Err(crate::StoreError::stored(
                    0,
                    "indexed result identity differs from its stored block",
                ));
            }
        }
        let duration_ms = match result.as_ref().map(|result| &result.block) {
            Some(Block::ToolResult {
                reported_duration_ms: Some(duration),
                ..
            }) => ValueState::known(*duration),
            _ => ValueState::Unknown {
                reason: Some("no reported duration".into()),
            },
        };
        // The conflict between the error flag and exit 0 is its own line; any
        // other failure shows the output line that names it. A collapsed read
        // leaves output unread, except to name a failure.
        let error_line = match (&fact.result, result.as_ref().map(|result| &result.block)) {
            (
                Some(result_fact),
                Some(Block::ToolResult {
                    outcome: ToolOutcome::Error { reason },
                    output,
                    ..
                }),
            ) if *reason != Some(ErrorReason::ExitCode(0)) => {
                let output = if full {
                    output.clone()
                } else {
                    match crate::ordered_blocks::one(self.conn()?, result_fact.pk.0)? {
                        Block::ToolResult { output, .. } => output,
                        _ => None,
                    }
                };
                output.as_deref().and_then(failure_line)
            }
            _ => None,
        };
        let pair = ToolPair {
            emit_at: call.source.reference.clone(),
            call,
            result,
            running: false,
            duration_ms,
            error_line,
            child: None,
            edit: None,
        };
        pair.validate()
            .map_err(|error| crate::StoreError::stored(0, error.to_string()))?;
        self.check()?;
        Ok(Some(pair))
    }

    fn tool_pair_fact(&self, message: &MessageId, ordinal: u64) -> Result<Option<PairFact>> {
        let resolved = crate::chain_names::resolved_message(self.conn()?, message)?;
        let message = &resolved;

        let conn = self.conn()?;
        let mut blocks = conn
            .all(
                &BlockAt::OF_MESSAGE,
                params![message.to_string(), ordinal.to_string()],
            )?
            .into_iter();
        let Some(input) = blocks.next() else {
            return Ok(None);
        };
        let input = BlockFact::of(input)?;
        if blocks.next().is_some() {
            return Err(crate::StoreError::stored(
                0,
                "one original block ordinal has conflicting current rows",
            ));
        }
        if !input.visible()? {
            return Ok(None);
        }
        self.check()?;
        let mut pairs = conn.all(&CallAndResult::OF_BLOCK, [input.pk])?.into_iter();
        let Some(pair) = pairs.next() else {
            return Ok(None);
        };
        let matched_row = pair.matched.unwrap_or(false);
        let result = match (
            pair.result_pk,
            pair.result_position,
            pair.result_ordinal,
            pair.result_message,
            pair.result_session,
        ) {
            (Some(pk), Some(idx), Some(ordinal), Some(message), Some(session)) => {
                Some(BlockFact::of(BlockAt {
                    pk,
                    position: idx,
                    ordinal,
                    message,
                    session,
                    call_id: pair.result_call_id,
                    version: pair.result_version,
                    layout: pair.result_layout,
                    blocks: pair.result_blocks,
                })?)
            }
            _ => None,
        };
        let call = BlockFact::of(BlockAt {
            pk: pair.pk,
            position: pair.position,
            ordinal: pair.ordinal,
            message: pair.message,
            session: pair.session,
            call_id: pair.call_id,
            version: pair.version,
            layout: pair.layout,
            blocks: pair.blocks,
        })?;
        let matched = matched_row
            && result
                .as_ref()
                .map(BlockFact::visible)
                .transpose()?
                .unwrap_or(false);
        if pairs.next().is_some() {
            return Err(crate::StoreError::stored(
                0,
                "one source block has conflicting current tool relations",
            ));
        }
        if !call.visible()? || (input.pk != call.pk && !matched) {
            return Ok(None);
        }
        Ok(Some(PairFact {
            call,
            result: result.filter(|_| matched),
        }))
    }
    fn tool_pair_block(&self, fact: &BlockFact, full: bool) -> Result<SourceBlock> {
        self.check()?;
        let conn = self.conn()?;
        let mut block = if full {
            crate::ordered_blocks::one_with(conn, fact.pk.0, true)?
        } else {
            let piece = block_piece(conn, fact.pk)?;
            crate::ordered_blocks::body(conn, &mut crate::content::Reader::default(), piece, false)?
        };
        if !full {
            match &mut block {
                Block::ToolCall { input, .. } => *input = serde_json::Value::Null,
                Block::ToolResult {
                    output,
                    data,
                    attachments,
                    ..
                } => {
                    *output = None;
                    *data = None;
                    attachments.clear();
                }
                _ => {}
            }
        }
        self.check()?;
        Ok(SourceBlock {
            source: fact.source(),
            block,
        })
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<BlockAt>();
    check.row::<CallAndResult>();
}

#[cfg(test)]
#[path = "tool_pairs/tests.rs"]
mod tests;

/// Where a block's piece is stored.
fn block_piece(conn: &rusqlite::Connection, block: BlockPk) -> Result<crate::content::Piece> {
    Ok(conn.query_row(
        "SELECT content_group,content_start,content_length,data_length FROM block WHERE pk=?1",
        [block],
        |row| crate::content::Piece::of(row, 0),
    )?)
}

//! Stored blocks decoded with their text: one block, a message's blocks, or
//! a session's text, thinking and tool-result blocks in session order. A
//! block's body and structured value are a piece of its message's stored
//! content ([`crate::content`]); its text is in `text`.
use crate::Result;
use crate::content::{Piece, Reader};
use crate::encoding::restore_text;
use polylogue_model::Block;
use rusqlite::Connection;
use std::ops::ControlFlow;

/// The columns [`decode`] reads, from `block b` joined with [`TEXT_JOIN`].
pub(crate) const COLUMNS: &str =
    "b.content_group,b.content_start,b.content_length,b.data_length,g.body,t.start,t.length";
/// A block's text, joined to `block b`.
pub(crate) const TEXT_JOIN: &str =
    "LEFT JOIN text t ON t.id=b.text_id LEFT JOIN content_group g ON g.id=t.group_id";

/// Visits the text, thinking and tool-result blocks of `session`'s messages
/// in session order, duplicate messages left out, until `visit` breaks.
pub(crate) fn visit_text(
    connection: &Connection,
    session: i64,
    mut visit: impl FnMut(Block) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(&format!(
        "SELECT {COLUMNS} FROM message m JOIN block b ON b.message_pk=m.pk {TEXT_JOIN}
         WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND b.kind IN ('text','thinking','tool_result')
         ORDER BY m.session_order,m.id,b.position"
    ))?;
    let mut rows = statement.query([session])?;
    let mut reader = Reader::default();
    while let Some(row) = rows.next()? {
        if visit(decode(connection, &mut reader, row, 0, false)?)?.is_break() {
            break;
        }
    }
    Ok(())
}

/// One block, decoded with its text and without its structured value.
pub(crate) fn one(connection: &Connection, block: i64) -> Result<Block> {
    one_with(connection, block, false)
}

/// One block, decoded with its text; with its structured value when `data`.
pub(crate) fn one_with(connection: &Connection, block: i64, data: bool) -> Result<Block> {
    let mut statement = connection.prepare_cached(&format!(
        "SELECT {COLUMNS} FROM block b {TEXT_JOIN} WHERE b.pk = ?1"
    ))?;
    let mut rows = statement.query([block])?;
    let row = rows.next()?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
    decode(connection, &mut Reader::default(), row, 0, data)
}

/// A message's blocks in order; without `structured`, tool results lack
/// their structured data, which is then neither read nor decoded.
pub(crate) fn of_message(
    connection: &Connection,
    message_pk: i64,
    structured: bool,
) -> Result<Vec<Block>> {
    let mut statement = connection.prepare_cached(&format!(
        "SELECT {COLUMNS} FROM block b {TEXT_JOIN} WHERE b.message_pk=?1 ORDER BY b.position"
    ))?;
    let mut rows = statement.query([message_pk])?;
    let mut reader = Reader::default();
    let mut blocks = Vec::new();
    while let Some(row) = rows.next()? {
        blocks.push(decode(connection, &mut reader, row, 0, structured)?);
    }
    Ok(blocks)
}

/// A block from the [`COLUMNS`] of `row` starting at `at`: its piece, then
/// its text's group body, start and length; with its structured value when
/// `data`.
pub(crate) fn decode(
    connection: &Connection,
    reader: &mut Reader,
    row: &rusqlite::Row<'_>,
    at: usize,
    data: bool,
) -> Result<Block> {
    let piece = Piece::of(row, at)?;
    let mut block = body(connection, reader, piece, data)?;
    if let Some(text) = crate::text_rows::column(row, at + 4)? {
        restore_text(&mut block, text);
    }
    Ok(block)
}

/// A block from its stored piece, its text still moved out; with its
/// structured value when `data`.
pub(crate) fn body(
    connection: &Connection,
    reader: &mut Reader,
    piece: Piece,
    data: bool,
) -> Result<Block> {
    let mut block: Block = reader.json(connection, piece.body())?;
    if data && let Some(value) = piece.data() {
        crate::encoding::set_data(&mut block, reader.json(connection, value)?);
    }
    Ok(block)
}

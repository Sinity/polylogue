//! Stored blocks decoded with their text: one block, or a session's text,
//! thinking and tool-result blocks in session order.
use crate::Result;
use crate::encoding::{restore_text, unpack_block};
use polylogue_model::Block;
use rusqlite::Connection;
use std::ops::ControlFlow;

/// Visits the text, thinking and tool-result blocks of `session`'s messages
/// in session order, duplicate messages left out, until `visit` breaks.
pub(crate) fn visit_text(
    connection: &Connection,
    session: i64,
    mut visit: impl FnMut(Block) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT b.body,g.body,t.start,t.length FROM message m JOIN block b ON b.message_pk=m.pk LEFT JOIN text t ON t.id=b.text_id LEFT JOIN text_group g ON g.id=t.group_id WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND b.kind IN ('text','thinking','tool_result') ORDER BY m.session_order,m.id,b.idx",
    )?;
    let mut rows = statement.query([session])?;
    while let Some(row) = rows.next()? {
        if visit(decode(row)?)?.is_break() {
            break;
        }
    }
    Ok(())
}

/// One block, decoded with its text.
pub(crate) fn one(connection: &Connection, block: i64) -> Result<Block> {
    let mut statement = connection.prepare_cached(
        "SELECT b.body, g.body, t.start, t.length FROM block b
         LEFT JOIN text t ON t.id = b.text_id LEFT JOIN text_group g ON g.id = t.group_id
         WHERE b.pk = ?1",
    )?;
    let mut rows = statement.query([block])?;
    let row = rows.next()?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
    decode(row)
}

/// A block from a row of its packed body, then its text's group body, start
/// and length.
fn decode(row: &rusqlite::Row<'_>) -> Result<Block> {
    let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
    if let Some(text) = crate::text_rows::column(row, 1)? {
        restore_text(&mut block, text);
    }
    Ok(block)
}

//! One stored block decoded with its source text.
use crate::Result;
use crate::encoding::{restore_text, unpack_block};
use polylogue_model::Block;
use rusqlite::Connection;

use std::ops::ControlFlow;

pub(crate) fn visit_text(
    connection: &Connection,
    session: i64,
    mut visit: impl FnMut(Block) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT b_payload.body,g.body,t.start,t.length FROM message m JOIN (block b LEFT JOIN message_version_block b_payload ON b_payload.pk=b.payload_pk) ON b.message_pk=m.pk LEFT JOIN text t ON t.id=b.text_id LEFT JOIN text_group g ON g.id=t.group_id WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND b.kind IN ('text','thinking','tool_result') ORDER BY m.session_order,m.id,b.idx",
    )?;
    let mut rows = statement.query([session])?;
    while let Some(row) = rows.next()? {
        let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
        if let Some(text) = crate::text_rows::column(row, 1)? {
            restore_text(&mut block, text);
        }
        if visit(block)?.is_break() {
            break;
        }
    }
    Ok(())
}

/// One block, decoded with its text.
pub(crate) fn one(connection: &Connection, block: i64) -> Result<Block> {
    let mut statement = connection.prepare_cached(
        "SELECT b_payload.body, g.body, t.start, t.length FROM block b LEFT JOIN message_version_block b_payload ON b_payload.pk=b.payload_pk
         LEFT JOIN text t ON t.id = b.text_id LEFT JOIN text_group g ON g.id = t.group_id
         WHERE b.pk = ?1",
    )?;
    let mut rows = statement.query([block])?;
    let row = rows.next()?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
    let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
    if let Some(text) = crate::text_rows::column(row, 1)? {
        restore_text(&mut block, text);
    }
    Ok(block)
}

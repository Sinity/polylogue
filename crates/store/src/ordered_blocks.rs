//! One stored block decoded with its source text.
use crate::Result;
use crate::encoding::{restore_text, unpack_block, unpack_text};
use polylogue_model::Block;
use rusqlite::Connection;

use std::ops::ControlFlow;

pub(crate) fn visit_text(
    connection: &Connection,
    session: i64,
    mut visit: impl FnMut(Block) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT bv.body,t.body FROM message m JOIN block b ON b.message_pk=m.pk JOIN block_value bv ON bv.pk=b.value_pk LEFT JOIN text t ON t.id=b.text_id WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND b.kind IN ('text','thinking','tool_result') ORDER BY m.session_order,m.id,b.idx",
    )?;
    let mut rows = statement.query([session])?;
    while let Some(row) = rows.next()? {
        let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
        if let Some(text) = row.get::<_, Option<Vec<u8>>>(1)? {
            restore_text(&mut block, unpack_text(&text)?);
        }
        if visit(block)?.is_break() {
            break;
        }
    }
    Ok(())
}

/// One block, decoded with its text.
pub(crate) fn one(connection: &Connection, block: i64) -> Result<Block> {
    let (body, text): (Vec<u8>, Option<Vec<u8>>) = connection
        .prepare_cached("SELECT bv.body, t.body FROM block b JOIN block_value bv ON bv.pk=b.value_pk LEFT JOIN text t ON t.id = b.text_id WHERE b.pk = ?1")?
        .query_row([block], |r| Ok((r.get(0)?, r.get(1)?)))?;
    let mut block = unpack_block(&body)?;
    if let Some(text) = text {
        restore_text(&mut block, unpack_text(&text)?);
    }
    Ok(block)
}

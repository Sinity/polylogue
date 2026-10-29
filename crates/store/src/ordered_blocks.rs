//! One stored block decoded with its source text.
use crate::Result;
use crate::encoding::{restore_text, unpack_block, unpack_text};
use polylogue_model::Block;
use rusqlite::Connection;

/// One block, decoded with its text.
pub(crate) fn one(connection: &Connection, block: i64) -> Result<Block> {
    let (body, text): (Vec<u8>, Option<Vec<u8>>) = connection
        .prepare_cached("SELECT b.body, t.body FROM block b LEFT JOIN text t ON t.id = b.text_id WHERE b.pk = ?1")?
        .query_row([block], |r| Ok((r.get(0)?, r.get(1)?)))?;
    let mut block = unpack_block(&body)?;
    if let Some(text) = text {
        restore_text(&mut block, unpack_text(&text)?);
    }
    Ok(block)
}

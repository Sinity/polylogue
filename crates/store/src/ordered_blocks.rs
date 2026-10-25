//! Source blocks on an existing connection, one decoded value at a time.
use crate::Result;
use crate::encoding::{restore_text, unpack_block, unpack_text};
use polylogue_model::Block;
use rusqlite::Connection;
use std::ops::ControlFlow;

pub(crate) struct StoredBlock {
    pub pk: i64,
    pub written_by: String,
    pub prompt_kind: Option<String>,
    pub block: Block,
}

pub(crate) fn visit(
    connection: &Connection,
    session: i64,
    mut visit: impl FnMut(StoredBlock) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT b.pk, m.written_by, m.prompt_kind, b.body, t.body
         FROM message m JOIN block b ON b.message_pk=m.pk
         LEFT JOIN text t ON t.id=b.text_id WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
         ORDER BY m.order_input_id, m.seq, m.record_position, m.id, b.idx",
    )?;
    let mut rows = statement.query([session])?;
    while let Some(row) = rows.next()? {
        let mut block = unpack_block(&row.get::<_, Vec<u8>>(3)?)?;
        if let Some(text) = row.get::<_, Option<Vec<u8>>>(4)? {
            restore_text(&mut block, unpack_text(&text)?);
        }
        if visit(StoredBlock {
            pk: row.get(0)?,
            written_by: row.get(1)?,
            prompt_kind: row.get(2)?,
            block,
        })?
        .is_break()
        {
            break;
        }
    }
    Ok(())
}

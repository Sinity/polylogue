//! Compressed snapshots leave decoding, folding and posting to the worker.
use super::{BATCH_BYTES, Group, Text, TextBatch};
use crate::Result;
use rusqlite::Connection;
use std::collections::BTreeMap;

pub(super) fn send(
    conn: &Connection,
    ids: &[i64],
    mut send: impl FnMut(TextBatch) -> Result<()>,
) -> Result<()> {
    let mut rows: BTreeMap<_, Option<(i64, i64, i64, i64)>> =
        ids.iter().map(|&id| (id, None)).collect();
    // Changed IDs drive PK lookups even when most texts are historical.
    // A missing row is an explicit removal from both search corpora.
    let mut statement = conn.prepare_cached(
        "SELECT t.id,t.indexed,t.group_id,t.start,t.length
         FROM json_each(?1) k CROSS JOIN text t ON t.id=k.value ORDER BY t.id",
    )?;
    let mut selected = statement.query([serde_json::to_string(ids)?])?;
    while let Some(row) = selected.next()? {
        rows.insert(
            row.get(0)?,
            Some((row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        );
    }
    drop(selected);
    let mut group_body = conn.prepare_cached("SELECT body FROM text_group WHERE id=?1")?;
    let mut batch = TextBatch::default();
    let mut groups = BTreeMap::new();
    let mut bytes = 0usize;
    for (id, stored) in rows {
        crate::cancellation::check_current()?;
        let (bits, group, start, length) = stored.unwrap_or((0, 0, 0, 0));
        let group = if bits == 0 {
            None
        } else if let Some(&index) = groups.get(&group) {
            Some(index)
        } else {
            let body: Vec<u8> = group_body.query_row([group], |row| row.get(0))?;
            // Current frames declare decoded size. An unknown-size frame or
            // a group above the window allowance is sent alone.
            let size = zstd::zstd_safe::get_frame_content_size(&body)
                .ok()
                .flatten()
                .and_then(|size| usize::try_from(size).ok())
                .unwrap_or(usize::MAX);
            let cost = size.saturating_add(body.len());
            if !batch.texts.is_empty() && bytes.saturating_add(cost) > BATCH_BYTES {
                send(std::mem::take(&mut batch))?;
                groups.clear();
                bytes = 0;
            }
            let index = batch.groups.len();
            batch.groups.push(Group {
                body,
                members: Vec::new(),
            });
            groups.insert(group, index);
            bytes = bytes.saturating_add(cost);
            Some(index)
        };
        if let Some(group) = group {
            batch.groups[group].members.push(batch.texts.len());
        }
        batch.texts.push(Text {
            id,
            bits,
            start,
            length,
            body: String::new(),
        });
        if bytes > BATCH_BYTES {
            send(std::mem::take(&mut batch))?;
            groups.clear();
            bytes = 0;
        }
    }
    if !batch.texts.is_empty() {
        send(batch)?;
    }
    Ok(())
}

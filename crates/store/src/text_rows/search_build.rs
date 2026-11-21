//! First-build search is created in the private index before installation.

use super::{groups, insert_entry};
use crate::{Result, StoreError};
use rusqlite::{Connection, Statement, TransactionBehavior};
use std::borrow::Cow;

const BATCH_BYTES: usize = 4 << 20;
const BATCH_TEXTS: usize = 1024;

struct IndexedText {
    id: i64,
    indexed: i64,
    group: i64,
    start: i64,
    length: i64,
    text: String,
}

/// Read each group's body once in a bounded window, then visit the texts in
/// their original rowid order.
pub(super) fn visit_indexed_texts(
    conn: &Connection,
    batch_bytes: usize,
    mut visit: impl FnMut(i64, i64, String) -> Result<()>,
) -> Result<()> {
    let mut select = conn.prepare(
        "SELECT id,indexed,group_id,start,length FROM text WHERE indexed<>0 ORDER BY id",
    )?;
    let mut group = conn.prepare("SELECT body FROM text_group WHERE id=?1")?;
    let mut batch = Vec::with_capacity(BATCH_TEXTS);
    let mut bytes = 0;
    let mut rows = select.query([])?;
    while let Some(row) = rows.next()? {
        crate::cancellation::check_current()?;
        let length = row.get(4)?;
        let size = usize::try_from(length)
            .map_err(|_| StoreError::stored(0, "a stored text has an invalid length"))?;
        if batch.len() == BATCH_TEXTS || size > batch_bytes - bytes {
            visit_batch(&mut group, &mut batch, &mut visit)?;
            bytes = 0;
        }
        batch.push(IndexedText {
            id: row.get(0)?,
            indexed: row.get(1)?,
            group: row.get(2)?,
            start: row.get(3)?,
            length,
            text: String::new(),
        });
        if size > batch_bytes {
            // Process a text above the window allowance alone.
            visit_batch(&mut group, &mut batch, &mut visit)?;
        } else {
            bytes += size;
        }
    }
    visit_batch(&mut group, &mut batch, &mut visit)
}

fn visit_batch(
    group: &mut Statement<'_>,
    batch: &mut Vec<IndexedText>,
    visit: &mut impl FnMut(i64, i64, String) -> Result<()>,
) -> Result<()> {
    batch.sort_unstable_by_key(|text| text.group);
    let mut remaining = batch.as_mut_slice();
    while let Some(first) = remaining.first() {
        crate::cancellation::check_current()?;
        let id = first.group;
        let end = remaining.partition_point(|text| text.group == id);
        let (members, rest) = remaining.split_at_mut(end);
        let mut rows = group.query([id])?;
        let row = rows
            .next()?
            .ok_or_else(|| StoreError::stored(0, "a text row has no group"))?;
        let body = row.get_ref(0)?.as_blob().map_err(rusqlite::Error::from)?;
        let decoded = groups::decode(body)?;
        for member in members {
            member.text = groups::slice_decoded(&decoded, member.start, member.length)?;
        }
        remaining = rest;
    }
    batch.sort_unstable_by_key(|text| text.id);
    for text in batch.drain(..) {
        visit(text.id, text.indexed, text.text)?;
    }
    Ok(())
}

/// Rebuilds both declared FTS tables in one transaction in the private index.
/// Reopening after a stop can build them again from the same committed texts.
pub(crate) fn build_search_indexes(conn: &Connection) -> Result<()> {
    let _part = crate::timing::part(crate::timing::Part::SearchBuild);
    crate::timing::time(crate::timing::Stage::SearchIndex, || {
        let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "INSERT INTO text_fts(text_fts) VALUES('delete-all');
             INSERT INTO output_fts(output_fts) VALUES('delete-all');",
        )?;
        let mut written = false;
        visit_indexed_texts(&tx, BATCH_BYTES, |id, indexed, text| {
            let folded = if text.is_ascii() {
                Cow::Borrowed(text.as_str())
            } else {
                Cow::Owned(polylogue_model::text::fold(&text))
            };
            for bit in [1, 2] {
                if indexed & bit != 0 {
                    insert_entry(&tx, id, bit, &folded)?;
                }
            }
            written = true;
            Ok(())
        })?;
        crate::fault::check("search.before_commit", conn.path().unwrap_or_default())?;
        crate::cancellation::check_current()?;
        if written {
            crate::archive::bump_counter(&tx)?;
        }
        tx.commit()?;
        Ok(())
    })
}

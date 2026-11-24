//! First-build search is created in the private index before installation.

use super::{folded_text, groups, insert_entry};
use crate::{Result, StoreError};
use rusqlite::{Connection, Statement, TransactionBehavior};

struct IndexedText {
    id: i64,
    indexed: i64,
    group: i64,
    start: i64,
    length: i64,
    text: String,
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
    for text in batch.drain(..) {
        visit(text.id, text.indexed, text.text)?;
    }
    Ok(())
}

/// The search entries of a build, written group by group from its texts'
/// marks. Each step writes a bounded batch in its own transaction, and the
/// next step starts after it, so the writer serves other work between
/// steps and nothing is written twice.
pub(crate) struct SearchBuild {
    started: bool,
    /// The last text entered, by (group, start, id).
    after: Option<(i64, i64, i64)>,
    written: bool,
}

/// Texts one step enters: each step's commit writes a new full-text
/// segment that later merges, so a step is large.
const STEP_TEXTS: i64 = 25_000;

impl SearchBuild {
    pub(crate) fn new() -> Self {
        Self {
            started: false,
            after: None,
            written: false,
        }
    }

    /// Writes the next batch; true once every marked text is entered.
    pub(crate) fn step(&mut self, conn: &Connection) -> Result<bool> {
        crate::timing::time(crate::timing::Stage::SearchIndex, || self.next(conn))
    }

    fn next(&mut self, conn: &Connection) -> Result<bool> {
        let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        if !self.started {
            // An index in group order for the walk, dropped when it ends.
            tx.execute_batch(
                "INSERT INTO text_fts(text_fts) VALUES('delete-all');
                 INSERT INTO output_fts(output_fts) VALUES('delete-all');
                 CREATE INDEX IF NOT EXISTS text_search_walk ON text(group_id,start,id) WHERE indexed<>0;",
            )?;
            tx.commit()?;
            self.started = true;
            return Ok(false);
        }
        let (group, start, id) = self.after.unwrap_or((i64::MIN, i64::MIN, i64::MIN));
        let rows: Vec<IndexedText> = tx
            .prepare_cached(
                "SELECT id,indexed,group_id,start,length FROM text
                 WHERE indexed<>0 AND (group_id,start,id)>(?1,?2,?3)
                 ORDER BY group_id,start,id LIMIT ?4",
            )?
            .query_map(rusqlite::params![group, start, id, STEP_TEXTS], |r| {
                Ok(IndexedText {
                    id: r.get(0)?,
                    indexed: r.get(1)?,
                    group: r.get(2)?,
                    start: r.get(3)?,
                    length: r.get(4)?,
                    text: String::new(),
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let Some(last) = rows.last() else {
            tx.execute("DROP INDEX IF EXISTS text_search_walk", [])?;
            crate::fault::check("search.before_commit", conn.path().unwrap_or_default())?;
            if self.written {
                crate::archive::bump_counter(&tx)?;
            }
            tx.commit()?;
            return Ok(true);
        };
        self.after = Some((last.group, last.start, last.id));
        let mut group = tx.prepare_cached("SELECT body FROM text_group WHERE id=?1")?;
        let mut batch = rows;
        visit_batch(&mut group, &mut batch, &mut |id, indexed, text| {
            let (folded, _) = folded_text(text);
            for bit in [1, 2] {
                if indexed & bit != 0 {
                    insert_entry(&tx, id, bit, &folded)?;
                }
            }
            Ok(())
        })?;
        drop(group);
        crate::cancellation::check_current()?;
        self.written = true;
        tx.commit()?;
        Ok(false)
    }
}

/// Rebuilds both declared FTS tables in the private index, step by step.
#[cfg(test)]
pub(crate) fn build_search_indexes(conn: &Connection) -> Result<()> {
    let mut search = SearchBuild::new();
    while !search.step(conn)? {}
    Ok(())
}

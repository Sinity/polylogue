//! Sequential logical reads through the operation's already verified blob copies.
use super::{Connection, PlainBytes, Plan};
use crate::excise::stored_text::{Chain, extent};
use crate::{Result, StoreError};
use rusqlite::{OptionalExtension, params};
use std::ops::Range;
use std::os::unix::fs::FileExt;

struct Segment {
    file_at: u64,
    end: u64,
}

pub(in super::super) struct Reader<'a> {
    plain: &'a mut PlainBytes,
    plan: &'a Plan,
    archive: &'a Connection,
    store: &'a hashed_blobs::Store,
    key: &'a Chain,
    at: u64,
    end: u64,
    segment: Option<Segment>,
}

impl<'a> Reader<'a> {
    pub(super) fn new(
        plain: &'a mut PlainBytes,
        plan: &'a Plan,
        archive: &'a Connection,
        store: &'a hashed_blobs::Store,
        key: &'a Chain,
        range: Range<u64>,
    ) -> Result<Self> {
        if range.start > range.end {
            return Err(StoreError::Invalid("logical byte range is reversed".into()));
        }
        Ok(Self {
            plain,
            plan,
            archive,
            store,
            key,
            at: range.start,
            end: range.end,
            segment: None,
        })
    }

    pub(in super::super) fn position(&self) -> u64 {
        self.at
    }

    pub(in super::super) fn read(&mut self, out: &mut [u8]) -> Result<usize> {
        if self.at == self.end || out.is_empty() {
            return Ok(0);
        }
        if self.segment.as_ref().is_none_or(|s| self.at == s.end) {
            let at = i64::try_from(self.at).map_err(|_| {
                StoreError::Invalid("logical byte position exceeds SQLite range".into())
            })?;
            let found = self
                .archive
                .prepare_cached(
                    "SELECT input,chain,offset,length,blob,blob_offset FROM extent
                 WHERE input=?1 AND chain=?2 AND offset<=?3 ORDER BY offset DESC LIMIT 1",
                )?
                .query_row(params![self.key.0, self.key.1, at], extent)
                .optional()?;
            let Some(mut e) = found else {
                return Err(StoreError::Invalid("revision bytes have a gap".into()));
            };
            let end = e
                .offset
                .checked_add(e.length)
                .ok_or_else(|| StoreError::Invalid("extent length overflow".into()))?;
            if e.offset > self.at || end <= self.at {
                return Err(StoreError::Invalid(
                    "revision bytes have a gap or overlap".into(),
                ));
            }
            let next = self.archive.prepare_cached("SELECT offset FROM extent WHERE input=?1 AND chain=?2 AND offset>?3 ORDER BY offset LIMIT 1")?
                .query_row(params![self.key.0,self.key.1,e.offset as i64],|r|r.get::<_,i64>(0)).optional()?;
            if next.is_some_and(|next| (next as u64) < end.min(self.end)) {
                return Err(StoreError::Invalid(
                    "revision bytes have a gap or overlap".into(),
                ));
            }
            let (start, length) = self.plain.load(self.plan, self.archive, self.store, &e)?;
            e.blob_offset = e
                .blob_offset
                .checked_add(self.at - e.offset)
                .ok_or_else(|| StoreError::Invalid("extent blob offset overflow".into()))?;
            let end = end.min(self.end);
            if e.blob_offset
                .checked_add(end - self.at)
                .is_none_or(|n| n > length)
            {
                return Err(StoreError::Invalid("extent exceeds its stored blob".into()));
            }
            self.segment = Some(Segment {
                file_at: start
                    .checked_add(e.blob_offset)
                    .ok_or_else(|| StoreError::Invalid("verified byte offset overflow".into()))?,
                end,
            });
        }
        let segment = self.segment.as_mut().expect("loaded logical segment");
        let take = out.len().min((segment.end - self.at) as usize);
        self.plain
            .file
            .read_exact_at(&mut out[..take], segment.file_at)?;
        segment.file_at += take as u64;
        self.at += take as u64;
        Ok(take)
    }

    pub(in super::super) fn read_exact(&mut self, mut out: &mut [u8]) -> Result<()> {
        while !out.is_empty() {
            let n = self.read(out)?;
            if n == 0 {
                return Err(StoreError::Invalid(
                    "revision ends before requested bytes".into(),
                ));
            }
            out = &mut out[n..];
        }
        Ok(())
    }
}

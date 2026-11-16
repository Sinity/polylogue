//! A tail body selects one chain of previous revisions. The temporary reverse
//! stack holds body references only and disappears when the lookup ends.
use super::*;
use polylogue_store::primary::{ByteCursor, ByteCursorCheck, ByteCursorQuery};
use std::io::{Read, Seek, SeekFrom, Write};

struct ReverseRefs {
    file: std::fs::File,
    end: u64,
}
impl ReverseRefs {
    fn push(&mut self, identity: &str) -> Result<()> {
        let length = u64::try_from(identity.len())
            .map_err(|_| Error::Invalid("body reference length overflow".into()))?;
        self.file.write_all(identity.as_bytes())?;
        self.file.write_all(&length.to_le_bytes())?;
        self.end = self
            .end
            .checked_add(length)
            .and_then(|end| end.checked_add(8))
            .ok_or_else(|| Error::Invalid("body reference stack overflow".into()))?;
        Ok(())
    }
    fn pop(&mut self) -> Result<Option<String>> {
        if self.end == 0 {
            return Ok(None);
        }
        let footer = self
            .end
            .checked_sub(8)
            .ok_or_else(|| Error::Invalid("invalid body reference stack".into()))?;
        self.file.seek(SeekFrom::Start(footer))?;
        let mut length = [0; 8];
        self.file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        let start = footer
            .checked_sub(length)
            .ok_or_else(|| Error::Invalid("invalid body reference length".into()))?;
        let size = usize::try_from(length)
            .map_err(|_| Error::Invalid("body reference length overflow".into()))?;
        let mut identity = vec![0; size];
        self.file.seek(SeekFrom::Start(start))?;
        self.file.read_exact(&mut identity)?;
        self.end = start;
        Ok(Some(
            String::from_utf8(identity).map_err(|error| Error::Invalid(error.to_string()))?,
        ))
    }
}

impl Sinex {
    pub fn byte_cursor(&self, query: &ByteCursorQuery) -> Result<Option<ByteCursor>> {
        query.validate()?;
        let generation = &query.generation;
        let through = self.queue()?.latest_order()?;
        let tail = match &query.tail_body {
            Some(tail) => tail.clone(),
            None => {
                let roots = self.queue()?.body_roots(generation, through)?;
                if roots.len() != 1 {
                    return Ok(None);
                }
                roots.into_iter().next().expect("one root")
            }
        };
        if self.queue()?.body_metadata(&tail)?.is_none() {
            return Ok(None);
        }
        let scratch = self.config.store.join("polylogue-staging");
        std::fs::create_dir_all(&scratch)?;
        let mut reversed = ReverseRefs {
            file: tempfile::tempfile_in(&scratch)?,
            end: 0,
        };
        let mut current = tail.clone();
        let mut newer: Option<(u64, u64)> = None;
        loop {
            let Some(header) = self.queue()?.body_metadata(&current)? else {
                return Ok(None);
            };
            if &header.generation != generation
                || header.order > through
                || newer.is_some_and(|(order, start)| {
                    header.order >= order || header.range.end != start
                })
            {
                return Ok(None);
            }
            reversed.push(&current)?;
            match header.previous {
                None => {
                    if header.range.start != generation.window().start {
                        return Ok(None);
                    }
                    break;
                }
                Some(previous) => {
                    newer = Some((header.order, header.range.start));
                    current = previous;
                }
            }
        }
        let mut check = ByteCursorCheck::new(generation.clone())?;
        let mut count = 0u64;
        let mut last_order = 0;
        while let Some(identity) = reversed.pop()? {
            let pinned = self.queue()?.pin_body_range(&identity)?;
            if pinned.order <= last_order || pinned.order > through {
                return Ok(None);
            }
            last_order = pinned.order;
            check.push(pinned.verify()?)?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("body chain length overflow".into()))?;
        }
        current = tail;
        loop {
            let children = self.queue()?.body_children(&current, generation, through)?;
            if children.len() > 1 {
                return Ok(None);
            }
            let Some(child) = children.into_iter().next() else {
                break;
            };
            let pinned = self.queue()?.pin_body_range(&child)?;
            if pinned.order <= last_order || pinned.order > through {
                return Ok(None);
            }
            last_order = pinned.order;
            check.push(pinned.verify()?)?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("body chain length overflow".into()))?;
            current = child;
        }
        if query.tail_body.is_none() && self.queue()?.body_count(generation, through)? != count {
            return Ok(None);
        }
        Ok(check.finish())
    }
}

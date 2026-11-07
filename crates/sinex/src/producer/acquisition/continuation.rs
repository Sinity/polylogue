//! A body anchor selects an actual predecessor lineage. The temporary reverse
//! stack contains body references only and disappears when the lookup ends.
use super::*;
use polylogue_store::primary::{RawAcquisitionCursor, RawAcquisitionProof, RawAcquisitionQuery};
use std::io::{Read, Seek, SeekFrom, Write};

struct ReverseRefs {
    file: std::fs::File,
    end: u64,
}
impl ReverseRefs {
    fn push(&mut self, identity: &str) -> Result<()> {
        let length = u64::try_from(identity.len())
            .map_err(|_| Error::Invalid("raw reference length overflow".into()))?;
        self.file.write_all(identity.as_bytes())?;
        self.file.write_all(&length.to_le_bytes())?;
        self.end = self
            .end
            .checked_add(length)
            .and_then(|end| end.checked_add(8))
            .ok_or_else(|| Error::Invalid("raw reference stack overflow".into()))?;
        Ok(())
    }
    fn pop(&mut self) -> Result<Option<String>> {
        if self.end == 0 {
            return Ok(None);
        }
        let footer = self
            .end
            .checked_sub(8)
            .ok_or_else(|| Error::Invalid("invalid raw reference stack".into()))?;
        self.file.seek(SeekFrom::Start(footer))?;
        let mut length = [0; 8];
        self.file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        let start = footer
            .checked_sub(length)
            .ok_or_else(|| Error::Invalid("invalid raw reference length".into()))?;
        let size = usize::try_from(length)
            .map_err(|_| Error::Invalid("raw reference length overflow".into()))?;
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
    pub fn raw_acquisition_cursor(
        &self,
        query: &RawAcquisitionQuery,
    ) -> Result<Option<RawAcquisitionCursor>> {
        query.validate()?;
        let generation = &query.generation;
        let through = self.queue()?.raw_cut()?;
        let anchor = match &query.anchor {
            Some(anchor) => anchor.clone(),
            None => {
                let roots = self.queue()?.raw_roots(generation, through)?;
                if roots.len() != 1 {
                    return Ok(None);
                }
                roots.into_iter().next().expect("one root")
            }
        };
        if self.queue()?.raw_metadata(&anchor)?.is_none() {
            return Ok(None);
        }
        let scratch = self.config.store.join("polylogue-staging");
        std::fs::create_dir_all(&scratch)?;
        let mut reversed = ReverseRefs {
            file: tempfile::tempfile_in(&scratch)?,
            end: 0,
        };
        let mut current = anchor.clone();
        let mut newer: Option<(u64, u64)> = None;
        loop {
            let Some(header) = self.queue()?.raw_metadata(&current)? else {
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
            match header.predecessor {
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
        let mut proof = RawAcquisitionProof::new(generation.clone())?;
        let mut count = 0u64;
        let mut last_order = 0;
        while let Some(identity) = reversed.pop()? {
            let pinned = self.queue()?.pin_raw(&identity)?;
            if pinned.order <= last_order || pinned.order > through {
                return Ok(None);
            }
            last_order = pinned.order;
            proof.push(pinned.verify()?)?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("raw chain length overflow".into()))?;
        }
        current = anchor;
        loop {
            let children = self.queue()?.raw_children(&current, generation, through)?;
            if children.len() > 1 {
                return Ok(None);
            }
            let Some(child) = children.into_iter().next() else {
                break;
            };
            let pinned = self.queue()?.pin_raw(&child)?;
            if pinned.order <= last_order || pinned.order > through {
                return Ok(None);
            }
            last_order = pinned.order;
            proof.push(pinned.verify()?)?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("raw chain length overflow".into()))?;
            current = child;
        }
        if query.anchor.is_none() && self.queue()?.raw_count(generation, through)? != count {
            return Ok(None);
        }
        Ok(proof.finish())
    }
}

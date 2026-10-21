//! One verified copy of each physical blob for this excision attempt. Logical
//! extents may revisit or reorder shared slices without decoding frames again.

use super::{Extent, READ_CHUNK};
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::Connection;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::os::unix::fs::{FileExt, PermissionsExt};

type Place = (u64, u64);

pub(super) struct PlainBytes {
    // Unlinked, mode 0600, under the process's managed temporary directory.
    file: File,
    length: u64,
    places: BTreeMap<BlobHash, Place>,
    #[cfg(test)]
    decoded: u64,
}

impl PlainBytes {
    pub(super) fn new() -> Result<Self> {
        let file = tempfile::tempfile()?;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            file,
            length: 0,
            places: BTreeMap::new(),
            #[cfg(test)]
            decoded: 0,
        })
    }

    pub(super) fn visit(
        &mut self,
        archive: &Connection,
        store: &hashed_blobs::Store,
        extent: &Extent,
        mut f: impl FnMut(&[u8], u64) -> Result<()>,
    ) -> Result<()> {
        let (start, length) = self.load(archive, store, extent)?;
        let end = extent
            .blob_offset
            .checked_add(extent.length)
            .ok_or_else(|| StoreError::Invalid("extent length overflow".into()))?;
        if end > length {
            return Err(StoreError::Invalid("extent exceeds its stored blob".into()));
        }
        let mut buffer = [0; READ_CHUNK];
        let mut at = 0;
        while at < extent.length {
            let take = (extent.length - at).min(buffer.len() as u64) as usize;
            self.file
                .read_exact_at(&mut buffer[..take], start + extent.blob_offset + at)?;
            let logical = extent
                .offset
                .checked_add(at)
                .ok_or_else(|| StoreError::Invalid("logical extent offset overflow".into()))?;
            f(&buffer[..take], logical)?;
            at += take as u64;
        }
        Ok(())
    }

    fn load(
        &mut self,
        archive: &Connection,
        store: &hashed_blobs::Store,
        extent: &Extent,
    ) -> Result<Place> {
        if let Some(place) = self.places.get(&extent.blob) {
            return Ok(*place);
        }
        let encoding: String = archive.query_row(
            "SELECT encoding FROM blob WHERE hash=?1",
            [&extent.blob.0[..]],
            |r| r.get(0),
        )?;
        if encoding != "plain" {
            let format = archive.query_row(
                "SELECT format FROM input WHERE id = ?1",
                [&extent.input],
                |r| r.get(0),
            )?;
            return Err(StoreError::ExcisionFormat {
                input: extent.input.clone(),
                format,
                detail: format!("blob encoding {encoding} needs its own text-excision decoder"),
            });
        }
        let start = self.length;
        self.file.seek(SeekFrom::Start(start))?;
        let copied = crate::blobs::copy_plain(archive, store, extent.blob, &mut self.file)?;
        #[cfg(test)]
        {
            self.decoded += copied.decoded_bytes;
        }
        self.length = start
            .checked_add(copied.bytes)
            .ok_or_else(|| StoreError::Invalid("temporary blob offsets overflow".into()))?;
        self.places.insert(extent.blob, (start, copied.bytes));
        Ok((start, copied.bytes))
    }
}

#[cfg(test)]
mod tests;

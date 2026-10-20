//! One verified copy of each physical blob for this excision attempt. Logical
//! extents may revisit or reorder shared slices without decoding frames again.

use super::{Extent, READ_CHUNK};
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, MAIN_DB};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
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
        let (rowid, place, encoding, size): (i64, String, String, i64) = archive.query_row(
            "SELECT rowid, place, encoding, size FROM blob WHERE hash = ?1",
            [&extent.blob.0[..]],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
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
        let size = u64::try_from(size).map_err(|_| StoreError::BlobCorrupt(extent.blob))?;
        let start = self.length;
        self.file.seek(SeekFrom::Start(start))?;
        let copied = match place.as_str() {
            "inline" => {
                // Incremental BLOB I/O also handles oversized historical inline
                // rows without allocating their compressed or decoded body whole.
                let blob = archive.blob_open(MAIN_DB, "blob", "data", rowid, true)?;
                let reader = zstd::stream::read::Decoder::new(blob)?;
                let n = self.copy(reader, extent.blob, size)?;
                #[cfg(test)]
                {
                    self.decoded += n;
                }
                n
            }
            "file" => {
                let mut reader =
                    store.read_plain(&hashed_blobs::Hash(extent.blob.0), Some(size), || Ok(()))?;
                let n = self.copy(&mut reader, extent.blob, size)?;
                #[cfg(test)]
                {
                    self.decoded += reader.stats().decoded_bytes;
                }
                n
            }
            _ => return Err(StoreError::Invalid(format!("unknown blob place {place}"))),
        };
        self.length = start
            .checked_add(copied)
            .ok_or_else(|| StoreError::Invalid("temporary blob offsets overflow".into()))?;
        self.places.insert(extent.blob, (start, copied));
        Ok((start, copied))
    }

    fn copy(&mut self, mut reader: impl Read, hash: BlobHash, expected: u64) -> Result<u64> {
        let mut buffer = [0; READ_CHUNK];
        let mut digest = blake3::Hasher::new();
        let mut length = 0u64;
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            length = length
                .checked_add(n as u64)
                .ok_or(StoreError::BlobCorrupt(hash))?;
            if length > expected {
                return Err(StoreError::BlobCorrupt(hash));
            }
            digest.update(&buffer[..n]);
            self.file.write_all(&buffer[..n])?;
        }
        if length != expected || digest.finalize().as_bytes() != &hash.0 {
            return Err(StoreError::BlobCorrupt(hash));
        }
        Ok(length)
    }
}

#[cfg(test)]
mod tests;

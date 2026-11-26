//! A directory of blobs, one file per hash: `bytes/<hex>` under the archive
//! root holds exactly one blob's encoding. A blob row names its file as a
//! container extent of the whole file, so the catalog reader opens it as it
//! opens a segment range. Each put links the prepared encoding into place,
//! syncs it and the directory once, and records its rows.
use super::{Home, Stored};
use crate::{Archive, Result};
use rusqlite::Connection;
use std::fs::File;
use std::path::PathBuf;

/// The directory under the archive root that holds the blob files.
const DIR: &str = "bytes";

/// Blob bytes kept as one file per hash, written by their own thread.
pub struct Directory {
    thread: super::PutThread,
}

impl Directory {
    pub fn start(archive: &Archive) -> Result<Self> {
        let dir = archive.root.join(DIR);
        std::fs::create_dir_all(&dir)?;
        File::open(&archive.root)?.sync_all()?;
        let home = DirectoryHome { dir };
        Ok(Self {
            thread: super::PutThread::start(archive, "polylogue-bytes", home)?,
        })
    }
}

impl super::Bytes for Directory {
    fn put_all(&self, blobs: Vec<super::PreparedBlob>) -> Result<super::Put> {
        self.thread.put_all(blobs)
    }

    fn open(
        &self,
        hash: &polylogue_model::BlobHash,
        range: std::ops::Range<u64>,
    ) -> Result<Box<dyn std::io::Read + Send>> {
        self.thread.open(hash, range)
    }

    fn close(&self) {
        self.thread.stop();
    }
}

struct DirectoryHome {
    dir: PathBuf,
}

impl Home for DirectoryHome {
    fn store(&mut self, conn: &Connection, blobs: Vec<super::PreparedBlob>) -> Result<Stored> {
        let mut extents = Vec::with_capacity(blobs.len());
        let mut bytes_written = 0;
        let mut written = std::collections::BTreeSet::new();
        for blob in blobs {
            if let Some(extent) =
                crate::blobs::load_extent(conn, "", &hashed_blobs::Hash(blob.hash().0))?
            {
                extents.push(extent);
                continue;
            }
            let blob = blob.into_prepared()?;
            let descriptor = blob.descriptor();
            let name = descriptor.hash.hex();
            let extent = hashed_blobs::Extent {
                segment: format!("{DIR}/{name}"),
                offset: 0,
                len: descriptor.stored_len,
                decoded_len: descriptor.decoded_len,
                hash: descriptor.hash,
                kind: descriptor.kind.clone(),
                index_hash: descriptor.index_hash,
                encoded_hash: descriptor.encoded_hash,
            };
            if written.insert(descriptor.hash) {
                let target = self.dir.join(&name);
                // A file a crash left before its row committed is this blob's
                // or a torn copy of it: it is written again.
                let partial = self.dir.join(format!("{name}.partial"));
                let mut source = blob.into_file();
                std::io::Seek::rewind(&mut source)?;
                let mut file = File::create(&partial)?;
                std::io::copy(&mut source, &mut file)?;
                crate::fault::check("segment.after_append", "")?;
                file.sync_all()?;
                std::fs::rename(&partial, &target)?;
                bytes_written += descriptor.stored_len;
            }
            extents.push(extent);
        }
        if !written.is_empty() {
            File::open(&self.dir)?.sync_all()?;
        }
        crate::fault::check("segment.after_sync", "")?;
        Ok(Stored {
            extents,
            end: None,
            syncs: u64::from(!written.is_empty()),
            bytes_written,
            acknowledge: None,
        })
    }

    fn recorded(&mut self, _: Stored) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) {}
}

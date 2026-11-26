//! The segment log: blobs appended to one segment file of the shared
//! container, one sync for every put waiting together.
//!
//! The record names how far the segment is committed (`segment.committed_end`),
//! and what lies past that end when a segment opens is cut off: the tail of a
//! put that a crash stopped before its rows committed, which no row names and
//! no caller was told is stored. The container cannot tell whole frames past
//! the end from a torn one without reading its own frame format, so the open
//! does not try: it compares the end the record names with the largest end
//! any commit recorded, which the thread writes beside the segment after each
//! commit, unsynced. A segment past the record's end by more than one put in
//! flight therefore means `archive.db` is older than its segment (put back from
//! a copy), and the open stops, naming both lengths, instead of cutting off
//! bytes a newer record named. That file lagging behind after a power loss
//! only ever lets the open cut what it would have cut anyway.
use super::{Home, Stored};
use crate::{Archive, ConnExt as _, Result, StoreError};
use rusqlite::Connection;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};

/// The segment new bytes are appended to.
pub(crate) const CURRENT: &str = "segments/current.hb";
/// The largest committed end any commit recorded for [`CURRENT`].
const RECORDED_END: &str = "segments/current.hb.committed";

/// The segment log of an archive, written by its own thread.
pub struct Segments {
    thread: super::PutThread,
}

impl Segments {
    /// Opens the archive's current segment, cutting an uncommitted tail, and
    /// starts the thread that appends to it. A segment holding bytes past the
    /// end its record names that a newer record committed stops the open.
    pub fn start(archive: &Archive) -> Result<Self> {
        let conn = crate::archive::open_read(&archive.archive_db())?;
        let writer = open(&conn, &archive.root)?;
        drop(conn);
        let home = SegmentHome {
            root: archive.root.clone(),
            writer: Some(writer),
        };
        Ok(Self {
            thread: super::PutThread::start(archive, "polylogue-segments", home)?,
        })
    }
}

impl super::Bytes for Segments {
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

struct SegmentHome {
    root: PathBuf,
    /// `None` after a failure, until the next put opens it again.
    writer: Option<hashed_blobs::SegmentWriter>,
}

impl Home for SegmentHome {
    fn store(&mut self, conn: &Connection, blobs: Vec<super::PreparedBlob>) -> Result<Stored> {
        if self.writer.is_none() {
            self.writer = Some(open(conn, &self.root)?);
        }
        let writer = self.writer.as_mut().expect("opened segment writer");
        let prepared = blobs.into_iter().map(super::PreparedBlob::into_prepared);
        let pending = writer.append_group(prepared, |hash| {
            crate::blobs::load_extent(conn, "", hash)
                .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))
        })?;
        crate::fault::check("segment.after_append", "")?;
        let synced = pending.wait()?;
        crate::fault::check("segment.after_sync", "")?;
        Ok(Stored {
            extents: synced.extents.clone(),
            end: Some(synced.end),
            syncs: synced.metrics.syncs,
            bytes_written: synced.metrics.appended_bytes,
            acknowledge: Some(synced),
        })
    }

    fn recorded(&mut self, stored: Stored) -> Result<()> {
        let synced = stored
            .acknowledge
            .ok_or_else(|| StoreError::internal("a segment group without its sync"))?;
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| StoreError::internal("a recorded group without its segment"))?;
        writer.acknowledge(&synced)?;
        record_end(&self.root, synced.end)
    }

    fn reset(&mut self) {
        self.writer = None;
    }
}

/// Opens the current segment at the end the record names, as the module
/// documentation describes.
fn open(conn: &Connection, root: &Path) -> Result<hashed_blobs::SegmentWriter> {
    let end: i64 = conn
        .scalar_optional("SELECT committed_end FROM segment WHERE name=?1", [CURRENT])?
        .unwrap_or(0);
    let end =
        u64::try_from(end).map_err(|_| StoreError::stored(0, "negative committed segment end"))?;
    let length = match std::fs::metadata(root.join(CURRENT)) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    if length > end
        && let Some(recorded) = recorded_end(root)?
        && recorded > end
    {
        return Err(StoreError::Blobs(hashed_blobs::Error::Corrupt {
            segment: CURRENT.into(),
            offset: end,
            len: length - end,
            what: format!(
                "the segment holds {length} bytes and archive.db names {end} of them; \
                 bytes up to {recorded} were committed by a newer archive.db, so this one \
                 is older than its segment and nothing is cut off"
            ),
        }));
    }
    Ok(hashed_blobs::SegmentWriter::open(
        root,
        CURRENT,
        end,
        &[],
        0o600,
    )?)
}

fn recorded_end(root: &Path) -> Result<Option<u64>> {
    match std::fs::read(root.join(RECORDED_END)) {
        Ok(bytes) => Ok(<[u8; 8]>::try_from(bytes.as_slice())
            .ok()
            .map(u64::from_le_bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Written after the commit that recorded `end` was synced, so it never
/// names more than a synced record does.
fn record_end(root: &Path, end: u64) -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(RECORDED_END))?;
    file.write_all_at(&end.to_le_bytes(), 0)?;
    Ok(())
}

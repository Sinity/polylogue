use super::*;
use snapshot::Extent;
use std::io::{Read, Seek};

pub struct ChainReader {
    cursor: Cursor,
    chain: StoredChain,
    store: Arc<hashed_blobs::Store>,
}

impl ChainReader {
    pub(super) fn new(chain: StoredChain, store: Arc<hashed_blobs::Store>) -> Self {
        let cursor = Cursor::forward(chain.length);
        Self {
            chain,
            store,
            cursor,
        }
    }
}

impl Read for ChainReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        self.cursor
            .read(out, self.chain.snapshot.as_deref(), &self.store)
            .map_err(std::io::Error::other)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Range,
    Forward,
}

pub(super) struct Cursor {
    access: Access,
    position: u64,
    end: u64,
    current: Option<Box<dyn Read>>,
    left: u64,
}

impl Cursor {
    pub fn forward(end: u64) -> Self {
        Self {
            access: Access::Forward,
            ..Self::range(0, end)
        }
    }

    pub fn range(position: u64, end: u64) -> Self {
        Self {
            access: Access::Range,
            position,
            end,
            current: None,
            left: 0,
        }
    }

    pub fn read(
        &mut self,
        out: &mut [u8],
        snapshot: Option<&Snapshot>,
        store: &hashed_blobs::Store,
    ) -> Result<usize> {
        if out.is_empty() || self.position == self.end {
            return Ok(0);
        }
        if self.left == 0 {
            let (extent, inline) = snapshot
                .ok_or_else(|| StoreError::Invalid("stored extent missing".into()))?
                .select(self.position, store)?;
            if !extent.inline && self.access == Access::Range {
                let length =
                    (extent.offset + extent.length - self.position).min(self.end - self.position);
                let want = out.len().min(length.try_into().unwrap_or(usize::MAX));
                file_range(store, &extent, self.position, &mut out[..want])?;
                self.position += want as u64;
                return Ok(want);
            }
            let mut reader: Box<dyn Read> = match inline {
                Some(reader) => reader,
                None => Box::new(store.read_plain(
                    &hashed_blobs::Hash(extent.hash.0),
                    Some(extent.size),
                    || Ok(()),
                )?),
            };
            let skip = extent.blob_offset + self.position - extent.offset;
            let skipped = std::io::copy(&mut reader.by_ref().take(skip), &mut std::io::sink())?;
            if skipped != skip {
                return Err(StoreError::Invalid("stored blob offset past EOF".into()));
            }
            self.left =
                (extent.offset + extent.length - self.position).min(self.end - self.position);
            self.current = Some(reader);
        }
        let want = out.len().min(self.left.try_into().unwrap_or(usize::MAX));
        let n = self.current.as_mut().unwrap().read(&mut out[..want])?;
        if n == 0 {
            return Err(StoreError::Invalid("stored blob ends before extent".into()));
        }
        self.left -= n as u64;
        self.position += n as u64;
        Ok(n)
    }
}

pub(super) fn inline(
    conn: &Connection,
    store: &hashed_blobs::Store,
    extent: &Extent,
) -> Result<Box<dyn Read>> {
    if extent.size < super::super::INLINE_LIMIT as u64 {
        let mut bytes = Vec::with_capacity(extent.size as usize);
        super::super::copy_plain(conn, store, extent.hash, &mut bytes)?;
        Ok(Box::new(std::io::Cursor::new(bytes)))
    } else {
        let mut file = tempfile::tempfile()?;
        super::super::copy_plain(conn, store, extent.hash, &mut file)?;
        file.rewind()?;
        Ok(Box::new(file))
    }
}

// The whole-name verification pass also supplies the requested bytes; do not
// decode the preceding file prefix a second time just to reach the range.
pub(super) fn file_range(
    store: &hashed_blobs::Store,
    extent: &Extent,
    position: u64,
    out: &mut [u8],
) -> Result<hashed_blobs::VerificationStats> {
    let start = extent.blob_offset + position - extent.offset;
    let end = start + out.len() as u64;
    let mut at = 0u64;
    store.verify_plain(
        &hashed_blobs::Hash(extent.hash.0),
        extent.size,
        |bytes| -> Result<()> {
            let next = at + bytes.len() as u64;
            let from = at.max(start);
            let to = next.min(end);
            if from < to {
                out[(from - start) as usize..(to - start) as usize]
                    .copy_from_slice(&bytes[(from - at) as usize..(to - at) as usize]);
            }
            at = next;
            Ok(())
        },
    )
}

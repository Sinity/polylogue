use super::*;
use std::io::{Read, Seek, SeekFrom};

pub struct ChainReader<S = crate::blobs::Blobs> {
    cursor: Cursor,
    chain: StoredChain,
    store: Arc<S>,
}
impl<S: super::super::PlainBlobs> ChainReader<S> {
    pub(super) fn new(chain: StoredChain, store: Arc<S>) -> Self {
        let cursor = Cursor::forward(chain.length);
        Self {
            chain,
            store,
            cursor,
        }
    }
}
impl<S: super::super::PlainBlobs> Read for ChainReader<S> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        self.cursor
            .read(out, self.chain.extents.as_deref(), self.store.as_ref())
            .map_err(std::io::Error::other)
    }
}
pub(super) struct Cursor {
    position: u64,
    end: u64,
    current: Option<hashed_blobs::BlobReader>,
    left: u64,
}
impl Cursor {
    pub fn forward(end: u64) -> Self {
        Self::range(0, end)
    }
    pub fn range(position: u64, end: u64) -> Self {
        Self {
            position,
            end,
            current: None,
            left: 0,
        }
    }
    pub fn read(
        &mut self,
        out: &mut [u8],
        extents: Option<&ExtentList>,
        store: &impl super::super::PlainBlobs,
    ) -> Result<usize> {
        if out.is_empty() || self.position == self.end {
            return Ok(0);
        }
        if self.left == 0 {
            let extent = extents
                .ok_or_else(|| StoreError::Invalid("stored byte range missing".into()))?
                .select(self.position)?;
            let mut reader = store.open_plain(&hashed_blobs::Hash(extent.hash.0), extent.size)?;
            reader.seek(SeekFrom::Start(
                extent.blob_offset + self.position - extent.offset,
            ))?;
            self.left =
                (extent.offset + extent.length - self.position).min(self.end - self.position);
            self.current = Some(reader);
        }
        let want = out.len().min(self.left.try_into().unwrap_or(usize::MAX));
        let count = self
            .current
            .as_mut()
            .expect("opened extent")
            .read(&mut out[..want])?;
        if count == 0 {
            return Err(StoreError::Invalid(
                "stored blob ends before its byte range".into(),
            ));
        }
        self.left -= count as u64;
        self.position += count as u64;
        Ok(count)
    }
}

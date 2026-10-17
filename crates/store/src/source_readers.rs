//! Two object-affine readers for one retained source capability. A snapshot's
//! base and patch cannot evict one another's checked object. Other source shapes
//! keep correctness and may evict; no global or pathname cache is introduced.
use crate::{BlobHash, Error, Result};
use std::{path::Path, sync::Mutex};
type Key = (String, i64, BlobHash);
struct State {
    key: Option<Key>,
    hashes: [Option<BlobHash>; 2],
}
pub(crate) struct SourceReaders {
    readers: [hashed_blobs::Reader; 2],
    state: Mutex<State>,
    #[cfg(test)]
    pub explicit_verification_bytes: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    pub explicit_verifications: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    pub force_shared_slot: std::sync::atomic::AtomicBool,
}
impl SourceReaders {
    pub fn new(root: &Path) -> Self {
        Self {
            readers: [
                hashed_blobs::Reader::new(root, vec![]),
                hashed_blobs::Reader::new(root, vec![]),
            ],
            state: Mutex::new(State {
                key: None,
                hashes: [None, None],
            }),
            #[cfg(test)]
            explicit_verification_bytes: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            explicit_verifications: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            force_shared_slot: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn select(
        &self,
        id: &str,
        chain: i64,
        binding: BlobHash,
        hash: BlobHash,
    ) -> Result<(&hashed_blobs::Reader, bool)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Invariant("source reader state poisoned".into()))?;
        if !matches!(&state.key,Some((old,c,b)) if old==id && *c==chain && *b==binding) {
            state.key = Some((id.into(), chain, binding));
            state.hashes = [None, None];
        }
        if let Some(index) = state.hashes.iter().position(|h| *h == Some(hash)) {
            return Ok((self.reader(index), false));
        }
        let index = state.hashes.iter().position(Option::is_none).unwrap_or(1);
        state.hashes[index] = Some(hash);
        Ok((self.reader(index), true))
    }
    fn reader(&self, index: usize) -> &hashed_blobs::Reader {
        #[cfg(test)]
        if self
            .force_shared_slot
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return &self.readers[0];
        }
        &self.readers[index]
    }
    pub fn forget(&self, hash: BlobHash) {
        if let Ok(mut state) = self.state.lock() {
            for slot in &mut state.hashes {
                if *slot == Some(hash) {
                    *slot = None;
                }
            }
        }
    }
    pub fn verified(&self, bytes: u64) {
        #[cfg(test)]
        {
            use std::sync::atomic::Ordering;
            self.explicit_verification_bytes
                .fetch_add(bytes, Ordering::Relaxed);
            self.explicit_verifications.fetch_add(1, Ordering::Relaxed);
        }
        #[cfg(not(test))]
        let _ = bytes;
    }
}

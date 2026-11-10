//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: every vector of the embedding key `similar:` ranks by, in
//!   memory, keyed by the model revision of the requesting read snapshot.
//! - **Query vectors**: the provider's vector for each recent `similar:` text,
//!   so a repeated or paged query neither pays nor waits for the provider
//!   again, and ranks by the same values.
//!
//! Entries are bound to their read position or embeddings snapshot revision.
//! SQL reads and quantization do not hold the vector cache mutex.

use crate::embeddings::Provider;
use polylogue_store::Reader;
use polylogue_store::vectors::{Loaded, Similarity};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Recent `similar:` texts whose query vectors are kept.
const QUERY_VECTORS: usize = 256;

/// The most recently used entries, by key, at most `capacity`.
struct Recent<V> {
    capacity: usize,
    entries: VecDeque<(String, V)>,
}

impl<V: Clone> Recent<V> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<V> {
        let at = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(at)?;
        let value = entry.1.clone();
        self.entries.push_back(entry);
        Some(value)
    }

    fn put(&mut self, key: String, value: V) {
        self.entries.retain(|(k, _)| k != &key);
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((key, value));
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Hits and misses since the daemon started, for `status`.
#[derive(Default)]
struct Counts {
    vector_loads: AtomicU64,
    query_vectors_reused: AtomicU64,
}

#[derive(Default)]
struct VectorCache {
    loaded: Option<Arc<Loaded>>,
    invalidation: Arc<()>,
}

pub struct Warm {
    vectors: Mutex<VectorCache>,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    counts: Counts,
    /// Session item indexes read by recent reads.
    pub(crate) items: super::items::ItemCache,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            vectors: Mutex::new(VectorCache::default()),
            query_vectors: Mutex::new(Recent::new(QUERY_VECTORS)),
            counts: Counts::default(),
            items: super::items::ItemCache::default(),
        }
    }
}

fn read_lock<'a, T>(
    lock: &'a Mutex<T>,
    reader: &Reader<'_>,
) -> Result<std::sync::MutexGuard<'a, T>, String> {
    loop {
        reader.check().map_err(|error| error.to_string())?;
        match lock.try_lock() {
            Ok(held) => return Ok(held),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err("read cache poisoned".into()),
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(5))
            }
        }
    }
}

impl Warm {
    /// The loaded vectors of the embedding key `similar:` ranks by, loading
    /// them when absent or stale. `None`: no vectors for this provider.
    pub fn vectors(
        &self,
        reader: &Reader<'_>,
        provider: &Provider,
    ) -> Result<Option<Arc<Loaded>>, String> {
        let prefix = provider.key_prefix();
        let store = reader.vectors().map_err(|e| e.to_string())?;
        let Some(key) = store.key_with_prefix(&prefix).map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        let revision = store.revision(&key).map_err(|e| e.to_string())?;
        self.cached_vectors(reader, &key, revision, || {
            store.load(&key).map_err(|e| e.to_string())
        })
        .map(Some)
    }

    fn cached_vectors(
        &self,
        reader: &Reader<'_>,
        key: &str,
        revision: i64,
        load: impl FnOnce() -> Result<Loaded, String>,
    ) -> Result<Arc<Loaded>, String> {
        let invalidation = {
            let held = read_lock(&self.vectors, reader)?;
            if let Some(loaded) = &held.loaded
                && loaded.key == key
                && loaded.revision == revision
            {
                return Ok(Arc::clone(loaded));
            }
            Arc::clone(&held.invalidation)
        };
        // SQL and quantization never hold the shared cache lock. The reader's
        // snapshot, rather than a process counter, identifies these values.
        let loaded = Arc::new(load()?);
        self.counts.vector_loads.fetch_add(1, Ordering::Relaxed);
        reader.check().map_err(|e| e.to_string())?;
        let mut held = read_lock(&self.vectors, reader)?;
        if Arc::ptr_eq(&invalidation, &held.invalidation) {
            if let Some(current) = &held.loaded
                && current.key == key
                && current.revision == loaded.revision
            {
                return Ok(Arc::clone(current));
            }
            let newer = held
                .loaded
                .as_ref()
                .is_some_and(|current| current.key == key && current.revision > loaded.revision);
            if !newer {
                held.loaded = Some(Arc::clone(&loaded));
            }
        }
        Ok(loaded)
    }

    /// Every stored text's similarity to `text`, for `similar:`.
    pub fn similarity(
        &self,
        reader: &Reader<'_>,
        provider: &Provider,
        text: &str,
    ) -> Result<Arc<Similarity>, String> {
        let Some(loaded) = self.vectors(reader, provider)? else {
            return Err("no vectors for this provider's model: `polylogue embed build`".into());
        };
        let key = format!("{}\0{text}", loaded.key);
        let cached = self
            .query_vectors
            .lock()
            .map_err(|_| "vector cache poisoned")?
            .get(&key);
        let query = match cached {
            Some(v) => {
                self.counts
                    .query_vectors_reused
                    .fetch_add(1, Ordering::Relaxed);
                v
            }
            None => {
                let v = Arc::new(
                    crate::embeddings::query_vector(provider, text).map_err(|e| e.to_string())?,
                );
                self.query_vectors
                    .lock()
                    .map_err(|_| "vector cache poisoned")?
                    .put(key, Arc::clone(&v));
                v
            }
        };
        loaded
            .similarity(&query)
            .map(Arc::new)
            .map_err(|e| e.to_string())
    }

    /// Drops everything; the next requests rebuild what they use.
    pub fn clear(&self) {
        if let Ok(mut v) = self.vectors.lock() {
            *v = VectorCache::default();
        }
        if let Ok(mut q) = self.query_vectors.lock() {
            q.clear();
        }
    }

    /// What `status` shows.
    pub fn summary(&self) -> crate::protocol::status::Warm {
        use crate::protocol::status::{Vectors, Warm};
        let vectors = self.vectors.lock().ok().and_then(|vectors| {
            vectors.loaded.as_ref().map(|loaded| Vectors {
                key: loaded.key.clone(),
                vectors: loaded.vectors(),
            })
        });
        Warm {
            vectors,
            vector_loads: self.counts.vector_loads.load(Ordering::Relaxed),
            query_vectors_reused: self.counts.query_vectors_reused.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
impl Warm {
    /// A vector load that, once started, waits for `release` before it reads.
    pub(crate) fn load_vectors_until(
        &self,
        reader: &Reader<'_>,
        started: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    ) {
        let _ = self.cached_vectors(reader, "synthetic/empty", 0, || {
            started.send(()).unwrap();
            let _ = release.recv();
            reader
                .vectors()
                .map_err(|e| e.to_string())?
                .load("synthetic/empty")
                .map_err(|e| e.to_string())
        });
    }
}

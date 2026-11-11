//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: every vector of the embedding key `similar:` ranks by, in
//!   memory, keyed by the model revision of the requesting read snapshot.
//! - **Query vectors**: the provider's vector for each recent `similar:` text,
//!   so a repeated or paged query neither pays nor waits for the provider
//!   again, and ranks by the same values. Concurrent reads of one text share
//!   one provider call: one provider call per query text is in flight.
//!
//! Entries are bound to their read position or embeddings snapshot revision.
//! SQL reads and quantization do not hold the vector cache mutex.

use crate::embeddings::Provider;
use polylogue_store::Reader;
use polylogue_store::vectors::{Loaded, Similarity};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

/// A provider call for one query text that later reads of the same text
/// wait for.
#[derive(Default)]
struct Flight {
    result: Mutex<Option<Result<Arc<Vec<f32>>, String>>>,
    done: std::sync::Condvar,
    #[cfg(test)]
    waiting: std::sync::atomic::AtomicUsize,
}

pub struct Warm {
    vectors: Mutex<VectorCache>,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    in_flight: Mutex<std::collections::HashMap<String, Arc<Flight>>>,
    counts: Counts,
    /// Session item indexes read by recent reads.
    pub(crate) items: super::items::ItemCache,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            vectors: Mutex::new(VectorCache::default()),
            query_vectors: Mutex::new(Recent::new(QUERY_VECTORS)),
            in_flight: Mutex::default(),
            counts: Counts::default(),
            items: super::items::ItemCache::default(),
        }
    }
}

/// The cache lock guards only pointer swaps, never SQL or quantization, so
/// a plain lock is held for microseconds.
fn cache_lock<T>(lock: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, String> {
    lock.lock().map_err(|_| "read cache poisoned".to_owned())
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
            let held = cache_lock(&self.vectors)?;
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
        let mut held = cache_lock(&self.vectors)?;
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
        let query = self.query_vector(&key, provider, text)?;
        loaded
            .similarity(&query)
            .map(Arc::new)
            .map_err(|e| e.to_string())
    }

    /// The provider's vector for `text`: kept from an earlier read, taken
    /// from a call another read has in flight, or asked for. Neither the
    /// call nor the wait for it holds an executor place.
    fn query_vector(
        &self,
        key: &str,
        provider: &Provider,
        text: &str,
    ) -> Result<Arc<Vec<f32>>, String> {
        let cached = cache_lock(&self.query_vectors)?.get(key);
        if let Some(vector) = cached {
            self.counts
                .query_vectors_reused
                .fetch_add(1, Ordering::Relaxed);
            return Ok(vector);
        }
        let (flight, leader) = {
            let mut in_flight = cache_lock(&self.in_flight)?;
            match in_flight.get(key) {
                Some(flight) => (Arc::clone(flight), false),
                None => {
                    let flight = Arc::new(Flight::default());
                    in_flight.insert(key.to_owned(), Arc::clone(&flight));
                    (flight, true)
                }
            }
        };
        if !leader {
            self.counts
                .query_vectors_reused
                .fetch_add(1, Ordering::Relaxed);
            return super::executor::outside(|| wait_for(&flight));
        }
        // Waiting reads hear how the call ended, however it ends.
        struct Land<'a> {
            warm: &'a Warm,
            key: &'a str,
            flight: Arc<Flight>,
        }
        impl Drop for Land<'_> {
            fn drop(&mut self) {
                let mut result = self
                    .flight
                    .result
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                result.get_or_insert_with(|| Err("the provider call did not finish".into()));
                drop(result);
                self.warm
                    .in_flight
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(self.key);
                self.flight.done.notify_all();
            }
        }
        let land = Land {
            warm: self,
            key,
            flight,
        };
        let result = super::executor::outside(|| {
            crate::embeddings::query_vector(provider, text)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });
        if let Ok(vector) = &result {
            cache_lock(&self.query_vectors)?.put(key.to_owned(), Arc::clone(vector));
        }
        *cache_lock(&land.flight.result)? = Some(result.clone());
        drop(land);
        result
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

/// Waits for another read's provider call; the read's deadline bounds the
/// wait and its cancellation wakes it.
fn wait_for(flight: &Arc<Flight>) -> Result<Arc<Vec<f32>>, String> {
    #[cfg(test)]
    flight.waiting.fetch_add(1, Ordering::Relaxed);
    let limit = polylogue_store::read::limit::ReadLimit::current();
    let _wake = limit.as_ref().map(|limit| {
        let flight = Arc::clone(flight);
        limit.cancellation.on_cancel(move || {
            let _held = flight.result.lock();
            flight.done.notify_all();
        })
    });
    let mut result = cache_lock(&flight.result)?;
    loop {
        if let Some(result) = result.as_ref() {
            return result.clone();
        }
        if let Some(limit) = &limit {
            limit.check().map_err(|error| error.to_string())?;
        }
        result = match limit
            .as_ref()
            .and_then(polylogue_store::read::limit::ReadLimit::remaining)
        {
            Some(left) => {
                flight
                    .done
                    .wait_timeout(result, left)
                    .map_err(|_| "read cache poisoned".to_owned())?
                    .0
            }
            None => flight
                .done
                .wait(result)
                .map_err(|_| "read cache poisoned".to_owned())?,
        };
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

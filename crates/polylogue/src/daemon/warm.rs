//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: a request extends the newest held set to its snapshot's
//!   model revision. Concurrent requests for the same model and revision
//!   share one load and its result. The two newest sets stay in memory;
//!   older snapshots load their own set without replacing newer values.
//! - **Query vectors**: the vector for each recent `similar:` text, so a
//!   repeated or paged query neither reads nor waits again and ranks by the
//!   same values. A text's vector is the one stored in `embeddings.db` when
//!   there is one; otherwise, with model calls on, the provider's answer,
//!   which is stored, so no text is paid for twice. Concurrent reads of one
//!   text share one lookup and one provider call: one provider call per
//!   query text is in flight.
//!
//! Entries are bound to their read position or embeddings snapshot revision.
//! SQL reads and quantization do not hold the vector cache mutex.

use crate::embeddings::Provider;
use polylogue_model::LockExt as _;
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

/// Work counters since the daemon started, for `status`: vector loads and
/// reused query vectors.
#[derive(Default)]
struct Counts {
    vector_loads: AtomicU64,
    query_vectors_reused: AtomicU64,
}

/// The two newest vector sets and requests currently loading a set.
#[derive(Default)]
struct VectorCache {
    sets: Vec<Arc<Loaded>>,
    loading: std::collections::HashMap<(String, i64), Arc<Flight<Loaded>>>,
}

impl VectorCache {
    fn exact(&self, key: &str, revision: i64) -> Option<Arc<Loaded>> {
        self.sets
            .iter()
            .find(|set| set.key == key && set.revision == revision)
            .cloned()
    }

    fn newest(&self, key: &str) -> Option<&Arc<Loaded>> {
        self.sets.last().filter(|set| set.key == key)
    }
}

/// One vector load or provider call shared by requests for the same key.
struct Flight<T> {
    result: Mutex<Option<Result<Arc<T>, String>>>,
    done: std::sync::Condvar,
}

impl<T> Default for Flight<T> {
    fn default() -> Self {
        Self {
            result: Mutex::new(None),
            done: std::sync::Condvar::new(),
        }
    }
}

/// The daemon's warm answers: loaded similarity vectors, query vectors and
/// in-flight provider calls.
pub struct Warm {
    clock: polylogue_model::WallClock,
    archive: polylogue_store::Archive,
    vectors: Mutex<VectorCache>,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    in_flight: Mutex<std::collections::HashMap<String, Arc<Flight<Vec<f32>>>>>,
    counts: Counts,
}

/// The cache lock guards only pointer swaps, never SQL or quantization, so
/// a plain lock is held for microseconds.
fn cache_lock<T>(lock: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, String> {
    lock.lock().map_err(|_| "read cache poisoned".to_owned())
}

impl Warm {
    pub fn new(archive: &polylogue_store::Archive, clock: polylogue_model::WallClock) -> Self {
        let origins = polylogue_sources::ORIGINS
            .iter()
            .map(|origin| origin.name.to_owned())
            .collect::<Vec<_>>();
        polylogue_store::selection::shipped_bindings(&origins)
            .expect("shipped query declarations prepare");
        Self {
            archive: archive.clone(),
            clock,
            vectors: Mutex::default(),
            query_vectors: Mutex::new(Recent::new(QUERY_VECTORS)),
            in_flight: Mutex::default(),
            counts: Counts::default(),
        }
    }

    /// The loaded vectors of the embedding key `similar:` ranks by, in the
    /// reader's snapshot. `None`: no vectors for this provider.
    pub fn vectors(
        &self,
        reader: &Reader<'_>,
        provider: &Provider,
    ) -> Result<Option<Arc<Loaded>>, String> {
        let prefix = provider.key_prefix();
        let key = reader
            .vectors()
            .and_then(|store| store.key_with_prefix(&prefix))
            .map_err(|e| e.to_string())?;
        key.map(|key| self.vectors_of(reader, &key)).transpose()
    }

    /// One load per model and revision, performed on the requesting reader.
    fn vectors_of(&self, reader: &Reader<'_>, key: &str) -> Result<Arc<Loaded>, String> {
        reader.check().map_err(|error| error.to_string())?;
        let revision = reader
            .vectors()
            .and_then(|store| store.revision(key))
            .map_err(|e| e.to_string())?;
        let address = (key.to_owned(), revision);
        let (flight, base) = {
            let mut cache = cache_lock(&self.vectors)?;
            if let Some(set) = cache.exact(key, revision) {
                return Ok(set);
            }
            if let Some(flight) = cache.loading.get(&address).cloned() {
                drop(cache);
                return wait_for(&flight);
            }
            let flight = Arc::new(Flight::default());
            cache.loading.insert(address.clone(), Arc::clone(&flight));
            let base = cache
                .newest(key)
                .filter(|set| set.revision < revision)
                .cloned();
            (flight, base)
        };
        // Even an unwinding load wakes the requests sharing its result.
        struct Loading<'a> {
            warm: &'a Warm,
            address: (String, i64),
            flight: Arc<Flight<Loaded>>,
        }
        impl Drop for Loading<'_> {
            fn drop(&mut self) {
                self.flight
                    .result
                    .locked()
                    .get_or_insert_with(|| Err("the vector load did not finish".into()));
                let mut cache = self.warm.vectors.locked();
                if cache
                    .loading
                    .get(&self.address)
                    .is_some_and(|flight| Arc::ptr_eq(flight, &self.flight))
                {
                    cache.loading.remove(&self.address);
                }
                self.flight.done.notify_all();
            }
        }
        let loading = Loading {
            warm: self,
            address,
            flight,
        };
        let result = reader
            .vectors()
            .and_then(|store| match base {
                Some(base) => store.append(&base),
                None => {
                    self.counts.vector_loads.fetch_add(1, Ordering::Relaxed);
                    store.load(key)
                }
            })
            .and_then(|loaded| {
                reader.check()?;
                Ok(Arc::new(loaded))
            })
            .map_err(|e| e.to_string());
        {
            let mut cache = cache_lock(&self.vectors)?;
            // Clearing the cache disconnects existing loads from publication.
            if let Ok(set) = &result
                && cache
                    .loading
                    .get(&loading.address)
                    .is_some_and(|flight| Arc::ptr_eq(flight, &loading.flight))
                && cache
                    .newest(key)
                    .is_none_or(|newest| newest.revision < revision)
            {
                let earlier = cache.newest(key).cloned();
                cache.sets = earlier.into_iter().chain([Arc::clone(set)]).collect();
            }
            *cache_lock(&loading.flight.result)? = Some(result.clone());
        }
        drop(loading);
        result
    }

    /// Texts' similarity to `text`, for `similar:`, scored as the query
    /// asks for them.
    pub fn similarity(
        &self,
        reader: &Reader<'_>,
        provider: &Provider,
        text: &str,
    ) -> Result<Arc<Similarity>, String> {
        let Some(loaded) = self.vectors(reader, provider)? else {
            return Err("no vectors for this provider's model: `polylogue embed build`".into());
        };
        let query = self.query_vector(provider, text)?;
        loaded
            .similarity(&query)
            .map(Arc::new)
            .map_err(|e| e.to_string())
    }

    /// The vector for `text`: kept from an earlier read, taken from a
    /// lookup or call another read has in flight, read from the stored
    /// query vectors, or, with model calls on, asked for and stored. It is
    /// asked for before the request takes a reader, so neither the call nor
    /// a wait for another read's call holds one.
    pub(crate) fn query_vector(
        &self,
        provider: &Provider,
        text: &str,
    ) -> Result<Arc<Vec<f32>>, String> {
        let text = &crate::embeddings::query_text(text);
        // A query vector depends on the provider's model and the text only.
        let key = &format!("{}\0{text}", provider.key_prefix());
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
            return wait_for(&flight);
        }
        // Waiting reads hear how the call ended, however it ends.
        struct Land<'a> {
            warm: &'a Warm,
            key: &'a str,
            flight: Arc<Flight<Vec<f32>>>,
        }
        impl Drop for Land<'_> {
            fn drop(&mut self) {
                let mut result = self.flight.result.locked();
                result.get_or_insert_with(|| Err("the provider call did not finish".into()));
                drop(result);
                self.warm.in_flight.locked().remove(self.key);
                self.flight.done.notify_all();
            }
        }
        let land = Land {
            warm: self,
            key,
            flight,
        };
        let result = self.stored_or_asked(provider, text).map(Arc::new);
        if let Ok(vector) = &result {
            cache_lock(&self.query_vectors)?.put(key.to_owned(), Arc::clone(vector));
        }
        *cache_lock(&land.flight.result)? = Some(result.clone());
        drop(land);
        result
    }

    /// The stored vector of `text` (NFC), or the provider's answer, stored
    /// under its query key, when model calls are on.
    fn stored_or_asked(&self, provider: &Provider, text: &str) -> Result<Vec<f32>, String> {
        // One page read at most: a lookup by key.
        const CACHE_KIB: i64 = 64;
        let stored = polylogue_store::vectors::Vectors::snapshot(&self.archive, CACHE_KIB)
            .and_then(|store| store.query_vector(&provider.key_prefix(), text))
            .map_err(|e| e.to_string())?;
        if let Some(vector) = stored {
            return Ok(vector);
        }
        if !provider.calls {
            return Err(crate::embeddings::CALLS_OFF.into());
        }
        let vector = crate::embeddings::query_vector(provider, text).map_err(|e| e.to_string())?;
        // A failed store keeps the ranking: the vector is kept in memory and
        // asked for again only after the daemon restarts.
        if let Err(error) = polylogue_store::vectors::Vectors::writer(&self.archive, CACHE_KIB)
            .and_then(|store| {
                store.put_query_vector(
                    &provider.query_key(vector.len()),
                    text,
                    &vector,
                    &self.clock,
                )
            })
        {
            polylogue_store::log!(error, "storing a query vector: {error}");
        }
        Ok(vector)
    }

    /// Drops everything; the next requests rebuild what they use.
    pub fn clear(&self) {
        if let Ok(mut v) = self.vectors.lock() {
            v.sets.clear();
            v.loading.clear();
        }
        if let Ok(mut q) = self.query_vectors.lock() {
            q.clear();
        }
    }

    /// What `status` shows.
    pub fn summary(&self) -> crate::protocol::status::Warm {
        use crate::protocol::status::{Vectors, Warm};
        let vectors = self.vectors.lock().ok().and_then(|vectors| {
            vectors.sets.last().map(|loaded| Vectors {
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

/// Waits for another request's vectors; the read's deadline bounds the
/// wait and its cancellation wakes it.
fn wait_for<T>(flight: &Arc<Flight<T>>) -> Result<Arc<T>, String> {
    polylogue_store::read::limit::ReadLimit::set_stage("vector wait");
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

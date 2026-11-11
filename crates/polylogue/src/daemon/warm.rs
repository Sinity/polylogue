//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: every vector of the embedding key `similar:` ranks by, in
//!   memory. One worker thread loads them, extends them with the vectors
//!   stored since whenever a vector write commits, and publishes each set
//!   with the model revision of the snapshot it read. A read uses the set
//!   of its own snapshot's revision, waiting for the worker when that set is
//!   not published yet; it neither loads nor extends vectors itself, except
//!   a read whose snapshot is older than both sets held, which loads its own.
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

/// The vectors the worker published, and what reads asked it for.
#[derive(Default)]
struct VectorCache {
    /// The newest set and the one it extended, newest last: a read from a
    /// snapshot taken before the newest commit still has its set.
    sets: Vec<Arc<Loaded>>,
    /// The embedding key reads rank by.
    wanted: Option<String>,
    /// Passes the worker finished, and how the last one failed.
    passes: u64,
    failure: Option<String>,
    /// Replaced by [`Warm::clear`]: a pass that started before cannot
    /// publish after it.
    invalidation: Arc<()>,
    stopping: bool,
    /// Reads that have waited for the worker.
    #[cfg(test)]
    waited: usize,
    #[cfg(test)]
    gate: Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>,
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

/// What the vectors worker shares with reads.
#[derive(Default)]
struct VectorShare {
    cache: Mutex<VectorCache>,
    published: std::sync::Condvar,
}

struct VectorWorker {
    wake: std::sync::mpsc::SyncSender<()>,
    thread: std::thread::JoinHandle<()>,
}

#[cfg(test)]
thread_local! {
    /// Vector loads and extensions run on this thread: a read must leave
    /// them to the worker.
    static LOADS_HERE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn loads_here() -> u64 {
    LOADS_HERE.with(std::cell::Cell::get)
}

fn counted_load<T>(load: impl FnOnce() -> T) -> T {
    #[cfg(test)]
    LOADS_HERE.with(|count| count.set(count.get() + 1));
    load()
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
    archive: polylogue_store::Archive,
    vectors: Arc<VectorShare>,
    worker: Mutex<Option<VectorWorker>>,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    in_flight: Mutex<std::collections::HashMap<String, Arc<Flight>>>,
    counts: Arc<Counts>,
}

impl Drop for Warm {
    fn drop(&mut self) {
        if let Ok(mut cache) = self.vectors.cache.lock() {
            cache.stopping = true;
        }
        if let Some(worker) = self.worker.get_mut().ok().and_then(Option::take) {
            let _ = worker.wake.try_send(());
            let _ = worker.thread.join();
        }
    }
}

/// The cache lock guards only pointer swaps, never SQL or quantization, so
/// a plain lock is held for microseconds.
fn cache_lock<T>(lock: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, String> {
    lock.lock().map_err(|_| "read cache poisoned".to_owned())
}

impl Warm {
    pub fn new(archive: &polylogue_store::Archive) -> Self {
        Self {
            archive: archive.clone(),
            vectors: Arc::default(),
            worker: Mutex::default(),
            query_vectors: Mutex::new(Recent::new(QUERY_VECTORS)),
            in_flight: Mutex::default(),
            counts: Arc::default(),
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

    /// The vectors of one embedding key in the reader's snapshot: the set
    /// the worker published for its revision, waited for when not yet
    /// published. The wait holds no executor place.
    fn vectors_of(&self, reader: &Reader<'_>, key: &str) -> Result<Arc<Loaded>, String> {
        let revision = reader
            .vectors()
            .and_then(|store| store.revision(key))
            .map_err(|e| e.to_string())?;
        if let Some(set) = cache_lock(&self.vectors.cache)?.exact(key, revision) {
            return Ok(set);
        }
        match super::executor::outside(|| self.wait_for_worker(key, revision))? {
            Some(set) => Ok(set),
            // The snapshot is older than both sets held: its own load serves
            // it and is not kept.
            None => {
                let loaded = counted_load(|| {
                    reader
                        .vectors()
                        .and_then(|store| store.load(key))
                        .map_err(|e| e.to_string())
                })?;
                reader.check().map_err(|e| e.to_string())?;
                Ok(Arc::new(loaded))
            }
        }
    }

    /// Asks the worker for `key` and waits until it publishes the set of
    /// `revision` (`Some`) or a newer one only (`None`), or fails. The read's
    /// deadline bounds the wait and its cancellation wakes it.
    fn wait_for_worker(&self, key: &str, revision: i64) -> Result<Option<Arc<Loaded>>, String> {
        let limit = polylogue_store::read::limit::ReadLimit::current();
        let _wake = limit.as_ref().map(|limit| {
            let share = Arc::clone(&self.vectors);
            limit.cancellation.on_cancel(move || {
                let _held = share.cache.lock();
                share.published.notify_all();
            })
        });
        let mut cache = cache_lock(&self.vectors.cache)?;
        #[cfg(test)]
        {
            cache.waited += 1;
        }
        let mut asked = None;
        loop {
            if let Some(set) = cache.exact(key, revision) {
                return Ok(Some(set));
            }
            if asked.is_some_and(|asked| cache.passes > asked) {
                if let Some(failure) = &cache.failure {
                    return Err(failure.clone());
                }
                if cache.newest(key).is_some_and(|set| set.revision > revision) {
                    return Ok(None);
                }
                // A pass that published nothing for us (it read an older
                // snapshot, or a clear discarded it): ask again.
                asked = None;
            }
            if asked.is_none() {
                cache.wanted = Some(key.to_owned());
                asked = Some(cache.passes);
                self.wake_worker()?;
                continue;
            }
            if let Some(limit) = &limit {
                limit.check().map_err(|error| error.to_string())?;
            }
            cache = match limit
                .as_ref()
                .and_then(polylogue_store::read::limit::ReadLimit::remaining)
            {
                Some(left) => {
                    self.vectors
                        .published
                        .wait_timeout(cache, left)
                        .map_err(|_| "read cache poisoned".to_owned())?
                        .0
                }
                None => self
                    .vectors
                    .published
                    .wait(cache)
                    .map_err(|_| "read cache poisoned".to_owned())?,
            };
        }
    }

    /// Wakes the worker, starting it on first use.
    fn wake_worker(&self) -> Result<(), String> {
        let mut worker = cache_lock(&self.worker)?;
        if worker.is_none() {
            let events = polylogue_store::vectors::write_events();
            let wake = events.wake_handle();
            let share = Arc::clone(&self.vectors);
            let counts = Arc::clone(&self.counts);
            let archive = self.archive.clone();
            let thread = std::thread::Builder::new()
                .name("polylogue-vectors".into())
                .spawn(move || run_worker(&archive, &share, &counts, &events))
                .map_err(|e| e.to_string())?;
            *worker = Some(VectorWorker { wake, thread });
        }
        if let Some(worker) = worker.as_ref() {
            let _ = worker.wake.try_send(());
        }
        Ok(())
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
    /// query vectors, or, with model calls on, asked for and stored.
    /// Neither the lookup, the call nor the wait for them holds an executor
    /// place.
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
        let result =
            super::executor::outside(|| self.stored_or_asked(provider, text).map(Arc::new));
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
                store.put_query_vector(&provider.query_key(vector.len()), text, &vector)
            })
        {
            eprintln!("polylogue: storing a query vector: {error}");
        }
        Ok(vector)
    }

    /// Drops everything; the next requests rebuild what they use.
    pub fn clear(&self) {
        if let Ok(mut v) = self.vectors.cache.lock() {
            v.sets.clear();
            v.wanted = None;
            v.failure = None;
            v.invalidation = Arc::new(());
        }
        if let Ok(mut q) = self.query_vectors.lock() {
            q.clear();
        }
    }

    /// What `status` shows.
    pub fn summary(&self) -> crate::protocol::status::Warm {
        use crate::protocol::status::{Vectors, Warm};
        let vectors = self.vectors.cache.lock().ok().and_then(|vectors| {
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

/// The vectors worker: on each cue (a vector write committed, or a read
/// asked), brings the wanted key's set to the revision of a fresh snapshot,
/// extending the newest set held, and publishes it.
fn run_worker(
    archive: &polylogue_store::Archive,
    share: &VectorShare,
    counts: &Counts,
    events: &polylogue_store::completion::CommitSubscription,
) {
    // A small page cache: the worker reads each stored row once.
    const CACHE_KIB: i64 = 2048;
    while events.recv().is_ok() {
        let (key, held, invalidation) = {
            let Ok(cache) = share.cache.lock() else {
                return;
            };
            if cache.stopping {
                return;
            }
            let Some(key) = cache.wanted.clone() else {
                continue;
            };
            let held = cache.newest(&key).cloned();
            (key, held, Arc::clone(&cache.invalidation))
        };
        #[cfg(test)]
        if let Ok(mut cache) = share.cache.lock()
            && let Some((started, release)) = cache.gate.take()
        {
            drop(cache);
            let _ = started.send(());
            let _ = release.recv();
        }
        let result = polylogue_store::vectors::Vectors::snapshot(archive, CACHE_KIB)
            .and_then(|store| {
                let revision = store.revision(&key)?;
                match held {
                    Some(held) if held.revision == revision => Ok(None),
                    Some(held) => counted_load(|| store.append(&held)).map(Some),
                    None => {
                        counts.vector_loads.fetch_add(1, Ordering::Relaxed);
                        counted_load(|| store.load(&key)).map(Some)
                    }
                }
            })
            .map_err(|e| e.to_string());
        let Ok(mut cache) = share.cache.lock() else {
            return;
        };
        if Arc::ptr_eq(&invalidation, &cache.invalidation) {
            match result {
                Ok(Some(loaded)) => {
                    let keep = cache.newest(&key).cloned();
                    cache.sets = keep.into_iter().chain([Arc::new(loaded)]).collect();
                    cache.failure = None;
                }
                Ok(None) => cache.failure = None,
                Err(error) => cache.failure = Some(error),
            }
        }
        cache.passes += 1;
        drop(cache);
        share.published.notify_all();
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
impl Warm {
    /// A vector load that, once the worker starts it, waits for `release`.
    pub(crate) fn load_vectors_until(
        &self,
        reader: &Reader<'_>,
        started: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    ) {
        if let Ok(mut cache) = self.vectors.cache.lock() {
            cache.gate = Some((started, release));
        }
        let _ = self.vectors_of(reader, "synthetic/empty");
    }
}

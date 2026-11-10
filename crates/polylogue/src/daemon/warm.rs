//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: every vector of the embedding key `similar:` ranks by, in
//!   memory, keyed by the model revision of the requesting read snapshot.
//! - **Query vectors**: the provider's vector for each recent `similar:` text,
//!   so a repeated or paged query neither pays nor waits for the provider
//!   again, and ranks by the same values.
//! - **Whether the index has caught up**, for reads: found by a worker of
//!   its own when the daemon starts, after the writer commits and when a
//!   read at a newer position asks, one at a time, so a read reports the
//!   last answer instead of walking every chain.
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
    /// Whether the index had caught up, with the change position it was read at.
    caught_up: Mutex<Option<(String, bool)>>,
    caught_up_error: Mutex<Option<String>>,
    /// Wakes the caught-up worker: the writer committed, or a read saw an
    /// answer older than its position.
    recheck: std::sync::Condvar,
    recheck_wanted: std::sync::atomic::AtomicBool,
    checking_stopped: std::sync::atomic::AtomicBool,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    counts: Counts,
    /// Session item indexes read by recent reads.
    pub(crate) items: super::items::ItemCache,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            vectors: Mutex::new(VectorCache::default()),
            caught_up: Mutex::new(None),
            caught_up_error: Mutex::new(None),
            recheck: std::sync::Condvar::new(),
            // The first check starts with the daemon.
            recheck_wanted: std::sync::atomic::AtomicBool::new(true),
            checking_stopped: std::sync::atomic::AtomicBool::new(false),
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

    /// Whether the index has caught up at `position`. The answer is computed
    /// once per change position, by the caught-up worker or by the first read
    /// at a new position; later pages and reads at that position reuse it.
    pub fn caught_up(&self, reader: &Reader<'_>, position: &str) -> polylogue_store::Result<bool> {
        if let Some((known, caught_up)) = self
            .caught_up
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            && known == position
        {
            return Ok(*caught_up);
        }
        let caught_up = reader
            .completeness(&polylogue_engine::reading_contracts())?
            .caught_up();
        *self
            .caught_up
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some((position.to_owned(), caught_up));
        Ok(caught_up)
    }

    /// Whether the index had caught up, as the caught-up worker last found
    /// it. A read never scans for this: when the position moved since, the
    /// worker is asked to check again and the read reports the last answer.
    /// Before the first check it is computed once.
    pub fn caught_up_for_read(
        &self,
        reader: &Reader<'_>,
        position: &str,
    ) -> polylogue_store::Result<bool> {
        let known = self
            .caught_up
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        match known {
            Some((at, caught_up)) => {
                if at != position {
                    self.request_check();
                }
                Ok(caught_up)
            }
            None => self.caught_up(reader, position),
        }
    }

    fn request_check(&self) {
        self.recheck_wanted.store(true, Ordering::Release);
        // Under the worker's lock, so the wake cannot fall between its check
        // and its wait.
        let _guard = self.caught_up.lock();
        self.recheck.notify_one();
    }

    /// The caught-up worker: finds whether the index has caught up when the
    /// daemon starts, whenever the writer committed (`commits`) and whenever
    /// a read asked for a newer answer, one at a time, until
    /// [`Warm::stop_checking`].
    pub fn caught_up_worker(
        &self,
        reads: &polylogue_store::ReadPool,
        commits: Option<polylogue_store::completion::CommitSubscription>,
    ) {
        let wake = commits
            .as_ref()
            .map(polylogue_store::completion::CommitSubscription::wake_handle);
        std::thread::scope(|scope| {
            if let Some(commits) = commits {
                scope.spawn(move || {
                    while commits.recv().is_ok() && !self.checking_stopped() {
                        self.request_check();
                    }
                });
            }
            loop {
                {
                    let Ok(held) = self.caught_up.lock() else {
                        break;
                    };
                    let Ok(held) = self.recheck.wait_while(held, |_| {
                        !self.checking_stopped() && !self.recheck_wanted.load(Ordering::Acquire)
                    }) else {
                        break;
                    };
                    drop(held);
                }
                if self.checking_stopped() {
                    break;
                }
                self.recheck_wanted.store(false, Ordering::Release);
                self.refresh_caught_up(reads);
            }
            self.checking_stopped.store(true, Ordering::Release);
            if let Some(wake) = wake {
                // The relay sees the stop on its next wake.
                let _ = wake.try_send(());
            }
        });
    }

    /// Ends [`Warm::caught_up_worker`].
    pub fn stop_checking(&self) {
        self.checking_stopped.store(true, Ordering::Release);
        let _guard = self.caught_up.lock();
        self.recheck.notify_all();
    }

    fn checking_stopped(&self) -> bool {
        self.checking_stopped.load(Ordering::Acquire)
    }

    fn refresh_caught_up(&self, reads: &polylogue_store::ReadPool) {
        let result = (|| {
            let reader = reads.reader()?;
            let position = reader.change_position()?.to_string();
            self.caught_up(&reader, &position).map(drop)
        })();
        *self
            .caught_up_error
            .lock()
            .unwrap_or_else(|error| error.into_inner()) =
            result.err().map(|error| error.to_string());
    }

    /// Why the last check of whether the index caught up failed.
    pub fn caught_up_error(&self) -> Option<String> {
        self.caught_up_error
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Drops everything; the next requests rebuild what they use.
    pub fn clear(&self) {
        if let Ok(mut v) = self.vectors.lock() {
            *v = VectorCache::default();
        }
        if let Ok(mut q) = self.query_vectors.lock() {
            q.clear();
        }
        if let Ok(mut c) = self.caught_up.lock() {
            *c = None;
        }
        self.recheck.notify_all();
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

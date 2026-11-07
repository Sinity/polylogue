//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: every vector of the embedding key `similar:` ranks by, in
//!   memory, keyed by the model revision of the requesting read snapshot.
//! - **Query vectors**: the provider's vector for each recent `similar:` text,
//!   so a repeated or paged query neither pays nor waits for the provider
//!   again, and ranks by the same values.
//! - **Archive counts** for `status`: counted by a worker of their own when
//!   the daemon starts and after the writer commits, while `status` answers
//!   from memory (no counts until the first count ends), so it stays fast
//!   during a long pass. Counts that take under 50 ms are simply taken again
//!   inside the request. One count runs at a time: commits made while it
//!   runs are counted once it ends, so the counts reach the writer's last
//!   commit one count after it, and `status` says which read position its
//!   counts are from.
//!
//! Entries are bound to their read position or embeddings snapshot revision.
//! SQL reads and quantization do not hold the vector cache mutex.

use crate::embeddings::Provider;
use polylogue_store::Reader;
use polylogue_store::vectors::{Loaded, Similarity};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Recent `similar:` texts whose query vectors are kept.
const QUERY_VECTORS: usize = 256;
/// A count this quick is repeated inside the `status` request.
const QUICK_COUNT: Duration = Duration::from_millis(50);

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

/// The archive counts `status` shows, and what they were read at.
#[derive(Clone)]
pub struct Counted {
    pub position: String,
    pub at: Instant,
    pub took: Duration,
    pub status: polylogue_store::Status,
}

#[derive(Default)]
struct VectorCache {
    loaded: Option<Arc<Loaded>>,
    invalidation: Arc<()>,
}

pub struct Warm {
    vectors: Mutex<VectorCache>,
    archive_counts: Mutex<Option<Counted>>,
    /// Whether the index had caught up, with the change position it was read at.
    caught_up: Mutex<Option<(String, bool)>>,
    count_error: Mutex<Option<String>>,
    /// Wakes the count worker: the writer committed, or a request saw
    /// counts older than its read.
    recount: std::sync::Condvar,
    recount_wanted: std::sync::atomic::AtomicBool,
    counting_stopped: std::sync::atomic::AtomicBool,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    counts: Counts,
    /// Session item indexes read by recent reads.
    pub(crate) items: super::items::ItemCache,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            vectors: Mutex::new(VectorCache::default()),
            archive_counts: Mutex::new(None),
            caught_up: Mutex::new(None),
            count_error: Mutex::new(None),
            recount: std::sync::Condvar::new(),
            // The first count starts with the daemon.
            recount_wanted: std::sync::atomic::AtomicBool::new(true),
            counting_stopped: std::sync::atomic::AtomicBool::new(false),
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

    /// The archive counts for `status`. When the last count was quick
    /// (under [`QUICK_COUNT`]) and is older than `position`, it counts again
    /// now, so a small archive's status is exact; otherwise it answers from
    /// memory and asks the worker for newer counts. `None` until the first
    /// count ends. After a failure, the worker retries while previous counts stay available.
    pub fn counts(&self, reader: &Reader<'_>, position: &str) -> Option<Counted> {
        let held = self.archive_counts.lock().ok()?.clone();
        match held {
            Some(c) if c.position == position => Some(c),
            Some(c) if c.took < QUICK_COUNT && self.count_error().is_none() => {
                match self.record_count(count(reader, position)) {
                    Some(counted) => Some(counted),
                    None => {
                        self.request_count();
                        Some(c)
                    }
                }
            }
            held => {
                self.request_count();
                held
            }
        }
    }

    /// Whether the index has caught up at `position`. The answer is computed
    /// once per change position, by the count worker or by the first read at
    /// a new position; later pages and reads at that position reuse it.
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

    /// Whether the index had caught up, as the count worker last found it.
    /// A read never scans for this: when the position moved since, the
    /// worker is asked to count again and the read reports the last answer.
    /// Before the first count it is computed once.
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
                    self.request_count();
                }
                Ok(caught_up)
            }
            None => self.caught_up(reader, position),
        }
    }

    fn request_count(&self) {
        self.recount_wanted.store(true, Ordering::Release);
        // Under the worker's lock, so the wake cannot fall between its check
        // and its wait.
        let _guard = self.archive_counts.lock();
        self.recount.notify_one();
    }

    /// The count worker: counts whenever the writer committed (`commits`)
    /// or a `status` asked for newer counts, one count at a time, until
    /// [`Warm::stop_counting`].
    pub fn count_worker(
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
                    while commits.recv().is_ok() && !self.counting_stopped() {
                        self.request_count();
                    }
                });
            }
            loop {
                {
                    let Ok(held) = self.archive_counts.lock() else {
                        break;
                    };
                    let Ok(held) = self.recount.wait_while(held, |_| {
                        !self.counting_stopped() && !self.recount_wanted.load(Ordering::Acquire)
                    }) else {
                        break;
                    };
                    drop(held);
                }
                if self.counting_stopped() {
                    break;
                }
                self.recount_wanted.store(false, Ordering::Release);
                self.refresh_counts(reads);
            }
            self.counting_stopped.store(true, Ordering::Release);
            if let Some(wake) = wake {
                // The relay sees the stop on its next wake.
                let _ = wake.try_send(());
            }
        });
    }

    /// Ends [`Warm::count_worker`].
    pub fn stop_counting(&self) {
        self.counting_stopped.store(true, Ordering::Release);
        let _guard = self.archive_counts.lock();
        self.recount.notify_all();
    }

    fn counting_stopped(&self) -> bool {
        self.counting_stopped.load(Ordering::Acquire)
    }

    fn refresh_counts(&self, reads: &polylogue_store::ReadPool) {
        let result = (|| {
            let reader = reads.reader().map_err(|error| error.to_string())?;
            let position = reader
                .change_position()
                .map_err(|error| error.to_string())?;
            let position = position.to_string();
            self.caught_up(&reader, &position)
                .map_err(|error| error.to_string())?;
            count(&reader, &position)
        })();
        self.record_count(result);
    }

    fn record_count(&self, result: Result<Counted, String>) -> Option<Counted> {
        match result {
            Ok(counted) => {
                *self
                    .archive_counts
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(counted.clone());
                *self
                    .count_error
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = None;
                Some(counted)
            }
            Err(error) => {
                *self
                    .count_error
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(error);
                None
            }
        }
    }

    pub fn count_error(&self) -> Option<String> {
        self.count_error
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
        if let Ok(mut c) = self.archive_counts.lock() {
            *c = None;
        }
        if let Ok(mut c) = self.caught_up.lock() {
            *c = None;
        }
        self.recount.notify_all();
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

fn count(reader: &Reader<'_>, position: &str) -> Result<Counted, String> {
    let started = Instant::now();
    let status = reader.status().map_err(|e| e.to_string())?;
    Ok(Counted {
        position: position.to_owned(),
        at: Instant::now(),
        took: started.elapsed(),
        status,
    })
}

#[cfg(test)]
mod tests;

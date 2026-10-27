//! Warm state: what the daemon keeps between requests so common answers
//! come without reloading or recomputing.
//!
//! - **Vectors**: every vector of the embedding key `near:` ranks by, in
//!   memory, reloaded after a commit stored or removed vectors.
//! - **Query vectors**: the provider's vector for each recent `near:` text,
//!   so a repeated or paged query neither pays nor waits for the provider
//!   again, and ranks by the same values.
//! - **Answers**: recent query pages, by request and read position. A page
//!   continuing a cursor is fixed by the cursor's clock and is good until
//!   the next write; a first page also reads the clock (`since:2h`, live
//!   states), so it is good for [`FIRST_PAGE_FOR`] at most.
//! - **Completion data**: unit and field names, values seen and recent
//!   session IDs, rebuilt on the first request after a write.
//! - **Archive counts** for `status`: counted by a worker of their own when
//!   the daemon starts and whenever a `status` finds them older than its
//!   read, while `status` answers from memory (no counts until the first
//!   count ends), so it stays fast during a long pass. Counts that take
//!   under 50 ms are simply taken again inside the request. A count of a large
//!   archive under load takes seconds; the worker leaves at least four
//!   times as long between counts, and `status` says which read position
//!   its counts are from.
//!
//! Every entry is checked against the read position (or the vector write
//! count) of the request that uses it, so a write is never answered from
//! before it.

use crate::embeddings::Provider;
use polylogue_store::Reader;
use polylogue_store::vectors::{self, Loaded, Similarity};
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a first page is answered again from memory.
pub const FIRST_PAGE_FOR: Duration = Duration::from_secs(5);
/// Recent `near:` texts whose query vectors are kept.
const QUERY_VECTORS: usize = 256;
/// Recent answers kept.
const ANSWERS: usize = 128;
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

#[derive(Clone)]
struct Answer {
    position: String,
    made: Instant,
    /// A first page, which read the clock.
    clocked: bool,
    value: Value,
}

/// Values completion offers for a query being typed.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Completion {
    pub units: Vec<String>,
    /// `name:` for every field of every unit.
    pub fields: Vec<String>,
    /// `field:value` for the values seen: repositories, origins,
    /// harnesses, tags, saved queries.
    pub values: Vec<String>,
    /// The most recently active sessions, newest first, with titles.
    pub sessions: Vec<(String, String)>,
}

/// Hits and misses since the daemon started, for `status`.
#[derive(Default)]
struct Counts {
    answers_reused: AtomicU64,
    answers_computed: AtomicU64,
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

pub struct Warm {
    vectors: Mutex<Option<Arc<Loaded>>>,
    archive_counts: Mutex<Option<Counted>>,
    /// Wakes the count worker: a request saw counts older than its read.
    recount: std::sync::Condvar,
    recount_wanted: std::sync::atomic::AtomicBool,
    query_vectors: Mutex<Recent<Arc<Vec<f32>>>>,
    answers: Mutex<Recent<Answer>>,
    completion: Mutex<Option<(String, Arc<Completion>)>>,
    counts: Counts,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            vectors: Mutex::new(None),
            archive_counts: Mutex::new(None),
            recount: std::sync::Condvar::new(),
            // The first count starts with the daemon.
            recount_wanted: std::sync::atomic::AtomicBool::new(true),
            query_vectors: Mutex::new(Recent::new(QUERY_VECTORS)),
            answers: Mutex::new(Recent::new(ANSWERS)),
            completion: Mutex::new(None),
            counts: Counts::default(),
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
    /// The loaded vectors of the embedding key `near:` ranks by, loading
    /// them when absent or stale. `None`: no vectors for this provider.
    pub fn vectors(
        &self,
        reader: &Reader<'_>,
        provider: &Provider,
    ) -> Result<Option<Arc<Loaded>>, String> {
        let prefix = provider.key_prefix();
        let mut held = read_lock(&self.vectors, reader)?;
        if let Some(loaded) = held.as_ref()
            && loaded.key.starts_with(&prefix)
            && loaded.writes == vectors::writes()
        {
            return Ok(Some(Arc::clone(loaded)));
        }
        let store = reader.vectors().map_err(|e| e.to_string())?;
        let Some(key) = store.key_with_prefix(&prefix).map_err(|e| e.to_string())? else {
            *held = None;
            return Ok(None);
        };
        let loaded = Arc::new(store.load(&key).map_err(|e| e.to_string())?);
        self.counts.vector_loads.fetch_add(1, Ordering::Relaxed);
        *held = Some(Arc::clone(&loaded));
        Ok(Some(loaded))
    }

    /// Every stored text's similarity to `text`, for `near:`.
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

    /// A recent answer to the same request at the same read position.
    pub fn answer(&self, request: &str, position: &str) -> Option<Value> {
        let found = self.answers.lock().ok()?.get(request)?;
        let fresh =
            found.position == position && (!found.clocked || found.made.elapsed() < FIRST_PAGE_FOR);
        if fresh {
            self.counts.answers_reused.fetch_add(1, Ordering::Relaxed);
            Some(found.value)
        } else {
            None
        }
    }

    pub fn keep_answer(&self, request: String, position: String, clocked: bool, value: &Value) {
        self.counts.answers_computed.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut answers) = self.answers.lock() {
            answers.put(
                request,
                Answer {
                    position,
                    made: Instant::now(),
                    clocked,
                    value: value.clone(),
                },
            );
        }
    }

    /// Completion data for the read position `position`, built on first
    /// use after a write.
    pub fn completion(
        &self,
        reader: &Reader<'_>,
        position: &str,
    ) -> Result<Arc<Completion>, String> {
        let mut held = read_lock(&self.completion, reader)?;
        if let Some((at, data)) = held.as_ref()
            && at == position
        {
            return Ok(Arc::clone(data));
        }
        let data = Arc::new(completion(reader).map_err(|e| e.to_string())?);
        *held = Some((position.to_owned(), Arc::clone(&data)));
        Ok(data)
    }

    /// The archive counts for `status`. When the last count was quick
    /// (under [`QUICK_COUNT`]) and is older than `position`, it counts again
    /// now, so a small archive's status is exact; otherwise it answers from
    /// memory and asks the worker for newer counts. `None` until the first
    /// count ends.
    pub fn counts(&self, reader: &Reader<'_>, position: &str) -> Option<Counted> {
        let held = self.archive_counts.lock().ok()?.clone();
        match held {
            Some(c) if c.position == position => Some(c),
            Some(c) if c.took < QUICK_COUNT => {
                let counted = count(reader, position).ok()?;
                if let Ok(mut held) = self.archive_counts.lock() {
                    *held = Some(counted.clone());
                }
                Some(counted)
            }
            held => {
                self.recount_wanted.store(true, Ordering::Release);
                self.recount.notify_one();
                held
            }
        }
    }

    /// The count worker: counts again when a `status` asked for newer
    /// counts, leaving at least four times the last count's duration
    /// between counts, until `stop` says so.
    pub fn count_worker(
        &self,
        reads: &polylogue_store::ReadPool,
        stop: &std::sync::mpsc::Receiver<()>,
    ) {
        loop {
            match stop.try_recv() {
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                _ => return,
            }
            let gap = {
                let Ok(held) = self.archive_counts.lock() else {
                    return;
                };
                let gap = held.as_ref().map_or(Duration::ZERO, |c| {
                    (c.took * 4)
                        .max(Duration::from_millis(200))
                        .saturating_sub(c.at.elapsed())
                });
                let wanted = self.recount_wanted.load(Ordering::Acquire);
                let wait = if gap.is_zero() && wanted {
                    Duration::ZERO
                } else if gap.is_zero() {
                    Duration::from_secs(1)
                } else {
                    gap
                };
                let Ok((held, _)) = self.recount.wait_timeout(held, wait) else {
                    return;
                };
                drop(held);
                gap
            };
            if !gap.is_zero() || !self.recount_wanted.swap(false, Ordering::AcqRel) {
                continue;
            }
            let Ok(reader) = reads.reader() else {
                continue;
            };
            let Ok(position) = reader.change_position().map(|p| p.to_string()) else {
                continue;
            };
            if let Ok(counted) = count(&reader, &position)
                && let Ok(mut held) = self.archive_counts.lock()
            {
                *held = Some(counted);
            }
        }
    }

    /// Drops everything; the next requests rebuild what they use.
    pub fn clear(&self) {
        if let Ok(mut v) = self.vectors.lock() {
            *v = None;
        }
        if let Ok(mut q) = self.query_vectors.lock() {
            q.clear();
        }
        if let Ok(mut a) = self.answers.lock() {
            a.clear();
        }
        if let Ok(mut c) = self.completion.lock() {
            *c = None;
        }
        if let Ok(mut c) = self.archive_counts.lock() {
            *c = None;
        }
        self.recount.notify_all();
    }

    /// What `status` shows.
    pub fn summary(&self) -> Value {
        let vectors = self.vectors.lock().ok().and_then(|v| {
            v.as_ref()
                .map(|l| serde_json::json!({"key": l.key, "vectors": l.vectors()}))
        });
        serde_json::json!({
            "vectors": vectors,
            "vector_loads": self.counts.vector_loads.load(Ordering::Relaxed),
            "query_vectors_reused": self.counts.query_vectors_reused.load(Ordering::Relaxed),
            "answers_reused": self.counts.answers_reused.load(Ordering::Relaxed),
            "answers_computed": self.counts.answers_computed.load(Ordering::Relaxed),
        })
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

/// Values completion offers, read from the index and the archive.
fn completion(reader: &Reader<'_>) -> Result<Completion, polylogue_store::StoreError> {
    use polylogue_query::ast::Unit;
    let mut data = Completion {
        units: Unit::all().map(|u| u.name().to_owned()).collect(),
        ..Completion::default()
    };
    for unit in Unit::all() {
        for name in polylogue_query::fields::names(unit) {
            let field = format!("{name}:");
            if !data.fields.contains(&field) {
                data.fields.push(field);
            }
        }
    }
    data.fields.sort();
    let column = |sql: &str,
                  field: &str,
                  out: &mut Vec<String>|
     -> Result<(), polylogue_store::StoreError> {
        for row in reader.query(sql, &[])? {
            if let Some(v) = row["v"].as_str().filter(|v| !v.is_empty()) {
                out.push(format!("{field}:{}", quoted(v)));
            }
        }
        Ok(())
    };
    let mut values = Vec::new();
    column(
        "SELECT DISTINCT repo AS v FROM session WHERE repo IS NOT NULL ORDER BY repo",
        "repo",
        &mut values,
    )?;
    column(
        "SELECT DISTINCT harness AS v FROM session WHERE harness IS NOT NULL ORDER BY harness",
        "harness",
        &mut values,
    )?;
    column(
        &format!(
            "SELECT DISTINCT n.key AS v FROM a.note n WHERE n.kind = 'tag' AND n.key IS NOT NULL \
             AND {} AND NOT {} ORDER BY n.key",
            polylogue_query::fields::current_note("n"),
            polylogue_query::fields::withdrawn_note("n"),
        ),
        "tag",
        &mut values,
    )?;
    column(
        &format!(
            "SELECT DISTINCT n.key AS v FROM a.note n WHERE n.kind = 'saved_query' AND n.key IS NOT NULL \
             AND {} AND NOT {} ORDER BY n.key",
            polylogue_query::fields::current_note("n"),
            polylogue_query::fields::withdrawn_note("n"),
        ),
        "query",
        &mut values,
    )?;
    for origin in polylogue_sources::origin::ORIGINS {
        values.push(format!("origin:{}", origin.name));
    }
    data.values = values;
    for row in reader.query(
        "SELECT id, coalesce(title, derived_title, '') AS title FROM session \
         WHERE kind = 'main' AND stub = 0 \
         ORDER BY last_activity_at IS NULL, last_activity_at DESC LIMIT 50",
        &[],
    )? {
        if let Some(id) = row["id"].as_str() {
            data.sessions.push((
                id.to_owned(),
                row["title"].as_str().unwrap_or("").to_owned(),
            ));
        }
    }
    Ok(data)
}

/// A value as the query language reads it back: bare when it is one plain
/// word, else in double quotes.
fn quoted(v: &str) -> String {
    if v.chars()
        .all(|c| c.is_alphanumeric() || "-_./@".contains(c))
    {
        v.to_owned()
    } else {
        format!("\"{}\"", v.replace('"', "\\\""))
    }
}

/// The completions of the word being typed: field names, values of the
/// field it names, unit names and session IDs, each starting with it.
pub fn completions(data: &Completion, word: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut add = |value: &str, help: &str| {
        if value.starts_with(word) && !out.iter().any(|(v, _)| v == value) {
            out.push((value.to_owned(), help.to_owned()));
        }
    };
    if word.contains(':') {
        for v in &data.values {
            add(v, "");
        }
        for (id, title) in &data.sessions {
            add(id, title);
        }
    } else {
        for u in &data.units {
            add(u, "unit");
        }
        for f in &data.fields {
            add(f, "field");
        }
        for (id, title) in &data.sessions {
            add(id, title);
        }
    }
    out
}

#[cfg(test)]
mod tests;

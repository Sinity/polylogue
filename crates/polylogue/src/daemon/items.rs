//! Session item indexes held in daemon memory between reads. A cached index
//! is checked against the store's two counters on every use, in O(1): an
//! append loads only the new entries, a rewrite loads the session again.
use polylogue_model::{
    SessionId,
    read::{EntryKey, EntryKind, FoldProfile, IndexEntry, RunningSums, Segment, visible_entries},
};
use polylogue_store::read::limit::ReadLimit;
use polylogue_store::{Reader, items::ItemVersions};
use std::collections::{HashMap, VecDeque};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// One session's entries and where each key sits.
#[derive(Debug)]
pub(crate) struct CachedIndex {
    pub session: SessionId,
    pub session_pk: i64,
    pub versions: ItemVersions,
    pub entries: Vec<IndexEntry>,
    pub places: HashMap<EntryKey, usize>,
    views: Mutex<Vec<(FoldProfile, Arc<RunningSums>)>>,
    /// Memory held by `entries` and `places`, measured when they change.
    held_bytes: usize,
}
impl CachedIndex {
    fn new(
        session: SessionId,
        session_pk: i64,
        versions: ItemVersions,
        entries: Vec<IndexEntry>,
    ) -> Self {
        let places = entries
            .iter()
            .enumerate()
            .map(|(at, entry)| (entry.key.clone(), at))
            .collect();
        let mut index = Self {
            session,
            session_pk,
            versions,
            entries,
            places,
            views: Mutex::new(Vec::new()),
            held_bytes: 0,
        };
        index.held_bytes = index.measure();
        index
    }
    fn append(&mut self, reader: &Reader<'_>, now: ItemVersions) -> polylogue_store::Result<()> {
        // The last message can grow. Earlier entries and their key positions
        // stay in place; only that final unit and the appended units change.
        let from = self
            .entries
            .iter()
            .rposition(|entry| matches!(entry.kind, EntryKind::Header | EntryKind::Event))
            .unwrap_or(0);
        let tail = reader.item_entries_from(self.session_pk, from as u64)?;
        for entry in &self.entries[from..] {
            self.places.remove(&entry.key);
        }
        self.entries.truncate(from);
        self.entries.extend(tail);
        for (at, entry) in self.entries.iter().enumerate().skip(from) {
            self.places.insert(entry.key.clone(), at);
        }
        for (profile, sums) in self.views.get_mut().unwrap_or_else(|e| e.into_inner()) {
            // An in-flight older read owns its snapshot. Ordinarily this Arc
            // is unique, and appending does not copy its unchanged prefix.
            let sums = Arc::make_mut(sums);
            let cut = sums.shown.partition_point(|shown| shown.entry < from);
            sums.shown.truncate(cut);
            sums.sums.truncate(cut + 1);
            let base = sums.total();
            let tail = visible_entries(
                &[Segment {
                    entries: &self.entries[from..],
                    inherited: false,
                }],
                profile,
            );
            sums.shown.extend(tail.shown.into_iter().map(|mut shown| {
                shown.entry += from;
                shown
            }));
            sums.sums.extend(
                tail.sums
                    .into_iter()
                    .skip(1)
                    .map(|sum| base.saturating_add(sum)),
            );
        }
        self.versions = now;
        self.held_bytes = self.measure();
        Ok(())
    }
    /// The heap memory of the entries and the key map: their tables at
    /// capacity and every string they own.
    fn measure(&self) -> usize {
        use std::mem::size_of;
        let entries = self.entries.capacity() * size_of::<IndexEntry>()
            + self.entries.iter().map(entry_heap_bytes).sum::<usize>();
        // A hash table slot holds the pair and one control byte.
        let places = self.places.capacity() * (size_of::<(EntryKey, usize)>() + 1)
            + self.places.keys().map(key_heap_bytes).sum::<usize>();
        entries + places
    }
    fn bytes(&self) -> usize {
        let views = self.views.lock().unwrap_or_else(|e| e.into_inner());
        self.held_bytes
            + views
                .iter()
                .map(|(_, sums)| {
                    sums.shown.capacity() * std::mem::size_of::<polylogue_model::read::Shown>()
                        + sums.sums.capacity() * std::mem::size_of::<u64>()
                })
                .sum::<usize>()
    }
}
impl Clone for CachedIndex {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            session_pk: self.session_pk,
            versions: self.versions,
            entries: self.entries.clone(),
            places: self.places.clone(),
            views: Mutex::new(self.views.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            held_bytes: self.held_bytes,
        }
    }
}

/// Heap bytes an entry key owns outside its own struct.
fn key_heap_bytes(key: &EntryKey) -> usize {
    match key {
        EntryKey::Message { message, .. } => {
            message.origin().as_str().len() + message.native().as_str().len()
        }
        EntryKey::Event { event } => event.capacity(),
    }
}

/// Heap bytes an entry owns outside its own struct.
fn entry_heap_bytes(entry: &IndexEntry) -> usize {
    let text = |value: &Option<String>| value.as_ref().map_or(0, String::capacity);
    key_heap_bytes(&entry.key)
        + text(&entry.target)
        + text(&entry.call_id)
        + text(&entry.event_kind)
        + entry.cleared.capacity() * std::mem::size_of::<String>()
        + entry.cleared.iter().map(String::capacity).sum::<usize>()
}

/// Memory the cache may hold. Host tuning belongs in configuration; this is
/// the default a daemon starts with.
const CACHE_BYTES: usize = 512 << 20;

#[derive(Debug, Default)]
pub(crate) struct ItemCache {
    held: Mutex<VecDeque<Arc<CachedIndex>>>,
    /// Sessions being loaded now. A second read of the same session waits
    /// for that load instead of loading the whole session again.
    loading: Mutex<HashMap<SessionId, Arc<Load>>>,
    #[cfg(test)]
    loads: AtomicUsize,
    #[cfg(test)]
    pause: Mutex<Option<Arc<tests::Pause>>>,
}

/// One load in progress, the readers waiting for it, and how it ended.
#[derive(Debug, Default)]
struct Load {
    state: Mutex<LoadState>,
    ended: Condvar,
    #[cfg(test)]
    waiting: AtomicUsize,
}

#[derive(Debug, Default)]
enum LoadState {
    #[default]
    Running,
    /// The session's index, or `None` when the session is not stored.
    Loaded(Option<Arc<CachedIndex>>),
    /// The load stopped with an error; each waiter loads for itself.
    Failed,
}

/// Counts a waiting reader for as long as it waits.
#[cfg(test)]
struct Waiting<'a>(&'a AtomicUsize);
#[cfg(test)]
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// How often a waiting read checks its own deadline and cancellation.
const WAIT_CHECK: Duration = Duration::from_millis(50);

impl ItemCache {
    /// The current index of a session, or `None` when it is not stored.
    pub fn get(
        &self,
        reader: &Reader<'_>,
        session: &SessionId,
    ) -> polylogue_store::Result<Option<Arc<CachedIndex>>> {
        let cached = {
            let held = self.held.lock().unwrap_or_else(|error| error.into_inner());
            held.iter().find(|index| &index.session == session).cloned()
        };
        let fresh = match cached {
            Some(cached) => {
                let Some(now) = reader.item_versions_of(cached.session_pk)? else {
                    return Ok(None);
                };
                if now == cached.versions {
                    self.touch(&cached);
                    return Ok(Some(cached));
                }
                if now.rewrite_version == cached.versions.rewrite_version {
                    // Release the cache's ownership before asking for mutable
                    // access. A concurrent read retains its older snapshot.
                    {
                        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
                        held.retain(|other| !Arc::ptr_eq(other, &cached));
                    }
                    let mut index = Arc::unwrap_or_clone(cached);
                    index.append(reader, now)?;
                    index
                } else {
                    return self.load_once(reader, session)?.map(Some).ok_or_else(|| {
                        polylogue_store::StoreError::NotFound(format!("session {session}"))
                    });
                }
            }
            None => return self.load_once(reader, session),
        };
        let fresh = Arc::new(fresh);
        self.insert(Arc::clone(&fresh));
        Ok(Some(fresh))
    }

    /// Full loads started since the cache was created.
    #[cfg(test)]
    fn loads(&self) -> usize {
        self.loads.load(Ordering::Relaxed)
    }

    /// Loads a whole session, or waits for a load of it already in progress.
    /// A waiting read still stops at its own deadline or cancellation; if the
    /// load it waited on fails, it loads the session itself.
    fn load_once(
        &self,
        reader: &Reader<'_>,
        session: &SessionId,
    ) -> polylogue_store::Result<Option<Arc<CachedIndex>>> {
        loop {
            let (load, leader) = {
                let mut loading = self.loading.lock().unwrap_or_else(|e| e.into_inner());
                match loading.get(session) {
                    Some(load) => (Arc::clone(load), false),
                    None => {
                        let load = Arc::new(Load::default());
                        loading.insert(session.clone(), Arc::clone(&load));
                        (load, true)
                    }
                }
            };
            if leader {
                return self.lead(reader, session, &load);
            }
            if let Some(done) = self.follow(&load)? {
                return Ok(done);
            }
        }
    }

    fn lead(
        &self,
        reader: &Reader<'_>,
        session: &SessionId,
        shared: &Load,
    ) -> polylogue_store::Result<Option<Arc<CachedIndex>>> {
        #[cfg(test)]
        self.loads.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        if let Some(pause) = self.pause.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            pause.hold();
        }
        let result = load(reader, session).map(|index| index.map(Arc::new));
        if let Ok(Some(index)) = &result {
            self.insert(Arc::clone(index));
        }
        *shared.state.lock().unwrap_or_else(|e| e.into_inner()) = match &result {
            Ok(index) => LoadState::Loaded(index.clone()),
            Err(_) => LoadState::Failed,
        };
        self.loading
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(session);
        shared.ended.notify_all();
        result
    }

    /// Waits for another read's load. `Ok(Some(_))` is its result; `Ok(None)`
    /// means it failed and this read should load the session itself.
    fn follow(&self, shared: &Load) -> polylogue_store::Result<Option<Option<Arc<CachedIndex>>>> {
        #[cfg(test)]
        shared.waiting.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        let _waiting = Waiting(&shared.waiting);
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &*state {
                LoadState::Loaded(index) => return Ok(Some(index.clone())),
                LoadState::Failed => return Ok(None),
                LoadState::Running => {}
            }
            if let Some(limit) = ReadLimit::current() {
                limit.check()?;
            }
            state = shared
                .ended
                .wait_timeout(state, WAIT_CHECK)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// A physical session's shown positions and costs under this profile.
    /// Preparation is shared by continuations and charged to this cache.
    pub fn visible(&self, index: &CachedIndex, profile: FoldProfile) -> Arc<RunningSums> {
        let prepared = {
            let mut views = index.views.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, sums)) = views.iter().find(|(held, _)| held == &profile) {
                Arc::clone(sums)
            } else {
                let sums = Arc::new(visible_entries(
                    &[Segment {
                        entries: &index.entries,
                        inherited: false,
                    }],
                    &profile,
                ));
                views.push((profile, Arc::clone(&sums)));
                sums
            }
        };
        self.trim();
        prepared
    }

    fn trim(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        trim(&mut held);
    }

    fn touch(&self, index: &Arc<CachedIndex>) {
        let mut held = self.held.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(at) = held.iter().position(|other| Arc::ptr_eq(other, index)) {
            let index = held.remove(at).expect("present");
            held.push_back(index);
        }
    }

    fn insert(&self, index: Arc<CachedIndex>) {
        let mut held = self.held.lock().unwrap_or_else(|error| error.into_inner());
        held.retain(|other| other.session != index.session);
        held.push_back(index);
        trim(&mut held);
    }
}

fn trim(held: &mut VecDeque<Arc<CachedIndex>>) {
    let mut total: usize = held.iter().map(|index| index.bytes()).sum();
    while total > CACHE_BYTES && held.len() > 1 {
        let oldest = held.pop_front().expect("more than one held");
        total -= oldest.bytes();
    }
}

fn load(reader: &Reader<'_>, session: &SessionId) -> polylogue_store::Result<Option<CachedIndex>> {
    Ok(reader.item_index(session)?.map(|stored| {
        CachedIndex::new(
            session.clone(),
            stored.session_pk,
            ItemVersions {
                append_seq: stored.append_seq,
                rewrite_version: stored.rewrite_version,
            },
            stored.entries,
        )
    }))
}

#[cfg(test)]
mod tests;

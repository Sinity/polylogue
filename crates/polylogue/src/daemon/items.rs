//! Session item indexes held in daemon memory between reads. A cached index
//! is checked against the store's two counters on every use, in O(1): an
//! append loads only the new entries, a rewrite loads the session again.
use polylogue_model::{
    SessionId,
    read::{EntryKey, EntryKind, FoldProfile, IndexEntry, RunningSums, Segment, visible_entries},
};
use polylogue_store::{Reader, items::ItemVersions};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// One session's entries and where each key sits.
#[derive(Debug)]
pub(crate) struct CachedIndex {
    pub session: SessionId,
    pub session_pk: i64,
    pub versions: ItemVersions,
    pub entries: Vec<IndexEntry>,
    pub places: HashMap<EntryKey, usize>,
    views: Mutex<Vec<(FoldProfile, Arc<RunningSums>)>>,
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
        Self {
            session,
            session_pk,
            versions,
            entries,
            places,
            views: Mutex::new(Vec::new()),
        }
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
        Ok(())
    }
    fn bytes(&self) -> usize {
        let views = self.views.lock().unwrap_or_else(|e| e.into_inner());
        self.entries.len() * ENTRY_BYTES
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
        }
    }
}

/// An estimate of one entry's memory, for the cache's size bound.
const ENTRY_BYTES: usize = 192;
/// Memory the cache may hold. Host tuning belongs in configuration; this is
/// the default a daemon starts with.
const CACHE_BYTES: usize = 512 << 20;

#[derive(Debug, Default)]
pub(crate) struct ItemCache {
    held: Mutex<VecDeque<Arc<CachedIndex>>>,
}

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
                    load(reader, session)?.ok_or(polylogue_store::StoreError::NotFound(format!(
                        "session {session}"
                    )))?
                }
            }
            None => match load(reader, session)? {
                Some(index) => index,
                None => return Ok(None),
            },
        };
        let fresh = Arc::new(fresh);
        self.insert(Arc::clone(&fresh));
        Ok(Some(fresh))
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

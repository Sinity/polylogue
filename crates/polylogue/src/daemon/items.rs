//! Session item indexes held in daemon memory between reads. A cached index
//! is checked against the store's two counters on every use, in O(1): an
//! append loads only the new entries, a rewrite loads the session again.
use polylogue_model::{
    SessionId,
    read::{EntryKey, EntryKind, IndexEntry},
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
        }
    }
    fn bytes(&self) -> usize {
        self.entries.len() * ENTRY_BYTES
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
                    // An append: reload from the last message or event, which
                    // may have grown, and keep everything before it.
                    let from = cached
                        .entries
                        .iter()
                        .rposition(|entry| {
                            matches!(entry.kind, EntryKind::Header | EntryKind::Event)
                        })
                        .unwrap_or(0);
                    let mut entries = cached.entries[..from].to_vec();
                    entries.extend(reader.item_entries_from(cached.session_pk, from as u64)?);
                    CachedIndex::new(session.clone(), cached.session_pk, now, entries)
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
        let mut total: usize = held.iter().map(|index| index.bytes()).sum();
        while total > CACHE_BYTES && held.len() > 1 {
            let oldest = held.pop_front().expect("more than one held");
            total -= oldest.bytes();
        }
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

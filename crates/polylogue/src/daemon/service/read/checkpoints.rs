//! Bounded private source state; eviction changes work, never the read answer.
use polylogue_render::model_pages::Checkpoint;
use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

const ENTRIES: usize = 64;
type Identity = (String, (u64, u64));

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Key {
    pub binding: String,
    pub content: String,
    pub generation: String,
    pub index: (u64, u64),
}

#[derive(Default)]
pub(crate) struct Cache {
    state: Mutex<State>,
}
#[derive(Default)]
struct State {
    identity: Option<Identity>,
    epoch: u64,
    entries: VecDeque<(Key, Checkpoint)>,
    #[cfg(test)]
    hits: usize,
}
impl State {
    fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.entries.clear();
    }
}
impl Cache {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub(crate) fn epoch(&self) -> u64 {
        self.state().epoch
    }
    pub(crate) fn clear(&self) {
        self.state().clear();
    }
    #[cfg(test)]
    pub(crate) fn test_counts(&self) -> (usize, usize) {
        let state = self.state();
        (state.entries.len(), state.hits)
    }
    pub(crate) fn mutation(&self) -> Mutation<'_> {
        self.clear();
        Mutation(self)
    }
    pub(super) fn get(&self, key: &Key, epoch: u64, message: usize) -> Option<Checkpoint> {
        let mut state = self.state();
        let identity = (key.generation.clone(), key.index);
        if state
            .identity
            .as_ref()
            .is_some_and(|previous| previous != &identity)
        {
            state.clear();
        }
        state.identity = Some(identity);
        if epoch != state.epoch {
            return None;
        }
        let found = state
            .entries
            .iter()
            .filter(|(entry, checkpoint)| entry == key && checkpoint.message() <= message)
            .max_by_key(|(_, checkpoint)| checkpoint.message())
            .map(|(_, checkpoint)| checkpoint.clone());
        #[cfg(test)]
        if found.is_some() {
            state.hits += 1;
        }
        found
    }
    pub(super) fn put(&self, key: Key, epoch: u64, checkpoint: Checkpoint) {
        let mut state = self.state();
        if epoch != state.epoch
            || state.identity.as_ref() != Some(&(key.generation.clone(), key.index))
        {
            return;
        }
        if let Some(at) = state.entries.iter().position(|(entry, previous)| {
            entry == &key && previous.message() == checkpoint.message()
        }) {
            state.entries.remove(at);
        }
        state.entries.push_back((key, checkpoint));
        while state.entries.len() > ENTRIES {
            state.entries.pop_front();
        }
    }
}

pub(crate) struct Mutation<'a>(&'a Cache);
impl Drop for Mutation<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

#[cfg(test)]
mod tests;

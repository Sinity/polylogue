//! Watch source directories and the nearest existing parents of missing ones.
//! Parents are watched nonrecursively to exclude unrelated state writes.
use crate::{Config, LoopMsg, collect};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::Sender;

pub(super) struct Roots {
    watcher: RecommendedWatcher,
    active: BTreeMap<PathBuf, bool>,
}

impl Roots {
    pub fn new(config: &Config, wake: Sender<LoopMsg>) -> notify::Result<Self> {
        let watcher = notify::recommended_watcher(move |result: notify::Result<notify::Event>| {
            if result.as_ref().is_ok_and(collect::requires_scan) {
                let _ = wake.send(LoopMsg::Changed);
            }
        })?;
        let mut roots = Self {
            watcher,
            active: BTreeMap::new(),
        };
        roots.refresh(config)?;
        Ok(roots)
    }

    pub fn refresh(&mut self, config: &Config) -> notify::Result<()> {
        let wanted: BTreeMap<_, _> = collect::roots(config)
            .into_iter()
            .map(|root| (root.path, root.recursive))
            .collect();
        for (path, recursive) in &wanted {
            if self.active.get(path) == Some(recursive) {
                continue;
            }
            let mode = if *recursive {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };
            self.watcher.watch(path, mode)?;
        }
        for path in self.active.keys() {
            if !wanted.contains_key(path)
                && let Err(error) = self.watcher.unwatch(path)
                && path.exists()
            {
                return Err(error);
            }
        }
        self.active = wanted;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

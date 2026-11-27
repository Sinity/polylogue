//! Watch source directories and the nearest existing parents of missing ones.
//! Parents are watched nonrecursively to exclude unrelated state writes.
use crate::{Config, LoopMsg, collect};
use crossbeam_channel::Sender;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use polylogue_model::LockExt;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// What the watcher saw since the last `take`.
#[derive(Default)]
pub(super) struct Changes {
    /// Paths to rescan.
    pub paths: BTreeSet<PathBuf>,
    /// A full scan of every source is needed.
    pub full: bool,
    /// Watcher errors, which the pass forwards into its report.
    pub errors: Vec<polylogue_model::Failure>,
}

impl Changes {
    fn record(&mut self, result: notify::Result<notify::Event>) {
        match result {
            Ok(event) if event.need_rescan() => self.full = true,
            Ok(event) if collect::requires_scan(&event) => {
                if event.paths.is_empty() {
                    self.full = true;
                } else {
                    self.paths.extend(event.paths);
                }
            }
            Ok(_) => {}
            Err(error) => {
                self.full = true;
                self.errors.push(polylogue_model::Failure::new(
                    "source watch",
                    &crate::EngineError::from(error),
                ));
            }
        }
    }

    fn pending(&self) -> bool {
        self.full || !self.paths.is_empty()
    }
}

pub(super) struct Roots {
    watcher: RecommendedWatcher,
    active: BTreeMap<PathBuf, bool>,
    changes: Arc<Mutex<Changes>>,
    retry: bool,
}

impl Roots {
    pub fn new(config: &Config, wake: Sender<LoopMsg>) -> notify::Result<Self> {
        let changes = Arc::new(Mutex::new(Changes::default()));
        let recorded = Arc::clone(&changes);
        let watcher = notify::recommended_watcher(move |result: notify::Result<notify::Event>| {
            let mut changes = recorded.locked();
            let pending = changes.pending();
            changes.record(result);
            if !pending && changes.pending() {
                let _ = wake.send(LoopMsg::Changed);
            }
        })?;
        let mut roots = Self {
            watcher,
            active: BTreeMap::new(),
            changes,
            retry: false,
        };
        if let Err(error) = roots.refresh(config) {
            roots.changes.locked().record(Err(error));
        }
        Ok(roots)
    }

    pub fn take(&mut self) -> Changes {
        let mut changes = std::mem::take(&mut *self.changes.locked());
        // Root metadata changes need a fresh coverage scan, including
        // permissions changing while no transcript changed.
        changes.full |= changes
            .paths
            .iter()
            .any(|path| self.active.contains_key(path));
        // A watched directory that was deleted loses its watch inside notify,
        // and a recreated one is not watched again unless the next refresh
        // watches it afresh.
        self.retry |= changes.full
            || changes
                .paths
                .iter()
                .any(|path| self.active.contains_key(path));
        changes
    }

    pub fn refresh(&mut self, config: &Config) -> notify::Result<()> {
        let wanted: BTreeMap<_, _> = collect::roots(config)
            .into_iter()
            .map(|root| (root.path, root.recursive))
            .collect();
        for (path, recursive) in &wanted {
            if !self.retry && self.active.get(path) == Some(recursive) {
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
            // A deleted root has already lost its watch, so only an unwatch
            // failure on a path that still exists is an error.
            if !wanted.contains_key(path)
                && let Err(error) = self.watcher.unwatch(path)
                && path.exists()
            {
                return Err(error);
            }
        }
        self.active = wanted;
        self.retry = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

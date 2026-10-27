//! Watch source directories and the nearest existing parents of missing ones.
//! Parents are watched nonrecursively to exclude unrelated state writes.
use crate::{Config, LoopMsg, collect};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(super) struct Changes {
    pub paths: BTreeSet<PathBuf>,
    pub full: bool,
    pub errors: Vec<String>,
}

impl Changes {
    fn record(&mut self, result: notify::Result<notify::Event>) {
        match result {
            Ok(event) if event.need_rescan() => self.full = true,
            Ok(event) if collect::requires_scan(&event) => self.paths.extend(event.paths),
            Ok(_) => {}
            Err(error) => {
                self.full = true;
                self.errors.push(error.to_string());
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
            let mut changes = recorded.lock().expect("watch changes");
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
            roots
                .changes
                .lock()
                .expect("watch changes")
                .record(Err(error));
        }
        Ok(roots)
    }

    pub fn take(&mut self) -> Changes {
        let changes = std::mem::take(&mut *self.changes.lock().expect("watch changes"));
        self.retry |= changes.full;
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

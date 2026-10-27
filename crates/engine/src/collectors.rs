//! Collector workers belong to the engine and finish before it closes.
use polylogue_sources::origin::{Collector, CollectorPaths};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{JoinHandle, spawn};

#[derive(Default)]
pub(crate) struct Collectors {
    stop: Arc<AtomicBool>,
    tasks: HashMap<&'static str, JoinHandle<Vec<String>>>,
}

impl Collectors {
    pub fn run(
        &mut self,
        name: &'static str,
        collector: &'static Collector,
        paths: &CollectorPaths<'_>,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if let Some(task) = self.tasks.get(name) {
            if !task.is_finished() {
                return errors;
            }
            let task = self.tasks.remove(name).expect("known collector");
            errors.extend(
                task.join()
                    .unwrap_or_else(|_| vec!["collector panicked".into()]),
            );
        }
        if collector.background {
            let home = paths.home.to_path_buf();
            let state_dir = paths.state_dir.to_path_buf();
            let spool = paths.spool.to_path_buf();
            let stop = Arc::clone(&self.stop);
            self.tasks.insert(
                name,
                spawn(move || {
                    (collector.sync)(
                        &CollectorPaths {
                            home: &home,
                            state_dir: &state_dir,
                            spool: &spool,
                        },
                        &stop,
                    )
                }),
            );
        } else {
            errors.extend((collector.sync)(paths, &self.stop));
        }
        errors
            .into_iter()
            .map(|error| format!("{name} collector: {error}"))
            .collect()
    }
}

impl Drop for Collectors {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for (_, task) in self.tasks.drain() {
            let _ = task.join();
        }
    }
}

#[cfg(test)]
mod tests;

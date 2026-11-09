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

impl crate::Engine {
    /// The collector cursor follows archived bytes, independently of parsing.
    /// Primary reads expose confirmed chains, so durable preparation alone
    /// cannot release the collector's pending revision.
    pub(crate) fn settle_collector(
        &self,
        origin: &str,
        collector: &Collector,
        paths: &CollectorPaths<'_>,
    ) -> Result<(), String> {
        let Some(settle) = collector.settle else {
            return Ok(());
        };
        settle(paths, &|path, hash, length| {
            let reader = self.reads.reader().map_err(|e| e.to_string())?;
            let known = reader
                .known_watched_input_at(&self.config.machine, path.as_os_str().as_encoded_bytes())
                .map_err(|e| e.to_string())?;
            let Some(known) = known else { return Ok(false) };
            if known.format != origin || known.stored != length || known.chain.is_none() {
                return Ok(false);
            }
            let facts = match std::fs::metadata(path) {
                Ok(metadata) => polylogue_store::FileFacts::of(&metadata),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(error.to_string()),
            };
            if known.facts != Some(facts) || facts.size != length {
                return Ok(false);
            }
            let chain = reader
                .stored_chain(
                    &known.input_id,
                    known.chain.expect("confirmed chain"),
                    length,
                )
                .map_err(|e| e.to_string())?;
            let mut bytes = chain.reader(Arc::clone(&self.blobs));
            let mut actual = blake3::Hasher::new();
            let read = std::io::copy(&mut bytes, &mut actual).map_err(|e| e.to_string())?;
            Ok(read == length && polylogue_model::BlobHash(*actual.finalize().as_bytes()) == hash)
        })
    }
}

//! Collector workers belong to the engine and finish before it closes.
use polylogue_sources::origin::{Collector, CollectorPaths};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;

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
            let archive = paths.archive.to_path_buf();
            let spool = paths.spool.to_path_buf();
            let stop = Arc::clone(&self.stop);
            let task = polylogue_model::spawn_named(format!("collect-{name}"), move || {
                (collector.sync)(
                    &CollectorPaths {
                        home: &home,
                        archive: &archive,
                        spool: &spool,
                    },
                    &stop,
                )
            });
            match task {
                Ok(task) => {
                    self.tasks.insert(name, task);
                }
                Err(error) => errors.push(format!("could not start its thread: {error}")),
            }
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

impl crate::Engine {
    /// The collector cursor follows archived bytes, independently of parsing.
    /// Primary reads expose confirmed chains, so a saved prepared write alone
    /// cannot release the collector's pending revision.
    pub(crate) fn advance_collector(
        &self,
        origin: &str,
        collector: &Collector,
        paths: &CollectorPaths<'_>,
    ) -> Result<(), String> {
        let Some(advance) = collector.advance else {
            return Ok(());
        };
        advance(paths, &|path, hash, length| {
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
            if !known
                .facts
                .is_some_and(|stored| stored.unchanged_as(&facts))
                || facts.size != length
            {
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

#[cfg(test)]
mod tests;

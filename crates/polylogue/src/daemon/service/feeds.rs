//! Completed batches prepared once per position, outside the cache lock.
use super::*;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

type Prepared = OnceLock<Result<tempfile::NamedTempFile, Response>>;
#[derive(Default)]
struct Files {
    position: String,
    files: HashMap<Option<String>, Arc<Prepared>>,
}
#[derive(Default)]
pub(super) struct Cache(Mutex<Files>);

impl Cache {
    pub fn read(&self, d: &Daemon, since: Option<&str>) -> Result<std::fs::File, Response> {
        let reader = d.reads.reader().map_err(|e| store_error(&e))?;
        let position = reader
            .change_position()
            .map_err(|e| store_error(&e))?
            .to_string();
        let entry = {
            let mut cache = self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cache.position != position {
                cache.files.clear();
                cache.position = position;
            }
            Arc::clone(cache.files.entry(since.map(str::to_owned)).or_default())
        };
        let file = entry
            .get_or_init(|| {
                let file = tempfile::NamedTempFile::new_in(d.archive.staging_dir())
                    .map_err(|e| error("io", e))?;
                let output = file.reopen().map_err(|e| error("io", e))?;
                super::super::query::feed::prepare_with(&reader, since, &d.query_clock()?, output)
                    .map_err(|f| Response::Error {
                        kind: f.kind,
                        message: f.message,
                        detail: f.detail,
                    })?;
                Ok(file)
            })
            .as_ref()
            .map_err(Clone::clone)?;
        file.reopen().map_err(|e| error("io", e))
    }
}

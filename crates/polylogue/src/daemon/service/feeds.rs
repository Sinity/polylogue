//! Completed batches shared by subscribers at the same position.
use super::*;
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct Cache {
    position: String,
    files: HashMap<(Option<String>, bool), tempfile::NamedTempFile>,
}

impl Cache {
    pub fn read(
        &mut self,
        d: &Daemon,
        since: Option<&str>,
        web: bool,
    ) -> Result<std::fs::File, Response> {
        let reader = d.reads.reader().map_err(|e| store_error(&e))?;
        let position = reader
            .change_position()
            .map_err(|e| store_error(&e))?
            .to_string();
        if position != self.position {
            self.files.clear();
            self.position = position;
        }
        let key = (since.map(str::to_owned), web);
        if !self.files.contains_key(&key) {
            let file =
                tempfile::NamedTempFile::new_in(&d.archive.root).map_err(|e| error("io", e))?;
            let output = file.reopen().map_err(|e| error("io", e))?;
            super::super::query::feed::prepare_with(&reader, since, now_ms(), output, |value| {
                if web {
                    super::super::request::output_values(value, |text| {
                        crate::web::access::scrub(text, &d.archive.root)
                    });
                }
            })
            .map_err(|f| {
                Response::Error {
                    kind: f.kind,
                    message: f.message,
                    detail: f.detail,
                }
                .masked()
            })?;
            self.files.insert(key.clone(), file);
        }
        self.files[&key].reopen().map_err(|e| error("io", e))
    }
}

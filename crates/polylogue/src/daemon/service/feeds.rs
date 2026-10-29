//! Finite feed batches use the accepting operation's captured evaluation clock.
use super::*;

#[derive(Default)]
pub(super) struct Cache;

impl Cache {
    pub fn read(
        &self,
        d: &Daemon,
        since: Option<&str>,
        operation: &Arc<crate::operation::Context>,
    ) -> Result<std::fs::File, Response> {
        let reader = d.reads.reader().map_err(|e| store_error(&e))?;
        let file =
            tempfile::NamedTempFile::new_in(d.archive.staging_dir()).map_err(|e| error("io", e))?;
        let output = file.reopen().map_err(|e| error("io", e))?;
        super::super::query::feed::prepare_with(&reader, since, &operation.clock, output).map_err(
            |f| Response::Error {
                kind: f.kind,
                message: f.message,
                detail: f.detail,
            },
        )?;
        file.reopen().map_err(|e| error("io", e))
    }
}

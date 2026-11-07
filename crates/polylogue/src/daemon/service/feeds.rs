//! Finite feed batches use the accepting operation's captured evaluation clock.
use super::*;

pub(super) fn read(
    d: &Daemon,
    since: Option<&str>,
    operation: &Arc<crate::operation::Context>,
) -> Result<std::fs::File, Failure> {
    let reader = d.reads.reader()?;
    let file = tempfile::NamedTempFile::new_in(d.archive.staging_dir())
        .map_err(|e| Failure::new("io", e))?;
    let output = file.reopen().map_err(|e| Failure::new("io", e))?;
    super::super::query::feed::prepare_with(&reader, since, &operation.clock, output)?;
    file.reopen().map_err(|e| Failure::new("io", e))
}

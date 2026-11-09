//! Finite feed batches use the accepting operation's captured evaluation clock.
use super::*;

/// One finite change-feed batch from the last applied position.
pub(crate) struct Changes;

/// The change feed's only argument.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChangesArguments {
    since: Option<String>,
}

impl crate::daemon::handler::Feed for Changes {
    type Arguments = ChangesArguments;
    fn produce(
        arguments: Self::Arguments,
        d: &Daemon,
        operation: &Arc<crate::operation::Context>,
        output: &mut dyn std::io::Write,
    ) -> Result<(), Failure> {
        let mut file = read(d, arguments.since.as_deref(), operation)?;
        std::io::copy(&mut file, output).map_err(|e| Failure::new("io", e))?;
        Ok(())
    }
}

fn read(
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

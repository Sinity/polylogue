//! Finite feed batches use the accepting operation's captured evaluation clock.
use super::{Daemon, Failure};
use std::sync::Arc;

/// One finite change-feed batch from the last applied position.
pub(crate) struct Changes;

/// The change feed's only argument.
#[derive(serde::Serialize, serde::Deserialize, command_surface::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChangesArguments {
    /// The last fully applied feed position (GEN:COUNTER:ARCHIVE)
    since: Option<String>,
}

impl crate::daemon::handler::Feed for Changes {
    type Arguments = ChangesArguments;
    fn produce(
        arguments: Self::Arguments,
        d: &Daemon,
        operation: &Arc<polylogue_reading::operation::Context>,
        output: &mut dyn std::io::Write,
    ) -> Result<(), Failure> {
        let mut file = read(d, arguments.since.as_deref(), operation)?;
        std::io::copy(&mut file, output).map_err(Failure::io)?;
        Ok(())
    }
}

fn read(
    d: &Daemon,
    since: Option<&str>,
    operation: &Arc<polylogue_reading::operation::Context>,
) -> Result<std::fs::File, Failure> {
    let reader = d.reads.reader()?;
    let file = tempfile::NamedTempFile::new_in(d.archive.staging_dir()).map_err(Failure::io)?;
    let output = file.reopen().map_err(Failure::io)?;
    super::super::query::feed::prepare_with(&reader, since, &operation.clock, output)?;
    file.reopen().map_err(Failure::io)
}

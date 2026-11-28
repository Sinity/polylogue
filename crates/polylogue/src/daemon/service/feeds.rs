//! The change feed: one finite batch of events from the last applied
//! position, under the accepting request's evaluation clock.
use super::Failure;
use crate::daemon::handler::{Call, Operation, Policy};
use command_surface::Answer;

/// One finite change-feed batch from the last applied position.
pub(crate) struct Changes;

/// The change feed's only argument.
#[derive(serde::Serialize, serde::Deserialize, command_surface::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChangesArguments {
    /// The last fully applied feed position (INDEX_INSTANCE:INDEX:ARCHIVE)
    after: Option<String>,
}

impl Operation for Changes {
    type Arguments = ChangesArguments;
    /// The position the batch completes, also on its `complete` event.
    type Reply = String;
    fn policy(
        _: &Self::Arguments,
        _: polylogue_model::authorship::Channel,
    ) -> Result<Policy, Failure> {
        Ok(Policy::READ)
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<String>, Failure> {
        let context = call.context()?;
        let reader = call.reader()?;
        let file = tempfile::NamedTempFile::new_in(call.daemon.archive.staging_dir())
            .map_err(Failure::io)?;
        let output = file.reopen().map_err(Failure::io)?;
        let mut batch = super::super::query::feed::prepare_with(
            &reader,
            arguments.after.as_deref(),
            &context.clock,
            output,
        )?;
        let position = reader.change_position()?.to_string();
        drop(reader);
        std::io::copy(&mut batch, &mut call.events()).map_err(Failure::io)?;
        Ok(Answer::ok(position))
    }
}

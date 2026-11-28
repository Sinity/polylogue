//! Archive identity and acknowledged completion for an executed query.
use super::{Ask, Failure, Options, Program};
use crate::daemon::{handler::Call, service};
use polylogue_model::{
    ChangePosition, RunId,
    query_objects::{ExecutionBinding, ExecutionPresentation, QueryObject, QueryRun, RunBinding},
};
use polylogue_reading::execution::{self, Completed};
use query_language::cursor::{AuthenticatedCursor, CursorContext};

pub(in crate::daemon) struct RecordedQuery<'a> {
    daemon: &'a crate::daemon::service::Daemon,
    context: std::sync::Arc<polylogue_reading::operation::Context>,
    object: QueryObject,
    run: QueryRun,
}

pub(in crate::daemon) struct StartedQuery<'a> {
    daemon: &'a crate::daemon::service::Daemon,
    run: QueryRun,
    hash: polylogue_model::BlobHash,
}

impl StartedQuery<'_> {
    pub(super) fn run(&self) -> &QueryRun {
        &self.run
    }

    pub(in crate::daemon) fn finish<T>(
        self,
        result: Result<Completed<T>, Failure>,
    ) -> Result<Completed<T>, Failure> {
        execution::finish(
            service::writer(self.daemon)?,
            &self.run,
            self.hash,
            || self.daemon.config.requests.now(),
            result.map_err(|failure| polylogue_reading::Error::Operation(failure.error)),
        )
        .map_err(Failure::from)
    }
}

impl<'a> RecordedQuery<'a> {
    pub(super) fn new(
        call: &Call<'a>,
        ask: &Ask<'_>,
        options: &Options,
        plan: &polylogue_query::plan::Plan,
        program: &Program,
        context: &CursorContext,
        read_position: ChangePosition,
    ) -> Result<Self, Failure> {
        let query_options = polylogue_store::query::execute::options_snapshot(
            options,
            ask.why,
            ask.ranking_identity.clone(),
        )?;
        let accepted = ask
            .context
            .accepted_request()
            .map(serde_json::to_value)
            .transpose()
            .map_err(Failure::internal)?;
        let binding = polylogue_store::selection::bind_execution(
            plan,
            program,
            context,
            query_options,
            ExecutionPresentation::Query,
            accepted,
        )?;
        Ok(Self::from_binding(
            call,
            ask,
            plan.query_object()?,
            binding,
            read_position,
        ))
    }

    pub(in crate::daemon) fn from_binding(
        call: &Call<'a>,
        ask: &Ask<'_>,
        object: QueryObject,
        binding: ExecutionBinding,
        read_position: ChangePosition,
    ) -> Self {
        let run = QueryRun {
            id: polylogue_reading::operation::new_run_id(),
            query_id: Some(object.id),
            at: ask.context.ran_at,
            read_position,
            machine: ask.context.machine.clone(),
            authorship: ask.context.authorship.clone(),
            binding: RunBinding::Original {
                execution: Box::new(binding),
            },
        };
        Self {
            daemon: call.daemon,
            context: ask.context.clone(),
            object,
            run,
        }
    }

    pub(super) fn id(&self) -> &RunId {
        &self.run.id
    }

    pub(in crate::daemon) fn record<T>(
        self,
        execute: impl FnOnce() -> Result<Completed<T>, Failure>,
    ) -> Result<T, Failure> {
        self.start()?
            .finish(execute())
            .map(|completed| completed.value)
    }

    pub(super) fn start(self) -> Result<StartedQuery<'a>, Failure> {
        let writer = service::writer(self.daemon)?;
        let context = self.context;
        let hash = self.run.binding_hash().map_err(Failure::internal)?;
        context.check_parent_lifetime()?;
        writer
            .archive()
            .query_object(self.object, context.parent_lifetime())?;
        writer
            .archive()
            .begin_query_run(self.run.clone(), context.parent_lifetime())?;
        Ok(StartedQuery {
            daemon: self.daemon,
            run: self.run,
            hash,
        })
    }
}

/// Resolve a sealed continuation against its original execution, before looking
/// up any current saved name. The caller's current read authority remains checked.
pub(super) fn replay(
    reader: &polylogue_store::Reader<'_>,
    cursor: &AuthenticatedCursor,
    ask: &Ask<'_>,
    current_options: &Options,
) -> Result<(polylogue_query::plan::Plan, Options, Option<u64>), Failure> {
    let id = cursor
        .context()
        .versions
        .get("run")
        .ok_or_else(|| Failure::stale_cursor("the cursor has no recorded execution"))?;
    let id = RunId::parse(id).map_err(Failure::internal)?;
    let run = reader
        .query_run(&id)?
        .ok_or_else(|| Failure::stale_cursor("the original query run is unavailable"))?;
    let outcome = reader
        .query_run_outcome(&run.id)?
        .ok_or_else(|| Failure::stale_cursor("the original query run has no recorded outcome"))?;
    let (plan, options) = replay_inputs(&run, ask, current_options)?;
    Ok((plan, options, outcome.count))
}

pub(super) fn replay_inputs(
    run: &QueryRun,
    ask: &Ask<'_>,
    current_options: &Options,
) -> Result<(polylogue_query::plan::Plan, Options), Failure> {
    let binding = run
        .binding
        .original()
        .filter(|binding| matches!(binding.presentation, ExecutionPresentation::Query))
        .ok_or_else(|| Failure::invalid_query("the cursor belongs to another operation"))?;
    if binding.replay.original_text != ask.text
        || binding.query_options.fields != current_options.fields
        || binding.query_options.why != ask.why
        || binding
            .query_options
            .self_session
            .as_ref()
            .map(ToString::to_string)
            != current_options.self_session
    {
        return Err(Failure::invalid_query(
            "the cursor belongs to a different question or query options",
        ));
    }
    let clock = binding
        .replay
        .context
        .clock
        .clone()
        .resolve(current_options.clock.day.zone().clone())
        .map_err(Failure::bad_value)?;
    let mut options = polylogue_store::query::execute::options_from_snapshot(
        &binding.query_options,
        clock,
        binding.replay.context.random_seed,
        ask.page_size,
    );
    options.origins.clone_from(&current_options.origins);
    options
        .provider_catalog
        .clone_from(&current_options.provider_catalog);
    Ok((
        polylogue_query::plan::replay_binding(&binding.replay)?,
        options,
    ))
}

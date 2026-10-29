//! Frozen execution inputs reuse the canonical query object and Store lowering.
use super::*;
use polylogue_model::query_objects::{
    ExecutionBinding, ExecutionPresentation, FrozenDefinition, QueryExecutionOptions, QueryObject,
    QueryRun, ReplayBinding,
};
use query_language::time::Clock;

impl PreparedSelection<'_> {
    /// Capture accepted inputs before exposing a continuation. Source facts and
    /// rendered headers remain under their original read authority.
    pub fn execution_binding(
        &self,
        query_options: QueryExecutionOptions,
        presentation: ExecutionPresentation,
        accepted_request: Option<Value>,
    ) -> Result<ExecutionBinding, SelectionError> {
        self.scope.check()?;
        if query_options.fields != self.execution_options.fields
            || query_options.self_session != self.execution_options.self_session
            || query_options.embeddings != self.execution_options.embeddings
        {
            return Err(invalid(
                "invalid_query",
                "execution options differ from the prepared selection",
            ));
        }
        let object = self.query_object()?;
        let definitions = self
            .definitions()
            .iter()
            .map(|definition| {
                let frozen = object
                    .definitions
                    .iter()
                    .find(|frozen| frozen.question == definition.question_id)
                    .ok_or_else(|| invalid("internal", "bound definition has no frozen body"))?;
                Ok(FrozenDefinition {
                    name: definition.name.clone(),
                    root: definition.root.clone(),
                    revision: definition.revision.clone(),
                    question: definition.question_id,
                    ast: frozen.identity_ast.clone(),
                })
            })
            .collect::<Result<Vec<_>, SelectionError>>()?;
        let context = &self.context;
        let mut versions = context.versions.clone();
        if let Some(identity) = &query_options.ranking_identity {
            versions.insert(
                "ranking_identity".into(),
                blake3::hash(identity.as_bytes()).to_hex().to_string(),
            );
        }
        let replay = ReplayBinding {
            bound_ast: object.identity_ast,
            source_outputs: object.source_outputs,
            definitions,
            context: CursorContext {
                question: context.question,
                clock: context.clock.wire(),
                resolved_times: self.program.resolved_times.clone(),
                random_seed: context.random_seed,
                read_positions: context.read_positions.clone(),
                versions,
            },
            scope_fingerprint: context.versions.get("scope").cloned(),
            original_text: self.plan.text().to_owned(),
        };
        self.scope.check()?;
        Ok(ExecutionBinding {
            replay,
            query_options,
            presentation,
            accepted_request,
        })
    }
}

impl Reader<'_> {
    /// The caller authenticates the current operation and restores the recorded
    /// clock through its owner policy before calling this method. This function
    /// does no settings/provider lookup and consumes no rendered source copies.
    pub fn prepare_query_run_selection<'read>(
        &'read self,
        object: &QueryObject,
        run: &QueryRun,
        clock: &Clock,
        ranking_identity: Option<&str>,
        page_size: Option<u64>,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        self.check()?;
        if run.question != object.id
            || run.binding.replay.context.question != object.id
            || run.binding.replay.bound_ast != object.identity_ast
            || run.binding.replay.source_outputs != object.source_outputs
        {
            return Err(invalid(
                "invalid_query",
                "execution binding names another query body",
            ));
        }
        if clock.wire() != run.binding.replay.context.clock {
            return Err(invalid(
                "stale_cursor",
                "the authenticated execution clock changed",
            ));
        }
        if run.binding.query_options.ranking_identity.as_deref() != ranking_identity {
            return Err(invalid(
                "stale_cursor",
                "the captured ranking basis is unavailable",
            ));
        }
        if self.query_scope_fingerprint()? != run.binding.replay.scope_fingerprint {
            return Err(invalid(
                "scope_changed",
                "the captured execution grant changed",
            ));
        }
        let expected_catalog = run
            .binding
            .replay
            .context
            .versions
            .get("catalog")
            .ok_or_else(|| {
                invalid(
                    "not_yet_available",
                    "execution has no native catalog binding",
                )
            })?;
        let options = crate::query::execute::options_from_snapshot(
            &run.binding.query_options,
            clock.clone(),
            run.binding.replay.context.random_seed,
            page_size,
        );
        let selection = self.prepare_query_object_selection(object, &options, expected_catalog)?;
        for key in ["compiler", "catalog", "scope", "ranking"] {
            if selection.context.versions.get(key) != run.binding.replay.context.versions.get(key) {
                return Err(invalid(
                    "stale_cursor",
                    "the captured execution version is unavailable",
                ));
            }
        }
        let old = &run.binding.replay.context.read_positions;
        let current = &selection.context.read_positions;
        for key in ["generation", "index_file"] {
            if old.get(key) != current.get(key) {
                return Err(invalid("stale_cursor", "the captured index is unavailable"));
            }
        }
        for key in ["index", "archive"] {
            if !old
                .get(key)
                .and_then(|value| value.parse::<i64>().ok())
                .zip(current.get(key).and_then(|value| value.parse::<i64>().ok()))
                .is_some_and(|(old, current)| old >= 0 && old <= current)
            {
                return Err(invalid(
                    "stale_cursor",
                    "the recorded read position is unavailable",
                ));
            }
        }
        if selection.program.resolved_times != run.binding.replay.context.resolved_times {
            return Err(invalid("stale_cursor", "the resolved query times changed"));
        }
        if selection.context.versions.get("resolved_times")
            != run.binding.replay.context.versions.get("resolved_times")
        {
            return Err(invalid(
                "stale_cursor",
                "the resolved-time commitment changed",
            ));
        }
        Ok(selection)
    }
}

#[cfg(test)]
mod tests;

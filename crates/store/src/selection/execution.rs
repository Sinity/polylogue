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
        mut query_options: QueryExecutionOptions,
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
        if query_options.direct_reference.is_some() && query_options.direct_reference != self.direct
        {
            return Err(invalid(
                "conflict",
                "direct presentation differs from the accepted selection",
            ));
        }
        if self.direct.is_some() && matches!(presentation, ExecutionPresentation::Query) {
            return Err(invalid(
                "conflict",
                "query execution requires semantic rows, not direct read presentation",
            ));
        }
        if let ExecutionPresentation::Read {
            direct_source: Some(source),
            ..
        } = &presentation
        {
            source
                .validate()
                .map_err(|error| invalid("invalid_query", error.to_string()))?;
            if self.direct.as_ref() != Some(&source.origin().reference) {
                return Err(invalid(
                    "conflict",
                    "span source differs from the accepted direct target",
                ));
            }
        }
        query_options.direct_reference = self.direct.clone();
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
        let binding = run.binding.original().ok_or_else(|| {
            invalid(
                "invalid_cursor",
                "execution replay requires the original root beginning",
            )
        })?;
        if run.question != Some(object.id)
            || binding.replay.context.question != object.id
            || binding.replay.bound_ast != object.identity_ast
            || binding.replay.source_outputs != object.source_outputs
        {
            return Err(invalid(
                "invalid_query",
                "execution binding names another query body",
            ));
        }
        if clock.wire() != binding.replay.context.clock {
            return Err(invalid(
                "stale_cursor",
                "the authenticated execution clock changed",
            ));
        }
        if binding.query_options.ranking_identity.as_deref() != ranking_identity {
            return Err(invalid(
                "stale_cursor",
                "the captured ranking basis is unavailable",
            ));
        }
        if self.query_scope_fingerprint()? != binding.replay.scope_fingerprint {
            return Err(invalid(
                "scope_changed",
                "the captured execution grant changed",
            ));
        }
        let expected_catalog = binding
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
            &binding.query_options,
            clock.clone(),
            binding.replay.context.random_seed,
            page_size,
        );
        let mut selection =
            self.prepare_query_object_selection(object, &options, expected_catalog)?;
        for key in ["compiler", "catalog", "scope", "ranking"] {
            if selection.context.versions.get(key) != binding.replay.context.versions.get(key) {
                return Err(invalid(
                    "stale_cursor",
                    "the captured execution version is unavailable",
                ));
            }
        }
        let old = &binding.replay.context.read_positions;
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
        if selection.program.resolved_times != binding.replay.context.resolved_times {
            return Err(invalid("stale_cursor", "the resolved query times changed"));
        }
        if selection.context.versions.get("resolved_times")
            != binding.replay.context.versions.get("resolved_times")
        {
            return Err(invalid(
                "stale_cursor",
                "the resolved-time commitment changed",
            ));
        }
        if let Some(reference) = &binding.query_options.direct_reference {
            let expected = polylogue_query::plan::prepare(
                &reference_query(reference)?,
                selection.plan.bindings(),
                |_| Ok(()),
            )?;
            if expected.question_id() != selection.question_id()
                || reference_unit(reference) != Some(selection.program.unit)
            {
                return Err(invalid(
                    "conflict",
                    "direct presentation names another bound question",
                ));
            }
            if matches!(reference,Reference::Assertion(revision) if revision.revision.is_none()) {
                return Err(invalid(
                    "not_yet_available",
                    "direct assertion has no pinned revision",
                ));
            }
            selection.direct = Some(reference.clone());
        }
        Ok(selection)
    }
}

#[cfg(test)]
mod tests;

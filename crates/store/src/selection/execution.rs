//! Frozen execution inputs reuse the canonical query object and Store lowering.
use super::*;
use polylogue_model::query_objects::{
    ExecutionBinding, ExecutionPresentation, FrozenDefinition, QueryExecutionOptions, ReplayBinding,
};

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
        let direct = self.direct.as_ref().map(ReadTarget::reference);
        if query_options.direct_reference.is_some() && query_options.direct_reference != direct {
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
            if direct.as_ref() != Some(&source.origin().reference) {
                return Err(invalid(
                    "conflict",
                    "span source differs from the accepted direct target",
                ));
            }
        }
        query_options.direct_reference = direct;
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
                    ast: frozen
                        .executable_ast()
                        .map_err(|reason| QueryError::Unavailable {
                            what: reason.to_string(),
                        })?
                        .clone(),
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
            bound_ast: object
                .body
                .into_query()
                .map_err(|reason| QueryError::Unavailable {
                    what: reason.to_string(),
                })?,
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

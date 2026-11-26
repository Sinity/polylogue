//! Saved execution inputs reuse the saved query object and Store lowering.
use super::*;
use polylogue_model::query_objects::{
    ExecutionBinding, ExecutionPresentation, FrozenDefinition, QueryExecutionOptions, ReplayBinding,
};

impl PreparedSelection<'_> {
    /// Capture accepted inputs before exposing a continuation. Source facts and
    /// rendered headers stay tied to their original read.
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
            return Err(SelectionError::invalid(
                polylogue_model::Code::InvalidQuery,
                "execution options differ from the prepared selection",
            ));
        }
        let direct = self.direct.as_ref().map(ReadTarget::reference);
        let references = self
            .fixed
            .as_ref()
            .map(|fixed| fixed.iter().map(ReadTarget::reference).collect())
            .or_else(|| direct.clone().map(|reference| vec![reference]));
        if query_options.references.is_some() && query_options.references != references {
            return Err(SelectionError::invalid(
                polylogue_model::Code::Conflict,
                "direct presentation differs from the accepted selection",
            ));
        }
        if (self.direct.is_some() || self.fixed.is_some())
            && matches!(presentation, ExecutionPresentation::Query)
        {
            return Err(SelectionError::invalid(
                polylogue_model::Code::Conflict,
                "query execution requires query rows, not direct read presentation",
            ));
        }
        if let ExecutionPresentation::Read {
            direct_source: Some(source),
            ..
        } = &presentation
        {
            source.validate().map_err(|error| {
                SelectionError::invalid(polylogue_model::Code::InvalidQuery, error.to_string())
            })?;
            if direct.as_ref() != Some(&source.origin().reference) {
                return Err(SelectionError::invalid(
                    polylogue_model::Code::Conflict,
                    "span source differs from the accepted direct target",
                ));
            }
        }
        query_options.references = references;
        let object = self.query_object()?;
        let definitions = self
            .definitions()
            .iter()
            .map(|definition| {
                let saved = object
                    .definitions
                    .iter()
                    .find(|saved| saved.query_id == definition.question_id)
                    .ok_or_else(|| {
                        SelectionError::invalid(
                            polylogue_model::Code::Internal,
                            "bound definition has no saved body",
                        )
                    })?;
                Ok(FrozenDefinition {
                    name: definition.name.clone(),
                    root: definition.root.clone(),
                    revision: definition.revision.clone(),
                    query_id: definition.question_id,
                    ast: saved.query.clone(),
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
        let mut source_outputs = object.source_outputs.clone();
        for output in object
            .definitions
            .iter()
            .flat_map(|definition| &definition.source_outputs)
        {
            if !source_outputs.contains(output) {
                source_outputs.push(output.clone());
            }
        }
        let replay = ReplayBinding {
            bound_ast: object.query,
            source_outputs,
            definitions,
            context: CursorContext {
                question: context.question,
                clock: context.clock.wire(),
                resolved_times: self.program.resolved_times.clone(),
                random_seed: context.random_seed,
                read_positions: context.read_positions.clone(),
                versions,
            },
            access_fingerprint: context.versions.get("access").cloned(),
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

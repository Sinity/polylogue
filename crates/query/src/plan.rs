//! A bound question keeps its source references instead of replacing identity with a plan hash.
use crate::{Bindings, QueryError, ast::*};
use query_language::{
    bind::BoundQuery,
    catalog::{Catalog, OutputSource, SourceOutput},
    identity::QuestionId,
};
use serde::Serialize;
use std::collections::BTreeMap;
mod objects;
mod values;
pub use objects::{
    catalog_fingerprint, question_id_from_reference, referenced_kept_results,
    referenced_question_ids, referenced_saved_names, replay, replay_binding, saved_query_bindings,
    validate_object,
};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct ResolvedDefinition {
    pub name: String,
    pub revision: String,
    pub root: String,
    pub question_id: QuestionId,
}
#[derive(Clone, Debug)]
pub struct Plan {
    saved_object: Option<polylogue_model::query_objects::QueryObject>,
    original: BoundQuery,
    expanded: BoundQuery,
    /// The expanded question written out, the text its spans locate.
    expanded_text: String,
    text: String,
    identity: Query,
    catalog: Arc<Catalog>,
    definitions: Vec<ResolvedDefinition>,
    bindings: Bindings,
    saved_definitions: Vec<polylogue_model::query_objects::SavedQueryDefinition>,
    identity_catalog: Arc<Catalog>,
    question_id: QuestionId,
}
impl Plan {
    /// Clock-free output metadata from this plan's already resolved relation.
    pub fn output_schema(&self) -> Result<query_language::catalog::OutputSchema, QueryError> {
        crate::compile::schema::bound_output_metadata(&self.expanded, &self.catalog)
            .map(|(produced, _)| produced)
    }
    /// Declarations of the physical columns this producer emits.
    pub fn output_schema_for_columns(
        &self,
        columns: &[crate::Column],
    ) -> Result<query_language::catalog::OutputSchema, QueryError> {
        use query_language::catalog::{Field, FieldKind};
        let (mut produced, scope) =
            crate::compile::schema::bound_output_metadata(&self.expanded, &self.catalog)?;
        let mut fields = Vec::with_capacity(columns.len());
        for column in columns {
            let mut field = produced
                .fields
                .iter()
                .find(|field| field.name == column.name)
                .or_else(|| scope.fields.iter().find(|field| field.name == column.name))
                .or_else(|| {
                    let mut candidates = scope
                        .fields
                        .iter()
                        .filter(|field| column_name(&field.name) == column.name);
                    let field = candidates.next()?;
                    candidates.next().is_none().then_some(field)
                })
                .cloned();
            if field.is_none() && column.kind != crate::ColKind::Text {
                field = Some(Field {
                    kind: crate::compile::source::column_kind(column.kind),
                    ..Field::default()
                });
            }
            let mut field = field.ok_or_else(|| QueryError::Unavailable {
                reason: format!(
                    "producer has no logical declaration for column {:?}",
                    column.name
                ),
            })?;
            field.name = column.name.clone();
            if column.reference.is_some() {
                // The producer's column fixes the reference target. Filter-only
                // catalog matching rules need not describe its emitted value.
                field.kind = FieldKind::Reference;
                field.unit = column.reference.map(|reference| reference.unit().into());
            }
            fields.push(field);
        }
        produced.fields = fields;
        produced.main_time = produced
            .main_time
            .filter(|name| produced.fields.iter().any(|field| &field.name == name));
        Ok(produced)
    }
    /// Queryable fields of this resolved relation, separate from columns.
    pub fn relation_scope(&self) -> Result<query_language::catalog::OutputSchema, QueryError> {
        let produced = self.output_schema()?;
        crate::compile::schema::relation_scope(&self.expanded, &produced)
    }
    pub fn bindings(&self) -> &Bindings {
        &self.bindings
    }
    pub fn query_object(&self) -> Result<polylogue_model::query_objects::QueryObject, QueryError> {
        objects::from_plan(self)
    }
    /// The expanded question bound whole for one execution: times resolved
    /// against its clock, its columns, origins and provider catalog.
    pub fn bound(&self, options: &crate::Options) -> Result<crate::bound::Bound, QueryError> {
        let typing =
            crate::bound::QuestionTyping::new(self.expanded.query(), self.expanded.typing());
        let binder = crate::filter::Binder {
            text: &self.expanded_text,
            origins: &options.origins,
            clock: &options.clock,
            provider_catalog: options.provider_catalog.as_deref(),
            saved: &self.bindings.saved,
            catalog: &self.catalog,
            expanding: &[],
            fields: &options.fields,
            kept: &self.bindings.kept,
            typing: Some(&typing),
        };
        crate::bound::Bound::of(
            self.expanded.query().clone(),
            Some(crate::bound::TypingAt {
                typing: self.expanded.typing(),
                base: Default::default(),
            }),
            &binder,
        )
    }
    pub fn expanded_query(&self) -> &BoundQuery {
        &self.expanded
    }
    /// The text the spans of [`Plan::expanded_query`] are positions in:
    /// binding writes a question out in its normal spelling.
    pub fn expanded_text(&self) -> &str {
        &self.expanded_text
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn bound_query(&self) -> &BoundQuery {
        &self.original
    }
    #[cfg(test)]
    pub fn identity_query(&self) -> &Query {
        &self.identity
    }
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }
    pub fn definitions(&self) -> &[ResolvedDefinition] {
        &self.definitions
    }
    pub fn question_id(&self) -> QuestionId {
        self.question_id
    }
    /// Figures declare the denominator of the root measurement, independently
    /// of count predicates used only to select rows.
    pub fn requires_denominator(&self) -> bool {
        self.expanded.query().transforms.iter().any(|transform| {
            matches!(transform.kind, TransformKind::Aggregate { .. })
                || crate::stages::measures(transform)
        })
    }
}

/// Product reference resolution is explicit and happens against the caller's pinned read.
pub fn prepare(
    text: &str,
    options: &Bindings,
    mut resolve: impl FnMut(&mut ConditionKind) -> Result<(), QueryError>,
) -> Result<Plan, QueryError> {
    prepare_inner(text, options, &mut resolve, &mut Vec::new())
}
fn prepare_inner(
    text: &str,
    options: &Bindings,
    resolve: &mut impl FnMut(&mut ConditionKind) -> Result<(), QueryError>,
    stack: &mut Vec<String>,
) -> Result<Plan, QueryError> {
    query_language::depth::deeper(|| {
        for kept in options.kept.values() {
            kept.validate_output_metadata()?;
        }
        let mut catalog = options.catalog();
        let mut query = crate::parse::with_catalog(text, &catalog)?;
        // The question is spelled canonically before its identity is checked,
        // which moves every span; the parse still locates the text as written.
        let mut written = query.clone();
        walk(
            &mut query,
            &mut |query| {
                if let Source::Result { reference, .. } = &mut query.source {
                    let name = reference.strip_prefix("result:").unwrap_or(reference);
                    if let Some(id) = options.kept_names.get(name) {
                        *reference = format!("result:{id}");
                    }
                }
                Ok(())
            },
            &mut |_| Ok(()),
        )?;
        let mut definitions = Vec::new();
        let mut saved_definitions: Vec<polylogue_model::query_objects::SavedQueryDefinition> =
            Vec::new();
        let mut identities = BTreeMap::<String, QuestionId>::new();
        walk(
            &mut query,
            &mut |query| {
                if let Source::Saved {
                    name, arguments, ..
                } = &query.source
                {
                    let mut source = query.clone();
                    source.transforms.clear();
                    source.attach.clear();
                    if let Source::Saved { condition, .. } = &mut source.source {
                        *condition = None;
                    }
                    let source_text = query_language::format::format_query(&source, &catalog)
                        .map_err(|e| QueryError::language(e, text))?;
                    let (_, schema) = options.output_metadata(&source_text)?;
                    let source = OutputSource::Saved {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    };
                    if catalog.source_output(&query.source).is_none() {
                        Arc::make_mut(&mut catalog)
                            .source_outputs
                            .push(SourceOutput { source, schema });
                    }
                }
                Ok(())
            },
            &mut |_| Ok(()),
        )?;
        normalize_references(&mut query)?;

        let bound = query_language::bind::bind(&query, &catalog)
            .map_err(|e| QueryError::language(e, text))?;
        query = bound.query().clone();
        values::normalize(&mut query, bound.typing())
            .map_err(|error| error.at_written(&mut query, &mut written))?;
        walk(&mut query, &mut |_| Ok(()), &mut |condition| {
            resolve(&mut condition.kind)
        })?;
        normalize_references(&mut query)?;
        let (original, _) =
            bind_spelled(&query, &catalog).map_err(|e| QueryError::language(e, text))?;
        let mut identity = original.query().clone();
        values::check(&mut identity, original.typing())
            .map_err(|error| error.at_written(&mut identity, &mut written))?;
        for name in referenced_saved_names(&identity) {
            if identities.contains_key(&name) {
                continue;
            }
            crate::compile::source::refuse_cycle(&name, stack, 0)?;
            let definition = crate::compile::saved_definition(text, 0, &name, options)?;
            if let crate::SavedQueryBody::Available {
                object: Some(object),
                ..
            } = &definition.body
            {
                let (node, dependency_nodes) =
                    objects::definition_with_dependencies(object, definition.hash)?;
                for node in std::iter::once(node).chain(dependency_nodes) {
                    if let Some(existing) = saved_definitions
                        .iter()
                        .find(|existing| existing.question == node.question)
                    {
                        if *existing != node {
                            return Err(QueryError::invalid(
                                "one saved question has conflicting captured bodies",
                            ));
                        }
                    } else {
                        saved_definitions.push(node);
                    }
                }
                definitions.push(ResolvedDefinition {
                    name: name.clone(),
                    revision: definition.revision.clone(),
                    root: definition.root.clone(),
                    question_id: definition.hash,
                });
                identities.insert(name, definition.hash);
                continue;
            }
            stack.push(name.clone());
            let nested = prepare_inner(definition.executable_text()?, options, resolve, stack)?;
            stack.pop();
            if nested.question_id != definition.hash {
                return Err(QueryError::invalid(format!(
                    "saved query {name:?} differs from its declared hash"
                )));
            }
            let object = nested.query_object()?;
            saved_definitions.push(objects::root_definition(&object));
            for dependency in object.definitions {
                if !saved_definitions
                    .iter()
                    .any(|known| known.question == dependency.question)
                {
                    saved_definitions.push(dependency);
                }
            }
            definitions.push(ResolvedDefinition {
                name: name.clone(),
                revision: definition.revision.clone(),
                root: definition.root.clone(),
                question_id: nested.question_id,
            });
            for definition in nested.definitions {
                if !definitions
                    .iter()
                    .any(|d| d.name == definition.name && d.revision == definition.revision)
                {
                    definitions.push(definition);
                }
            }
            identities.insert(name, nested.question_id);
        }
        walk(
            &mut identity,
            &mut |query| {
                if let Source::Unit { implicit, .. } = &mut query.source {
                    *implicit = false;
                }
                if let Source::Saved { name, .. } = &mut query.source {
                    *name = identities[name].to_string();
                }
                Ok(())
            },
            &mut |condition| {
                if let ConditionKind::Saved { name, .. } = &mut condition.kind {
                    *name = identities[name].to_string();
                }
                Ok(())
            },
        )?;
        // Saved sources are named by their question identity; without one
        // the identity catalog is the catalog itself.
        let mut identity_catalog = catalog.clone();
        if identity_catalog.source_outputs.iter().any(
            |output| matches!(&output.source, OutputSource::Saved { name, .. } if identities.contains_key(name)),
        ) {
            for output in &mut Arc::make_mut(&mut identity_catalog).source_outputs {
                if let OutputSource::Saved { name, .. } = &mut output.source
                    && let Some(hash) = identities.get(name)
                {
                    *name = hash.to_string();
                }
            }
        }
        let question_id =
            query_language::identity::question_id(&identity, &identity_catalog, &BTreeMap::new())
                .map_err(|e| QueryError::language(e, text))?;
        let (expanded, _) =
            crate::compile::source::expand_ast(original.query().clone(), text, options)?;

        let (expanded, expanded_text) =
            bind_spelled(&expanded, &catalog).map_err(|e| QueryError::language(e, text))?;
        crate::stages::check_measure_last(&expanded.query().transforms, &written.transforms)?;
        let mut bindings = options.clone();
        // Names are presentation bindings, never additional stored results.
        // Every source in this plan now names the resolved immutable ID.
        let aliases = &bindings.kept_names;
        bindings.kept.retain(|name, _| !aliases.contains_key(name));
        Ok(Plan {
            saved_object: None,
            expanded,
            expanded_text,
            text: text.to_owned(),
            original,
            identity,
            catalog,
            definitions,
            bindings,
            saved_definitions,
            identity_catalog,
            question_id,
        })
    })
}

/// A question bound from its own canonical text, and that text. A plan pairs
/// each tree it keeps with its spelling: the binder and compiler place
/// refusals and search terms in the expanded text, and a saved question's
/// body is placed in its stored syntax, by the tree's spans. Binding first
/// gives the nodes substituted parameter values read as, which the text
/// then spells.
pub(crate) fn bind_spelled(
    expanded: &Query,
    catalog: &Catalog,
) -> Result<(BoundQuery, String), query_language::Error> {
    let elaborated = query_language::bind::bind(expanded, catalog)?;
    let text = query_language::format::format_query(elaborated.query(), catalog)?;
    let parsed = query_language::parse::parse(&text, catalog)?;
    Ok((query_language::bind::bind(&parsed, catalog)?, text))
}

impl QueryError {
    /// A value refusal located in the bound question's canonical spelling,
    /// moved to the same condition in the question as written.
    fn at_written(self, bound: &mut Query, written: &mut Query) -> Self {
        let Self::BadValue {
            field,
            value,
            position,
            message,
        } = self
        else {
            return self;
        };
        // Positions in the bound spelling are not those of its conditions'
        // starts, so the refused condition is found by its field and value and
        // placed by its order among the conditions.
        let conditions = |query: &mut Query| {
            let mut conditions = vec![];
            let _ = walk(query, &mut |_| Ok(()), &mut |condition| {
                let holds = matches!(
                    &condition.kind,
                    ConditionKind::Match { field: name, value: Value::Scalar { text, .. } }
                        if *name == field && *text == value
                ) || matches!(
                    &condition.kind,
                    ConditionKind::Match { field: name, value: Value::Set { values } }
                        if *name == field && values.iter().any(|member| matches!(member, Value::Scalar { text, .. } if *text == value))
                );
                conditions.push((condition.span.start, holds));
                Ok(())
            });
            conditions
        };
        let (bound, written) = (conditions(bound), conditions(written));
        let position = match bound.iter().position(|(_, holds)| *holds) {
            Some(index) if bound.len() == written.len() => written[index].0,
            _ => position,
        };
        Self::BadValue {
            field,
            value,
            position,
            message,
        }
    }
}

pub(crate) fn walk(
    query: &mut Query,
    on_query: &mut impl FnMut(&mut Query) -> Result<(), QueryError>,
    on_condition: &mut impl FnMut(&mut Condition) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        on_query(query)?;
        match &mut query.source {
            Source::Unit { condition, .. }
            | Source::Saved { condition, .. }
            | Source::Result { condition, .. } => {
                if let Some(c) = condition {
                    visit_condition(c, on_query, on_condition)?;
                }
            }
        }
        for attachment in &mut query.attach {
            for window in &mut attachment.windows {
                if let Window::Where(c) = window {
                    visit_condition(c, on_query, on_condition)?;
                }
            }
        }
        for stage in &mut query.transforms {
            match &mut stage.kind {
                TransformKind::Where { condition: c }
                | TransformKind::Scope {
                    condition: Some(c), ..
                } => visit_condition(c, on_query, on_condition)?,
                TransformKind::Combine { query, .. }
                | TransformKind::Join { query, .. }
                | TransformKind::Diff { query } => walk(query, on_query, on_condition)?,
                TransformKind::Extension { arguments, .. } => {
                    visit_extensions(arguments, on_query, on_condition)?
                }
                _ => {}
            }
        }
        Ok(())
    })
}
fn visit_condition(
    c: &mut Condition,
    on_query: &mut impl FnMut(&mut Query) -> Result<(), QueryError>,
    on_condition: &mut impl FnMut(&mut Condition) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        on_condition(c)?;
        match &mut c.kind {
            ConditionKind::And { conditions }
            | ConditionKind::Or { conditions }
            | ConditionKind::Sequence {
                steps: conditions, ..
            } => {
                for c in conditions {
                    visit_condition(c, on_query, on_condition)?;
                }
            }
            ConditionKind::Not { condition: c }
            | ConditionKind::Exists {
                condition: Some(c), ..
            }
            | ConditionKind::Count {
                condition: Some(c), ..
            } => visit_condition(c, on_query, on_condition)?,
            ConditionKind::In { query, .. } | ConditionKind::Order { query, .. } => {
                walk(query, on_query, on_condition)?
            }
            ConditionKind::Extension { arguments, .. } => {
                visit_extensions(arguments, on_query, on_condition)?
            }
            _ => {}
        }
        Ok(())
    })
}
fn visit_extensions(
    arguments: &mut [ExtensionArgument],
    on_query: &mut impl FnMut(&mut Query) -> Result<(), QueryError>,
    on_condition: &mut impl FnMut(&mut Condition) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        for argument in arguments {
            match argument {
                ExtensionArgument::Query(query) => walk(query, on_query, on_condition)?,
                ExtensionArgument::Condition(condition) => {
                    visit_condition(condition, on_query, on_condition)?
                }
                ExtensionArgument::Optional(arguments) | ExtensionArgument::List(arguments) => {
                    visit_extensions(arguments, on_query, on_condition)?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

fn normalize_references(query: &mut Query) -> Result<(), QueryError> {
    walk(query, &mut |_| Ok(()), &mut |condition| {
        let ConditionKind::Reference { reference: text } = &condition.kind else {
            return Ok(());
        };
        let Some(reference) = polylogue_model::Reference::parse(text)
            .or_else(|_| polylogue_model::Reference::parse(text))
            .ok()
            .filter(|reference| crate::catalog::reference_unit(reference).is_some())
        else {
            return Ok(());
        };
        condition.kind = ConditionKind::Reference {
            reference: reference.to_string(),
        };
        Ok(())
    })
}

#[cfg(test)]
mod tests;

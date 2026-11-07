//! A bound question retains its source references instead of replacing identity with a plan hash.
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
    catalog_fingerprint, question_id_from_reference, referenced_question_ids,
    referenced_saved_names, replay, saved_query_bindings, validate_object,
};

#[derive(Clone, Debug, Serialize)]
pub struct ResolvedDefinition {
    pub name: String,
    pub revision: String,
    pub root: String,
    pub question_id: QuestionId,
}
#[derive(Clone, Debug)]
pub struct Plan {
    retained_object: Option<polylogue_model::query_objects::QueryObject>,
    original: BoundQuery,
    expanded: BoundQuery,
    text: String,
    identity: Query,
    catalog: Catalog,
    definitions: Vec<ResolvedDefinition>,
    bindings: Bindings,
    frozen: Vec<polylogue_model::query_objects::FrozenQueryDefinition>,
    identity_catalog: Catalog,
    question_id: QuestionId,
}
impl Plan {
    /// Clock-free output metadata from this plan's already resolved relation.
    pub fn output_schema(&self) -> Result<query_language::catalog::OutputSchema, QueryError> {
        crate::compile::schema::bound_output_metadata(&self.expanded, &self.catalog)
            .map(|(produced, _)| produced)
    }
    /// Declarations of the physical columns this producer actually emits.
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
                        .filter(|field| field.name.replace(['.', '-'], "_") == column.name);
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
                what: format!(
                    "producer has no logical declaration for column {:?}",
                    column.name
                ),
            })?;
            field.name = column.name.clone();
            if column.reference.is_some() {
                // The physical producer proves the reference target. Filter-only
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
    /// A re-kept field inherits uncertainty from its actual input; fixed
    /// expressions such as text(E) establish a new declaration.
    pub fn output_proof_for_columns(
        &self,
        columns: &[crate::Column],
        actual: &BTreeMap<String, crate::Kept>,
    ) -> Result<polylogue_model::kept::SchemaProof, QueryError> {
        let schema = self.output_schema_for_columns(columns)?;
        let query = self
            .expanded
            .typing()
            .query(&Default::default())
            .expect("bound root query");
        let scope = query
            .transforms
            .last()
            .map_or(&query.source, |stage| &stage.after);
        let unproven_fields = columns
            .iter()
            .filter_map(|column| {
                let info = scope.field(&column.name).or_else(|| {
                    scope
                        .fields
                        .iter()
                        .find(|info| info.field.name.replace(['.', '-'], "_") == column.name)
                });
                info.and_then(|info| crate::compile::unproven_match_policy(info, actual))
                    .map(|_| column.name.clone())
            })
            .collect();
        Ok(polylogue_model::kept::SchemaProof::Captured {
            schema,
            unproven_fields,
        })
    }
    /// Queryable fields of this exact resolved relation, separate from columns.
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
    pub fn expanded_query(&self) -> &BoundQuery {
        &self.expanded
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn bound_query(&self) -> &BoundQuery {
        &self.original
    }
    pub fn identity_query(&self) -> &Query {
        &self.identity
    }
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }
    pub fn definitions(&self) -> &[ResolvedDefinition] {
        &self.definitions
    }
    pub fn question_id(&self) -> QuestionId {
        self.question_id
    }
    /// Figures declare the denominator of the root measurement, independently
    /// of count predicates used merely to select ordinary rows.
    pub fn requires_denominator(&self) -> bool {
        self.expanded.query().transforms.iter().any(|transform| {
            matches!(transform.kind, TransformKind::Aggregate { .. })
                || matches!(&transform.kind, TransformKind::Extension { name, .. } if name == crate::catalog::MEASURE)
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
    for kept in options.kept.values() {
        kept.validate_output_metadata()?;
    }
    let mut catalog = options.catalog();
    let mut query = crate::parse::with_catalog(text, &catalog)?;
    let mut definitions = Vec::new();
    let mut frozen: Vec<polylogue_model::query_objects::FrozenQueryDefinition> = Vec::new();
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
                    catalog.source_outputs.push(SourceOutput { source, schema });
                }
            }
            Ok(())
        },
        &mut |_| Ok(()),
    )?;
    canonical_references(&mut query)?;

    let bound =
        query_language::bind::bind(&query, &catalog).map_err(|e| QueryError::language(e, text))?;
    query = bound.query().clone();
    walk(&mut query, &mut |_| Ok(()), &mut |condition| {
        resolve(&mut condition.kind)
    })?;
    canonical_references(&mut query)?;
    let original =
        query_language::bind::bind(&query, &catalog).map_err(|e| QueryError::language(e, text))?;
    values::check(&original)?;
    let mut identity = original.query().clone();
    let names = std::cell::RefCell::new(Vec::<String>::new());
    walk(
        &mut identity,
        &mut |query| {
            if let Source::Saved { name, .. } = &query.source {
                names.borrow_mut().push(name.clone());
            }
            Ok(())
        },
        &mut |condition| {
            if let ConditionKind::Saved { name, .. } = &condition.kind {
                names.borrow_mut().push(name.clone());
            }
            Ok(())
        },
    )?;
    for name in names.into_inner() {
        if identities.contains_key(&name) {
            continue;
        }
        if stack.contains(&name) {
            return Err(QueryError::Invalid {
                message: format!("recursive saved query {name:?}"),
                position: 0,
            });
        }
        let definition = crate::compile::saved_definition(text, 0, &name, options)?;
        if let crate::SavedQueryBody::Frozen {
            object: Some(object),
            ..
        } = &definition.body
        {
            let (node, closure) = objects::retained_definition(object, definition.hash)?;
            for node in std::iter::once(node).chain(closure) {
                if let Some(existing) = frozen
                    .iter()
                    .find(|existing| existing.question == node.question)
                {
                    if *existing != node {
                        return Err(QueryError::Invalid {
                            message: "one retained question has conflicting captured bodies".into(),
                            position: 0,
                        });
                    }
                } else {
                    frozen.push(node);
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
            return Err(QueryError::Invalid {
                message: format!("saved query {name:?} differs from its declared semantic hash"),
                position: 0,
            });
        }
        let object = nested.query_object()?;
        frozen.push(objects::frozen_root(&object));
        for dependency in object.definitions {
            if !frozen
                .iter()
                .any(|known| known.question == dependency.question)
            {
                frozen.push(dependency);
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
    let mut identity_catalog = catalog.clone();
    for output in &mut identity_catalog.source_outputs {
        if let OutputSource::Saved { name, .. } = &mut output.source
            && let Some(hash) = identities.get(name)
        {
            *name = hash.to_string();
        }
    }
    let question_id =
        query_language::identity::question_id(&identity, &identity_catalog, &BTreeMap::new())
            .map_err(|e| QueryError::language(e, text))?;
    let (expanded, _) =
        crate::compile::source::expand_ast(original.query().clone(), text, options)?;

    let expanded = query_language::bind::bind(&expanded, &catalog)
        .map_err(|e| QueryError::language(e, text))?;
    Ok(Plan {
        retained_object: None,
        expanded,
        text: text.to_owned(),
        original,
        identity,
        catalog,
        definitions,
        bindings: options.clone(),
        frozen,
        identity_catalog,
        question_id,
    })
}

pub(crate) fn walk(
    query: &mut Query,
    on_query: &mut impl FnMut(&mut Query) -> Result<(), QueryError>,
    on_condition: &mut impl FnMut(&mut Condition) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
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
}
fn visit_condition(
    c: &mut Condition,
    on_query: &mut impl FnMut(&mut Query) -> Result<(), QueryError>,
    on_condition: &mut impl FnMut(&mut Condition) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
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
}
fn visit_extensions(
    arguments: &mut [ExtensionArgument],
    on_query: &mut impl FnMut(&mut Query) -> Result<(), QueryError>,
    on_condition: &mut impl FnMut(&mut Condition) -> Result<(), QueryError>,
) -> Result<(), QueryError> {
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
}

fn canonical_references(query: &mut Query) -> Result<(), QueryError> {
    walk(query, &mut |_| Ok(()), &mut |condition| {
        let ConditionKind::Reference { reference: text } = &condition.kind else {
            return Ok(());
        };
        let Some(reference) = polylogue_model::Reference::parse(text)
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

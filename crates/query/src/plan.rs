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
        crate::compile::schema::infer(self.expanded.query(), &self.catalog)
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
        self.expanded.query().stages.iter().any(|stage| {
            matches!(
                stage.kind,
                StageKind::Aggregate { .. } | StageKind::Measure { .. }
            )
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
    let mut catalog = options.catalog();
    let mut query = crate::parse::with_catalog(text, &catalog)?;
    let mut definitions = Vec::new();
    let mut frozen = Vec::new();
    let mut identities = BTreeMap::<String, QuestionId>::new();
    walk(
        &mut query,
        &mut |query| {
            if let Source::Saved {
                name, arguments, ..
            } = &query.source
            {
                let mut source = query.clone();
                source.stages.clear();
                source.attach.clear();
                if let Source::Saved { condition, .. } = &mut source.source {
                    *condition = None;
                }
                let source_text = query_language::format::format_query(&source, &catalog)
                    .map_err(|e| QueryError::language(e, text))?;
                let (schema, relation_scope) = options.output_metadata(&source_text)?;
                let source = OutputSource::Saved {
                    name: name.clone(),
                    arguments: arguments.clone(),
                };
                if catalog.source_output(&query.source).is_none() {
                    catalog.source_outputs.push(SourceOutput {
                        source,
                        schema,
                        relation_scope: Some(relation_scope),
                    });
                }
            }
            Ok(())
        },
        &mut |_| Ok(()),
    )?;
    canonical_references(&mut query)?;
    crate::compile::schema::supply_matches(&query, &mut catalog)?;
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
        frozen.push(polylogue_model::query_objects::FrozenQueryDefinition {
            question: object.id,
            normal_form: object.normal_form,
            parameters: object.parameters,
            identity_ast: object.identity_ast,
            source_outputs: object.source_outputs,
        });
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
    let rendered = query_language::format::format_query(original.query(), &catalog)
        .map_err(|e| QueryError::language(e, text))?;
    let (expanded, _) = crate::compile::source::expand(&rendered, options)?;
    crate::compile::schema::supply_matches(&expanded, &mut catalog)?;
    let expanded = query_language::bind::bind(&expanded, &catalog)
        .map_err(|e| QueryError::language(e, text))?;
    Ok(Plan {
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
        | Source::Result { condition, .. }
        | Source::RecallLast { condition, .. } => {
            if let Some(c) = condition {
                visit_condition(c, on_query, on_condition)?;
            }
        }
        _ => {}
    }
    for attachment in &mut query.attach {
        for window in &mut attachment.windows {
            if let Window::Where(c) = window {
                visit_condition(c, on_query, on_condition)?;
            }
        }
    }
    for stage in &mut query.stages {
        match &mut stage.kind {
            StageKind::Where { condition: c }
            | StageKind::Scope {
                condition: Some(c), ..
            } => visit_condition(c, on_query, on_condition)?,
            StageKind::Combine { query, .. } | StageKind::Temporal { query, .. } => {
                walk(query, on_query, on_condition)?
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
        ConditionKind::In { query, .. } => walk(query, on_query, on_condition)?,
        _ => {}
    }
    Ok(())
}

fn canonical_references(query: &mut Query) -> Result<(), QueryError> {
    walk(query, &mut |_| Ok(()), &mut |condition| {
        let text = match &condition.kind {
            ConditionKind::Reference { reference } => reference.clone(),
            ConditionKind::Match {
                field,
                value: Value::Scalar { text, .. },
            } => format!("{field}:{text}"),
            _ => return Ok(()),
        };
        let Some(reference) = polylogue_model::Reference::parse(&text)
            .ok()
            .filter(|reference| crate::catalog::reference_unit(reference).is_some())
        else {
            return Ok(());
        };
        let text = reference.to_string();
        let (kind, value) = text.split_once(':').expect("native reference namespace");
        condition.kind = ConditionKind::Match {
            field: kind.into(),
            value: Value::Scalar {
                text: value.into(),
                quoted: true,
            },
        };
        Ok(())
    })
}

#[cfg(test)]
mod tests;

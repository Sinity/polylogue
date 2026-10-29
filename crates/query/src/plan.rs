//! A bound question retains its source references instead of replacing identity with a plan hash.
use crate::{Options, QueryError, ast::*};
use query_language::{
    bind::BoundQuery,
    catalog::{Catalog, OutputSource, SourceOutput},
    identity::QuestionId,
};
use serde::Serialize;
use std::collections::BTreeMap;

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
    identity: Query,
    catalog: Catalog,
    definitions: Vec<ResolvedDefinition>,
    question_id: QuestionId,
}
impl Plan {
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
}

/// Product reference resolution is explicit and happens against the caller's pinned read.
pub fn prepare(
    text: &str,
    options: &Options,
    mut resolve: impl FnMut(&mut ConditionKind) -> Result<(), QueryError>,
) -> Result<Plan, QueryError> {
    prepare_inner(text, options, &mut resolve, &mut Vec::new())
}
fn prepare_inner(
    text: &str,
    options: &Options,
    resolve: &mut impl FnMut(&mut ConditionKind) -> Result<(), QueryError>,
    stack: &mut Vec<String>,
) -> Result<Plan, QueryError> {
    let mut catalog = options.catalog();
    let mut query = crate::parse::with_catalog(text, &catalog)?;
    let mut definitions = Vec::new();
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
                let schema = options.output_schema(&source_text)?;
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
    let bound =
        query_language::bind::bind(&query, &catalog).map_err(|e| QueryError::language(e, text))?;
    query = bound.query().clone();
    walk(&mut query, &mut |_| Ok(()), &mut |condition| {
        resolve(&mut condition.kind)
    })?;
    let original =
        query_language::bind::bind(&query, &catalog).map_err(|e| QueryError::language(e, text))?;
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
        let definition = match options.saved.get(&name).map(Vec::as_slice) {
            Some([one]) => one,
            Some(_) => {
                return Err(QueryError::Invalid {
                    message: format!("ambiguous saved query {name:?}"),
                    position: 0,
                });
            }
            None => {
                return Err(QueryError::Invalid {
                    message: format!("unknown saved query {name:?}"),
                    position: 0,
                });
            }
        };
        stack.push(name.clone());
        let nested = prepare_inner(&definition.text, options, resolve, stack)?;
        stack.pop();
        definitions.push(ResolvedDefinition {
            name: name.clone(),
            revision: definition.revision.clone(),
            root: definition.first_note.clone(),
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
    let question_id = query_language::identity::question_id(&identity, &catalog, &BTreeMap::new())
        .map_err(|e| QueryError::language(e, text))?;
    Ok(Plan {
        original,
        identity,
        catalog,
        definitions,
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

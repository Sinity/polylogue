//! Product definitions become shared syntax before the shared binder validates it.
use super::*;
use query_language::catalog::{
    Catalog, Field, FieldKind, OutputSchema, OutputSource, SourceOutput,
};

pub(crate) fn catalog(options: &Bindings) -> Catalog {
    let mut catalog = crate::catalog::catalog(&options.origins, &options.command_words);
    for (id, kept) in &options.kept {
        for (part, columns) in [
            (ResultPart::Rows, &kept.columns),
            (ResultPart::Matches, &kept.match_columns),
        ] {
            let schema = schema(&catalog, &kept.unit, columns);
            catalog.source_outputs.push(SourceOutput {
                source: OutputSource::Result {
                    reference: format!("result:{id}"),
                    part,
                },
                schema,
            });
        }
    }
    catalog
}

pub(crate) fn schema(catalog: &Catalog, unit_name: &str, columns: &[Column]) -> OutputSchema {
    let unit = catalog.unit(unit_name);
    let fields: Vec<Field> = columns
        .iter()
        .map(|column| {
            let kind = match column.kind {
                ColKind::Text => FieldKind::Text,
                ColKind::Number => FieldKind::Number,
                ColKind::Time => FieldKind::Time,
                ColKind::Money => FieldKind::Money,
                ColKind::Duration => FieldKind::Duration,
                ColKind::Bool => FieldKind::Bool,
                ColKind::Ids => FieldKind::List(Box::new(FieldKind::Reference)),
                ColKind::Json => FieldKind::Json,
            };
            let mut field = unit
                .and_then(|unit| {
                    unit.fields.iter().find(|field| {
                        (field.name == column.name
                            || field.name.replace(['.', '-'], "_") == column.name)
                            && (field.kind == kind
                                || column.kind == ColKind::Text
                                    && matches!(
                                        field.kind,
                                        FieldKind::Match
                                            | FieldKind::Text
                                            | FieldKind::Substring
                                            | FieldKind::Subtree
                                            | FieldKind::Reference
                                    ))
                    })
                })
                .cloned()
                .unwrap_or_default();
            field.name = column.name.clone();
            field.kind = if column.reference.is_some() && column.kind == ColKind::Text {
                FieldKind::Reference
            } else if column.kind == ColKind::Text
                && matches!(
                    field.kind,
                    FieldKind::Match
                        | FieldKind::Text
                        | FieldKind::Substring
                        | FieldKind::Subtree
                        | FieldKind::Reference
                )
            {
                field.kind.clone()
            } else {
                kind
            };
            field
        })
        .collect();
    let main_time = unit
        .and_then(|unit| unit.main_time.clone())
        .filter(|name| fields.iter().any(|field| &field.name == name));
    OutputSchema {
        unit: Some(unit_name.to_owned()),
        fields,
        main_time,
    }
}

pub(crate) fn expand(
    text: &str,
    options: &Bindings,
) -> Result<(Query, Vec<Definition>), QueryError> {
    let mut query = parse_query(text, options)?;
    let mut definitions = vec![];
    expand_query(&mut query, text, options, &mut vec![], &mut definitions)?;
    Ok((query, definitions))
}

fn expand_query(
    query: &mut Query,
    text: &str,
    options: &Bindings,
    stack: &mut Vec<String>,
    definitions: &mut Vec<Definition>,
) -> Result<(), QueryError> {
    if let Source::Saved {
        name,
        arguments,
        condition,
        ..
    } = query.source.clone()
    {
        if stack.contains(&name) {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{name} refers to itself (through {})",
                    stack.join(" -> ")
                ),
                position: 0,
            });
        }
        let definition = saved::saved_definition(text, 0, &name, options)?;
        let mut base = parse_query(&definition.text, options)?;
        saved::Parameters::new(&name, &arguments).bind(&mut base, 0)?;
        stack.push(name.clone());
        expand_query(&mut base, &definition.text, options, stack, definitions)?;
        stack.pop();
        if let Some(condition) = condition {
            if !base.stages.is_empty() || !base.attach.is_empty() {
                return Err(QueryError::Invalid {
                    message: format!(
                        "query:{name} is a shaped result (it has stages); read it as `query:{name} | where …`, it cannot take conditions"
                    ),
                    position: condition.span.start,
                });
            }
            let Source::Unit {
                condition: base_condition,
                ..
            } = &mut base.source
            else {
                return Err(not_built("inline conditions on this saved source"));
            };
            *base_condition = Some(match base_condition.take() {
                None => condition,
                Some(previous) => Condition {
                    span: condition.span,
                    kind: ConditionKind::And {
                        conditions: vec![previous, condition],
                    },
                },
            });
        }
        base.stages.append(&mut query.stages);
        base.attach.append(&mut query.attach);
        *query = base;
        let used = Definition {
            name,
            revision: definition.revision.clone(),
        };
        if !definitions.contains(&used) {
            definitions.push(used);
        }
    }
    match &mut query.source {
        Source::Unit {
            condition: Some(condition),
            ..
        }
        | Source::Result {
            condition: Some(condition),
            ..
        }
        | Source::RecallLast {
            condition: Some(condition),
            ..
        } => expand_condition(condition, text, options, stack, definitions)?,
        _ => {}
    }
    for attachment in &mut query.attach {
        for window in &mut attachment.windows {
            if let Window::Where(condition) = window {
                expand_condition(condition, text, options, stack, definitions)?;
            }
        }
    }
    for stage in &mut query.stages {
        match &mut stage.kind {
            StageKind::Where { condition }
            | StageKind::Scope {
                condition: Some(condition),
                ..
            } => expand_condition(condition, text, options, stack, definitions)?,
            StageKind::Combine { query, .. } | StageKind::Temporal { query, .. } => {
                expand_query(query, text, options, stack, definitions)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn expand_condition(
    condition: &mut Condition,
    text: &str,
    options: &Bindings,
    stack: &mut Vec<String>,
    definitions: &mut Vec<Definition>,
) -> Result<(), QueryError> {
    match &mut condition.kind {
        ConditionKind::And { conditions }
        | ConditionKind::Or { conditions }
        | ConditionKind::Sequence {
            steps: conditions, ..
        } => {
            for condition in conditions {
                expand_condition(condition, text, options, stack, definitions)?;
            }
        }
        ConditionKind::Not { condition }
        | ConditionKind::Exists {
            condition: Some(condition),
            ..
        }
        | ConditionKind::Count {
            condition: Some(condition),
            ..
        } => expand_condition(condition, text, options, stack, definitions)?,
        ConditionKind::In { query, .. } => expand_query(query, text, options, stack, definitions)?,
        _ => {}
    }
    Ok(())
}

pub(crate) fn column_kind(kind: ColKind) -> FieldKind {
    match kind {
        ColKind::Text => FieldKind::Text,
        ColKind::Ids => FieldKind::List(Box::new(FieldKind::Reference)),
        ColKind::Json => FieldKind::Json,
        ColKind::Number => FieldKind::Number,
        ColKind::Money => FieldKind::Money,
        ColKind::Time => FieldKind::Time,
        ColKind::Duration => FieldKind::Duration,
        ColKind::Bool => FieldKind::Bool,
    }
}

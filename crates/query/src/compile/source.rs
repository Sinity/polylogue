//! Product definitions become shared syntax before the shared binder validates it.
use super::*;
use query_language::catalog::{Catalog, FieldKind, OutputSource, SourceOutput};

pub(crate) fn catalog(options: &Bindings) -> std::sync::Arc<Catalog> {
    let mut shared = crate::catalog::catalog(&options.origins);
    if options.kept.is_empty() {
        return shared;
    }
    let catalog = std::sync::Arc::make_mut(&mut shared);
    for (id, kept) in &options.kept {
        for part in [ResultPart::Rows, ResultPart::Matches] {
            let schema = kept.output_schema(part);
            catalog.source_outputs.push(SourceOutput {
                source: OutputSource::Result {
                    reference: format!("result:{id}"),
                    part,
                },
                schema,
            });
        }
    }
    shared
}

pub(crate) fn expand(
    text: &str,
    options: &Bindings,
) -> Result<(Query, Vec<Definition>), QueryError> {
    expand_ast(parse_query(text, options)?, text, options)
}

/// Expand the already parsed question; text is kept only for diagnostics.
pub(crate) fn expand_ast(
    mut query: Query,
    text: &str,
    options: &Bindings,
) -> Result<(Query, Vec<Definition>), QueryError> {
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
    query_language::depth::deeper(|| {
        if let Source::Saved {
            name,
            arguments,
            condition,
            ..
        } = query.source.clone()
        {
            let (mut base, definition) = saved_body(
                &name,
                &arguments,
                text,
                0,
                options,
                &catalog(options),
                stack,
            )?;
            stack.push(name.clone());
            expand_query(
                &mut base,
                definition.executable_text()?,
                options,
                stack,
                definitions,
            )?;
            stack.pop();
            if let Some(condition) = condition {
                if !base.transforms.is_empty() || !base.attach.is_empty() {
                    return Err(QueryError::Invalid {
                        message: format!(
                            "query:{name} is a shaped result (it has transforms); read it as `query:{name} | where …`, it cannot take conditions"
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
                // The saved relation bound its own defaults, and the caller's
                // conditions only narrow it. Its text stays spliced in, so its
                // top-level `exists` or `seq` still feeds `| matches`; its
                // membership beside the caller's conditions keeps a top-level
                // condition such as `hidden:*` from lifting a default it bound.
                let saved = Condition {
                    span: Span::default(),
                    kind: ConditionKind::Saved {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                };
                let mut conditions: Vec<Condition> = base_condition.take().into_iter().collect();
                conditions.push(saved);
                conditions.push(condition.clone());
                *base_condition = Some(Condition {
                    span: condition.span,
                    kind: ConditionKind::And { conditions },
                });
            }
            base.transforms.append(&mut query.transforms);
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
        for stage in &mut query.transforms {
            match &mut stage.kind {
                TransformKind::Where { condition }
                | TransformKind::Scope {
                    condition: Some(condition),
                    ..
                } => expand_condition(condition, text, options, stack, definitions)?,
                TransformKind::Combine { query, .. }
                | TransformKind::Join { query, .. }
                | TransformKind::Diff { query } => {
                    expand_query(query, text, options, stack, definitions)?
                }
                TransformKind::Extension { arguments, .. } => {
                    expand_extensions(arguments, text, options, stack, definitions)?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

fn expand_condition(
    condition: &mut Condition,
    text: &str,
    options: &Bindings,
    stack: &mut Vec<String>,
    definitions: &mut Vec<Definition>,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
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
            ConditionKind::In { query, .. } | ConditionKind::Order { query, .. } => {
                expand_query(query, text, options, stack, definitions)?
            }
            ConditionKind::Extension { arguments, .. } => {
                expand_extensions(arguments, text, options, stack, definitions)?
            }
            _ => {}
        }
        Ok(())
    })
}
fn expand_extensions(
    arguments: &mut [ExtensionArgument],
    text: &str,
    options: &Bindings,
    stack: &mut Vec<String>,
    definitions: &mut Vec<Definition>,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        for argument in arguments {
            match argument {
                ExtensionArgument::Query(query) => {
                    expand_query(query, text, options, stack, definitions)?
                }
                ExtensionArgument::Condition(condition) => {
                    expand_condition(condition, text, options, stack, definitions)?
                }
                ExtensionArgument::Optional(arguments) | ExtensionArgument::List(arguments) => {
                    expand_extensions(arguments, text, options, stack, definitions)?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

/// Refuses a saved query that names itself through the ones expanding
/// around it, naming the chain (docs/query.md: Grammar).
pub(crate) fn refuse_cycle(
    name: &str,
    expanding: &[String],
    position: usize,
) -> Result<(), QueryError> {
    if expanding.iter().any(|outer| outer == name) {
        return Err(QueryError::Invalid {
            message: format!(
                "query:{name} refers to itself (through {} -> {name})",
                expanding.join(" -> ")
            ),
            position,
        });
    }
    Ok(())
}

/// The body of the saved query `name` with its parameters bound to
/// `arguments`: the query its declared identity hashed, never its current
/// text, and the definition it came from. `expanding` holds the saved
/// queries already being expanded around this one, outermost first; naming
/// one of them again is the one cycle refusal.
pub(crate) fn saved_body<'o>(
    name: &str,
    arguments: &[Argument],
    text: &str,
    position: usize,
    definitions: &'o impl Definitions,
    catalog: &Catalog,
    expanding: &[String],
) -> Result<(Query, &'o SavedQuery), QueryError> {
    refuse_cycle(name, expanding, position)?;
    let definition = saved::saved_definition(text, position, name, definitions)?;
    let mut body = match &definition.body {
        crate::SavedQueryBody::Available {
            object: Some(object),
            ..
        } => {
            if definition.hash == object.id {
                object.query.clone()
            } else {
                object
                    .definitions
                    .iter()
                    .find(|node| node.query_id == definition.hash)
                    .ok_or_else(|| {
                        QueryError::invalid("saved query differs from its declared hash")
                    })?
                    .query
                    .clone()
            }
        }
        _ => crate::parse::with_catalog(definition.executable_text()?, catalog)?,
    };
    saved::Parameters::new(name, arguments).bind(&mut body, position)?;
    Ok((body, definition))
}

pub fn column_kind(kind: ColKind) -> FieldKind {
    match kind {
        ColKind::Text => FieldKind::Text,
        ColKind::Ids => FieldKind::List(Box::new(FieldKind::Reference)),
        ColKind::Json => FieldKind::Json,
        ColKind::Number => FieldKind::Number,
        ColKind::Money => FieldKind::Money,
        ColKind::Time => FieldKind::Time,
        ColKind::Duration | ColKind::DurationSeconds => FieldKind::Duration,
        ColKind::Bool => FieldKind::Bool,
    }
}

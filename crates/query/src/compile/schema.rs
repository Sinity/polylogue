//! Output columns are a property of bound syntax, independent of a clock or SQL.
use super::*;
use query_language::catalog::{
    Catalog, Field, FieldKind, MatchOutput, MatchPattern, MatchSource, OutputSchema,
};
fn error(message: impl Into<String>) -> QueryError {
    QueryError::Invalid {
        message: message.into(),
        position: 0,
    }
}
fn unit_schema(unit: &str, catalog: &Catalog) -> Result<OutputSchema, QueryError> {
    let unit = catalog
        .unit(unit)
        .ok_or_else(|| error(format!("unknown output unit {unit:?}")))?;
    let product = crate::units::Unit::from_name(&unit.name)
        .ok_or_else(|| error(format!("unknown native output unit {:?}", unit.name)))?;
    let spec = crate::units::spec(product);
    // The queryable catalog also contains reference atoms and filter-only fields.
    // Frozen output schemas describe exactly what the native producer emits.
    let fields = spec
        .columns
        .iter()
        .map(|column| {
            unit.fields
                .iter()
                .find(|field| field.name == column.name)
                .cloned()
                .unwrap_or_else(|| Field {
                    name: column.name.into(),
                    kind: super::source::column_kind(column.kind),
                    ..Field::default()
                })
        })
        .collect();
    Ok(OutputSchema {
        unit: Some(unit.name.clone()),
        fields,
        main_time: unit.main_time.clone(),
        match_patterns: Vec::new(),
    })
}
/// The product's existing projection declarations, shared by initial binding,
/// replay, completion metadata and the actual producer conformance laws.
pub(crate) fn match_output(
    source: &MatchSource,
    catalog: &Catalog,
) -> Result<OutputSchema, QueryError> {
    let field = |name: &str, kind| Field {
        name: name.into(),
        kind,
        ..Field::default()
    };
    let fields = match &source.pattern {
        MatchPattern::Exists { unit } => {
            let unit =
                crate::units::Unit::from_name(unit).ok_or_else(|| error("unknown match unit"))?;
            let spec = crate::units::spec(unit);
            let child = unit_schema(unit.name(), catalog)?;
            let mut fields = Vec::new();
            if !(source.parent_unit == "sessions"
                && spec.columns.iter().any(|column| column.name == "session"))
            {
                fields.push(field("parent_id", FieldKind::Text));
            }
            fields.push(field("match", FieldKind::Number));
            fields.extend(child.fields);
            fields
        }
        MatchPattern::Sequence { unit, steps } => {
            if unit != "tool_calls" {
                return Err(not_built("sequence Matches outside tool calls"));
            }
            let mut fields = vec![
                field("session", FieldKind::Reference),
                field("match", FieldKind::Number),
            ];
            for step in 1..=*steps {
                for name in ["message", "command", "outcome", "at"] {
                    let mut column = catalog
                        .field("tool_calls", name)
                        .cloned()
                        .ok_or_else(|| error(format!("missing native tool-call field {name}")))?;
                    column.name = format!("step{step}_{name}");
                    fields.push(column);
                }
            }
            for gap in 1..*steps {
                let prefix = if *steps == 2 {
                    "gap".into()
                } else {
                    format!("gap{gap}")
                };
                fields.push(field(&format!("{prefix}_steps"), FieldKind::Number));
                fields.push(field(&format!("{prefix}_time"), FieldKind::Duration));
            }
            fields
        }
    };
    Ok(OutputSchema {
        unit: Some(source.parent_unit.clone()),
        fields,
        main_time: None,
        match_patterns: Vec::new(),
    })
}

pub(crate) fn supply_matches(query: &Query, catalog: &mut Catalog) -> Result<(), QueryError> {
    let mut contains_matches = false;
    crate::plan::walk(
        &mut query.clone(),
        &mut |query| {
            contains_matches |= query
                .stages
                .iter()
                .any(|stage| matches!(stage.kind, StageKind::Matches));
            Ok(())
        },
        &mut |_| Ok(()),
    )?;
    if !contains_matches {
        return Ok(());
    }
    let text = query_language::format::format_query(query, catalog)
        .map_err(|error| QueryError::language(error, ""))?;
    for source in query_language::parse::matches_requirements(&text, catalog)
        .map_err(|error| QueryError::language(error, &text))?
    {
        if catalog.match_output(&source).is_none() {
            let schema = match_output(&source, catalog)?;
            catalog.match_outputs.push(MatchOutput { source, schema });
        }
    }
    Ok(())
}
pub(crate) fn infer(query: &Query, catalog: &Catalog) -> Result<OutputSchema, QueryError> {
    output_metadata(query, catalog).map(|(produced, _)| produced)
}
pub(crate) fn output_metadata(
    query: &Query,
    catalog: &Catalog,
) -> Result<(OutputSchema, OutputSchema), QueryError> {
    let mut snapshot = catalog.clone();
    supply_matches(query, &mut snapshot)?;
    let catalog = &snapshot;
    let bound = query_language::bind::bind(query, catalog)
        .map_err(|error| QueryError::language(error, ""))?;
    let query = bound.query();
    let mut schema = match &query.source {
        Source::Unit { unit, .. } | Source::RecallLast { unit, .. } => unit_schema(unit, catalog)?,
        Source::Saved { .. } | Source::Result { .. } => catalog
            .source_output(&query.source)
            .cloned()
            .ok_or_else(|| error("output source is not bound"))?,
        _ => return Err(not_built("this source output schema")),
    };
    let typing = bound
        .typing()
        .query(&query_language::scope::NodePath::default())
        .expect("binding captures the root query scope");
    for (index, stage) in query.stages.iter().enumerate() {
        match &stage.kind {
            StageKind::Scope { unit, .. } => {
                schema = unit_schema(unit, catalog)?;
            }
            StageKind::Group { .. } | StageKind::Aggregate { .. } | StageKind::Fields { .. } => {
                let after = &typing.stages[index].after;
                schema.unit = after.unit.clone();
                // SQL output order is product-owned; semantic field declarations
                // always come from the shared scope captured for this exact stage.
                let columns = match &stage.kind {
                    StageKind::Group { keys } => Some(
                        keys.iter()
                            .map(Operand::column)
                            .chain(std::iter::once("count".into()))
                            .collect::<Vec<_>>(),
                    ),
                    StageKind::Aggregate { aggregates } => Some(
                        schema
                            .fields
                            .iter()
                            .filter(|field| {
                                after.field(&field.name).is_some()
                                    && !aggregates
                                        .iter()
                                        .any(|aggregate| aggregate.column() == field.name)
                            })
                            .map(|field| field.name.clone())
                            .chain(aggregates.iter().map(Aggregate::column))
                            .collect(),
                    ),
                    StageKind::Fields { fields } => Some(fields.clone()),
                    _ => None,
                };
                schema.fields = if let Some(columns) = columns {
                    columns
                        .iter()
                        .map(|name| {
                            after
                                .field(name)
                                .map(|info| info.field.clone())
                                .ok_or_else(|| error(format!("bound output has no field {name:?}")))
                        })
                        .collect::<Result<_, _>>()?
                } else {
                    after.fields.iter().map(|info| info.field.clone()).collect()
                };
                match &stage.kind {
                    StageKind::Group { .. } | StageKind::Aggregate { .. } => {
                        schema.main_time = None
                    }
                    StageKind::Fields { .. } => {
                        for field in &mut schema.fields {
                            field.name = flat(&field.name);
                        }
                        schema.main_time = schema
                            .main_time
                            .filter(|name| schema.fields.iter().any(|field| &field.name == name));
                    }
                    _ => unreachable!(),
                }
                // These are product evidence columns alongside the semantic aggregate.
                // The shared scope supplies the aggregate's name and type.
                if let StageKind::Aggregate { aggregates } = &stage.kind {
                    for aggregate in aggregates {
                        if matches!(
                            aggregate.function,
                            AggregateFunction::Sum
                                | AggregateFunction::Avg
                                | AggregateFunction::Min
                                | AggregateFunction::Max
                                | AggregateFunction::Percentile(_)
                        ) && let Some(field) = aggregate.fields.first()
                        {
                            schema.fields.push(Field {
                                name: format!("unknown_{}", flat(field)),
                                kind: FieldKind::Integer,
                                ..Field::default()
                            });
                        }
                    }
                }
            }
            StageKind::Matches => {
                let after = &typing.stages[index].after;
                let source = after
                    .fields
                    .iter()
                    .find_map(|field| match &field.origin {
                        query_language::scope::Origin::Matches { source } => Some(source),
                        _ => None,
                    })
                    .ok_or_else(|| error("bound Matches output has no projection provenance"))?;
                schema = catalog
                    .match_output(source)
                    .cloned()
                    .ok_or_else(|| error("bound Matches output has no declaration"))?;
            }
            StageKind::Combine { query, .. } | StageKind::Temporal { query, .. } => {
                if infer(query, catalog)?.unit != schema.unit {
                    return Err(error("combined outputs have different units"));
                }
            }
            StageKind::Where { .. }
            | StageKind::Sort { .. }
            | StageKind::Limit { .. }
            | StageKind::Offset { .. }
            | StageKind::First { .. }
            | StageKind::Distinct
            | StageKind::Fill
            | StageKind::AsOf { .. }
            | StageKind::Measure { .. } => {}
            StageKind::Extension { .. } => return Err(not_built("extension output schema")),
        }
    }
    let text = query_language::format::format_query(query, catalog)
        .map_err(|error| QueryError::language(error, ""))?;
    schema.match_patterns = query_language::parse::relation_patterns(&text, catalog)
        .map_err(|error| QueryError::language(error, &text))?
        .ok_or_else(|| error("output relation qualifiers are unresolved"))?;
    let scope = relation_scope(&bound, &schema)?;
    Ok((schema, scope))
}

/// Capture declarations from this exact bound tree, without retaining operand
/// paths or origins that would point into a different syntax tree on replay.
pub(crate) fn relation_scope(
    bound: &query_language::bind::BoundQuery,
    produced: &OutputSchema,
) -> Result<OutputSchema, QueryError> {
    let typing = bound
        .typing()
        .query(&query_language::scope::NodePath::default())
        .ok_or_else(|| error("binding has no root relation scope"))?;
    let scope = typing
        .stages
        .last()
        .map_or(&typing.source, |stage| &stage.after);
    let mut fields: Vec<Field> = scope.fields.iter().map(|info| info.field.clone()).collect();
    // Product evidence columns are supplied by the same actual projection.
    for field in &produced.fields {
        if !fields.iter().any(|known| known.name == field.name) {
            fields.push(field.clone());
        }
    }
    Ok(OutputSchema {
        unit: scope.unit.clone(),
        fields,
        main_time: produced.main_time.clone(),
        match_patterns: produced.match_patterns.clone(),
    })
}

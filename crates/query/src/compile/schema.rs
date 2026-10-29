//! Output columns are a property of bound syntax, independent of a clock or SQL.
use super::*;
use query_language::catalog::{Catalog, Field, FieldKind, OutputSchema};
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
    Ok(OutputSchema {
        unit: Some(unit.name.clone()),
        fields: unit.fields.clone(),
        main_time: unit.main_time.clone(),
    })
}
fn value_kind(schema: &OutputSchema, operand: &Operand) -> Result<Field, QueryError> {
    match operand {
        Operand::Field { name } => schema
            .fields
            .iter()
            .find(|f| f.name.eq_ignore_ascii_case(name) || f.name.replace(['.', '-'], "_") == *name)
            .cloned()
            .ok_or_else(|| error(format!("unknown output field {name:?}"))),
        Operand::Function { name, arguments } => {
            let argument = arguments
                .first()
                .ok_or_else(|| error(format!("{name} requires a field")))?;
            let mut field = value_kind(schema, argument)?;
            field.name = match argument {
                Operand::Field { name: field } => {
                    format!("{}_{}", flat(field), name.to_ascii_lowercase())
                }
                _ => name.to_ascii_lowercase(),
            };
            field.values.clear();
            field.help.clear();
            field.read_as.clear();
            match name.to_ascii_lowercase().as_str() {
                "day" | "week" | "month" => field.kind = FieldKind::Time,
                "lower" => {}
                _ => return Err(not_built("this output key function")),
            }
            Ok(field)
        }
    }
}
fn aggregate(schema: &OutputSchema, aggregate: &Aggregate) -> Result<Vec<Field>, QueryError> {
    let mut field = match aggregate.fields.first() {
        Some(name) => value_kind(schema, &Operand::field(name))?,
        None => Field::default(),
    };
    field.name = aggregate.column();
    field.values.clear();
    field.help.clear();
    field.read_as.clear();
    field.kind = match aggregate.function {
        AggregateFunction::Count | AggregateFunction::CountDistinct => FieldKind::Number,
        _ => field.kind,
    };
    let mut out = vec![field];
    if matches!(
        aggregate.function,
        AggregateFunction::Sum
            | AggregateFunction::Avg
            | AggregateFunction::Min
            | AggregateFunction::Max
            | AggregateFunction::Percentile(_)
    ) && let Some(name) = aggregate.fields.first()
    {
        out.push(Field {
            name: format!("unknown_{}", flat(name)),
            kind: FieldKind::Number,
            ..Field::default()
        });
    }
    Ok(out)
}
fn conditions(query: &Query) -> Vec<&Condition> {
    let source = match &query.source {
        Source::Unit { condition, .. }
        | Source::Saved { condition, .. }
        | Source::Result { condition, .. }
        | Source::RecallLast { condition, .. } => condition.as_ref(),
        _ => None,
    };
    fn flatten(c: &Condition) -> Vec<&Condition> {
        match &c.kind {
            ConditionKind::And { conditions } => conditions.iter().flat_map(flatten).collect(),
            _ => vec![c],
        }
    }
    source.into_iter().flat_map(flatten).collect()
}
fn matches_schema(
    query: &Query,
    parent: &OutputSchema,
    catalog: &Catalog,
) -> Result<OutputSchema, QueryError> {
    let matches = conditions(query)
        .into_iter()
        .filter(|c| {
            matches!(
                c.kind,
                ConditionKind::Exists { .. } | ConditionKind::Sequence { .. }
            )
        })
        .collect::<Vec<_>>();
    let [condition] = matches.as_slice() else {
        return Err(error("matches needs one top-level exists or sequence"));
    };
    let field = |name: &str, kind| Field {
        name: name.into(),
        kind,
        ..Field::default()
    };
    let fields = match &condition.kind {
        ConditionKind::Exists { unit, .. } => {
            let unit =
                crate::units::Unit::from_name(unit).ok_or_else(|| error("unknown match unit"))?;
            let spec = crate::units::spec(unit);
            let child = unit_schema(unit.name(), catalog)?;
            let mut fields = vec![];
            if !(parent.unit.as_deref() == Some("sessions")
                && spec.columns.iter().any(|c| c.name == "session"))
            {
                fields.push(field("parent_id", FieldKind::Text));
            }
            fields.push(field("match", FieldKind::Number));
            for column in spec.columns {
                fields.push(
                    child
                        .fields
                        .iter()
                        .find(|f| f.name == column.name)
                        .cloned()
                        .unwrap_or_else(|| {
                            field(column.name, super::source::column_kind(column.kind))
                        }),
                );
            }
            fields
        }
        ConditionKind::Sequence { steps, .. } => {
            let mut fields = vec![
                field("session", FieldKind::Reference),
                field("match", FieldKind::Number),
            ];
            for n in 1..=steps.len() {
                for (suffix, kind) in [
                    ("message", FieldKind::Text),
                    ("command", FieldKind::Text),
                    ("outcome", FieldKind::Text),
                    ("at", FieldKind::Time),
                ] {
                    fields.push(field(&format!("step{n}_{suffix}"), kind));
                }
            }
            for n in 1..steps.len() {
                let prefix = if steps.len() == 2 {
                    "gap".into()
                } else {
                    format!("gap{n}")
                };
                fields.push(field(&format!("{prefix}_steps"), FieldKind::Number));
                fields.push(field(&format!("{prefix}_time"), FieldKind::Duration));
            }
            fields
        }
        _ => unreachable!(),
    };
    Ok(OutputSchema {
        unit: parent.unit.clone(),
        fields,
        main_time: None,
    })
}
pub(crate) fn infer(query: &Query, catalog: &Catalog) -> Result<OutputSchema, QueryError> {
    let mut schema = match &query.source {
        Source::Unit { unit, .. } | Source::RecallLast { unit, .. } => unit_schema(unit, catalog)?,
        Source::Saved { .. } | Source::Result { .. } => catalog
            .source_output(&query.source)
            .cloned()
            .ok_or_else(|| error("output source is not bound"))?,
        _ => return Err(not_built("this source output schema")),
    };
    let mut group = None;
    for stage in &query.stages {
        match &stage.kind {
            StageKind::Group { keys } => {
                let mut fields = keys
                    .iter()
                    .map(|key| value_kind(&schema, key))
                    .collect::<Result<Vec<_>, _>>()?;
                for field in &mut fields {
                    field.name = flat(&field.name);
                }
                group = Some(schema.clone());
                schema.fields = fields;
                schema.fields.push(Field {
                    name: "count".into(),
                    kind: FieldKind::Number,
                    ..Field::default()
                });
                schema.main_time = None;
            }
            StageKind::Aggregate { aggregates } => {
                let source = group.as_ref().unwrap_or(&schema);
                let fields = aggregates
                    .iter()
                    .map(|a| aggregate(source, a))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                if group.is_some() {
                    schema.fields.retain(|f| f.name != "count");
                    schema.fields.extend(fields);
                } else {
                    schema.fields = fields;
                }
                schema.main_time = None;
            }
            StageKind::Fields { fields } => {
                schema.fields = fields
                    .iter()
                    .map(|name| value_kind(&schema, &Operand::field(name)))
                    .collect::<Result<Vec<_>, _>>()?;
                for field in &mut schema.fields {
                    field.name = flat(&field.name);
                }
                schema.main_time = schema
                    .main_time
                    .filter(|name| schema.fields.iter().any(|f| &f.name == name));
            }
            StageKind::Matches => schema = matches_schema(query, &schema, catalog)?,
            StageKind::Scope { unit, .. } => {
                schema = unit_schema(unit, catalog)?;
                group = None;
            }
            StageKind::Combine { query, .. } | StageKind::Temporal { query, .. } => {
                let other = infer(query, catalog)?;
                if other.unit != schema.unit {
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
    Ok(schema)
}

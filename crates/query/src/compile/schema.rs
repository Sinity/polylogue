//! Output columns are a property of bound syntax, independent of a clock or SQL.
use super::*;
use query_language::catalog::{Catalog, Field, FieldKind, OutputSchema};
// A daemon's origin choices validate queries; saved fields describe output types.
fn output_field(info: &query_language::scope::FieldInfo) -> Field {
    use query_language::scope::{FieldInfo, Origin, TypeRule};
    fn origin_choices(info: &FieldInfo) -> bool {
        match &info.origin {
            Origin::Catalog { field, .. } => field.rsplit('.').next() == Some("origin"),
            Origin::Group { input, .. } | Origin::Matched { input } => origin_choices(input),
            Origin::Function {
                arguments: inputs,
                rule: TypeRule::Input(index),
                ..
            }
            | Origin::Aggregate {
                inputs,
                rule: TypeRule::Input(index),
                ..
            } => inputs.get(*index).is_some_and(origin_choices),
            _ => false,
        }
    }
    let mut field = info.field.clone();
    if origin_choices(info) {
        field.values.clear();
    }
    field
}

fn unit_schema(unit: &str, catalog: &Catalog) -> Result<OutputSchema, QueryError> {
    let unit = catalog
        .unit(unit)
        .ok_or_else(|| QueryError::invalid(format!("unknown output unit {unit:?}")))?;
    let product = crate::units::Unit::from_name(&unit.name).ok_or_else(|| {
        if crate::units::DesignedUnit::named(&unit.name).is_some() {
            not_built(&format!("unit {}", unit.name))
        } else {
            QueryError::invalid(format!("unknown built-in output unit {:?}", unit.name))
        }
    })?;
    let spec = crate::units::spec(product);
    // The queryable catalog also contains reference atoms and filter-only fields.
    // Saved output schemas describe what the built-in producer emits.
    let fields = spec
        .columns
        .iter()
        .map(|column| {
            let mut field = unit
                .fields
                .iter()
                .find(|field| field.name == column.name)
                .cloned()
                .unwrap_or_else(|| Field {
                    name: column.name.into(),
                    kind: super::source::column_kind(column.kind),
                    ..Field::default()
                });
            if let Some(declaration) =
                crate::fields::lookup(product, column.name).map(|field| field.declaration())
            {
                if let Some(kind) = declaration.value_kind {
                    field.kind = super::source::column_kind(kind);
                }
                if declaration.origin_values {
                    field.values.clear();
                }
            }
            field
        })
        .collect();
    Ok(OutputSchema {
        match_units: None,
        unit: Some(unit.name.clone()),
        fields,
        main_time: unit.main_time.clone(),
    })
}
pub(crate) fn infer(query: &Query, catalog: &Catalog) -> Result<OutputSchema, QueryError> {
    output_metadata(query, catalog).map(|(produced, _)| produced)
}
pub(crate) fn output_metadata(
    query: &Query,
    catalog: &Catalog,
) -> Result<(OutputSchema, OutputSchema), QueryError> {
    let bound = query_language::bind::bind(query, catalog)
        .map_err(|error| QueryError::language(error, ""))?;
    bound_output_metadata(&bound, catalog)
}
pub(crate) fn bound_output_metadata(
    bound: &query_language::bind::BoundQuery,
    catalog: &Catalog,
) -> Result<(OutputSchema, OutputSchema), QueryError> {
    let query = bound.query();
    let mut schema = match &query.source {
        Source::Unit { unit, .. } => unit_schema(unit, catalog)?,
        Source::Saved { .. } | Source::Result { .. } => catalog
            .source_output(&query.source)
            .cloned()
            .ok_or_else(|| QueryError::invalid("output source is not bound"))?,
    };
    let typing = bound
        .typing()
        .query(&query_language::scope::NodePath::default())
        .expect("binding captures the root query scope");
    let mut group_input = None;
    for (index, stage) in query.transforms.iter().enumerate() {
        if matches!(stage.kind, TransformKind::Group { .. }) {
            group_input = Some(schema.clone());
        }
        match &stage.kind {
            TransformKind::First { .. } if group_input.is_some() => {
                schema = group_input.take().expect("grouped first input");
            }
            TransformKind::Scope { unit, .. } => {
                schema = unit_schema(unit, catalog)?;
            }
            TransformKind::Group { .. }
            | TransformKind::Aggregate { .. }
            | TransformKind::Fields { .. } => {
                let after = &typing.transforms[index].after;
                schema.unit = after.unit.clone();
                // SQL output order is the product's; field declarations
                // always come from the shared scope captured for this stage.
                let columns = match &stage.kind {
                    TransformKind::Group { keys } => Some(
                        keys.iter()
                            .map(|column| column.column())
                            .chain(std::iter::once("count".into()))
                            .collect::<Vec<_>>(),
                    ),
                    TransformKind::Aggregate { aggregates } => Some(
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
                    TransformKind::Fields { columns } => {
                        Some(columns.iter().map(|column| column.column()).collect())
                    }
                    _ => None,
                };
                schema.fields = if let Some(columns) = columns {
                    columns
                        .iter()
                        .map(|name| {
                            after.field(name).map(output_field).ok_or_else(|| {
                                QueryError::invalid(format!("bound output has no field {name:?}"))
                            })
                        })
                        .collect::<Result<_, _>>()?
                } else {
                    after.fields.iter().map(output_field).collect()
                };
                match &stage.kind {
                    TransformKind::Group { .. } | TransformKind::Aggregate { .. } => {
                        schema.main_time = None
                    }
                    TransformKind::Fields { .. } => {
                        schema.main_time = schema
                            .main_time
                            .filter(|name| schema.fields.iter().any(|field| &field.name == name));
                    }
                    _ => unreachable!(),
                }
                // These are extra product columns beside the aggregate.
                // The shared scope supplies the aggregate's name and type.
                if let TransformKind::Aggregate { aggregates } = &stage.kind {
                    for aggregate in aggregates {
                        if matches!(
                            aggregate.function,
                            AggregateFunction::Sum
                                | AggregateFunction::Avg
                                | AggregateFunction::Min
                                | AggregateFunction::Max
                                | AggregateFunction::Percentile(_)
                        ) && let Some(field) = aggregate.arguments.first()
                        {
                            schema.fields.push(Field {
                                name: format!("unknown_{}", column_name(&field.column())),
                                kind: FieldKind::Integer,
                                ..Field::default()
                            });
                        }
                    }
                }
            }
            TransformKind::Matches => {
                let after = &typing.transforms[index].after;
                let units = typing.transforms[index]
                    .before
                    .match_units
                    .as_deref()
                    .unwrap_or_default();
                let [unit] = units else {
                    return Err(QueryError::invalid("matches has no unique step unit"));
                };
                let step_unit = crate::units::Unit::from_name(unit)
                    .ok_or_else(|| QueryError::invalid("unknown matched unit"))?;
                let names = ["parent", "match", "position", "step", "step.position"]
                    .into_iter()
                    .map(str::to_owned)
                    .chain(
                        crate::units::spec(step_unit)
                            .columns
                            .iter()
                            .map(|column| format!("step.{}", column.name)),
                    )
                    .chain(
                        schema
                            .fields
                            .iter()
                            .map(|field| format!("parent.{}", field.name)),
                    );
                schema = OutputSchema {
                    match_units: None,
                    unit: None,
                    fields: names
                        .filter_map(|name| after.field(&name).map(output_field))
                        .collect(),
                    main_time: None,
                };
            }
            TransformKind::Combine { query, .. } => {
                if infer(query, catalog)?.unit != schema.unit {
                    return Err(QueryError::invalid("combined outputs have different units"));
                }
            }
            TransformKind::Join { query, .. } => {
                let inner = infer(query, catalog)?;
                let after = &typing.transforms[index].after;
                for field in inner.fields {
                    let name = format!("match.{}", field.name);
                    if let Some(info) = after.field(&name) {
                        schema.fields.push(output_field(info));
                    }
                }
                if let Some(info) = after.field(query_language::catalog::MATCH_DISTANCE) {
                    schema.fields.push(output_field(info));
                }
            }
            TransformKind::Diff { .. } => {
                let after = &typing.transforms[index].after;
                for name in ["change", "old_rank", "new_rank"] {
                    if let Some(info) = after.field(name) {
                        schema.fields.push(output_field(info));
                    }
                }
            }
            TransformKind::Where { .. }
            | TransformKind::Sort { .. }
            | TransformKind::Limit { .. }
            | TransformKind::Offset { .. }
            | TransformKind::First { .. }
            | TransformKind::Distinct
            | TransformKind::Fill
            | TransformKind::AsOf { .. } => {}
            TransformKind::Extension { .. } => return Err(not_built("extension output schema")),
        }
        if !matches!(
            stage.kind,
            TransformKind::Group { .. } | TransformKind::Aggregate { .. }
        ) {
            group_input = None;
        }
    }
    let scope = relation_scope(bound, &schema)?;
    Ok((schema, scope))
}

/// Capture declarations from this bound tree, without keeping operand
/// paths or origins that would point into a different syntax tree on replay.
pub(crate) fn relation_scope(
    bound: &query_language::bind::BoundQuery,
    produced: &OutputSchema,
) -> Result<OutputSchema, QueryError> {
    let typing = bound
        .typing()
        .query(&query_language::scope::NodePath::default())
        .ok_or_else(|| QueryError::invalid("binding has no root relation scope"))?;
    let scope = typing
        .transforms
        .last()
        .map_or(&typing.source, |stage| &stage.after);
    let mut fields: Vec<Field> = scope.fields.iter().map(output_field).collect();
    // Extra product columns come from the same field selection.
    for field in &produced.fields {
        if !fields.iter().any(|known| known.name == field.name) {
            fields.push(field.clone());
        }
    }
    Ok(OutputSchema {
        match_units: scope.match_units.clone(),
        unit: scope.unit.clone(),
        fields,
        main_time: produced.main_time.clone(),
    })
}

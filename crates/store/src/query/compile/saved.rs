//! Saved queries (`query:NAME`) and kept results (`result:<id>`): a
//! saved query is expanded into the query that names it, and a kept result
//! reads back the rows it stored.

use super::*;

pub(super) struct BoundRelation {
    name: String,
    arguments: Vec<Argument>,
    unit: Unit,
    cond: Option<Condition>,
}

impl<'a> Compiler<'a> {
    /// `query:NAME` as a condition: a saved relation over the same unit,
    /// its parameters bound to the values given. A shaped result cannot be
    /// a condition, and units that disagree are an error.
    pub(super) fn saved_cond(
        &mut self,
        name: &str,
        arguments: &[Argument],
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        query_language::depth::deeper(|| {
            let (unit, cond) = self.saved_relation(name, arguments, span)?;
            same_unit(name, unit, scope.unit, span)?;
            let Some(cond) = cond else {
                return Ok("1".into());
            };
            self.saved_stack.push(name.to_owned());
            let sql = self.cond(&cond, scope, false);
            self.saved_stack.pop();
            Ok(format!("({})", sql?))
        })
    }

    /// Resolves and binds each saved relation once for both defaults and SQL.
    pub(super) fn saved_relation(
        &mut self,
        name: &str,
        arguments: &[Argument],
        span: Span,
    ) -> Result<(Unit, Option<Condition>), QueryError> {
        query_language::depth::deeper(|| {
            if self.saved_stack.iter().any(|n| n == name) {
                return Err(QueryError::Invalid {
                    message: format!(
                        "query:{name} refers to itself (through {})",
                        self.saved_stack.join(" -> ")
                    ),
                    position: span.start,
                });
            }
            if let Some(bound) = self
                .saved_relations
                .iter()
                .find(|bound| bound.name == name && bound.arguments == arguments)
            {
                return Ok((bound.unit, bound.cond.clone()));
            }
            let definition = saved_definition(self.text, span.start, name, self.opts)?;
            let mut query = parse_query(definition.executable_text()?, self.opts)?;
            Parameters::new(name, arguments).bind(&mut query, span.start)?;
            if !query.transforms.is_empty() || !query.attach.is_empty() {
                return Err(QueryError::Invalid {
                    message: format!(
                        "query:{name} is a shaped result (it has stages), so it cannot be a condition; \
                     read it as `query:{name} | where …`"
                    ),
                    position: span.start,
                });
            }
            let (unit, cond) = match query.source {
                Source::Unit {
                    unit, condition, ..
                } => (bind_unit(&unit)?, condition),
                Source::Saved {
                    name: nested,
                    arguments,
                    condition,
                    ..
                } => {
                    self.saved_stack.push(name.to_owned());
                    let relation = self.saved_relation(&nested, &arguments, span);
                    self.saved_stack.pop();
                    let (unit, base) = relation?;
                    let condition = match (base, condition) {
                        (Some(base), Some(condition)) => Some(Condition {
                            span: condition.span,
                            kind: ConditionKind::And {
                                conditions: vec![base, condition],
                            },
                        }),
                        (base, condition) => base.or(condition),
                    };
                    (unit, condition)
                }
                _ => return Err(not_built("a saved source of this kind as a condition")),
            };
            self.used(name, definition);
            self.saved_relations.push(BoundRelation {
                name: name.to_owned(),
                arguments: arguments.to_vec(),
                unit,
                cond: cond.clone(),
            });
            Ok((unit, cond))
        })
    }

    pub(super) fn saved_defaults(
        &mut self,
        scope: &RowContext,
        top: &[Condition],
    ) -> Result<Vec<Condition>, QueryError> {
        let mut bound = vec![];
        for condition in top {
            let ConditionKind::Saved { name, arguments } = &condition.kind else {
                bound.push(condition.clone());
                continue;
            };
            let (unit, cond) = self.saved_relation(name, arguments, condition.span)?;
            same_unit(name, unit, scope.unit, condition.span)?;
            if let Some(cond) = cond {
                self.saved_stack.push(name.clone());
                let defaults = self.saved_defaults(scope, &top_level(&cond));
                self.saved_stack.pop();
                bound.extend(defaults?);
            }
        }
        Ok(bound)
    }

    /// Records the revision a saved query's name resolved to.
    fn used(&mut self, name: &str, definition: &SavedQuery) {
        let used = Definition {
            name: name.to_owned(),
            revision: definition.revision.clone(),
        };
        if !self.definitions.contains(&used) {
            self.definitions.push(used);
        }
    }
}

fn same_unit(name: &str, selected: Unit, expected: Unit, span: Span) -> Result<(), QueryError> {
    if selected != expected {
        return Err(QueryError::Invalid {
            message: format!(
                "query:{name} selects {}, not {}",
                selected.name(),
                expected.name()
            ),
            position: span.start,
        });
    }
    Ok(())
}

/// A query whose source is a kept result (`result:<id>`, `.rows` or
/// `.matches`): its stored values, read as they were kept, never re-run.
/// Only stages may follow.
impl Compiler<'_> {
    pub(super) fn kept_program(mut self, query: &Query) -> Result<Program, QueryError> {
        let Source::Result {
            reference,
            part,
            condition,
            ..
        } = &query.source
        else {
            return Err(QueryError::Invalid {
                message: "a kept source is required".into(),
                position: 0,
            });
        };
        if condition.is_some() || !query.attach.is_empty() {
            return Err(QueryError::Invalid {
                message: "A kept result takes stages, not conditions or attached rows".into(),
                position: 0,
            });
        }
        let id = reference.strip_prefix("result:").unwrap_or(reference);
        let opts = self.opts;
        let kept = opts.kept.get(id).ok_or_else(|| QueryError::Unavailable {
            what: format!("kept result {id:?}"),
        })?;
        let matched = *part == ResultPart::Matches;
        let part = if matched { "matches" } else { "rows" };
        let columns = if matched {
            &kept.match_columns
        } else {
            &kept.columns
        };
        if matched && columns.is_empty() {
            return Err(QueryError::Invalid {
                message: format!("result:{id} kept no match rows"),
                position: 0,
            });
        }
        let native = Unit::all().find(|unit| unit.name() == kept.unit);
        let unit = native.unwrap_or(Unit::Sessions);
        let shape = if matched {
            Shape::Matches
        } else {
            kept.shape.unwrap_or(Shape::Rows)
        };
        let id_param = self.text_param(id);
        let mut select = vec![
            "j.key AS _pk".into(),
            format!("json_array({id_param}, '{part}') AS _kept_source"),
        ];
        let mut slots = std::collections::BTreeMap::new();
        let mut time_text_columns = std::collections::BTreeSet::new();
        for (index, column) in columns.iter().enumerate() {
            let path = self.text_param(&format!(
                "$.{}",
                serde_json::to_string(&column.name).expect("column name is JSON text")
            ));
            let value = if matches!(column.kind, ColKind::Ids | ColKind::Json) {
                format!("j.value -> {path}")
            } else if column.kind == ColKind::Money {
                // Kept as exact dollars; read back as nano-USD from its text.
                format!("pl_kept_money(j.value -> {path})")
            } else {
                format!("json_extract(j.value, {path})")
            };
            let slot = format!("_kept_value{index}");
            select.push(format!("{value} AS {}", column_sql(&slot)));
            slots.insert(column.name.clone(), slot);
            if column.kind == ColKind::Time {
                time_text_columns.insert(column.name.clone());
            }
        }
        let row_basis = if !matched && kept.native_identity && shape == Shape::Rows {
            if let Some(unit) = native {
                let mut identity = vec![];
                let count = units::spec(unit).identity.len();
                for index in 0..count {
                    let slot = format!("_kept_identity{index}");
                    select.push(format!("pl_kept_identity(k.rows_json -> ('$.identities[' || j.key || ']'), {index}, {count}) AS {slot}"));
                    identity.push(slot);
                }
                RowBasis::Native {
                    unit,
                    columns: identity,
                }
            } else {
                RowBasis::Distinct {
                    columns: vec!["_kept_source".into(), "_pk".into()],
                }
            }
        } else {
            RowBasis::Distinct {
                columns: vec!["_kept_source".into(), "_pk".into()],
            }
        };
        let program = Program {
            query: self.text.into(),
            row_basis,
            unit,
            shape,
            sql: format!(
                "SELECT {} FROM source_kept_result k, json_each(k.rows_json, '$.{part}') j WHERE k.id={id_param} ORDER BY j.key",
                select.join(", ")
            ),
            row_count_sql: None,
            row_membership_sql: None,
            params: std::mem::take(&mut self.params),
            columns: columns.clone(),
            column_slots: slots,
            native_sources: Some(vec![]),
            time_text_columns,
            ranked: false,
            snippet_terms: vec![],
            id_checks: vec![],
            change_generations: vec![],
            assertion_cutoffs: vec![],
            attach: vec![],
            notes: vec![],
            incomplete: vec![],
            why: vec![],
            why_defaults: vec![],
            why_of: format!(
                "SELECT count(*) FROM source_kept_result k, json_each(k.rows_json, '$.{part}') j WHERE k.id={id_param}"
            ),
            page_size: opts.page_size.unwrap_or(50),
            near: None,
            counts_near: false,
            definitions: vec![],
            clock: opts.clock.clone(),
            resolved_times: vec![],
            measure: None,
            order: vec![OrderKey::ascending("_pk")],
        };
        self.program_over(query, program)
    }
}

/// The definition a saved query's name resolves to: the current revision
/// of the one note family with that name.
pub use polylogue_query::compile::saved_definition;

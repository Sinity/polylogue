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
        let (unit, cond) = self.saved_relation(name, arguments, span)?;
        same_unit(name, unit, scope.unit, span)?;
        let Some(cond) = cond else {
            return Ok("1".into());
        };
        self.saved_stack.push(name.to_owned());
        let sql = self.cond(&cond, scope, false);
        self.saved_stack.pop();
        Ok(format!("({})", sql?))
    }

    /// Resolves and binds each saved relation once for both defaults and SQL.
    pub(super) fn saved_relation(
        &mut self,
        name: &str,
        arguments: &[Argument],
        span: Span,
    ) -> Result<(Unit, Option<Condition>), QueryError> {
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
        let mut query = parse_query(&definition.text, self.opts)?;
        Parameters::new(name, arguments).bind(&mut query, span.start)?;
        if !query.stages.is_empty() || !query.attach.is_empty() {
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
pub(super) fn compile_kept(
    text: &str,
    query: &Query,
    opts: &Options,
) -> Result<Option<Program>, QueryError> {
    let Source::Result {
        reference,
        part,
        condition,
        ..
    } = &query.source
    else {
        return Ok(None);
    };
    if condition.is_some() || !query.attach.is_empty() {
        return Err(QueryError::Invalid {
            message: "A kept result takes stages, not conditions or attached rows".into(),
            position: 0,
        });
    }
    let mut query = query.clone();
    let id = reference.strip_prefix("result:").unwrap_or(reference);
    let part = match part {
        ResultPart::Rows => "rows",
        ResultPart::Matches => "matches",
    };
    let kept = opts.kept.get(id).ok_or_else(|| {
        QueryError::syntax(text, 0, format!("no kept result has the ID {id:?}"), None)
    })?;
    let columns = if part == "matches" {
        if kept.match_columns.is_empty() {
            return Err(QueryError::Invalid {
                message: format!("result:{id} kept no match rows (its query has no exists or seq)"),
                position: 0,
            });
        }
        &kept.match_columns
    } else {
        &kept.columns
    };
    query.source = Source::Unit {
        unit: Unit::Sessions.name().into(),
        implicit: false,
        condition: None,
    };
    let unit = Unit::all()
        .find(|u| u.name() == kept.unit)
        .unwrap_or(Unit::Sessions);
    let mut c = Compiler::new(text, opts);
    let id_param = c.text_param(id);
    let select: Vec<String> = columns
        .iter()
        .map(|col| {
            if matches!(col.kind, ColKind::Ids | ColKind::Json) {
                format!("(j.value -> '$.\"{0}\"') AS \"{0}\"", col.name)
            } else {
                format!("json_extract(j.value, '$.\"{0}\"') AS \"{0}\"", col.name)
            }
        })
        .collect();
    let rel = MatchRel {
        sql: format!(
            "SELECT j.key AS _pk, {} FROM source_kept_result k, json_each(k.rows_json, '$.{part}') j \
             WHERE k.id = {id_param} ORDER BY j.key",
            select.join(", ")
        ),
        columns: columns.iter().map(|c| (c.name.clone(), c.kind)).collect(),
        order: vec![OrderKey::ascending("_pk")],
        recursive: false,
    };
    let mut program = c.program_over(&query, rel)?;
    if part == "rows" && program.shape == Shape::Matches {
        program.shape = Shape::Rows;
        program.unit = unit;
        program.columns = columns.clone();
    }
    Ok(Some(program))
}

/// The definition a saved query's name resolves to: the current revision
/// of the one note family with that name.
pub use polylogue_query::compile::saved_definition;

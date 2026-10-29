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
pub(super) fn saved_definition<'o>(
    text: &str,
    position: usize,
    name: &str,
    opts: &'o Options,
) -> Result<&'o SavedQuery, QueryError> {
    match opts.saved.get(name).map(Vec::as_slice) {
        Some([one]) => Ok(one),
        Some(several @ [_, _, ..]) => Err(QueryError::Invalid {
            message: format!(
                "query:{name} names {} saved queries ({}); withdraw or rename all but one",
                several.len(),
                several
                    .iter()
                    .map(|s| format!("note:{}", s.first_note))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            position,
        }),
        _ => {
            let known: Vec<&str> = opts.saved.keys().map(String::as_str).collect();
            Err(QueryError::syntax(
                text,
                position,
                format!(
                    "no saved query named {name:?}{}",
                    if known.is_empty() {
                        String::new()
                    } else {
                        format!(" (saved: {})", known.join(", "))
                    }
                ),
                None,
            ))
        }
    }
}

/// A saved query's parameters: each `$word` standing for a value in its
/// text is replaced by the value the call gives, in the parsed query, so a
/// value never changes the query's shape.
pub(super) struct Parameters {
    name: String,
    arguments: Vec<Argument>,
    collect_only: bool,
}

impl Parameters {
    pub(super) fn new(name: &str, arguments: &[Argument]) -> Self {
        Self {
            name: name.to_owned(),
            arguments: arguments.to_vec(),
            collect_only: false,
        }
    }

    pub(super) fn names(query: &mut Query) -> Result<Vec<String>, QueryError> {
        let mut collector = Self::new("", &[]);
        collector.collect_only = true;
        let mut names = vec![];
        collector.query(query, &mut names, 0)?;
        Ok(names)
    }

    pub(super) fn bind(&self, query: &mut Query, position: usize) -> Result<(), QueryError> {
        let mut used = vec![];
        self.query(query, &mut used, position)?;
        if let Some(unused) = self.arguments.iter().find(|a| !used.contains(&a.parameter)) {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{} has no parameter {}{}",
                    self.name,
                    unused.parameter,
                    if used.is_empty() {
                        String::new()
                    } else {
                        format!(" (its parameters: {})", used.join(", "))
                    }
                ),
                position,
            });
        }
        Ok(())
    }

    fn query(
        &self,
        query: &mut Query,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        match &mut query.source {
            Source::Unit {
                condition: Some(condition),
                ..
            }
            | Source::RecallLast {
                condition: Some(condition),
                ..
            }
            | Source::Result {
                condition: Some(condition),
                ..
            } => self.cond(condition, used, position)?,
            Source::Saved {
                arguments,
                condition,
                ..
            } => {
                for argument in arguments {
                    self.value(&mut argument.value, used, position)?;
                }
                if let Some(condition) = condition {
                    self.cond(condition, used, position)?;
                }
            }
            _ => {}
        }
        for child in &mut query.attach {
            for window in &mut child.windows {
                if let Window::Where(cond) = window {
                    self.cond(cond, used, position)?;
                }
            }
        }
        for stage in &mut query.stages {
            match &mut stage.kind {
                StageKind::Where { condition: cond }
                | StageKind::Scope {
                    condition: Some(cond),
                    ..
                } => self.cond(cond, used, position)?,
                StageKind::Combine { query, .. } => self.query(query, used, position)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn cond(
        &self,
        c: &mut Condition,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        match &mut c.kind {
            ConditionKind::And { conditions }
            | ConditionKind::Or { conditions }
            | ConditionKind::Sequence {
                steps: conditions, ..
            } => {
                for condition in conditions {
                    self.cond(condition, used, position)?;
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
            } => self.cond(condition, used, position)?,
            ConditionKind::In { query, .. } => self.query(query, used, position)?,
            ConditionKind::Match { value, .. }
            | ConditionKind::Compare { value, .. }
            | ConditionKind::Since { value, .. }
            | ConditionKind::Until { value, .. }
            | ConditionKind::Similar { value }
            | ConditionKind::TimeRange { value, .. } => self.value(value, used, position)?,
            ConditionKind::Between { from, to, .. } => {
                self.value(from, used, position)?;
                self.value(to, used, position)?;
            }
            ConditionKind::Text { text, mode } => {
                if let Some(argument) = self.argument(text, used, position)? {
                    *text = value_text(&argument.value)?.into();
                    *mode = if matches!(argument.value, Value::Scalar { quoted: true, .. }) {
                        TextMode::Phrase
                    } else {
                        TextMode::Word
                    };
                }
            }
            ConditionKind::Saved { arguments, .. } => {
                for argument in arguments {
                    self.value(&mut argument.value, used, position)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn value(
        &self,
        value: &mut Value,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        match value {
            Value::Parameter { name } => {
                let marker = format!("${name}");
                if let Some(argument) = self.argument(&marker, used, position)? {
                    *value = argument.value.clone();
                }
            }
            Value::Scalar {
                text,
                quoted: false,
            } => {
                if let Some(argument) = self.argument(text, used, position)? {
                    *value = argument.value.clone();
                }
            }
            Value::Set { values } => {
                for value in values {
                    self.value(value, used, position)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The value given for `$word`, or `None` when `text` is no parameter.
    fn argument(
        &self,
        text: &str,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<Option<&Argument>, QueryError> {
        let Some(parameter) = parameter_name(text) else {
            return Ok(None);
        };
        if !used.iter().any(|u| u == parameter) {
            used.push(parameter.to_owned());
        }
        if self.collect_only {
            return Ok(None);
        }
        let argument = self
            .arguments
            .iter()
            .find(|a| a.parameter == parameter)
            .ok_or_else(|| QueryError::Invalid {
                message: format!(
                    "query:{} needs a value for {parameter}: `query:{}({parameter}=…)`",
                    self.name, self.name
                ),
                position,
            })?;
        Ok(Some(argument))
    }
}

/// `$name`: a parameter of a saved query.
fn parameter_name(text: &str) -> Option<&str> {
    let name = text.strip_prefix('$')?;
    let mut chars = name.chars();
    (chars.next()?.is_ascii_alphabetic() && chars.all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .then_some(name)
}

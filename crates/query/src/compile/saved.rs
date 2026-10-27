//! Saved queries (`query:NAME`) and kept results (`result:<id>`): a
//! saved query is expanded into the query that names it, and a kept result
//! reads back the rows it stored.

use super::*;

pub(super) struct BoundRelation {
    name: String,
    arguments: Vec<Argument>,
    unit: Unit,
    cond: Option<Cond>,
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
        if unit != scope.unit {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{name} selects {}, not {}",
                    unit.name(),
                    scope.unit.name()
                ),
                position: span.start,
            });
        }
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
    ) -> Result<(Unit, Option<Cond>), QueryError> {
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
        let mut query = parse(&definition.text)?;
        Parameters::new(name, arguments, None).bind(&mut query, span.start)?;
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
            Source::Unit { unit, cond } => (unit, cond),
            Source::Sessions { cond } => (Unit::Sessions, Some(cond)),
            Source::Empty => (Unit::Sessions, None),
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
        top: &[Cond],
    ) -> Result<Vec<Cond>, QueryError> {
        let mut bound = vec![];
        for condition in top {
            let CondKind::Saved { name, arguments } = &condition.kind else {
                bound.push(condition.clone());
                continue;
            };
            let (unit, cond) = self.saved_relation(name, arguments, condition.span)?;
            if unit != scope.unit {
                return Err(QueryError::Invalid {
                    message: format!(
                        "query:{name} selects {}, not {}",
                        unit.name(),
                        scope.unit.name()
                    ),
                    position: condition.span.start,
                });
            }
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

/// A query whose source is a kept result (`result:<id>`, `.rows` or
/// `.matches`): its stored values, read as they were kept, never re-run.
/// Only stages may follow.
pub(super) fn compile_kept(text: &str, opts: &Options) -> Result<Option<Program>, QueryError> {
    let lead = text.len() - text.trim_start().len();
    let trimmed = text.trim_start();
    let Some(after) = trimmed
        .get(..7)
        .filter(|p| p.eq_ignore_ascii_case("result:"))
        .map(|_| &trimmed[7..])
    else {
        return Ok(None);
    };
    let end = after
        .find(|c: char| c.is_whitespace() || c == '|')
        .unwrap_or(after.len());
    let word = &after[..end];
    let (id, part) = match word.rsplit_once('.') {
        Some((id, p @ ("rows" | "matches"))) => (id, p),
        _ => (word, "rows"),
    };
    let rest = after[end..].trim();
    if !rest.is_empty() && !rest.starts_with('|') {
        return Err(QueryError::Invalid {
            message: format!(
                "result:{id} is a kept result: read it with stages (`result:{id} | group by …`), not conditions"
            ),
            position: lead,
        });
    }
    let kept = opts.kept.get(id).ok_or_else(|| {
        QueryError::syntax(
            text,
            lead + 7,
            format!("no kept result has the ID {id:?}"),
            None,
        )
    })?;
    let columns = if part == "matches" {
        if kept.match_columns.is_empty() {
            return Err(QueryError::Invalid {
                message: format!("result:{id} kept no match rows (its query has no exists or seq)"),
                position: lead,
            });
        }
        &kept.match_columns
    } else {
        &kept.columns
    };
    let stages = format!("sessions {rest}");
    let query = parse(&stages)?;
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
            "SELECT j.key AS _pk, {} FROM a.kept_result k, json_each(k.rows_json, '$.{part}') j \
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

/// A query that starts with `query:NAME` reads that saved query as its
/// source: the rest of the text only narrows it. A saved relation (`U where
/// C`) takes the rest as further conditions; a shaped result (with stages)
/// takes only further stages.
pub(super) struct BoundSource {
    /// The saved query with the rest of the text joined to it.
    pub text: String,
    pub definition: Definition,
    parameters: Parameters,
}

impl BoundSource {
    /// Binds the saved query's parameters in the parsed text; a `$word`
    /// the caller wrote stays a word.
    pub fn bind(&self, query: &mut Query) -> Result<(), QueryError> {
        self.parameters.bind(query, 0)
    }
}

pub(super) fn bind_saved_source(
    text: &str,
    opts: &Options,
) -> Result<Option<BoundSource>, QueryError> {
    let lead = text.len() - text.trim_start().len();
    let starts = text[lead..]
        .get(..6)
        .is_some_and(|p| p.eq_ignore_ascii_case("query:"));
    if !starts {
        return Ok(None);
    }
    let query = parse(text)?;
    let first = match &query.source {
        Source::Sessions { cond } => match &cond.kind {
            CondKind::And { all } => &all[0],
            _ => cond,
        },
        _ => return Ok(None),
    };
    let CondKind::Saved { name, arguments } = &first.kind else {
        return Ok(None);
    };
    if first.span.start != lead {
        return Ok(None);
    }
    let definition = saved_definition(text, lead + 6, name, opts)?;
    let saved = definition.text.as_str();
    let rest = text[first.span.end..].trim();
    let shape = parse(saved)?;
    let shaped = !shape.stages.is_empty() || !shape.attach.is_empty();
    let (conds, stages) = match rest.find('|') {
        Some(i) => (rest[..i].trim(), &rest[i..]),
        None => (rest, ""),
    };
    let bound = |text: String, region: std::ops::Range<usize>| {
        Ok(Some(BoundSource {
            text,
            definition: Definition {
                name: name.clone(),
                revision: definition.revision.clone(),
            },
            parameters: Parameters::new(name, arguments, Some(region)),
        }))
    };
    if shaped {
        if !conds.is_empty() {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{name} is a shaped result (it has stages); read it as \
                     `query:{name} | where …`, it cannot take conditions"
                ),
                position: lead,
            });
        }
        return bound(format!("{saved} {stages}"), 0..saved.len());
    }
    let (unit, cond) = match &shape.source {
        Source::Unit { unit, cond } => (*unit, cond.as_ref()),
        Source::Sessions { cond } => (Unit::Sessions, Some(cond)),
        Source::Empty => (Unit::Sessions, None),
    };
    let Some(cond) = cond else {
        // Every row by the unit's defaults, narrowed by the rest.
        let joined = if conds.is_empty() {
            unit.name().to_owned()
        } else {
            format!("{} where {conds}", unit.name())
        };
        return bound(format!("{joined} {stages}").trim_end().to_owned(), 0..0);
    };
    let head = format!("{} where (", unit.name());
    let cond = &saved[cond.span.start..cond.span.end];
    let region = head.len()..head.len() + cond.len();
    let mut joined = format!("{head}{cond})");
    if !conds.is_empty() {
        joined = format!("{joined} {conds}");
    }
    bound(format!("{joined} {stages}").trim_end().to_owned(), region)
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
struct Parameters {
    name: String,
    arguments: Vec<Argument>,
    /// The part of the text the saved query's own words are in; elsewhere
    /// a `$word` is the caller's and stays as written.
    region: Option<std::ops::Range<usize>>,
}

impl Parameters {
    fn new(name: &str, arguments: &[Argument], region: Option<std::ops::Range<usize>>) -> Self {
        Self {
            name: name.to_owned(),
            arguments: arguments.to_vec(),
            region,
        }
    }

    fn bind(&self, query: &mut Query, position: usize) -> Result<(), QueryError> {
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
            Source::Unit { cond: Some(c), .. } | Source::Sessions { cond: c } => {
                self.cond(c, used, position)?
            }
            _ => {}
        }
        for child in &mut query.attach {
            for window in &mut child.windows {
                if let Window::Cond { cond } = window {
                    self.cond(cond, used, position)?;
                }
            }
        }
        for stage in &mut query.stages {
            match &mut stage.kind {
                StageKind::Where { cond }
                | StageKind::Scope {
                    cond: Some(cond), ..
                } => self.cond(cond, used, position)?,
                StageKind::Combine { query, .. } => self.query(query, used, position)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn cond(
        &self,
        c: &mut Cond,
        used: &mut Vec<String>,
        position: usize,
    ) -> Result<(), QueryError> {
        let inside = self
            .region
            .as_ref()
            .is_none_or(|r| r.start <= c.span.start && c.span.end <= r.end);
        match &mut c.kind {
            CondKind::And { all: list } | CondKind::Or { any: list } => {
                for x in list {
                    self.cond(x, used, position)?;
                }
            }
            CondKind::Seq { steps: list, .. } => {
                for x in list {
                    self.cond(x, used, position)?;
                }
            }
            CondKind::Not { cond } | CondKind::Exists { cond, .. } => {
                self.cond(cond, used, position)?
            }
            CondKind::Count {
                cond: Some(cond), ..
            } => self.cond(cond, used, position)?,
            CondKind::In { query, .. } => self.query(query, used, position)?,
            CondKind::Match { value, .. } if inside => {
                if let Value::One { text, quoted } = value
                    && let Some(argument) = self.argument(text, used, position)?
                {
                    *text = argument.text.clone();
                    *quoted = argument.quoted;
                }
            }
            CondKind::Compare { value, .. } if inside => {
                if let Some(argument) = self.argument(value, used, position)? {
                    *value = argument.text.clone();
                }
            }
            CondKind::Between { from, to, .. } if inside => {
                for bound in [from, to] {
                    if let Some(argument) = self.argument(bound, used, position)? {
                        *bound = argument.text.clone();
                    }
                }
            }
            CondKind::Text { text, phrase } if inside => {
                if let Some(argument) = self.argument(text, used, position)? {
                    *text = argument.text.clone();
                    *phrase = argument.quoted;
                }
            }
            CondKind::Saved { arguments, .. } if inside => {
                for inner in arguments {
                    if let Some(argument) = self.argument(&inner.text, used, position)? {
                        inner.text = argument.text.clone();
                        inner.quoted = argument.quoted;
                    }
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
        if !used.iter().any(|u| u == parameter) {
            used.push(parameter.to_owned());
        }
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

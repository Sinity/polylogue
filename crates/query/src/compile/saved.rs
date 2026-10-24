//! Saved queries (`query:NAME`) and kept results (`result:<id>`): a
//! saved query is expanded into the query that names it, and a kept result
//! reads back the rows it stored.

use super::*;

impl<'a> Compiler<'a> {
    /// `query:NAME` as a condition: a saved relation over the same unit
    /// (spec query.md, Saved queries). A shaped result cannot be a
    /// condition, and units that disagree are an error.
    pub(super) fn saved_cond(
        &mut self,
        name: &str,
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        let opts = self.opts;
        let saved = saved_text(self.text, span.start, name, opts)?;
        if self.saved_stack.iter().any(|n| n == name) {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{name} refers to itself (through {})",
                    self.saved_stack.join(" -> ")
                ),
                position: span.start,
            });
        }
        let shape = parse(saved)?;
        if !shape.stages.is_empty() || !shape.attach.is_empty() {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{name} is a shaped result (it has stages), so it cannot be a condition; \
                     read it as `query:{name} | where …`"
                ),
                position: span.start,
            });
        }
        let (unit, cond) = match shape.source {
            Source::Unit { unit, cond } => (unit, cond),
            Source::Sessions { cond } => (Unit::Sessions, Some(cond)),
            Source::Empty => (Unit::Sessions, None),
        };
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
/// source (spec query.md, Saved queries): the rest of the text only narrows
/// it. A saved relation (`U where C`) takes the rest as further conditions;
/// a shaped result (with stages) takes only further stages.
pub(super) fn bind_saved_source(text: &str, opts: &Options) -> Result<Option<String>, QueryError> {
    let lead = text.len() - text.trim_start().len();
    let trimmed = text.trim_start();
    let Some(after) = trimmed
        .get(..6)
        .filter(|p| p.eq_ignore_ascii_case("query:"))
        .map(|_| &trimmed[6..])
    else {
        return Ok(None);
    };
    let end = after
        .find(|c: char| c.is_whitespace() || c == '|')
        .unwrap_or(after.len());
    let name = &after[..end];
    let rest = after[end..].trim();
    let saved = saved_text(text, lead + 6, name, opts)?;
    let shape = parse(saved)?;
    let shaped = !shape.stages.is_empty() || !shape.attach.is_empty();
    let (conds, stages) = match rest.find('|') {
        Some(i) => (rest[..i].trim(), &rest[i..]),
        None => (rest, ""),
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
        return Ok(Some(format!("{saved} {stages}")));
    }
    let base = match &shape.source {
        Source::Unit { unit, cond } => {
            let c = cond
                .as_ref()
                .map(|c| format!(" ({})", &saved[c.span.start..c.span.end]))
                .unwrap_or_default();
            format!("{} where{c}", unit.name())
        }
        Source::Sessions { cond } => {
            format!(
                "sessions where ({})",
                &saved[cond.span.start..cond.span.end]
            )
        }
        Source::Empty => "sessions where kind:*".to_owned(),
    };
    let joined = if conds.is_empty() {
        base
    } else {
        format!("{base} {conds}")
    };
    Ok(Some(format!("{joined} {stages}").trim_end().to_owned()))
}

pub(super) fn saved_text<'o>(
    text: &str,
    position: usize,
    name: &str,
    opts: &'o Options,
) -> Result<&'o str, QueryError> {
    opts.saved.get(name).map(String::as_str).ok_or_else(|| {
        let known: Vec<&str> = opts.saved.keys().map(String::as_str).collect();
        QueryError::syntax(
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
        )
    })
}

//! One typed field-expression selector for lowering and per-session facets.
use super::*;
struct Expressions<'a> {
    text: &'a str,
    now: i64,
    aliases: &'a mut usize,
}
impl Expressions<'_> {
    fn fresh(&mut self) -> Aliases {
        *self.aliases += 1;
        Aliases::suffixed(*self.aliases)
    }
    fn unknown_field(&self, field: &str, unit: Unit, span: Span) -> QueryError {
        unknown_field(self.text, field, unit, span)
    }
    fn expr(
        &mut self,
        field: &str,
        scope: &RowContext,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        let lower = field.to_ascii_lowercase();
        let span = Span {
            start: self.text[at..].find(field).map_or(at, |i| at + i),
            end: at,
        };
        let a = &scope.a;
        if lower == "session" && fields::has_session(scope.unit) {
            return Ok((format!("{}.id", a.s), ColKind::Text));
        }
        if let Some((prefix, rest)) = lower.split_once('.')
            && let Some(rel) = units::spec(scope.unit).relation(prefix)
        {
            let sub = match rel.via {
                Via::Same => RowContext {
                    unit: rel.target,
                    a: a.clone(),
                },
                Via::Session(alias) => RowContext {
                    unit: Unit::Sessions,
                    a: Aliases {
                        s: a.expand(alias),
                        ..a.clone()
                    },
                },
                Via::Exists {
                    value: Some(template),
                    ..
                } => {
                    let n = self.fresh();
                    let (e, k) = self.expr(
                        rest,
                        &RowContext {
                            unit: rel.target,
                            a: n.clone(),
                        },
                        at,
                    )?;
                    return Ok((
                        relation_sql(template, a, &n, rel.target, "").replace("{e}", &e),
                        k,
                    ));
                }
                Via::Exists { value: None, .. } => {
                    return Err(not_built(&format!(
                        "grouping or sorting by {prefix}.{rest} on {}",
                        scope.unit.name()
                    )));
                }
            };
            return self.expr(rest, &sub, at);
        }
        let def = if let Some(d) = fields::lookup(scope.unit, &lower) {
            d
        } else if fields::has_session(scope.unit)
            && let Some(d) = fields::lookup(Unit::Sessions, &lower)
        {
            d
        } else {
            return Err(self.unknown_field(field, scope.unit, span));
        };
        let Some(expr) = def.expr(a, self.now) else {
            scalar(def.column_kind(), def.name, at)?;
            return Err(QueryError::Invalid {
                message: format!("{} cannot be grouped or sorted by", def.name),
                position: span.start,
            });
        };
        Ok((expr, def.column_kind()))
    }
}
pub(super) fn expression(
    text: &str,
    unit: Unit,
    aliases: Aliases,
    field: &str,
    now: i64,
    at: usize,
    next: &mut usize,
) -> Result<(String, ColKind), QueryError> {
    Expressions {
        text,
        now,
        aliases: next,
    }
    .expr(field, &RowContext { unit, a: aliases }, at)
}

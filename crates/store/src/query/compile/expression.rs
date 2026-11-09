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
        query_language::depth::deeper(|| {
            let lower = field.to_ascii_lowercase();
            let span = Span { start: at, end: at };
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
            let expr = match def.stored_unknown {
                Some(word) => format!(
                    "CASE WHEN lower({expr}) = '{}' THEN NULL ELSE ({expr}) END",
                    word.replace('\'', "''")
                ),
                None => expr,
            };
            Ok((expr, def.column_kind()))
        })
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

impl Compiler<'_> {
    /// Expressions retain native units: nanodollars and milliseconds.
    pub(super) fn sql_expression(
        &mut self,
        value: &Expression,
        scope: &RowContext,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        self.sql_expression_with(value, at, &mut |compiler, field, at| {
            compiler.expr(field, scope, at)
        })
    }

    /// The same expression rules apply to native rows and derived columns.
    pub(super) fn sql_expression_with<F>(
        &mut self,
        value: &Expression,
        at: usize,
        field: &mut F,
    ) -> Result<(String, ColKind), QueryError>
    where
        F: FnMut(&mut Self, &str, usize) -> Result<(String, ColKind), QueryError>,
    {
        query_language::depth::deeper(|| match value {
            Expression::Field { name } => field(self, name, at),
            Expression::Value { value } => self.expression_literal(value, None, at),
            Expression::Arithmetic {
                operator,
                left,
                right,
            } => {
                let (left, left_kind, right, right_kind) =
                    self.expression_pair(left, right, Some(*operator), at, field)?;
                let kind =
                    arithmetic_column(*operator, left_kind, right_kind).ok_or_else(|| {
                        QueryError::Invalid {
                            message: "arithmetic requires compatible numeric dimensions".into(),
                            position: at,
                        }
                    })?;
                let sql = match operator {
                    Arithmetic::Divide => format!("(1.0 * ({left})) / NULLIF(({right}), 0)"),
                    _ => format!("({left}) {} ({right})", operator.symbol()),
                };
                Ok((format!("pl_finite({sql})"), kind))
            }
            Expression::Function { name, arguments } if name == "glob" => {
                let [text, pattern] = arguments.as_slice() else {
                    return Err(expression_arity(name, 2, at));
                };
                let mut arguments = Vec::with_capacity(2);
                for argument in [text, pattern] {
                    let (sql, kind) = if let Expression::Value { value } = argument {
                        self.expression_literal(value, Some(ColKind::Text), at)?
                    } else {
                        self.sql_expression_with(argument, at, field)?
                    };
                    if kind != ColKind::Text {
                        return Err(QueryError::Invalid {
                            message: "glob takes text and a text pattern".into(),
                            position: at,
                        });
                    }
                    arguments.push(sql);
                }
                Ok((
                    format!("pl_glob({}, {})", arguments[0], arguments[1]),
                    ColKind::Bool,
                ))
            }
            Expression::Function { name, arguments } if name == "case_eq" => {
                let [left, right, result] = arguments.as_slice() else {
                    return Err(expression_arity(name, 3, at));
                };
                let (left, _, right, _) = self.expression_pair(left, right, None, at, field)?;
                let (result, kind) = self.sql_expression_with(result, at, field)?;
                Ok((
                    format!("CASE WHEN ({left}) = ({right}) THEN ({result}) ELSE NULL END"),
                    kind,
                ))
            }
            Expression::Function { name, arguments } if name == "text" => {
                let [argument] = arguments.as_slice() else {
                    return Err(expression_arity(name, 1, at));
                };
                if matches!(
                    argument,
                    Expression::Value {
                        value: Value::Unknown
                    }
                ) {
                    return Ok(("NULL".into(), ColKind::Text));
                }
                let (sql, kind) = self.sql_expression_with(argument, at, field)?;
                let json = match kind {
                    ColKind::Text => 0,
                    ColKind::Json => 1,
                    _ => {
                        return Err(QueryError::Invalid {
                            message: "text takes a string value".into(),
                            position: at,
                        });
                    }
                };
                Ok((format!("pl_query_text({sql}, {json})"), ColKind::Text))
            }
            Expression::Function { name, arguments } => {
                let [argument] = arguments.as_slice() else {
                    return Err(expression_arity(name, 1, at));
                };
                let (sql, kind) = self.sql_expression_with(argument, at, field)?;
                match name.as_str() {
                    "abs"
                        if matches!(
                            kind,
                            ColKind::Number
                                | ColKind::Money
                                | ColKind::Duration
                                | ColKind::DurationSeconds
                        ) =>
                    {
                        Ok((format!("pl_abs({sql})"), kind))
                    }
                    "abs" => Err(QueryError::Invalid {
                        message: "abs takes a number, money, or duration".into(),
                        position: at,
                    }),
                    "len" => match kind {
                        ColKind::Text => Ok((format!("pl_text_length({sql})"), ColKind::Number)),
                        ColKind::Ids | ColKind::Json => {
                            Ok((format!("pl_json_length({sql})"), ColKind::Number))
                        }
                        _ => Err(QueryError::Invalid {
                            message: "len takes text or a list".into(),
                            position: at,
                        }),
                    },
                    "hour" | "day" | "week" | "weekday" | "month" | "month_day" | "year"
                        if kind == ColKind::Time =>
                    {
                        Ok((
                            format!("pl_{name}({sql})"),
                            if matches!(name.as_str(), "hour" | "year") {
                                ColKind::Number
                            } else {
                                ColKind::Text
                            },
                        ))
                    }
                    "hour" | "day" | "week" | "weekday" | "month" | "month_day" | "year" => {
                        Err(QueryError::Invalid {
                            message: format!("{name} takes a time"),
                            position: at,
                        })
                    }
                    _ => Err(not_built(&format!("function {name}"))),
                }
            }
        })
    }

    fn expression_pair<F>(
        &mut self,
        left: &Expression,
        right: &Expression,
        arithmetic: Option<Arithmetic>,
        at: usize,
        field: &mut F,
    ) -> Result<(String, ColKind, String, ColKind), QueryError>
    where
        F: FnMut(&mut Self, &str, usize) -> Result<(String, ColKind), QueryError>,
    {
        if let (Expression::Value { value: left }, Expression::Value { value: right }) =
            (left, right)
            && !matches!(arithmetic, Some(Arithmetic::Multiply | Arithmetic::Divide))
        {
            let is_money = |value: &Value| matches!(value,Value::Scalar { text,quoted:false } if text.starts_with('$') || text.to_ascii_lowercase().ends_with("usd"));
            if is_money(left) || is_money(right) {
                let (left, left_kind) = self.expression_literal(left, Some(ColKind::Money), at)?;
                let (right, right_kind) =
                    self.expression_literal(right, Some(ColKind::Money), at)?;
                return Ok((left, left_kind, right, right_kind));
            }
        }
        let contextual = |kind| match arithmetic {
            Some(Arithmetic::Add | Arithmetic::Subtract) if kind == ColKind::Money => Some(kind),
            Some(_) => None,
            None => Some(kind),
        };
        if let Expression::Value { value } = left {
            let (right_sql, right_kind) = self.sql_expression_with(right, at, field)?;
            let (left_sql, left_kind) =
                self.expression_literal(value, contextual(right_kind), at)?;
            Ok((left_sql, left_kind, right_sql, right_kind))
        } else {
            let (left_sql, left_kind) = self.sql_expression_with(left, at, field)?;
            let (right_sql, right_kind) = if let Expression::Value { value } = right {
                self.expression_literal(value, contextual(left_kind), at)?
            } else {
                self.sql_expression_with(right, at, field)?
            };
            Ok((left_sql, left_kind, right_sql, right_kind))
        }
    }

    fn expression_literal(
        &mut self,
        value: &Value,
        expected: Option<ColKind>,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        let bad = || QueryError::Invalid {
            message: "expected a finite scalar expression value".into(),
            position: at,
        };
        if let Value::Set { values } = value {
            let parts = values
                .iter()
                .map(|value| {
                    self.expression_literal(value, expected, at)
                        .map(|(sql, _)| sql)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((format!("json_array({})", parts.join(",")), ColKind::Json));
        }
        let Value::Scalar { text, quoted } = value else {
            return match value {
                Value::Unknown => Ok(("NULL".into(), expected.unwrap_or(ColKind::Number))),
                _ => Err(bad()),
            };
        };
        if expected == Some(ColKind::Text) {
            return Ok((self.text_param(text), ColKind::Text));
        }
        if expected == Some(ColKind::Money) {
            let number = time::money_nusd(text).ok_or_else(bad)?;
            return Ok((self.bind(Param::Int(number)), ColKind::Money));
        }
        if expected == Some(ColKind::Time) {
            let value = self
                .time("time", text, Span { start: at, end: at })?
                .start_ms();
            return Ok((self.bind(Param::Int(value)), ColKind::Time));
        }
        if matches!(expected, Some(ColKind::Duration | ColKind::DurationSeconds)) {
            let number = time::duration_ms(text).ok_or_else(bad)?;
            return Ok((self.bind(Param::Int(number)), expected.unwrap()));
        }
        if expected == Some(ColKind::Bool) && matches!(text.as_str(), "true" | "false") {
            return Ok((
                self.bind(Param::Int(i64::from(text == "true"))),
                ColKind::Bool,
            ));
        }
        if !quoted || expected == Some(ColKind::Number) {
            if let Ok(number) = text.parse::<i64>() {
                return Ok((self.bind(Param::Int(number)), ColKind::Number));
            }
            if let Ok(number) = text.parse::<f64>()
                && number.is_finite()
            {
                return Ok((self.bind(Param::Real(number)), ColKind::Number));
            }
            if let Some(number) = time::duration_ms(text) {
                return Ok((self.bind(Param::Int(number)), ColKind::Duration));
            }
            if (text.starts_with('$') || text.to_ascii_lowercase().ends_with("usd"))
                && let Some(number) = time::money_nusd(text)
            {
                return Ok((self.bind(Param::Int(number)), ColKind::Money));
            }
        }
        if expected == Some(ColKind::Number) {
            return Err(bad());
        }
        Ok((self.text_param(text), ColKind::Text))
    }

    pub(super) fn expression_comparison<F>(
        &mut self,
        left: &Expression,
        op: Op,
        right: &Expression,
        at: usize,
        field: &mut F,
    ) -> Result<String, QueryError>
    where
        F: FnMut(&mut Self, &str, usize) -> Result<(String, ColKind), QueryError>,
    {
        if let Expression::Value {
            value: Value::Set { values },
        } = right
        {
            let parts = values
                .iter()
                .map(|value| {
                    self.expression_comparison(
                        left,
                        if op == Op::Ne { Op::Eq } else { op },
                        &Expression::value(value.clone()),
                        at,
                        field,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let comparison = if parts.is_empty() {
                "0".into()
            } else {
                format!("({})", parts.join(" OR "))
            };
            return Ok(if op == Op::Ne {
                format!("NOT ({comparison})")
            } else {
                comparison
            });
        }
        let (left, _, right, _) = self.expression_pair(left, right, None, at, field)?;
        Ok(match op {
            Op::Contains => format!("instr(pl_lower({left}), pl_lower({right})) > 0"),
            Op::Prefix => format!("instr(({left}), ({right})) = 1"),
            _ => format!("({left}) {} ({right})", op.sql()),
        })
    }

    pub(super) fn natural_comparison<F>(
        &mut self,
        left: &Expression,
        right: &Expression,
        natural: query_language::catalog::NaturalMatch,
        at: usize,
        field: &mut F,
    ) -> Result<String, QueryError>
    where
        F: FnMut(&mut Self, &str, usize) -> Result<(String, ColKind), QueryError>,
    {
        use query_language::catalog::NaturalMatch;
        match natural {
            NaturalMatch::Equal | NaturalMatch::Same => {
                return self.expression_comparison(left, Op::Eq, right, at, field);
            }
            NaturalMatch::Contains => {
                return self.expression_comparison(left, Op::Contains, right, at, field);
            }
            NaturalMatch::Period => {
                return Err(not_built(
                    "period matching on derived time values without native period evidence",
                ));
            }
            _ => {}
        }
        if let Expression::Value {
            value: Value::Set { values },
        } = right
        {
            let parts = values
                .iter()
                .map(|value| {
                    self.natural_comparison(
                        left,
                        &Expression::value(value.clone()),
                        natural,
                        at,
                        field,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(if parts.is_empty() {
                "0".into()
            } else {
                format!("({})", parts.join(" OR "))
            });
        }
        let (left, left_kind, right, right_kind) =
            self.expression_pair(left, right, None, at, field)?;
        if left_kind != ColKind::Text || right_kind != ColKind::Text {
            return Err(QueryError::Invalid {
                message: "this natural match requires string values".into(),
                position: at,
            });
        }
        let function = if natural == NaturalMatch::Words {
            "pl_query_word"
        } else {
            "pl_query_under"
        };
        Ok(format!("{function}({left}, {right})"))
    }
}
fn expression_arity(name: &str, count: usize, at: usize) -> QueryError {
    QueryError::Invalid {
        message: format!("{name} takes {count} arguments"),
        position: at,
    }
}
fn arithmetic_column(operator: Arithmetic, left: ColKind, right: ColKind) -> Option<ColKind> {
    use query_language::catalog::FieldKind as K;
    let kind = |k| match k {
        ColKind::Number => K::Number,
        ColKind::Money => K::Money,
        ColKind::Time => K::Time,
        ColKind::Duration | ColKind::DurationSeconds => K::Duration,
        ColKind::Json => K::Json,
        _ => K::Text,
    };
    query_language::scope::arithmetic_kind(operator, &kind(left), &kind(right)).map(|kind| {
        match kind {
            K::Money => ColKind::Money,
            K::Time => ColKind::Time,
            K::Duration
                if matches!(left, ColKind::DurationSeconds)
                    || matches!(right, ColKind::DurationSeconds) =>
            {
                ColKind::DurationSeconds
            }
            K::Duration => ColKind::Duration,
            K::Json => ColKind::Json,
            _ => ColKind::Number,
        }
    })
}

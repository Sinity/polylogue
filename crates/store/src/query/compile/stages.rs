//! Stages after the conditions: `| sort by`, `| group by` with its
//! aggregates and conditions on groups, and attached rows.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn sort(
        &mut self,
        key: &Expression,
        desc: bool,
        scope: &RowContext,
        at: usize,
    ) -> Result<Vec<String>, QueryError> {
        let identity: Vec<String> = units::spec(scope.unit)
            .identity
            .iter()
            .map(|term| scope.a.expand(term))
            .collect();
        if matches!(key, Expression::Function { name, .. } if name == "random") {
            return Ok(match parse_random_seed(key)?.or(self.opts.random_seed) {
                Some(seed) => {
                    let name = match identity.as_slice() {
                        [one] => format!("CAST({one} AS TEXT)"),
                        terms => format!("json_array({})", terms.join(", ")),
                    };
                    let mut order = vec![format!("pl_random({name}, '{seed}')")];
                    order.extend(identity);
                    order
                }
                None => vec!["random()".into()],
            });
        }
        let (expr, kind) = self.sql_expression(key, scope, at)?;
        scalar(kind, &key.column(), at)?;
        if let Expression::Field { name } = key {
            self.show(name, &expr, kind, Self::field_reference(name, scope));
        }
        let dir = if desc { " DESC" } else { "" };
        let mut order = vec![format!("({expr}) IS NULL"), format!("({expr}){dir}")];
        order.extend(identity);
        Ok(order)
    }

    /// Finds the unit a field belongs to before its name becomes an alias.
    pub(super) fn field_reference(field: &str, scope: &RowContext) -> Option<ColumnReference> {
        let lower = field.to_ascii_lowercase();
        if lower == "session" && fields::has_session(scope.unit) {
            return Some(ColumnReference::Session);
        }
        if let Some((prefix, rest)) = lower.split_once('.')
            && let Some(rel) = units::spec(scope.unit).relation(prefix)
        {
            return match rel.via {
                Via::Same | Via::Session(_) => {
                    Self::field_reference(rest, &RowContext::plain(rel.target))
                }
                Via::Exists { .. } => None,
            };
        }
        fields::lookup(scope.unit, &lower)
            .or_else(|| {
                fields::has_session(scope.unit)
                    .then(|| fields::lookup(Unit::Sessions, &lower))
                    .flatten()
            })
            .and_then(|def| def.sql)
            .and_then(|e| e.reference())
    }

    /// The same typed selector supplies query lowering and native facets.
    pub(super) fn expr(
        &mut self,
        field: &str,
        scope: &RowContext,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        expression::expression(
            self.text,
            scope.unit,
            scope.a.clone(),
            field,
            self.opts.clock.now.as_millisecond(),
            at,
            &mut self.aliases,
        )
    }

    pub(super) fn group_key(
        &mut self,
        key: &Expression,
        scope: &RowContext,
        at: usize,
    ) -> Result<(String, String, ColKind), QueryError> {
        let (expr, kind) = self.sql_expression(key, scope, at)?;
        scalar(kind, &key.column(), at)?;
        Ok((key.column(), Self::display_value(&expr, kind), kind))
    }

    pub(super) fn agg(
        &mut self,
        f: &Aggregate,
        scope: &RowContext,
        at: usize,
    ) -> Result<Vec<(String, String, ColKind)>, QueryError> {
        let col = f.column();
        if !f.function.arguments().contains(&f.arguments.len()) {
            return Err(QueryError::Invalid {
                message: format!(
                    "{} received the wrong number of arguments",
                    f.function.name()
                ),
                position: at,
            });
        }
        if f.arguments.is_empty() {
            return Ok(vec![(col, "count(*)".into(), ColKind::Number)]);
        }
        let inputs = f
            .arguments
            .iter()
            .map(|argument| self.sql_expression(argument, scope, at))
            .collect::<Result<Vec<_>, _>>()?;
        let (expr, kind) = &inputs[0];
        let kind = *kind;
        if !matches!(
            f.function,
            AggregateFunction::Count
                | AggregateFunction::CountDistinct
                | AggregateFunction::Histogram
        ) {
            for ((_, kind), argument) in inputs.iter().zip(&f.arguments) {
                scalar(*kind, &argument.column(), at)?;
            }
        }
        let value = |sql: String, kind| {
            if kind == ColKind::Money {
                format!("({sql}) / 1e9")
            } else {
                sql
            }
        };
        let (sql, result_kind) = match &f.function {
            AggregateFunction::Count => (format!("count({expr})"), ColKind::Number),
            AggregateFunction::CountDistinct => {
                (format!("count(DISTINCT {expr})"), ColKind::Number)
            }
            AggregateFunction::Sum => (value(format!("sum({expr})"), kind), kind),
            AggregateFunction::Avg => (value(format!("avg({expr})"), kind), kind),
            AggregateFunction::Min => (value(format!("min({expr})"), kind), kind),
            AggregateFunction::Max => (value(format!("max({expr})"), kind), kind),
            AggregateFunction::Percentile(p) => {
                (value(format!("pl_percentile({expr}, {p})"), kind), kind)
            }
            AggregateFunction::Covered => {
                let (end, end_kind) = &inputs[1];
                if *end_kind != kind {
                    return Err(QueryError::Invalid {
                        message: "covered endpoints require the same dimension".into(),
                        position: at,
                    });
                }
                let result_kind = if kind == ColKind::Time {
                    ColKind::Duration
                } else {
                    kind
                };
                (
                    value(format!("pl_covered({expr}, {end})"), result_kind),
                    result_kind,
                )
            }
            AggregateFunction::Correlation => (
                format!("pl_correlation({expr}, {})", inputs[1].0),
                ColKind::Number,
            ),
            AggregateFunction::Histogram => (
                format!("pl_histogram({expr}, '{}')", column_kind_name(kind)),
                ColKind::Json,
            ),
        };
        let mut out = vec![(col, sql, result_kind)];
        for (argument, (sql, _)) in f.arguments.iter().zip(&inputs) {
            if let Expression::Field { .. } = argument {
                let name = format!("unknown_{}", argument.column());
                if !out.iter().any(|(prior, _, _)| prior == &name) {
                    out.push((name, format!("count(*) - count({sql})"), ColKind::Number));
                }
            }
        }
        if let Expression::Field { name: field } = &f.arguments[0] {
            let field = field.as_str();
            let flat_field = flat(field);
            if scope.unit == Unit::Sessions {
                let derivation = if crate::session_sql::COUNT_FIELDS
                    .iter()
                    .any(|(name, _)| *name == field)
                {
                    Some("counts")
                } else if matches!(
                    field,
                    "wall_span"
                        | "clustered_time"
                        | "tool_time"
                        | "tool_time_union"
                        | "tool_untimed"
                        | "ends_with"
                        | "last_stop_reason"
                        | "waiting_model"
                        | "waiting_human"
                ) {
                    Some("session_span")
                } else {
                    None
                };
                if let Some(name) = derivation {
                    out.push((
                        format!("pending_{flat_field}"),
                        format!(
                            "coalesce(sum(NOT ({})),0)",
                            crate::session_sql::current(&scope.a.s, name)
                        ),
                        ColKind::Number,
                    ));
                }
            }
            if scope.unit == Unit::Sessions && matches!(field, "tool_time" | "tool_time_union") {
                out.push((
                    format!("unmeasured_calls_{flat_field}"),
                    format!(
                        "CASE WHEN count(*)=0 THEN 0 ELSE sum({}) END",
                        crate::session_sql::span(&scope.a.s, "tool_untimed")
                    ),
                    ColKind::Number,
                ));
            }
            // A session cost known only in part (some request-bearing messages
            // unpriced) is a lower bound, and the total says how many. A tree
            // cost is one too when a session in the tree is, or when a
            // subagent the tree spawned is not in the archive.
            if scope.unit == Unit::Sessions
                && matches!(f.function, AggregateFunction::Sum | AggregateFunction::Avg)
            {
                let s = &scope.a.s;
                let unpriced = |sessions: &str| {
                    format!(
                        "EXISTS (SELECT 1 FROM message xp WHERE {} AND xp.role = 'assistant' AND xp.cost_nusd IS NULL)",
                        fields::own_messages("xp", Some(&format!("IN ({sessions})")))
                    )
                };
                let partial = match field {
                    "cost" => Some(unpriced(&format!("SELECT {s}.pk"))),
                    "tree_cost" => {
                        let tree = fields::subtree(s);
                        Some(format!(
                            "({} OR EXISTS (SELECT 1 FROM link xl JOIN session xs ON xs.id = xl.parent_id \
                         WHERE xl.kind = 'subagent' AND xl.child_pk IS NULL AND xs.pk IN ({tree})))",
                            unpriced(&tree)
                        ))
                    }
                    _ => None,
                };
                if let Some(partial) = partial {
                    out.push((
                        format!("partial_{flat_field}"),
                        format!("count(CASE WHEN ({expr}) IS NOT NULL AND {partial} THEN 1 END)"),
                        ColKind::Number,
                    ));
                }
            }
        }
        Ok(out)
    }

    /// `| where …` after grouping: conditions on the group's columns.
    pub(super) fn group_cond(
        &mut self,
        c: &Condition,
        keys: Option<&[(String, String, ColKind)]>,
        aggs: Option<&[(String, String, ColKind)]>,
    ) -> Result<String, QueryError> {
        let cols: Vec<&(String, String, ColKind)> = keys
            .unwrap_or(&[])
            .iter()
            .chain(aggs.unwrap_or(&[]).iter())
            .collect();
        let find = |field: &str| {
            cols.iter()
                .find(|(n, _, _)| n == field || *n == flat(field))
                .map(|(n, _, k)| (column_sql(n), *k))
        };
        let missing = |field: &str, at: usize| QueryError::Invalid {
            message: format!(
                "after grouping, conditions name the group's columns ({}); {field:?} is not one",
                cols.iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            position: at,
        };
        Ok(match &c.kind {
            ConditionKind::And { conditions: all } => format!(
                "({})",
                all.iter()
                    .map(|x| self.group_cond(x, keys, aggs))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" AND ")
            ),
            ConditionKind::Or { conditions: any } => format!(
                "({})",
                any.iter()
                    .map(|x| self.group_cond(x, keys, aggs))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" OR ")
            ),
            ConditionKind::Not { condition: cond } => {
                format!("NOT ({})", self.group_cond(cond, keys, aggs)?)
            }
            ConditionKind::Compare { left, op, right } => {
                self.expression_comparison(left, *op, right, c.span.start, &mut |_, field, at| {
                    let (name, kind) = find(field).ok_or_else(|| missing(field, at))?;
                    let sql = format!("g.{name}");
                    Ok((
                        if kind == ColKind::Money {
                            format!("({sql}) * 1e9")
                        } else {
                            sql
                        },
                        kind,
                    ))
                })?
            }
            ConditionKind::Between { left, from, to } => {
                let mut resolve = |_: &mut Self, field: &str, at| {
                    let (name, kind) = find(field).ok_or_else(|| missing(field, at))?;
                    let sql = format!("g.{name}");
                    Ok((
                        if kind == ColKind::Money {
                            format!("({sql}) * 1e9")
                        } else {
                            sql
                        },
                        kind,
                    ))
                };
                let lower = self.expression_comparison(
                    left,
                    Op::Ge,
                    &Expression::value(from.clone()),
                    c.span.start,
                    &mut resolve,
                )?;
                let upper = self.expression_comparison(
                    left,
                    Op::Le,
                    &Expression::value(to.clone()),
                    c.span.start,
                    &mut resolve,
                )?;
                format!("({lower}) AND ({upper})")
            }
            ConditionKind::Match {
                field,
                value: Value::Scalar { text, .. },
            } => {
                let (n, _) = find(field).ok_or_else(|| missing(field, c.span.start))?;
                let p = self.text_param(&text.to_lowercase());
                format!("lower(g.{n}) = {p}")
            }
            _ => {
                return Err(QueryError::Invalid {
                    message:
                        "after grouping, `| where` compares the group's columns, such as count >= 3"
                            .into(),
                    position: c.span.start,
                });
            }
        })
    }

    pub(super) fn attach(
        &mut self,
        child: &Attachment,
        parent: &RowContext,
    ) -> Result<Attach, QueryError> {
        let saved = std::mem::take(&mut self.params);
        let saved_ctes = std::mem::take(&mut self.ctes);
        let saved_shown = std::mem::take(&mut self.shown);
        let saved_rank = self.rank.take();
        let saved_near_rank = self.near_rank.take();
        let saved_near_join = self.near_join.take();
        let saved_fusion = self.fusion.take();
        let unit = bind_unit(&child.unit)?;
        let sc = RowContext::plain(unit);
        let conds: Vec<Condition> = child
            .windows
            .iter()
            .filter_map(|w| match w {
                Window::Where(cond) => Some(cond.clone()),
                _ => None,
            })
            .collect();
        let top: Vec<Condition> = conds.iter().flat_map(top_level).collect();
        let mut w = self.defaults(&sc, &top)?;
        for c in &conds {
            w.push(self.cond(c, &sc, false)?);
        }
        let mut select = vec![];
        let mut columns = vec![];
        for f in &child.fields {
            let lower = f.to_ascii_lowercase();
            let (expr, kind) = if unit == Unit::Messages && lower == "text" {
                (
                    format!(
                        "(SELECT group_concat(pl_text(tx.body), char(10)) FROM (SELECT xb.text_id FROM block xb WHERE xb.message_pk = {}.pk AND xb.kind = 'text' ORDER BY xb.idx) xbs JOIN text tx ON tx.id = xbs.text_id)",
                        sc.a.m
                    ),
                    ColKind::Text,
                )
            } else {
                self.expr(f, &sc, child.span.start)?
            };
            let e = match kind {
                ColKind::Money => format!("({expr}) / 1e9"),
                _ => expr,
            };
            select.push(format!("{e} AS {}", flat(f)));
            columns.push(Column {
                name: flat(f),
                kind,
                reference: Self::field_reference(f, &sc),
            });
        }
        let session = if parent.unit == Unit::Sessions {
            Ok("?PARENT".to_owned())
        } else {
            Err(not_built(&format!(
                "attaching {} to {}",
                unit.name(),
                parent.unit.name()
            )))
        };
        let parent_link = self.under_parent(parent.unit, &sc, "?PARENT", session)?;
        w.push(parent_link);
        let ctes = if self.ctes.is_empty() {
            String::new()
        } else {
            format!("WITH {} ", self.ctes.join(", "))
        };
        let order = self.default_order(&sc);
        let n = self.params.len() + 1;
        let sql = format!(
            "{ctes}SELECT {} FROM {} WHERE {} ORDER BY {}",
            if select.is_empty() {
                "*".into()
            } else {
                select.join(", ")
            },
            base_from(unit, &sc.a),
            and_all(&w),
            order.join(", ")
        )
        .replace("?PARENT", &format!("?{n}"));
        let params = std::mem::replace(&mut self.params, saved);
        self.ctes = saved_ctes;
        self.shown = saved_shown;
        self.rank = saved_rank;
        self.near_rank = saved_near_rank;
        self.near_join = saved_near_join;
        self.fusion = saved_fusion;
        let (mut first, mut last) = (None, None);
        for w in &child.windows {
            match w {
                Window::First(n) => first = Some(*n),
                Window::Last(n) => last = Some(*n),
                Window::Where(..) => {}
            }
        }
        Ok(Attach {
            name: unit.name().to_owned(),
            sql,
            params,
            columns,
            first,
            last,
        })
    }
}

fn column_kind_name(kind: ColKind) -> &'static str {
    match kind {
        ColKind::Text => "text",
        ColKind::Ids | ColKind::Json => "json",
        ColKind::Number => "number",
        ColKind::Money => "money",
        ColKind::Time => "time",
        ColKind::Duration => "duration",
        ColKind::Bool => "bool",
    }
}

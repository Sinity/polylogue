//! Stages after the conditions: `| sort by`, `| group by` with its
//! aggregates and conditions on groups, and attached rows.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn sort(
        &mut self,
        key: &SortKey,
        desc: bool,
        scope: &RowContext,
        at: usize,
    ) -> Result<Vec<String>, QueryError> {
        let identity: Vec<String> = units::spec(scope.unit)
            .identity
            .iter()
            .map(|term| scope.a.expand(term))
            .collect();
        Ok(match key {
            SortKey::Random { seed } => match seed.or(self.opts.random_seed) {
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
            },
            SortKey::Field { field } => {
                let (expr, kind) = self.expr(field, scope, at)?;
                scalar(kind, field, at)?;
                self.show(field, &expr, kind, Self::field_reference(field, scope));
                let dir = if desc { " DESC" } else { "" };
                let mut order = vec![format!("({expr}) IS NULL"), format!("({expr}){dir}")];
                order.extend(identity.iter().map(|term| format!("{term}{dir}")));
                order
            }
        })
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

    /// The value of a field, for sorting and grouping.
    pub(super) fn expr(
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
        let Some(expr) = def.expr(a, self.opts.clock.now) else {
            return Err(QueryError::Invalid {
                message: format!("{} cannot be grouped or sorted by", def.name),
                position: span.start,
            });
        };
        Ok((expr, def.column_kind()))
    }

    pub(super) fn group_key(
        &mut self,
        key: &GroupKey,
        scope: &RowContext,
        at: usize,
    ) -> Result<(String, String, ColKind), QueryError> {
        let (expr, kind) = self.expr(&key.field, scope, at)?;
        scalar(kind, &key.field, at)?;
        let name = flat(&key.field);
        Ok(match key.bucket {
            None => {
                let e = match kind {
                    ColKind::Money => format!("({expr}) / 1e9"),
                    _ => expr,
                };
                (name, e, kind)
            }
            Some(b) => {
                if kind != ColKind::Time {
                    return Err(QueryError::Invalid {
                        message: format!("{}({}) needs a time field", bucket_name(b), key.field),
                        position: at,
                    });
                }
                let zone = self.text_param(self.opts.clock.day.zone_name().unwrap_or("UTC"));
                let start = self.opts.clock.day.start_minutes;
                (
                    format!("{}_{name}", bucket_name(b)),
                    format!("pl_{}({expr}, {zone}, {start})", bucket_name(b)),
                    ColKind::Text,
                )
            }
        })
    }

    pub(super) fn agg(
        &mut self,
        f: &AggFn,
        scope: &RowContext,
        at: usize,
    ) -> Result<Vec<(String, String, ColKind)>, QueryError> {
        let col = f.column();
        let Some(field) = f.field() else {
            return Ok(vec![(col, "count(*)".into(), ColKind::Number)]);
        };
        let (expr, kind) = self.expr(field, scope, at)?;
        scalar(kind, field, at)?;
        let value = |e: String| match kind {
            ColKind::Money => format!("{e} / 1e9"),
            _ => e,
        };
        let flat_field = flat(field);
        let unknown = (
            format!("unknown_{flat_field}"),
            format!("sum(({expr}) IS NULL)"),
            ColKind::Number,
        );
        let mut out = match f {
            AggFn::Count => unreachable!("handled above"),
            AggFn::Sum { .. } => vec![(col, value(format!("sum({expr})")), kind), unknown],
            AggFn::Avg { .. } => vec![(col, value(format!("avg({expr})")), kind), unknown],
            AggFn::Min { .. } => vec![(col, value(format!("min({expr})")), kind), unknown],
            AggFn::Max { .. } => vec![(col, value(format!("max({expr})")), kind), unknown],
            AggFn::Percentile { p, .. } => {
                vec![
                    (col, value(format!("pl_percentile({expr}, {p})")), kind),
                    unknown,
                ]
            }
            AggFn::CountDistinct { .. } => {
                vec![(col, format!("count(DISTINCT {expr})"), ColKind::Number)]
            }
        };
        // A session cost known only in part (some request-bearing messages
        // unpriced) is a lower bound, and the total says how many.
        if scope.unit == Unit::Sessions
            && matches!(field, "cost" | "tree_cost")
            && matches!(f, AggFn::Sum { .. } | AggFn::Avg { .. })
        {
            out.push((
                format!("partial_{flat_field}"),
                format!(
                    "sum(({expr}) IS NOT NULL AND EXISTS (SELECT 1 FROM message xp WHERE {} AND xp.role = 'assistant' AND xp.cost_nusd IS NULL))",
                    fields::own_messages(&scope.a.s, "xp")
                ),
                ColKind::Number,
            ));
        }
        Ok(out)
    }

    /// `| where …` after grouping: conditions on the group's columns.
    pub(super) fn group_cond(
        &mut self,
        c: &Cond,
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
                .find(|(n, _, _)| *n == flat(field))
                .map(|(n, _, k)| (n.clone(), *k))
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
            CondKind::And { all } => format!(
                "({})",
                all.iter()
                    .map(|x| self.group_cond(x, keys, aggs))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" AND ")
            ),
            CondKind::Or { any } => format!(
                "({})",
                any.iter()
                    .map(|x| self.group_cond(x, keys, aggs))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" OR ")
            ),
            CondKind::Not { cond } => format!("NOT ({})", self.group_cond(cond, keys, aggs)?),
            CondKind::Compare { field, op, value } => {
                let (n, _) = find(field).ok_or_else(|| missing(field, c.span.start))?;
                let v = match value.parse::<f64>() {
                    Ok(x) if x.is_finite() => format!("{x}"),
                    _ => self.text_param(value),
                };
                format!("g.{n} {} {v}", op.sql())
            }
            CondKind::Match {
                field,
                value: Value::One { text, .. },
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
        child: &Child,
        parent: &RowContext,
    ) -> Result<Attach, QueryError> {
        let saved = std::mem::take(&mut self.params);
        let saved_ctes = std::mem::take(&mut self.ctes);
        let saved_shown = std::mem::take(&mut self.shown);
        let saved_rank = self.rank.take();
        let saved_near_rank = self.near_rank.take();
        let saved_near_join = self.near_join.take();
        let saved_fusion = self.fusion.take();
        let sc = RowContext::plain(child.unit);
        let conds: Vec<Cond> = child
            .windows
            .iter()
            .filter_map(|w| match w {
                Window::Cond { cond } => Some(cond.clone()),
                _ => None,
            })
            .collect();
        let top: Vec<Cond> = conds.iter().flat_map(top_level).collect();
        let mut w = self.defaults(&sc, &top);
        for c in &conds {
            w.push(self.cond(c, &sc, false)?);
        }
        let mut select = vec![];
        let mut columns = vec![];
        for f in &child.fields {
            let lower = f.to_ascii_lowercase();
            let (expr, kind) = if child.unit == Unit::Messages && lower == "text" {
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
                child.unit.name(),
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
            base_from(child.unit, &sc.a),
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
                Window::First { n } => first = Some(*n),
                Window::Last { n } => last = Some(*n),
                Window::Cond { .. } => {}
            }
        }
        Ok(Attach {
            name: child.unit.name().to_owned(),
            sql,
            params,
            columns,
            first,
            last,
        })
    }
}

//! Stages after the conditions: `| sort by`, `| group by` with its
//! aggregates and conditions on groups, and attached rows.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn note_field(&mut self, field: &str) {
        if field == "clustered_time" || field.ends_with(".clustered_time") {
            let note = format!(
                "clustered_time uses a {}-minute gap, extended across paired tools",
                polylogue_model::derived::CLUSTER_GAP_MS / 60_000
            );
            if !self.notes.contains(&note) {
                self.notes.push(note);
            }
        }
    }

    /// Finds the unit a field belongs to before its name becomes an alias.
    pub(super) fn field_reference(field: &str, scope: &RowContext) -> Option<ColumnReference> {
        let lower = field.to_ascii_lowercase();
        if lower == "session" && fields::has_session(scope.unit) {
            return polylogue_query::fields::column_reference(scope.unit, &lower);
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
            .filter(|def| def.sql.is_some())
            .and_then(|def| def.reference())
    }

    /// The same typed selector supplies query lowering and native facets.
    pub(super) fn expr(
        &mut self,
        field: &str,
        scope: &RowContext,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        query_language::depth::deeper(|| {
            self.note_field(field);
            expression::expression(
                self.text,
                scope.unit,
                scope.a.clone(),
                field,
                self.opts.clock.now.as_millisecond(),
                at,
                &mut self.aliases,
            )
        })
    }

    pub(super) fn agg(
        &mut self,
        f: &Aggregate,
        scope: &RowContext,
        at: usize,
    ) -> Result<Vec<(String, String, ColKind)>, QueryError> {
        let mut out = self.agg_with(f, at, &mut |compiler, field, at| {
            compiler.expr(field, scope, at)
        })?;
        if let Some(Expression::Field { name: field }) = f.arguments.first() {
            let (expr, _) = self.sql_expression(&f.arguments[0], scope, at)?;
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

    pub(super) fn agg_with<F>(
        &mut self,
        f: &Aggregate,
        at: usize,
        field: &mut F,
    ) -> Result<Vec<(String, String, ColKind)>, QueryError>
    where
        F: FnMut(&mut Self, &str, usize) -> Result<(String, ColKind), QueryError>,
    {
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
            .map(|argument| self.sql_expression_with(argument, at, field))
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
        let value = |sql: String, kind| Self::display_value(&sql, kind);
        let (sql, result_kind) = match &f.function {
            AggregateFunction::Count => (format!("count({expr})"), ColKind::Number),
            AggregateFunction::CountDistinct => {
                (format!("count(DISTINCT {expr})"), ColKind::Number)
            }
            // A sum over no rows is 0; a sum whose members are all unknown
            // stays unknown and the unknown count says so.
            AggregateFunction::Sum => (
                value(
                    format!(
                        "CASE WHEN count(*) = 0 THEN 0 ELSE pl_finite_aggregate(sum({expr}), count({expr})) END"
                    ),
                    kind,
                ),
                kind,
            ),
            // An average amount is integer nano-USD, rounded once.
            AggregateFunction::Avg if kind == ColKind::Money => (
                format!("pl_rounded_quotient(sum({expr}), count({expr}))"),
                kind,
            ),
            AggregateFunction::Avg => (
                value(
                    format!("pl_finite_aggregate(avg({expr}), count({expr}))"),
                    kind,
                ),
                kind,
            ),
            AggregateFunction::Min => (value(format!("min({expr})"), kind), kind),
            AggregateFunction::Max => (value(format!("max({expr})"), kind), kind),
            // The percentile aggregate ranks floats; an amount comes back as
            // integer nano-USD.
            AggregateFunction::Percentile(p) if kind == ColKind::Money => {
                (format!("pl_money(pl_percentile({expr}, {p}))"), kind)
            }
            AggregateFunction::Percentile(p) => {
                (value(format!("pl_percentile({expr}, {p})"), kind), kind)
            }
            AggregateFunction::Covered => {
                let (end, end_kind) = &inputs[1];
                let duration = |kind| matches!(kind, ColKind::Duration | ColKind::DurationSeconds);
                if *end_kind != kind && !(duration(*end_kind) && duration(kind)) {
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
        Ok(out)
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
            select.push(format!("{expr} AS {}", flat(f)));
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
        ColKind::DurationSeconds => "duration_seconds",
        ColKind::Bool => "bool",
    }
}

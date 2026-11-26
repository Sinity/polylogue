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
            .filter(|id| fields::rule(*id).sql.is_some())
            .and_then(|id| id.declaration().column_reference())
    }

    /// The same typed selector supplies query lowering and unit facets.
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
            let flat_field = column_name(field);
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
                let unpriced = |sessions: &str| crate::measure::unpriced_in(sessions);
                let partial = match field {
                    "cost" => Some(unpriced(&format!("SELECT {s}.pk"))),
                    "tree_cost" => {
                        let tree = fields::descendants(s);
                        Some(format!(
                            "({} OR EXISTS (SELECT 1 FROM link xl JOIN session xs ON xs.id = xl.parent_id \
                         WHERE xl.kind IN ({}) AND NOT EXISTS (SELECT 1 FROM session xc WHERE xc.id = xl.child_id) AND xs.pk IN ({tree})))",
                            unpriced(&tree),
                            fields::STRUCTURAL_KINDS
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
            // Amounts rank as integers, so every nano-USD stays exact.
            AggregateFunction::Percentile(p) if kind == ColKind::Money => {
                (format!("pl_integer_percentile({expr}, {p})"), kind)
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

    /// Rows attached to each of a page's rows, compiled as their own
    /// statement with their own parameters.
    pub(super) fn attach(
        &mut self,
        child: &polylogue_query::selection::Attached,
        parent: &RowContext,
        measured: bool,
    ) -> Result<Attach, QueryError> {
        let mut compiler = self.nested(self.text);
        compiler.aliases = self.aliases;
        let attach = compiler.attached(child, parent, measured);
        self.aliases = compiler.aliases;
        self.absorb(compiler);
        attach
    }

    fn attached(
        &mut self,
        child: &polylogue_query::selection::Attached,
        parent: &RowContext,
        measured: bool,
    ) -> Result<Attach, QueryError> {
        let span = child.span;
        let child = child.rows.as_ref().map_err(Clone::clone)?;
        let unit = child.unit;
        let sc = RowContext::plain(unit);
        let defaults = child.defaults.get(measured);
        let mut w = self.defaults(&sc, defaults);
        for filter in &child.filters {
            w.push(self.lower(filter, &sc, Cx::plain(measured))?);
        }
        let mut select = vec![];
        let mut columns = vec![];
        for f in &child.fields {
            let lower = f.to_ascii_lowercase();
            let (expr, kind) = if unit == Unit::Messages && lower == "text" {
                (
                    format!(
                        "(SELECT group_concat(pl_text_at(xg.body, tx.start, tx.length), char(10)) FROM (SELECT xb.text_id FROM block xb WHERE xb.message_pk = {}.pk AND xb.kind = 'text' ORDER BY xb.idx) xbs JOIN text tx ON tx.id = xbs.text_id JOIN content_group xg ON xg.id = tx.group_id)",
                        sc.a.m
                    ),
                    ColKind::Text,
                )
            } else {
                self.expr(f, &sc, span.start)?
            };
            select.push(format!("{expr} AS {}", column_name(f)));
            columns.push(Column {
                name: column_name(f),
                kind,
                reference: Self::field_reference(f, &sc),
            });
        }
        let session = if parent.unit == Unit::Sessions {
            Ok("_parents.value".to_owned())
        } else {
            Err(not_built(&format!(
                "attaching {} to {}",
                unit.name(),
                parent.unit.name()
            )))
        };
        let parent_link = self.under_parent(parent.unit, &sc, "_parents.value", session)?;
        w.push(parent_link);
        self.sessions_within_bounds(&sc, defaults, &mut w, true);
        let ctes = if self.ctes.is_empty() {
            String::new()
        } else {
            format!("WITH {} ", self.ctes.join(", "))
        };
        let (first, last) = (child.first, child.last);
        // `[first:n]` keeps each parent's first n children and `[last:n]` the
        // last n of those.
        let kept = match (first, last) {
            (None, None) => "1".to_owned(),
            (Some(first), None) => format!("_ordinal <= {first}"),
            (None, Some(last)) => format!("_ordinal > _sibling_count - {last}"),
            (Some(first), Some(last)) => {
                format!("_ordinal <= {first} AND _ordinal > min(_sibling_count, {first}) - {last}")
            }
        };
        let order = self.default_order(&sc, None).join(", ");
        let n = self.params.len() + 1;
        if child.fields.is_empty() {
            // No field list attaches the unit's own columns.
            let (_, own) = row_columns(unit, &sc.a, self.opts.clock.now.as_millisecond());
            for (name, expression, kind, reference) in own {
                select.push(format!("{expression} AS {}", column_sql(&name)));
                columns.push(Column {
                    name,
                    kind,
                    reference,
                });
            }
        }
        let select = select.join(", ");
        let shown = columns
            .iter()
            .map(|column| column_sql(&column.name))
            .collect::<Vec<_>>()
            .join(", ");
        // The page's parents drive the join, so each parent's children are
        // read through its key.
        let sql = format!(
            "{ctes}SELECT _parent, {shown} FROM (SELECT _parents.value AS _parent, {select}, \
             row_number() OVER (PARTITION BY _parents.value ORDER BY {order}) AS _ordinal, \
             count(*) OVER (PARTITION BY _parents.value) AS _sibling_count \
             FROM json_each(?{n}) _parents CROSS JOIN {} WHERE {}) WHERE {kept} ORDER BY _parent, _ordinal",
            base_from(unit, &sc.a),
            and_all(&w),
        );
        Ok(Attach {
            name: unit.name().to_owned(),
            sql,
            params: std::mem::take(&mut self.params),
            columns,
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

//! `exists`, `seq` and their `| matches` rows: conditions over related rows
//! and over ordered steps within a session.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn exists(
        &mut self,
        unit: Unit,
        cond: Option<&Condition>,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let (from, rows, child) = self.related_rows(unit, cond, scope, span)?;
        let Some(cond) = cond else {
            return Ok(format!("EXISTS (SELECT 1 FROM {from} WHERE {rows})"));
        };
        let inner = self.cond(cond, &child, false)?;
        // True when some child satisfies the condition, false when every
        // child fails it, unknown otherwise, so `NOT exists` leaves out a
        // row whose children are all unknown for it.
        Ok(format!(
            "(CASE WHEN EXISTS (SELECT 1 FROM {from} WHERE {rows} AND {inner}) THEN 1 \
             WHEN EXISTS (SELECT 1 FROM {from} WHERE {rows} AND ({inner}) IS NULL) THEN NULL \
             ELSE 0 END)"
        ))
    }

    /// `count(UNIT where COND) OP N`: how many related rows match. A count
    /// is always known, so the condition and its `NOT` partition the rows.
    pub(super) fn counted(
        &mut self,
        unit: Unit,
        cond: Option<&Condition>,
        op: Op,
        n: u64,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let (from, mut rows, child) = self.related_rows(unit, cond, scope, span)?;
        if let Some(cond) = cond {
            rows = format!("{rows} AND {}", self.cond(cond, &child, false)?);
        }
        let n = i64::try_from(n).unwrap_or(i64::MAX);
        Ok(format!(
            "(SELECT count(*) FROM {from} WHERE {rows}) {} {n}",
            op.sql()
        ))
    }

    /// The rows of `unit` related to one row of `scope`, by `unit`'s
    /// defaults (which `cond` may name): their FROM, their WHERE, and the
    /// aliases a condition on them is compiled over.
    fn related_rows(
        &mut self,
        unit: Unit,
        cond: Option<&Condition>,
        scope: &RowContext,
        span: Span,
    ) -> Result<(String, String, RowContext), QueryError> {
        let child = RowContext {
            unit,
            a: self.fresh(),
        };
        let key = scope.a.expand(units::spec(scope.unit).key);
        let session = self.session_alias(scope, span).map(|s| format!("{s}.pk"));
        let link = self.under_parent(scope.unit, &child, &key, session)?;
        let top = cond.map(top_level).unwrap_or_default();
        let mut w = self.defaults(&child, &top)?;
        w.push(link);
        Ok((base_from(unit, &child.a), and_all(&w), child))
    }

    /// `seq(a -> b -> c)`: tool calls of one session in session order,
    /// each step a condition on one call.
    pub(super) fn seq(
        &mut self,
        steps: &[Condition],
        gaps: &[Gap],
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        let ord_cte = "tc_order";
        if !self.ctes.iter().any(|c| c.starts_with(ord_cte)) {
            self.ctes.push(format!(
                "{ord_cte} AS (SELECT t.pk AS pk, t.session_pk AS session_pk, t.at AS at, \
                 row_number() OVER (PARTITION BY t.session_pk ORDER BY m.order_input_id, m.seq, m.record_position, m.id, cb.idx) AS n \
                 FROM tool_call t JOIN block cb ON cb.pk = t.call_block_pk JOIN message m ON m.pk = cb.message_pk \
                 WHERE {} AND {})",
                fields::own_messages("m", None), fields::counts_as_action("t")
            ));
        }
        let mut aliases = vec![];
        let mut conds = vec![];
        for step in steps {
            let a = self.fresh();
            let sc = RowContext {
                unit: Unit::ToolCalls,
                a: Aliases {
                    s: s.clone(),
                    ..a.clone()
                },
            };
            conds.push(self.cond(step, &sc, false)?);
            aliases.push(a);
        }
        // Build from the last step outwards: each step EXISTS inside the
        // previous one, ordered after it.
        let mut inner = String::new();
        for i in (0..steps.len()).rev() {
            let a = &aliases[i];
            let o = format!("o{}", a.t);
            let mut w = vec![format!("{o}.session_pk = {s}.pk"), conds[i].clone()];
            if i > 0 {
                let prev = format!("o{}", aliases[i - 1].t);
                match &gaps[i - 1] {
                    Gap::Later => w.push(format!("{o}.n > {prev}.n")),
                    Gap::Next => w.push(format!("{o}.n = {prev}.n + 1")),
                    Gap::Within {
                        distance: Distance::Steps { count },
                    } => {
                        w.push(format!("{o}.n > {prev}.n"));
                        w.push(format!("{o}.n - {prev}.n <= {count}"));
                    }
                    // Both times known, the later strictly later: a missing
                    // or reversed time does not satisfy a window.
                    Gap::Within {
                        distance: Distance::Time { duration },
                    } => {
                        let ms = time::duration_ms(value_text(duration)?)
                            .ok_or_else(|| not_built("this sequence duration"))?;
                        w.push(format!("{o}.n > {prev}.n"));
                        w.push(format!("{o}.at > {prev}.at"));
                        w.push(format!("{o}.at - {prev}.at <= {ms}"));
                    }
                }
            }
            if !inner.is_empty() {
                w.push(inner.clone());
            }
            inner = format!(
                "EXISTS (SELECT 1 FROM {ord_cte} {o} JOIN tool_call {t} ON {t}.pk = {o}.pk \
                 JOIN block {cb} ON {cb}.pk = {t}.call_block_pk JOIN message {m} ON {m}.pk = {cb}.message_pk \
                 WHERE {})",
                and_all(&w),
                t = a.t,
                cb = a.cb,
                m = a.m
            );
        }
        Ok(inner)
    }

    /// `| matches` of `exists unit(cond)`: one row per parent row and
    /// matching child, numbered within the parent in the child's order.
    pub(super) fn exists_matches(
        &mut self,
        unit: Unit,
        cond: Option<&Condition>,
        parent: &RowContext,
        parent_rows: &str,
        span: Span,
    ) -> Result<MatchRel, QueryError> {
        let (child_from, children, child) = self.related_rows(unit, cond, parent, span)?;
        let mut w = vec![children];
        if let Some(cond) = cond {
            w.push(self.cond(cond, &child, false)?);
        }
        w.push(format!("({parent_rows})"));
        let parent_pk = parent.a.expand(units::spec(parent.unit).key);
        let (child_pk, child_cols) =
            row_columns(unit, &child.a, self.opts.clock.now.as_millisecond());
        let parent_ref = row_reference(parent)?;
        let child_ref = row_reference(&child)?;
        let child_order = self.default_order(&child);
        let parent_order = self.default_order(parent);
        let mut select = vec![
            format!("{child_pk} AS _pk"),
            format!("{parent_pk} AS _match_parent_pk"),
            format!("{parent_ref} AS parent"),
            format!(
                "row_number() OVER (PARTITION BY {parent_pk} ORDER BY {}) AS match",
                child_order.join(", ")
            ),
            "1 AS position".into(),
            format!("{child_ref} AS step"),
        ];
        let mut columns = match_columns();
        let catalog = self.opts.catalog();
        if catalog.field(unit.name(), "position").is_some() {
            let (position, kind) = self.expr("position", &child, span.start)?;
            select.push(format!("{position} AS {}", column_sql("step.position")));
            columns.push(("step.position".into(), kind));
        }
        for (name, expression, kind, _) in child_cols {
            if catalog.field(unit.name(), &name).is_some() {
                let name = format!("step.{name}");
                select.push(format!("{expression} AS {}", column_sql(&name)));
                columns.push((name, kind));
            }
        }
        let (hidden, order) = hidden_order(&[parent_order, child_order].concat(), "_m");
        select.extend(hidden);
        let sql = format!(
            "SELECT {} FROM {}, {} WHERE {} ORDER BY {}",
            select.join(", "),
            base_from(parent.unit, &parent.a),
            child_from,
            and_all(&w),
            order_by(&order)
        );
        Ok(MatchRel {
            sql,
            columns,
            order,
            recursive: false,
        })
    }

    /// `| matches` of `seq(...)`: per session, the earliest match first,
    /// then the next match after its last step, never overlapping. Each
    /// step takes the earliest call that still lets the rest complete.
    pub(super) fn seq_matches(
        &mut self,
        steps: &[Condition],
        gaps: &[Gap],
        parent: &RowContext,
        parent_rows: &str,
        span: Span,
    ) -> Result<MatchRel, QueryError> {
        let s = self.session_alias(parent, span)?;
        let ord = "tc_order";
        let k = steps.len();
        let mut al = vec![];
        let mut conds = vec![];
        for step in steps {
            let a = self.fresh();
            let sc = RowContext {
                unit: Unit::ToolCalls,
                a: a.clone(),
            };
            conds.push(self.cond(step, &sc, false)?);
            al.push(a);
        }
        let o = |i: usize| format!("o{}", al[i].t);
        let src = |i: usize| {
            let a = &al[i];
            format!(
                "{ord} {o} JOIN tool_call {t} ON {t}.pk = {o}.pk JOIN block {cb} ON {cb}.pk = {t}.call_block_pk \
                 JOIN message {m} ON {m}.pk = {cb}.message_pk JOIN session {sa} ON {sa}.pk = {o}.session_pk",
                o = o(i),
                t = a.t,
                cb = a.cb,
                m = a.m,
                sa = a.s
            )
        };
        let gap = |i: usize, prev_n: &str, prev_at: &str| -> Result<Vec<String>, QueryError> {
            let oi = o(i);
            Ok(match &gaps[i - 1] {
                Gap::Later => vec![format!("{oi}.n > {prev_n}")],
                Gap::Next => vec![format!("{oi}.n = {prev_n} + 1")],
                Gap::Within {
                    distance: Distance::Steps { count },
                } => vec![
                    format!("{oi}.n > {prev_n}"),
                    format!("{oi}.n - {prev_n} <= {count}"),
                ],
                Gap::Within {
                    distance: Distance::Time { duration },
                } => {
                    let ms = time::duration_ms(value_text(duration)?)
                        .ok_or_else(|| not_built("this sequence duration"))?;
                    vec![
                        format!("{oi}.n > {prev_n}"),
                        format!("{oi}.at > {prev_at}"),
                        format!("{oi}.at - {prev_at} <= {ms}"),
                    ]
                }
            })
        };
        // rest[i]: steps i.. can still follow step i-1 (alias o(i-1)).
        let mut rest = vec![String::new(); k + 1];
        for i in (1..k).rev() {
            let prev = o(i - 1);
            let mut w = vec![
                format!("{}.session_pk = {prev}.session_pk", o(i)),
                conds[i].clone(),
            ];
            w.extend(gap(i, &format!("{prev}.n"), &format!("{prev}.at"))?);
            if !rest[i + 1].is_empty() {
                w.push(rest[i + 1].clone());
            }
            rest[i] = format!("EXISTS (SELECT 1 FROM {} WHERE {})", src(i), and_all(&w));
        }
        let pick =
            |i: usize, sess: &str, prev_n: &str, prev_at: &str| -> Result<String, QueryError> {
                let mut w = vec![format!("{}.session_pk = {sess}", o(i)), conds[i].clone()];
                if i == 0 {
                    w.push(format!("{}.n > {prev_n}", o(i)));
                } else {
                    w.extend(gap(i, prev_n, prev_at)?);
                }
                if !rest[i + 1].is_empty() {
                    w.push(rest[i + 1].clone());
                }
                Ok(format!(
                    "(SELECT {oi}.n FROM {} WHERE {} ORDER BY {oi}.n LIMIT 1)",
                    src(i),
                    and_all(&w),
                    oi = o(i)
                ))
            };
        let name = format!("seqm{}", self.aliases);
        let owners_sql = format!(
            "SELECT DISTINCT {s}.pk AS pk FROM {} WHERE {parent_rows}",
            base_from(parent.unit, &parent.a)
        );
        let mut next = vec![];
        for i in 1..k {
            next.push(format!(
                "WHEN {} THEN {}",
                i - 1,
                pick(i, "r.session_pk", "r.n", "r.at")?
            ));
        }
        next.push(format!(
            "WHEN {} THEN {}",
            k - 1,
            pick(0, "r.session_pk", "r.n", "NULL")?
        ));
        let last = k - 1;
        self.ctes.push(format!(
            "{name}(session_pk, mt, step, n, at, pk) AS (\
             SELECT ox.session_pk, 1, 0, ox.n, ox.at, ox.pk FROM ({owners_sql}) ow \
             JOIN {ord} ox ON ox.session_pk = ow.pk AND ox.n = {first} \
             UNION ALL \
             SELECT r.session_pk, CASE WHEN r.step = {last} THEN r.mt + 1 ELSE r.mt END, \
             CASE WHEN r.step = {last} THEN 0 ELSE r.step + 1 END, ox.n, ox.at, ox.pk \
             FROM {name} r JOIN {ord} ox ON ox.session_pk = r.session_pk AND ox.n = CASE r.step {cases} END)",
            first = pick(0, "ow.pk", "0", "NULL")?,
            cases = next.join(" ")
        ));
        let child = RowContext {
            unit: Unit::ToolCalls,
            a: self.fresh(),
        };
        let (_, child_cols) =
            row_columns(child.unit, &child.a, self.opts.clock.now.as_millisecond());
        let catalog = self.opts.catalog();
        let mut select = vec![
            "q.pk AS _pk".into(),
            format!(
                "{} AS _match_parent_pk",
                parent.a.expand(units::spec(parent.unit).key)
            ),
            format!("{} AS parent", row_reference(parent)?),
            "q.mt AS match".into(),
            "q.step + 1 AS position".into(),
            "q.n AS \"step.position\"".into(),
            format!("{} AS step", row_reference(&child)?),
        ];
        let mut columns = match_columns();
        columns.push(("step.position".into(), ColKind::Number));
        for (name, expression, kind, _) in child_cols {
            if catalog.field(child.unit.name(), &name).is_some() {
                let name = format!("step.{name}");
                select.push(format!("{expression} AS {}", column_sql(&name)));
                columns.push((name, kind));
            }
        }
        // A recursive prefix is exposed only after its final step exists.
        let sql = format!(
            "SELECT {} FROM {name} q JOIN ({}) ON {} = q.pk, {} \
             WHERE {s}.pk=q.session_pk AND ({parent_rows}) AND EXISTS (SELECT 1 FROM {name} complete WHERE complete.session_pk=q.session_pk \
             AND complete.mt=q.mt AND complete.step={last}) ORDER BY parent,q.mt,q.step",
            select.join(", "),
            base_from(child.unit, &child.a),
            child.a.expand(units::spec(child.unit).key),
            base_from(parent.unit, &parent.a)
        );
        Ok(MatchRel {
            sql,
            columns,
            order: vec![
                OrderKey::ascending("parent"),
                OrderKey::ascending("match"),
                OrderKey::ascending("position"),
            ],
            recursive: true,
        })
    }
}

fn match_columns() -> Vec<(String, ColKind)> {
    vec![
        ("parent".into(), ColKind::Text),
        ("match".into(), ColKind::Number),
        ("position".into(), ColKind::Number),
        ("step".into(), ColKind::Text),
    ]
}
fn row_reference(scope: &RowContext) -> Result<String, QueryError> {
    let a = &scope.a;
    Ok(match scope.unit {
        Unit::Sessions => format!("'session:' || {}.id", a.s),
        Unit::Messages => format!("'message:' || {}.id", a.m),
        Unit::Blocks => format!("'block:' || {}.id || '#' || {}.idx", a.m, a.b),
        Unit::ToolCalls => format!("'block:' || {}.id || '#' || {}.idx", a.m, a.cb),
        Unit::Documents => format!("'document:' || {}.id", a.row),
        Unit::Events => format!("'event:' || {}.id", a.row),
        Unit::Assertions => format!("'assertion:' || {}.id", a.row),
        Unit::Deliveries => format!("'delivery:' || {}.id", a.row),
        _ => return Err(not_built("match references at this row grain")),
    })
}

/// Matches retain the entering relation's visible values under `parent.`.
/// Reuse those values and metadata, including projected aliases, rather than
/// reading native fields with the same names from a different row scope.
pub(super) fn project_match_parent(result: &mut Program, parent: &Program) {
    let mut projection = vec!["matched.*".to_owned()];
    for column in &parent.columns {
        let name = format!("parent.{}", column.name);
        projection.push(format!(
            "parent_values.{} AS {}",
            column_sql(parent.column_slot(&column.name)),
            column_sql(&name)
        ));
        if parent.time_text_columns.contains(&column.name) {
            result.time_text_columns.insert(name.clone());
        }
        let mut column = column.clone();
        column.name = name;
        result.columns.push(column);
    }
    let mut order = Vec::new();
    for (index, key) in parent.order.iter().enumerate() {
        let column = format!("_match_parent_order{index}");
        projection.push(format!(
            "parent_values.{} AS {}",
            column_sql(&key.column),
            column_sql(&column)
        ));
        order.push(OrderKey {
            column,
            descending: key.descending,
        });
    }
    order.extend(result.order.iter().cloned());
    result.order = order;
    result.sql = format!(
        "SELECT {} FROM ({}) matched JOIN ({}) parent_values ON parent_values._pk = matched._match_parent_pk ORDER BY {}",
        projection.join(", "),
        result.sql,
        parent.sql,
        order_by(&result.order)
    );
}

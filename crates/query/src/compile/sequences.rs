//! `exists`, `seq` and their `| matches` rows: conditions over related rows
//! and over ordered steps within a session.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn exists(
        &mut self,
        unit: Unit,
        cond: &Cond,
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        let child = Scope {
            unit,
            a: self.fresh(),
        };
        let key = scope.a.expand(units::spec(scope.unit).key);
        let session = self.session_alias(scope, span).map(|s| format!("{s}.pk"));
        let link = self.under_parent(scope.unit, &child, &key, session)?;
        let top = top_level(cond);
        let mut w = self.defaults(&child, &top);
        w.push(link);
        w.push(self.cond(cond, &child, false)?);
        Ok(format!(
            "EXISTS (SELECT 1 FROM {} WHERE {})",
            base_from(unit, &child.a),
            and_all(&w)
        ))
    }

    /// `seq(a -> b -> c)`: tool calls of one session in session order,
    /// each step a condition on one call.
    pub(super) fn seq(
        &mut self,
        steps: &[Cond],
        gaps: &[Gap],
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        let ord_cte = "tc_order";
        if !self.ctes.iter().any(|c| c.starts_with(ord_cte)) {
            self.ctes.push(format!(
                "{ord_cte} AS (SELECT t.pk AS pk, t.session_pk AS session_pk, t.at AS at, \
                 row_number() OVER (PARTITION BY t.session_pk ORDER BY m.order_input_id, m.seq, m.record_position, m.id, cb.idx) AS n \
                 FROM tool_call t JOIN block cb ON cb.pk = t.call_block_pk JOIN message m ON m.pk = cb.message_pk \
                 WHERE m.inherited_from IS NULL AND m.duplicate_of IS NULL \
                 AND COALESCE(m.on_active_path, 1) = 1 AND {})",
                fields::counted_call("t")
            ));
        }
        let mut aliases = vec![];
        let mut conds = vec![];
        for step in steps {
            let a = self.fresh();
            let sc = Scope {
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
                match gaps[i - 1] {
                    Gap::Later => w.push(format!("{o}.n > {prev}.n")),
                    Gap::Next => w.push(format!("{o}.n = {prev}.n + 1")),
                    Gap::Within { ms } => {
                        w.push(format!("{o}.n > {prev}.n"));
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

    /// `| matches` of `exists unit(cond)`: one row per owner and matching
    /// child, numbered within the owner in the child's order.
    pub(super) fn exists_matches(
        &mut self,
        unit: Unit,
        cond: &Cond,
        owner: &Scope,
        owners: &str,
        span: Span,
    ) -> Result<MatchRel, QueryError> {
        let child = Scope {
            unit,
            a: self.fresh(),
        };
        let key = owner.a.expand(units::spec(owner.unit).key);
        let session = self.session_alias(owner, span).map(|s| format!("{s}.pk"));
        let link = self.under_parent(owner.unit, &child, &key, session)?;
        let mut w = self.defaults(&child, &top_level(cond));
        w.push(link);
        w.push(self.cond(cond, &child, false)?);
        w.push(format!("({owners})"));
        let (owner_pk, owner_cols) = row_columns(owner.unit, &owner.a, self.opts.clock.now);
        let (child_pk, child_cols) = row_columns(unit, &child.a, self.opts.clock.now);
        let mut columns = vec![];
        let mut select = vec![format!("{child_pk} AS _pk")];
        let child_names: Vec<&str> = child_cols.iter().map(|(n, _, _, _)| n.as_str()).collect();
        if !(owner.unit == Unit::Sessions && child_names.contains(&"session")) {
            let id = owner_cols
                .iter()
                .find(|(n, _, _, _)| n == "id")
                .map_or_else(|| owner_pk.clone(), |(_, e, _, _)| e.clone());
            select.push(format!("{id} AS owner"));
            columns.push(("owner".to_owned(), ColKind::Text));
        }
        let child_order = self.default_order(&child);
        let owner_order = self.default_order(owner);
        select.push(format!(
            "row_number() OVER (PARTITION BY {owner_pk} ORDER BY {}) AS match",
            child_order.join(", ")
        ));
        columns.push(("match".to_owned(), ColKind::Number));
        for (n, e, k, _) in &child_cols {
            select.push(format!("{e} AS {n}"));
            columns.push((n.clone(), *k));
        }
        let (hidden, order) = hidden_order(&[owner_order, child_order].concat(), "_m");
        select.extend(hidden);
        let sql = format!(
            "SELECT {} FROM {}, {} WHERE {} ORDER BY {}",
            select.join(", "),
            base_from(owner.unit, &owner.a),
            base_from(unit, &child.a),
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
        steps: &[Cond],
        gaps: &[Gap],
        owner: &Scope,
        owners: &str,
        span: Span,
    ) -> Result<MatchRel, QueryError> {
        let s = self.session_alias(owner, span)?;
        let ord = "tc_order";
        let k = steps.len();
        let mut al = vec![];
        let mut conds = vec![];
        for step in steps {
            let a = self.fresh();
            let sc = Scope {
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
        let gap = |i: usize, prev_n: &str, prev_at: &str| -> Vec<String> {
            let oi = o(i);
            match gaps[i - 1] {
                Gap::Later => vec![format!("{oi}.n > {prev_n}")],
                Gap::Next => vec![format!("{oi}.n = {prev_n} + 1")],
                Gap::Within { ms } => vec![
                    format!("{oi}.n > {prev_n}"),
                    format!("{oi}.at > {prev_at}"),
                    format!("{oi}.at - {prev_at} <= {ms}"),
                ],
            }
        };
        // rest[i]: steps i.. can still follow step i-1 (alias o(i-1)).
        let mut rest = vec![String::new(); k + 1];
        for i in (1..k).rev() {
            let prev = o(i - 1);
            let mut w = vec![
                format!("{}.session_pk = {prev}.session_pk", o(i)),
                conds[i].clone(),
            ];
            w.extend(gap(i, &format!("{prev}.n"), &format!("{prev}.at")));
            if !rest[i + 1].is_empty() {
                w.push(rest[i + 1].clone());
            }
            rest[i] = format!("EXISTS (SELECT 1 FROM {} WHERE {})", src(i), and_all(&w));
        }
        let pick = |i: usize, sess: &str, prev_n: &str, prev_at: &str| {
            let mut w = vec![format!("{}.session_pk = {sess}", o(i)), conds[i].clone()];
            if i == 0 {
                w.push(format!("{}.n > {prev_n}", o(i)));
            } else {
                w.extend(gap(i, prev_n, prev_at));
            }
            if !rest[i + 1].is_empty() {
                w.push(rest[i + 1].clone());
            }
            format!(
                "(SELECT {oi}.n FROM {} WHERE {} ORDER BY {oi}.n LIMIT 1)",
                src(i),
                and_all(&w),
                oi = o(i)
            )
        };
        let name = format!("seqm{}", self.aliases);
        let owners_sql = format!(
            "SELECT DISTINCT {s}.pk AS pk FROM {} WHERE {owners}",
            base_from(owner.unit, &owner.a)
        );
        let mut next = vec![];
        for i in 1..k {
            next.push(format!(
                "WHEN {} THEN {}",
                i - 1,
                pick(i, "r.session_pk", "r.n", "r.at")
            ));
        }
        next.push(format!(
            "WHEN {} THEN {}",
            k - 1,
            pick(0, "r.session_pk", "r.n", "NULL")
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
            first = pick(0, "ow.pk", "0", "NULL"),
            cases = next.join(" ")
        ));
        let mut inner = vec![
            "q.session_pk AS session_pk".to_owned(),
            "q.mt AS mt".to_owned(),
            "min(CASE WHEN q.step = 0 THEN q.pk END) AS _pk".to_owned(),
        ];
        for i in 0..k {
            inner.push(format!("max(CASE WHEN q.step = {i} THEN q.pk END) AS p{i}"));
            inner.push(format!("max(CASE WHEN q.step = {i} THEN q.n END) AS n{i}"));
            inner.push(format!("max(CASE WHEN q.step = {i} THEN q.at END) AS a{i}"));
        }
        let mut select = vec![
            "g._pk AS _pk".to_owned(),
            "(SELECT ss.id FROM session ss WHERE ss.pk = g.session_pk) AS session".to_owned(),
            "g.mt AS match".to_owned(),
        ];
        let mut columns = vec![
            ("session".to_owned(), ColKind::Text),
            ("match".to_owned(), ColKind::Number),
        ];
        for i in 0..k {
            let n = i + 1;
            select.push(format!(
                "(SELECT mm.id FROM tool_call tt JOIN block bb ON bb.pk = tt.call_block_pk \
                 JOIN message mm ON mm.pk = bb.message_pk WHERE tt.pk = g.p{i}) AS step{n}_message"
            ));
            select.push(format!(
                "(SELECT tt.command FROM tool_call tt WHERE tt.pk = g.p{i}) AS step{n}_command"
            ));
            select.push(format!(
                "(SELECT tt.outcome FROM tool_call tt WHERE tt.pk = g.p{i}) AS step{n}_outcome"
            ));
            select.push(format!("g.a{i} AS step{n}_at"));
            columns.push((format!("step{n}_message"), ColKind::Text));
            columns.push((format!("step{n}_command"), ColKind::Text));
            columns.push((format!("step{n}_outcome"), ColKind::Text));
            columns.push((format!("step{n}_at"), ColKind::Time));
        }
        for i in 1..k {
            let (steps_col, time_col) = if k == 2 {
                ("gap_steps".to_owned(), "gap_time".to_owned())
            } else {
                (format!("gap{i}_steps"), format!("gap{i}_time"))
            };
            select.push(format!("g.n{i} - g.n{} - 1 AS {steps_col}", i - 1));
            select.push(format!("g.a{i} - g.a{} AS {time_col}", i - 1));
            columns.push((steps_col, ColKind::Number));
            columns.push((time_col, ColKind::Duration));
        }
        let sql = format!(
            "SELECT {} FROM (SELECT {} FROM {name} q GROUP BY q.session_pk, q.mt) g ORDER BY g.session_pk, g.mt",
            select.join(", "),
            inner.join(", ")
        );
        Ok(MatchRel {
            sql,
            columns,
            order: vec![OrderKey::ascending("session"), OrderKey::ascending("match")],
            recursive: true,
        })
    }
}

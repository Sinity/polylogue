//! Assembling one program from a parsed query: the unit's rows, its
//! conditions, stages and the columns shown, ranked and ordered.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn program_with(
        mut self,
        q: &Query,
        preset: Option<MatchRel>,
    ) -> Result<Program, QueryError> {
        let (mut scope, wheres, top) = match &q.source {
            Source::Empty => (Scope::plain(Unit::Sessions), vec![], vec![]),
            Source::Sessions { cond } => {
                let top = top_level(cond);
                (Scope::plain(Unit::Sessions), vec![(cond.clone())], top)
            }
            Source::Unit { unit, cond } => {
                let top = cond.as_ref().map(top_level).unwrap_or_default();
                (Scope::plain(*unit), cond.iter().cloned().collect(), top)
            }
        };
        let mut where_sql: Vec<String> = self.defaults(&scope, &top);
        let defaults_only = where_sql.clone();
        let mut why = vec![];
        for c in &wheres {
            let sql = self.cond(c, &scope, true)?;
            where_sql.push(sql);
        }
        for c in &top {
            // `near:` ranks rather than filters: it keeps every row.
            let ranking = matches!(&c.kind, CondKind::Match { field, .. } if field.eq_ignore_ascii_case("near"));
            if ranking {
                continue;
            }
            let rank = self.rank.clone();
            let alone = self.cond(c, &scope, false)?;
            self.rank = rank;
            why.push((self.text[c.span.start..c.span.end].to_owned(), alone));
        }
        let mut order: Option<Vec<String>> = None;
        let mut group: Option<Vec<(String, String, ColKind)>> = None;
        let mut aggs: Option<Vec<(String, String, ColKind)>> = None;
        let mut having: Vec<String> = vec![];
        let mut group_order: Option<Vec<String>> = None;
        let mut selection: Option<(Option<u64>, Option<u64>)> = None;
        let mut distinct = false;

        let mut matched: Option<MatchRel> = preset;
        let mut measure: Option<Measure> = None;
        for stage in &q.stages {
            if measure.is_some() {
                return Err(QueryError::Invalid {
                    message: "`| measure WHAT` is the last stage".into(),
                    position: stage.span.start,
                });
            }
            let grouped = group.is_some() || aggs.is_some();
            if let Some(rel) = &matched {
                match &stage.kind {
                    StageKind::Limit { n } => {
                        let (_, off) = selection.unwrap_or((None, None));
                        selection = Some((Some(*n), off));
                    }
                    StageKind::Offset { n } => {
                        let (lim, _) = selection.unwrap_or((None, None));
                        selection = Some((lim, Some(*n)));
                    }
                    StageKind::Count | StageKind::Agg { .. } if aggs.is_some() => {
                        return Err(QueryError::Invalid {
                            message: "a query aggregates once".into(),
                            position: stage.span.start,
                        });
                    }
                    StageKind::Count => {
                        aggs = Some(vec![("count".into(), "count(*)".into(), ColKind::Number)]);
                    }
                    StageKind::Agg { fns } => {
                        let mut cols = vec![];
                        for f in fns {
                            cols.extend(match_agg(f, rel, stage.span.start)?);
                        }
                        aggs = Some(cols);
                    }
                    StageKind::GroupBy { keys } if !grouped => {
                        let mut cols = vec![];
                        for k in keys {
                            let (name, kind) = match_column(rel, &k.field, stage.span.start)?;
                            if k.bucket.is_some() {
                                return Err(not_built(
                                    "time buckets after `| matches`",
                                    "architecture 7, query language",
                                ));
                            }
                            cols.push((name.clone(), name, kind));
                        }
                        group = Some(cols);
                    }
                    StageKind::SortBy {
                        key: SortKey::Field { field },
                        descending,
                        then,
                    } if !grouped && then.is_empty() => {
                        let (name, _) = match_column(rel, field, stage.span.start)?;
                        let dir = if *descending { " DESC" } else { "" };
                        let mut sorted = vec![format!("{name} IS NULL"), format!("{name}{dir}")];
                        for identity in ["owner", "session", "match"] {
                            if rel.columns.iter().any(|(name, _)| name == identity) {
                                sorted.push(format!("\"{identity}\"{dir}"));
                            }
                        }
                        order = Some(sorted);
                    }
                    _ => {
                        return Err(not_built(
                            "this stage after `| matches` (limit, offset, sort by, group by, count and agg are built)",
                            "architecture 7, query language",
                        ));
                    }
                }
                continue;
            }
            match &stage.kind {
                StageKind::Scope { unit, cond } => {
                    if grouped || scope.unit != Unit::Sessions {
                        return Err(QueryError::Invalid {
                            message: "a `| UNIT where …` stage follows a session query".into(),
                            position: stage.span.start,
                        });
                    }
                    let sessions =
                        format!("SELECT s.pk FROM session s WHERE {}", and_all(&where_sql));
                    let next = Scope::plain(*unit);
                    let ntop = cond.as_ref().map(top_level).unwrap_or_default();
                    let mut w = self.defaults(&next, &ntop);
                    let link = if fields::has_session(*unit) {
                        format!("{}.pk IN ({sessions})", next.a.s)
                    } else {
                        self.session_link(&next, &sessions)?
                    };
                    w.push(link);
                    if let Some(c) = cond {
                        w.push(self.cond(c, &next, true)?);
                    }
                    scope = next;
                    where_sql = w;
                    why.clear();
                    order = None;
                    selection = None;
                }
                StageKind::Where { cond } if !grouped => {
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    where_sql.push(self.cond(cond, &scope, false)?);
                }
                StageKind::Where { cond } => {
                    having.push(self.group_cond(cond, group.as_deref(), aggs.as_deref())?);
                }
                StageKind::SortBy {
                    key,
                    descending,
                    then,
                } if !grouped => {
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let mut keys = self.sort(key, *descending, &scope, stage.span.start)?;
                    // The key's own tie-break goes last.
                    let pk = if !then.is_empty() && keys.len() > 1 {
                        keys.pop()
                    } else {
                        None
                    };
                    for (field, desc) in then {
                        let (expr, kind) = self.expr(field, &scope, stage.span.start)?;
                        scalar(kind, field, stage.span.start)?;
                        self.show(field, &expr, kind, Self::field_reference(field, &scope));
                        let dir = if *desc { " DESC" } else { "" };
                        keys.push(format!("({expr}) IS NULL"));
                        keys.push(format!("({expr}){dir}"));
                    }
                    keys.extend(pk);
                    order = Some(keys);
                }
                StageKind::SortBy { then, .. } if !then.is_empty() => {
                    return Err(QueryError::Invalid {
                        message: "a grouped or matched sort takes one key".into(),
                        position: stage.span.start,
                    });
                }
                StageKind::SortBy {
                    key, descending, ..
                } => {
                    group_order = Some(group_sort(
                        key,
                        *descending,
                        group.as_deref(),
                        aggs.as_deref(),
                        self.opts.random_seed,
                    )?);
                }
                StageKind::GroupBy { keys } => {
                    if grouped {
                        return Err(QueryError::Invalid {
                            message: "a query groups once".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let mut cols = vec![];
                    for k in keys {
                        cols.push(self.group_key(k, &scope, stage.span.start)?);
                    }
                    group = Some(cols);
                }
                StageKind::Count => {
                    if aggs.is_some() {
                        return Err(QueryError::Invalid {
                            message: "a query aggregates once".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() && group.is_none() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    aggs = Some(vec![("count".into(), "count(*)".into(), ColKind::Number)]);
                }
                StageKind::Agg { fns } => {
                    if aggs.is_some() {
                        return Err(QueryError::Invalid {
                            message: "a query aggregates once".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() && group.is_none() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let mut cols = vec![];
                    for f in fns {
                        cols.extend(self.agg(f, &scope, stage.span.start)?);
                    }
                    aggs = Some(cols);
                }
                StageKind::Limit { n } => {
                    let (_, off) = selection.unwrap_or((None, None));
                    selection = Some((Some(*n), off));
                }
                StageKind::Offset { n } => {
                    let (lim, _) = selection.unwrap_or((None, None));
                    selection = Some((lim, Some(*n)));
                }
                StageKind::Distinct => distinct = true,
                StageKind::Measure { what } => {
                    if grouped || scope.unit != Unit::Sessions || measure.is_some() {
                        return Err(QueryError::Invalid {
                            message: "`| measure WHAT` follows a session query and ends it".into(),
                            position: stage.span.start,
                        });
                    }
                    measure = Some(Measure {
                        what: what.clone(),
                        everything: wheres.is_empty(),
                    });
                }
                StageKind::Matches => {
                    if grouped || selection.is_some() || distinct {
                        return Err(QueryError::Invalid {
                            message:
                                "`| matches` comes before limit, offset, distinct and grouping"
                                    .into(),
                            position: stage.span.start,
                        });
                    }
                    let found: Vec<&Cond> = top
                        .iter()
                        .filter(|c| {
                            matches!(c.kind, CondKind::Exists { .. } | CondKind::Seq { .. })
                        })
                        .collect();
                    let [c] = found.as_slice() else {
                        return Err(QueryError::Invalid {
                            message: format!(
                                "`| matches` needs exactly one top-level exists(...) or seq(...); this query has {}",
                                found.len()
                            ),
                            position: stage.span.start,
                        });
                    };
                    let owners = and_all(&where_sql);
                    matched = Some(match &c.kind {
                        CondKind::Exists { unit, cond } => {
                            self.exists_matches(*unit, cond, &scope, &owners, c.span)?
                        }
                        CondKind::Seq { steps, gaps } => {
                            self.seq_matches(steps, gaps, &scope, &owners, c.span)?
                        }
                        _ => unreachable!("filtered above"),
                    });
                    order = None;
                    self.notes.push("one row per match".into());
                }
            }
        }

        let recursive = matched.as_ref().is_some_and(|m| m.recursive);
        let ctes = if self.ctes.is_empty() {
            String::new()
        } else if recursive {
            format!("WITH RECURSIVE {} ", self.ctes.join(", "))
        } else {
            format!("WITH {} ", self.ctes.join(", "))
        };
        let from = self.from(&scope)?;
        let wh = and_all(&where_sql);
        let unit = scope.unit;

        let (sql, shape, columns, order) = if let Some(rel) = matched {
            let limit = |sql: String| match selection {
                Some((lim, off)) => format!(
                    "SELECT * FROM ({sql}) LIMIT {} OFFSET {}",
                    lim.map_or(-1, |n| n.min(i64::MAX as u64) as i64),
                    off.unwrap_or(0)
                ),
                None => sql,
            };
            if group.is_some() || aggs.is_some() {
                let keys = group.clone().unwrap_or_default();
                let aggs = aggs
                    .unwrap_or_else(|| vec![("count".into(), "count(*)".into(), ColKind::Number)]);
                let mut select: Vec<String> =
                    keys.iter().map(|(n, e, _)| format!("{e} AS {n}")).collect();
                select.extend(aggs.iter().map(|(n, e, _)| format!("{e} AS {n}")));
                let columns: Vec<Column> = keys
                    .iter()
                    .chain(aggs.iter())
                    .map(|(n, _, k)| Column {
                        name: n.clone(),
                        kind: *k,
                        reference: None,
                    })
                    .collect();
                if keys.is_empty() {
                    let sql = format!("{ctes}SELECT {} FROM ({}) mr", select.join(", "), rel.sql);
                    (sql, Shape::Summary, columns, vec![])
                } else {
                    let group_by = keys
                        .iter()
                        .map(|(n, _, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let order = order.unwrap_or_else(|| {
                        let mut o = vec!["count DESC".to_owned()];
                        o.extend(keys.iter().map(|(n, _, _)| format!("{n} IS NULL, {n}")));
                        o.retain(|x| {
                            x != "count DESC" || aggs.iter().any(|(n, _, _)| n == "count")
                        });
                        o
                    });
                    let (hidden, keys) = hidden_order(&order, "_o");
                    let sql = format!(
                        "{ctes}SELECT *, {} FROM (SELECT {} FROM ({}) mr GROUP BY {group_by}) g ORDER BY {}",
                        hidden.join(", "),
                        select.join(", "),
                        rel.sql,
                        order_by(&keys)
                    );
                    (limit(sql), Shape::Groups, columns, keys)
                }
            } else {
                let columns: Vec<Column> = rel
                    .columns
                    .iter()
                    .map(|(n, k)| Column {
                        name: n.clone(),
                        kind: *k,
                        reference: None,
                    })
                    .collect();
                let (sql, keys) = match order {
                    Some(o) => {
                        let (hidden, mut keys) = hidden_order(&o, "_o");
                        keys.extend(rel.order.iter().cloned());
                        let sql = format!(
                            "{ctes}SELECT *, {} FROM ({}) mr ORDER BY {}",
                            hidden.join(", "),
                            rel.sql,
                            order_by(&keys)
                        );
                        (sql, keys)
                    }
                    None => (format!("{ctes}{}", rel.sql), rel.order.clone()),
                };
                (limit(sql), Shape::Matches, columns, keys)
            }
        } else if group.is_some() || aggs.is_some() {
            let keys = group.clone().unwrap_or_default();
            let aggs =
                aggs.unwrap_or_else(|| vec![("count".into(), "count(*)".into(), ColKind::Number)]);
            let mut select: Vec<String> =
                keys.iter().map(|(n, e, _)| format!("{e} AS {n}")).collect();
            select.extend(aggs.iter().map(|(n, e, _)| format!("{e} AS {n}")));
            let mut columns: Vec<Column> = keys
                .iter()
                .chain(aggs.iter())
                .map(|(n, _, k)| Column {
                    name: n.clone(),
                    kind: *k,
                    reference: None,
                })
                .collect();
            if keys.is_empty() {
                let sql = format!("{ctes}SELECT {} FROM {from} WHERE {wh}", select.join(", "));
                (sql, Shape::Summary, std::mem::take(&mut columns), vec![])
            } else {
                let group_by = keys
                    .iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let inner = format!(
                    "SELECT {} FROM {from} WHERE {wh} GROUP BY {group_by}",
                    select.join(", ")
                );
                let order = group_order.unwrap_or_else(|| {
                    let mut o = vec![];
                    if aggs.iter().any(|(n, _, _)| n == "count") {
                        o.push("count DESC".to_owned());
                    }
                    o.extend(keys.iter().map(|(n, _, _)| format!("{n} IS NULL, {n}")));
                    o
                });
                let (hidden, keys) = hidden_order(&order, "_o");
                let mut sql = format!(
                    "{ctes}SELECT *, {} FROM ({inner}) g WHERE {} ORDER BY {}",
                    hidden.join(", "),
                    and_all(&having),
                    order_by(&keys)
                );
                if let Some((lim, off)) = selection {
                    sql = format!(
                        "SELECT * FROM ({sql} LIMIT {} OFFSET {})",
                        lim.map_or(-1, |n| n.min(i64::MAX as u64) as i64),
                        off.unwrap_or(0)
                    );
                }
                (sql, Shape::Groups, columns, keys)
            }
        } else {
            if selection.is_some() {
                self.apply_selection(&scope, &mut where_sql, &order, selection.take());
            }
            let wh = and_all(&where_sql);
            let order = order.unwrap_or_else(|| self.default_order(&scope));
            let (pk, mut cols) = row_columns(unit, &scope.a, self.opts.clock.now);
            if !self.opts.fields.is_empty() {
                cols.clear();
                let fields = self.opts.fields.clone();
                for f in &fields {
                    let (e, k) = self.expr(f, &scope, 0)?;
                    cols.push((flat(f), e, k, Self::field_reference(f, &scope)));
                }
                self.shown.clear();
            }
            let mut columns: Vec<Column> = cols
                .iter()
                .map(|(n, _, k, reference)| Column {
                    name: n.clone(),
                    kind: *k,
                    reference: *reference,
                })
                .collect();
            for (n, e, k, reference) in std::mem::take(&mut self.shown) {
                if !columns.iter().any(|c| c.name == n) {
                    columns.push(Column {
                        name: n.clone(),
                        kind: k,
                        reference,
                    });
                    cols.push((n, e, k, reference));
                }
            }
            if let Some(rank) = &self.rank {
                cols.push(("rank".into(), rank.clone(), ColKind::Number, None));
                columns.push(Column {
                    name: "rank".into(),
                    kind: ColKind::Number,
                    reference: None,
                });
            }
            let mut select: Vec<String> = vec![format!("{pk} AS _pk")];
            select.extend(cols.iter().map(|(n, e, _, _)| format!("{e} AS {n}")));
            let (sql, keys) = if distinct {
                // Distinct rows are identified by their values alone.
                let keys: Vec<OrderKey> = columns
                    .iter()
                    .map(|c| OrderKey::ascending(&c.name))
                    .collect();
                let names = columns
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "{ctes}SELECT min(_pk) AS _pk, {names} FROM (SELECT {} FROM {from} WHERE {wh}) GROUP BY {names} ORDER BY {}",
                    select.join(", "),
                    order_by(&keys)
                );
                (sql, keys)
            } else {
                let (hidden, mut keys) = hidden_order(&order, "_o");
                select.extend(hidden);
                keys.push(OrderKey::ascending("_pk"));
                let sql = format!(
                    "{ctes}SELECT {} FROM {from} WHERE {wh} ORDER BY {}",
                    select.join(", "),
                    order_by(&keys)
                );
                (sql, keys)
            };
            (sql, Shape::Rows, columns, keys)
        };

        let mut attach = vec![];
        if !q.attach.is_empty() {
            if shape != Shape::Rows {
                return Err(QueryError::Invalid {
                    message: "attached rows apply to rows, not to groups or summaries".into(),
                    position: q.attach[0].span.start,
                });
            }
            for child in &q.attach {
                attach.push(self.attach(child, &scope)?);
            }
        }

        let why = why
            .into_iter()
            .map(|(clause, cond)| WhyClause {
                sql: format!(
                    "{ctes}SELECT count(*) FROM {from} WHERE {}",
                    and_all(&[defaults_only.clone(), vec![cond]].concat())
                ),
                clause,
            })
            .collect();
        let why_of = format!(
            "{ctes}SELECT count(*) FROM {from} WHERE {}",
            and_all(&defaults_only)
        );
        let page_size =
            self.opts
                .page_size
                .unwrap_or(if unit == Unit::Sessions && shape == Shape::Rows {
                    20
                } else {
                    50
                });
        Ok(Program {
            query: self.text.to_owned(),
            unit,
            shape,
            sql,
            params: self.params,
            columns,
            ranked: self.rank.is_some(),
            text_terms: self.text_terms,
            id_checks: self.id_checks,
            change_generations: self.change_generations,
            attach,
            notes: self.notes,
            degraded: self.degraded,
            why,
            why_of,
            page_size,
            near: self.near,
            now: self.opts.clock.now,
            zone: self.opts.clock.day.zone_name().unwrap_or("UTC").to_owned(),
            day_start_minutes: self.opts.clock.day.start_minutes,
            measure,
            order,
        })
    }

    /// A selection (`| limit`, `| offset`) turned into a membership test,
    /// so later stages filter, group or count only the selected rows.
    pub(super) fn apply_selection(
        &mut self,
        scope: &Scope,
        where_sql: &mut Vec<String>,
        order: &Option<Vec<String>>,
        selection: Option<(Option<u64>, Option<u64>)>,
    ) {
        let Some((lim, off)) = selection else {
            return;
        };
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now);
        let order = order.clone().unwrap_or_else(|| self.default_order(scope));
        let from = self.from(scope).unwrap_or_default();
        let inner = format!(
            "SELECT {pk} FROM {from} WHERE {} ORDER BY {} LIMIT {} OFFSET {}",
            and_all(where_sql),
            order.join(", "),
            lim.map_or(-1, |n| n.min(i64::MAX as u64) as i64),
            off.unwrap_or(0)
        );
        where_sql.push(format!("{pk} IN ({inner})"));
    }

    /// The FROM clause of a unit, joining the rows its fields read.
    pub(super) fn from(&self, scope: &Scope) -> Result<String, QueryError> {
        Ok(base_from(scope.unit, &scope.a))
    }

    /// The default filters a unit binds, dropped by a top-level condition
    /// naming their field (spec query.md, Defaults, binding).
    pub(super) fn defaults(&mut self, scope: &Scope, top: &[Cond]) -> Vec<String> {
        let named = named_fields(top, scope.unit);
        let reference = top.iter().any(|c| self.is_reference(c));
        units::spec(scope.unit)
            .defaults
            .iter()
            .filter(|d| !(d.unless_reference && reference))
            .filter(|d| !d.unless.iter().any(|f| named.contains(*f)))
            .map(|d| d.sql.render(&scope.a, self.opts.clock.now))
            .collect()
    }

    pub(super) fn default_order(&self, scope: &Scope) -> Vec<String> {
        let a = &scope.a;
        let spec = units::spec(scope.unit);
        let time = a.expand(spec.time);
        if let Some(rank) = &self.rank {
            return vec![
                format!("{rank} IS NULL"),
                rank.clone(),
                format!("{time} IS NULL"),
                format!("{time} DESC"),
            ];
        }
        let desc = if spec.newest_first { " DESC" } else { "" };
        let mut o = vec![format!("{time} IS NULL"), format!("{time}{desc}")];
        if spec.session_order {
            // Rows at the same time keep session order.
            o.push(format!("{}.order_input_id", a.m));
            o.push(format!("{}.seq", a.m));
        }
        o.push(format!("{}{desc}", a.expand(spec.identity)));
        o
    }
}

//! Assembling one program from a parsed query: the unit's rows, its
//! conditions, stages and the columns shown, ranked and ordered.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn program_with(
        mut self,
        q: &Query,
        preset: Option<MatchRel>,
    ) -> Result<Program, QueryError> {
        self.stages_start = q.stages.first().map_or(usize::MAX, |s| s.span.start);
        let (mut scope, wheres, top) = match &q.source {
            Source::Empty => (RowContext::plain(Unit::Sessions), vec![], vec![]),
            Source::Sessions { cond } => {
                let top = top_level(cond);
                (RowContext::plain(Unit::Sessions), vec![(cond.clone())], top)
            }
            Source::Unit { unit, cond } => {
                let top = cond.as_ref().map(top_level).unwrap_or_default();
                (
                    RowContext::plain(*unit),
                    cond.iter().cloned().collect(),
                    top,
                )
            }
        };
        let mut where_sql: Vec<String> = self.defaults(&scope, &top)?;
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
        let mut pending = combine::PendingFusion::default();
        for stage in &q.stages {
            if measure.is_some() {
                return Err(QueryError::Invalid {
                    message: "`| measure WHAT` is the last stage".into(),
                    position: stage.span.start,
                });
            }
            let fusing = matches!(
                stage.kind,
                StageKind::Combine {
                    how: Combination::Union | Combination::Fuse,
                    ..
                }
            );
            if !fusing && !pending.is_empty() {
                self.fuse_pending(
                    std::mem::take(&mut pending),
                    &scope,
                    &mut where_sql,
                    &mut order,
                )?;
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
                                return Err(not_built("time buckets after `| matches`"));
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
                        for identity in ["parent_id", "session", "match"] {
                            if rel.columns.iter().any(|(name, _)| name == identity) {
                                sorted.push(format!("\"{identity}\"{dir}"));
                            }
                        }
                        order = Some(sorted);
                    }
                    _ => {
                        return Err(not_built(
                            "this stage after `| matches` (limit, offset, sort by, group by, count and agg are built)",
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
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let sessions = format!(
                        "SELECT s.pk FROM {} WHERE {}",
                        self.from(&scope)?,
                        and_all(&where_sql)
                    );
                    // The session ranking selects sessions; children have their own order.
                    self.near_request = None;
                    self.rank = None;
                    self.near_rank = None;
                    self.near_join = None;
                    self.fusion = None;
                    self.shown.clear();
                    let next = RowContext::plain(*unit);
                    let ntop = cond.as_ref().map(top_level).unwrap_or_default();
                    let mut w = self.defaults(&next, &ntop)?;
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
                    // A fused rank and the lists it names belong to the
                    // sessions.
                    self.fusion = None;
                    self.shown.retain(|(name, ..)| name != "found_in");
                }
                StageKind::Combine { how, query } => {
                    if grouped {
                        return Err(QueryError::Invalid {
                            message: "rows combine before they are grouped or counted".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let at = stage.span.start;
                    match how {
                        Combination::Union => pending
                            .unions
                            .push(self.combined_rows(query, scope.unit, at)?),
                        Combination::Fuse => pending
                            .fuses
                            .push(self.combined_rows(query, scope.unit, at)?),
                        Combination::Intersect | Combination::Except => {
                            self.membership(*how, query, &scope, &mut where_sql, at)?
                        }
                    }
                    why.clear();
                }
                StageKind::Where { cond } if !grouped => {
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    // A later filter is one more condition on the same rows:
                    // its words rank them as they would in the `where`,
                    // unless a sort already ordered them.
                    let sql = self.cond(cond, &scope, order.is_none());
                    where_sql.push(sql?);
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
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    measure = Some(Measure {
                        what: what.clone(),
                        population_sql: String::new(),
                        selection_sql: String::new(),
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
                    let parent_rows = and_all(&where_sql);
                    matched = Some(match &c.kind {
                        CondKind::Exists { unit, cond } => {
                            self.exists_matches(*unit, cond, &scope, &parent_rows, c.span)?
                        }
                        CondKind::Seq { steps, gaps } => {
                            self.seq_matches(steps, gaps, &scope, &parent_rows, c.span)?
                        }
                        _ => unreachable!("filtered above"),
                    });
                    order = None;
                    self.notes.push("one row per match".into());
                }
            }
        }

        let rows_answer =
            matched.is_none() && group.is_none() && aggs.is_none() && measure.is_none();
        if rows_answer {
            self.setup_near(true);
        }

        self.fuse_pending(
            std::mem::take(&mut pending),
            &scope,
            &mut where_sql,
            &mut order,
        )?;
        self.fuse_text_and_near(&scope, &mut where_sql)?;
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
            if let Some(rank) = self.ranking() {
                cols.push(("rank".into(), rank, ColKind::Number, None));
                columns.push(Column {
                    name: "rank".into(),
                    kind: ColKind::Number,
                    reference: None,
                });
            }
            let mut select: Vec<String> = vec![format!("{pk} AS _pk")];
            select.extend(cols.iter().map(|(n, e, _, _)| format!("{e} AS {n}")));
            select.push(format!(
                "{} AS _near",
                self.near_rank.as_deref().unwrap_or("1")
            ));
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
                    "{ctes}SELECT min(_pk) AS _pk, {names}, max(_near) AS _near FROM (SELECT {} FROM {from} WHERE {wh}) GROUP BY {names} ORDER BY {}",
                    select.join(", "),
                    order_by(&keys)
                );
                (sql, keys)
            } else {
                let (hidden, keys) = hidden_order(&order, "_o");
                select.extend(hidden);
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
        if let Some(measure) = &mut measure {
            measure.selection_sql = sql.clone();
            measure.population_sql = format!(
                "SELECT {s}.pk AS _pk FROM session {s} WHERE {defaults}{parameters}",
                s = scope.a.s,
                defaults = and_all(&defaults_only),
                parameters = if self.params.is_empty() {
                    String::new()
                } else {
                    format!(" AND (?{0} IS ?{0})", self.params.len())
                }
            );
        }
        let why_of = format!(
            "{ctes}SELECT count(*) FROM {from} WHERE {}",
            and_all(&defaults_only)
        );
        let ranked = self.ranking().is_some();
        let page_size =
            self.opts
                .page_size
                .unwrap_or(if unit == Unit::Sessions && shape == Shape::Rows {
                    20
                } else {
                    50
                });
        if self.near_unavailable {
            let note = "semantic search is not enabled; rows are ordered by time and unranked";
            if !self.incomplete.iter().any(|n| n == note) {
                self.incomplete.push(note.into());
            }
        }
        if columns.iter().any(|c| c.name.ends_with("clustered_time")) {
            self.notes.push(format!(
                "clustered_time uses a {}-minute gap, extended across paired tools",
                polylogue_model::derived::CLUSTER_GAP_MS / 60_000
            ));
        }
        Ok(Program {
            query: self.text.to_owned(),
            unit,
            shape,
            sql,
            params: self.params,
            columns,
            ranked,
            text_terms: self.text_terms,
            id_checks: self.id_checks,
            change_generations: self.change_generations,
            attach,
            notes: self.notes,
            incomplete: self.incomplete,
            why,
            why_of,
            page_size,
            counts_near: shape == Shape::Rows && self.near_rank.is_some(),
            near: self.near,
            definitions: self.definitions,
            now: self.opts.clock.now,
            zone: self.opts.clock.day.zone_name().unwrap_or("UTC").to_owned(),
            day_start_minutes: self.opts.clock.day.start_minutes,
            measure,
            order,
        })
    }

    /// Words and `near:` together: the conditions and both orders are
    /// evaluated once, over the rows the conditions keep, and each row's
    /// fused rank joins it. Done once, when the rows are complete or when a
    /// combining stage needs their order.
    pub(super) fn fuse_text_and_near(
        &mut self,
        scope: &RowContext,
        where_sql: &mut Vec<String>,
    ) -> Result<(), QueryError> {
        if self.fusion.is_some() {
            return Ok(());
        }
        let (Some(text), Some(near)) = (self.rank.clone(), self.near_rank.clone()) else {
            return Ok(());
        };
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now);
        let rows = format!("f{}", self.ctes.len() + 1);
        self.ctes.push(format!(
            "{rows} AS MATERIALIZED (SELECT {pk} AS k, {text} AS t, {near} AS n FROM {} WHERE {})",
            self.from(scope)?,
            and_all(where_sql)
        ));
        let fused = format!("f{}", self.ctes.len() + 1);
        self.ctes.push(format!(
            "{fused} AS MATERIALIZED (SELECT k, -(1.0 / ({RANK_FUSION_OFFSET} + row_number() OVER (ORDER BY t IS NULL, t, k))) \
             - CASE WHEN n IS NULL THEN 0.0 ELSE 1.0 / ({RANK_FUSION_OFFSET} + row_number() OVER (ORDER BY n IS NULL, n, k)) END AS r \
             FROM {rows})"
        ));
        // The fused rows are the rows the conditions keep: the join
        // replaces evaluating them again.
        self.fusion = Some(Fusion {
            join: format!("JOIN {fused} fused ON fused.k = {pk}"),
            rank: "fused.r".into(),
        });
        where_sql.clear();
        Ok(())
    }

    /// A selection (`| limit`, `| offset`) turned into a membership test,
    /// so later stages filter, group or count only the selected rows.
    pub(super) fn apply_selection(
        &mut self,
        scope: &RowContext,
        where_sql: &mut Vec<String>,
        order: &Option<Vec<String>>,
        selection: Option<(Option<u64>, Option<u64>)>,
    ) {
        let Some((lim, off)) = selection else {
            return;
        };
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now);
        if order.is_none() {
            self.setup_near(true);
        }
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
    pub(super) fn from(&self, scope: &RowContext) -> Result<String, QueryError> {
        let mut from = base_from(scope.unit, &scope.a);
        let fused = self.fusion.as_ref().map(|f| &f.join);
        for join in [self.near_join.as_ref(), fused].into_iter().flatten() {
            from.push(' ');
            from.push_str(join);
        }
        Ok(from)
    }

    /// The default filters a unit binds, dropped by a top-level condition
    /// naming their field.
    pub(super) fn defaults(
        &mut self,
        scope: &RowContext,
        top: &[Cond],
    ) -> Result<Vec<String>, QueryError> {
        // OR and NOT keep the outer defaults: only top-level ANDs expand.
        let bound = self.saved_defaults(scope, top)?;
        let named = named_fields(&bound, scope.unit);
        let reference = bound.iter().any(|c| self.is_reference(c));
        Ok(units::spec(scope.unit)
            .defaults
            .iter()
            .filter(|d| !(d.unless_reference && reference))
            .filter(|d| !d.unless.iter().any(|f| named.contains(*f)))
            .map(|d| d.sql.render(&scope.a, self.opts.clock.now))
            .collect())
    }

    /// How the rows rank, lower first: by full-text relevance, by
    /// similarity to a `near:` text, or, with both, by reciprocal rank
    /// fusion of the two orders over the rows the conditions keep (each
    /// row scores `1/(60 + its place)` in each order it is in; a row with
    /// no vector is in the text order only).
    pub(super) fn ranking(&self) -> Option<String> {
        if let Some(fusion) = &self.fusion {
            return Some(fusion.rank.clone());
        }
        self.rank.clone().or_else(|| self.near_rank.clone())
    }

    pub(super) fn default_order(&self, scope: &RowContext) -> Vec<String> {
        let a = &scope.a;
        let spec = units::spec(scope.unit);
        let time = a.expand(spec.time);
        if let Some(rank) = self.ranking() {
            // Equal ranks at equal times still need one order, or a page
            // continuing after one of them would skip the others.
            let mut o = vec![
                format!("{rank} IS NULL"),
                rank,
                format!("{time} IS NULL"),
                format!("{time} DESC"),
            ];
            o.extend(
                spec.identity
                    .iter()
                    .map(|term| format!("{} DESC", a.expand(term))),
            );
            return o;
        }
        let desc = if spec.newest_first { " DESC" } else { "" };
        let mut o = vec![format!("{time} IS NULL"), format!("{time}{desc}")];
        if spec.session_order {
            // Rows at the same time keep session order.
            o.push(format!("{}.order_input_id", a.m));
            o.push(format!("{}.seq", a.m));
            o.push(format!("{}.record_position", a.m));
        }
        o.extend(
            spec.identity
                .iter()
                .map(|term| format!("{}{desc}", a.expand(term))),
        );
        o
    }
}

//! Assembling one program from a parsed query: the unit's rows, its
//! conditions, stages and the columns shown, ranked and ordered.

use super::*;

impl<'a> Compiler<'a> {
    /// A selection and the `stages` that read its rows as they are.
    pub(super) fn native_program_with(
        mut self,
        selection: &Selection,
        stages: &[Transform],
        preset: Option<MatchRel>,
    ) -> Result<Program, QueryError> {
        self.measuring = selection.measured;
        self.rank_index = selection.search.rank;
        let mut scope = RowContext::plain(selection.unit);
        if let Some(semantic) = &selection.semantic {
            if !self.source_near_allowed {
                return Err(custom_fields::similar_misplaced(semantic.span));
            }
            self.near_request = Some((
                Near {
                    text: semantic.text.clone(),
                },
                RowContext {
                    unit: semantic.unit,
                    a: scope.a.clone(),
                },
            ));
        }
        let mut top = selection.top_level();
        let applied: Vec<(String, String)> = selection
            .defaults
            .iter()
            .map(|&id| (id.describe(), units::default_sql(id, &scope.a)))
            .collect();
        let mut where_sql: Vec<String> = applied.iter().map(|(_, sql)| sql.clone()).collect();
        let defaults_only = where_sql.clone();
        if let Some(input) = &self.input_rows {
            let key = scope.a.expand(units::spec(scope.unit).key);
            where_sql.push(format!("{key} IN (SELECT _pk FROM ({input}))"));
        }
        let mut why = vec![];
        // An order condition makes the source a relation boundary, read by
        // the relation that holds it.
        if let Some(filter) = &selection.filter
            && !selection.written.iter().any(contains_order)
        {
            where_sql.push(self.lower(filter, &scope, true)?);
        }
        self.snippet_terms.extend(selection.search.terms.clone());
        for c in &top {
            if contains_order(c) {
                continue;
            }
            let rank = self.rank.clone();
            let alone = self.cond(c, &scope, false)?;
            self.rank = rank;
            // Bound conditions retain their original spans, while the compiler's
            // text is a formatted expansion. Render the bound clause itself.
            let clause = Query {
                source: Source::Unit {
                    unit: scope.unit.name().into(),
                    implicit: false,
                    condition: Some(c.clone()),
                },
                transforms: vec![],
                attach: vec![],
            };
            let catalog = self.catalog.clone().unwrap_or_else(|| self.opts.catalog());
            let clause = query_language::format::format_query(&clause, &catalog)
                .map_err(|error| QueryError::language(error, self.text))?;
            let prefix = format!("{} where ", scope.unit.name());
            why.push((
                clause
                    .strip_prefix(&prefix)
                    .expect("unit clause")
                    .to_owned(),
                alone,
            ));
        }
        if selection.order == Order::Sorted {
            self.near_request = None;
        }
        let mut matched = preset;
        let mut measure: Option<Measure> = None;
        for stage in stages {
            if measure.is_some() {
                return Err(QueryError::Invalid {
                    message: "`| measure WHAT` is the last stage".into(),
                    position: stage.span.start,
                });
            }
            match &stage.kind {
                TransformKind::Scope {
                    unit,
                    condition: cond,
                } => {
                    if scope.unit != Unit::Sessions {
                        return Err(QueryError::Invalid {
                            message: "a `| UNIT where …` stage follows a session query".into(),
                            position: stage.span.start,
                        });
                    }
                    let sessions = format!(
                        "SELECT s.pk FROM {} WHERE {}",
                        self.from(&scope)?,
                        and_all(&where_sql)
                    );
                    // The session ranking selects sessions; children have their own order.
                    self.near_request = None;
                    self.rank = None;
                    self.rank_index = None;
                    self.near_rank = None;
                    self.near_join = None;
                    self.fusion = None;
                    self.shown.clear();
                    let unit = bind_unit(unit)?;
                    let next = RowContext::plain(unit);
                    let ntop = cond.as_ref().map(top_level).unwrap_or_default();
                    let mut w = self.defaults(&next, &ntop)?;
                    let link = if fields::has_session(unit) {
                        format!("{}.pk IN ({sessions})", next.a.s)
                    } else {
                        self.session_link(&next, &sessions)?
                    };
                    w.push(link);
                    // Snippets look for the terms of the rows shown.
                    self.snippet_terms.clear();
                    if let Some(c) = cond {
                        let filter = self.binder().bind(c, unit)?;
                        let search = polylogue_query::search::Search::of(Some(&filter), unit);
                        self.rank_index = search.rank;
                        w.push(self.lower(&filter, &next, true)?);
                        self.snippet_terms.extend(search.terms);
                    }
                    scope = next;
                    top = ntop;
                    where_sql = w;
                    why.clear();
                    // A fused rank and the lists it names belong to the
                    // sessions.
                    self.fusion = None;
                    self.shown.retain(|(name, ..)| name != "found_in");
                }
                TransformKind::Extension { name, arguments }
                    if name == polylogue_query::catalog::MEASURE =>
                {
                    let [ExtensionArgument::Value(what), rest @ ..] = arguments.as_slice() else {
                        return Err(not_built("this measure declaration"));
                    };
                    if rest.iter().any(|argument| !matches!(argument, ExtensionArgument::Optional(values) if values.is_empty())) {
                        return Err(not_built("measure by"));
                    }
                    if scope.unit != Unit::Sessions || measure.is_some() {
                        return Err(QueryError::Invalid {
                            message: "`| measure WHAT` follows a session query and ends it".into(),
                            position: stage.span.start,
                        });
                    }
                    measure = Some(Measure {
                        what: value_text(what)?.into(),
                        population_sql: String::new(),
                        selection_sql: String::new(),
                        everything: selection.written.is_empty(),
                        since: None,
                        until: None,
                    });
                }
                TransformKind::Matches => {
                    let found: Vec<&Condition> = top
                        .iter()
                        .filter(|c| {
                            matches!(
                                c.kind,
                                ConditionKind::Exists { .. } | ConditionKind::Sequence { .. }
                            )
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
                        ConditionKind::Exists { unit, condition } => self.exists_matches(
                            bind_unit(unit)?,
                            condition.as_deref(),
                            &scope,
                            &parent_rows,
                            c.span,
                        )?,
                        ConditionKind::Sequence {
                            steps,
                            gaps,
                            all: false,
                            ..
                        } => self.seq_matches(steps, gaps, &scope, &parent_rows, c.span)?,
                        ConditionKind::Sequence { all: true, .. } => {
                            return Err(not_built(
                                "overlapping sequence Matches projection (seq all)",
                            ));
                        }
                        _ => {
                            return Err(QueryError::Invalid {
                                message: "matches requires an exists or sequence qualifier".into(),
                                position: c.span.start,
                            });
                        }
                    });
                    self.notes.push("one row per step of each match".into());
                }
                TransformKind::AsOf { .. } => return Err(relations::as_of_unavailable()),
                _ => return Err(not_built("this shared query stage")),
            }
        }

        let rows_answer = matched.is_none() && measure.is_none();
        if rows_answer {
            self.setup_near(true, &scope, &where_sql);
        }
        let count_relation = if rows_answer {
            Some((self.from(&scope)?, and_all(&where_sql)))
        } else {
            None
        };
        self.fuse_text_and_near(&scope, &mut where_sql)?;
        let recursive = matched.as_ref().is_some_and(|m| m.recursive);
        let ctes = if self.ctes.is_empty() {
            String::new()
        } else {
            format!(
                "WITH {}{} ",
                if recursive { "RECURSIVE " } else { "" },
                self.ctes.join(", ")
            )
        };
        let from = self.from(&scope)?;
        let wh = and_all(&where_sql);
        let unit = scope.unit;
        let mut row_count = None;
        let mut row_membership = None;
        let mut keyed = None;
        let (sql, shape, columns, order) =
            if let Some(rel) = matched {
                let columns = rel.columns;
                (
                    format!("{ctes}{}", rel.sql),
                    Shape::Matches,
                    columns,
                    rel.order,
                )
            } else {
                let order = self.default_order(&scope);
                let (pk, mut cols) =
                    row_columns(unit, &scope.a, self.opts.clock.now.as_millisecond());
                if let Projection::Fields(fields) = &selection.projection {
                    cols.clear();
                    for field in fields {
                        let (expression, kind) = self.expr(field, &scope, 0)?;
                        cols.push((
                            flat(field),
                            Self::display_value(&expression, kind),
                            kind,
                            Self::field_reference(field, &scope),
                        ));
                    }
                    self.shown.clear();
                }
                let mut columns: Vec<Column> = cols
                    .iter()
                    .map(|(name, _, kind, reference)| Column {
                        name: name.clone(),
                        kind: *kind,
                        reference: *reference,
                    })
                    .collect();
                for (name, expression, kind, reference) in std::mem::take(&mut self.shown) {
                    if !columns.iter().any(|c| c.name == name) {
                        columns.push(Column {
                            name: name.clone(),
                            kind,
                            reference,
                        });
                        cols.push((name, expression, kind, reference));
                    }
                }
                // The page's rows read their own columns; the rank shown comes
                // from the key, which holds it already.
                let row_select: Vec<String> =
                    [
                        format!("{pk} AS _pk"),
                        format!("{} AS _recency", scope.a.expand(units::spec(unit).time)),
                    ]
                    .into_iter()
                    .chain(cols.iter().map(|(name, expression, _, _)| {
                        format!("{expression} AS {}", column_sql(name))
                    }))
                    .chain(units::spec(unit).identity.iter().enumerate().map(
                        |(index, expression)| {
                            format!("{} AS _identity{index}", scope.a.expand(expression))
                        },
                    ))
                    .collect();
                if let Some(rank) = self.ranking() {
                    cols.push(("rank".into(), rank, ColKind::Number, None));
                    columns.push(Column {
                        name: "rank".into(),
                        kind: ColKind::Number,
                        reference: None,
                    });
                }
                let mut select = vec![
                    format!("{pk} AS _pk"),
                    format!("{} AS _recency", scope.a.expand(units::spec(unit).time)),
                ];
                if let Some(rank) = self.ranking() {
                    select.push(format!("{rank} AS _rank"));
                }
                select.extend(cols.iter().map(|(name, expression, _, _)| {
                    format!("{expression} AS {}", column_sql(name))
                }));
                select.push(format!(
                    "{} AS _near",
                    self.near_rank.as_deref().unwrap_or("1")
                ));
                for (index, expression) in units::spec(unit).identity.iter().enumerate() {
                    select.push(format!(
                        "{} AS _identity{index}",
                        scope.a.expand(expression)
                    ));
                }
                let without = self.near_rank.as_ref().map_or_else(
                    || "0".to_owned(),
                    |near| format!("count(*) FILTER (WHERE ({near}) IS NULL)"),
                );
                let (count_from, count_where) = count_relation
                    .as_ref()
                    .map(|(from, wh)| (from.as_str(), wh.as_str()))
                    .unwrap_or((&from, &wh));
                row_count = Some(format!(
                    "{ctes}SELECT count(*), {without} FROM {count_from} WHERE {count_where}"
                ));
                row_membership = Some(format!(
                    "{ctes}SELECT {pk} AS _pk,{} AS _near FROM {count_from} WHERE {count_where}",
                    self.near_rank.as_deref().unwrap_or("1")
                ));
                let (hidden, keys) = hidden_order(&order, "_o");
                if measure.is_none() {
                    let mut key_select = vec![format!("{pk} AS _pk")];
                    if let Some(rank) = self.ranking() {
                        key_select.push(format!("{rank} AS _rank"));
                    }
                    key_select.push(format!(
                        "{} AS _near",
                        self.near_rank.as_deref().unwrap_or("1")
                    ));
                    key_select.extend(hidden.iter().cloned());
                    keyed = Some((
                        format!(
                            "{ctes}SELECT {} FROM {from} WHERE {wh}",
                            key_select.join(", ")
                        ),
                        format!(
                            "{ctes}SELECT {} FROM {}",
                            row_select.join(", "),
                            base_from(unit, &scope.a)
                        ),
                        pk.clone(),
                    ));
                }
                select.extend(hidden);
                let sql = format!(
                    "{ctes}SELECT {} FROM {from} WHERE {wh} ORDER BY {}",
                    select.join(", "),
                    order_by(&keys)
                );
                (sql, Shape::Rows, columns, keys)
            };
        let mut attach = vec![];
        if !selection.attach.is_empty() {
            if shape != Shape::Rows {
                return Err(QueryError::Invalid {
                    message: "attached rows apply to rows, not to groups or summaries".into(),
                    position: selection.attach[0].span.start,
                });
            }
            for child in &selection.attach {
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
            (measure.since, measure.until) = self.session_window;
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
        let why_defaults = applied
            .iter()
            .enumerate()
            .map(|(i, (describe, _))| {
                let others: Vec<String> = defaults_only
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, sql)| sql.clone())
                    .collect();
                WhyClause {
                    clause: describe.clone(),
                    sql: format!(
                        "{ctes}SELECT count(*) FROM {from} WHERE {}",
                        and_all(&others)
                    ),
                }
            })
            .collect();
        let ranked = self.ranking().is_some();
        let counts_near = shape == Shape::Rows && self.near_rank.is_some();
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
        let row_basis = if shape == Shape::Matches {
            RowBasis::Group {
                columns: vec!["parent".into(), "match".into(), "position".into()],
            }
        } else {
            RowBasis::Native {
                unit,
                columns: units::spec(unit)
                    .identity
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("_identity{i}"))
                    .collect(),
            }
        };
        // Only the answer's own rows are read key first; a stage or a
        // combination over them reads them whole.
        let statements = match (keyed, row_count) {
            (Some((keys, rows, key)), Some(count)) if self.answer => Statements::Keyed(KeyedRows {
                keys,
                rows,
                key,
                count,
            }),
            (_, count) => Statements::Derived(DerivedRows {
                rows: sql,
                count,
                members: row_membership,
            }),
        };
        Ok(Program {
            row_basis,
            query: self.text.to_owned(),
            unit,
            shape,
            statements,
            params: self.params,
            columns,
            column_slots: Default::default(),
            native_sources: None,
            time_text_columns: Default::default(),
            ranked,
            snippet_terms: self.snippet_terms,
            id_checks: self.id_checks,
            change_generations: self.change_generations,
            assertion_cutoffs: self.assertion_cutoffs,
            attach,
            notes: self.notes,
            incomplete: self.incomplete,
            why,
            why_of,
            why_defaults,
            page_size,
            counts_near,
            near: self.near,
            definitions: self.definitions,
            clock: self.opts.clock.clone(),
            resolved_times: self.resolved_times.into_inner(),
            measure,
            order,
        })
    }

    /// Words and `similar:` together: the conditions and both orders are
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
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now.as_millisecond());
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

    /// The FROM clause of a unit, joining the rows its fields read.
    pub(super) fn from(&self, scope: &RowContext) -> Result<String, QueryError> {
        let mut from = self.row_source(scope);
        if let Some(fusion) = &self.fusion {
            from.push(' ');
            from.push_str(&fusion.join);
        }
        Ok(from)
    }

    fn row_source(&self, scope: &RowContext) -> String {
        let mut from = base_from(scope.unit, &scope.a);
        if let Some(join) = &self.near_join {
            from.push(' ');
            from.push_str(join);
        }
        from
    }

    /// The default filters a unit binds, dropped by a top-level condition
    /// naming their field.
    pub(super) fn defaults(
        &mut self,
        scope: &RowContext,
        top: &[Condition],
    ) -> Result<Vec<String>, QueryError> {
        Ok(
            polylogue_query::defaults::bind(scope.unit, top, &self.opts.origins, self.measuring)
                .into_iter()
                .map(|id| units::default_sql(id, &scope.a))
                .collect(),
        )
    }

    /// How the rows rank, lower first: by full-text relevance, by
    /// similarity to a `similar:` text, or, with both, by reciprocal rank
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
            o.push(format!("{}.session_order", a.m));
        }
        o.extend(
            spec.identity
                .iter()
                .map(|term| format!("{}{desc}", a.expand(term))),
        );
        o
    }
}

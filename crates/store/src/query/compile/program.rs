//! Assembling one program from a parsed query: the unit's rows, its
//! conditions, stages and the columns shown, ranked and ordered.

use super::*;

/// One selection's SQL while it is lowered: its rows, conditions and stages.
struct SelectionSql {
    scope: RowContext,
    top: Vec<Condition>,
    applied: Vec<(units::DefaultId, String)>,
    defaults_only: Vec<String>,
    where_sql: Vec<String>,
    why: Vec<(String, String)>,
    matched: Option<MatchRel>,
    measure: Option<Measure>,
    explicit_order: Option<Vec<String>>,
}

struct RowStatements {
    sql: String,
    shape: Shape,
    columns: Vec<Column>,
    order: Vec<OrderKey>,
    count: Option<String>,
    members: Option<String>,
    keyed: Option<(String, String)>,
}

impl RowStatements {
    fn row_kind(&self, unit: Unit) -> RowKind {
        if self.shape == Shape::Matches {
            RowKind::Group {
                columns: vec!["parent".into(), "match".into(), "position".into()],
            }
        } else {
            RowKind::UnitRows {
                unit,
                columns: units::spec(unit)
                    .identity
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("_identity{i}"))
                    .collect(),
            }
        }
    }
}

impl<'a> Compiler<'a> {
    pub(super) fn unit_rows_program_with(
        mut self,
        selection: &Selection,
        stages: &[Transform],
        preset: Option<MatchRel>,
    ) -> Result<Program, QueryError> {
        let mut parts = self.selection_sql(selection, preset)?;
        let defaults: Vec<_> = parts.applied.iter().map(|(id, _)| *id).collect();
        if defaults.contains(&units::DefaultId::LogicalSessions) {
            self.fold_into_logical_sessions(&parts.scope, &defaults, &mut parts.where_sql)?;
        }
        self.apply_stages(selection, stages, &mut parts)?;
        let count_relation = self.finish_conditions(selection, &mut parts)?;
        let recursive = parts.matched.as_ref().is_some_and(|m| m.recursive);
        let ctes = if self.ctes.is_empty() {
            String::new()
        } else {
            format!(
                "WITH {}{} ",
                if recursive { "RECURSIVE " } else { "" },
                self.ctes.join(", ")
            )
        };
        let rows = self.row_statements(selection, &parts, &ctes, count_relation)?;
        let mut attach = vec![];
        if !selection.attach.is_empty() {
            if rows.shape != Shape::Rows {
                return Err(QueryError::Invalid {
                    message: "attached rows apply to rows, not to groups or summaries".into(),
                    position: selection.attach[0].span.start,
                });
            }
            for child in &selection.attach {
                attach.push(self.attach(child, &parts.scope)?);
            }
        }

        let from = self.from(&parts.scope)?;
        let why = self.explanations(&parts, &ctes, &from);
        Ok(self.into_program(parts, rows, attach, why))
    }

    fn selection_sql(
        &mut self,
        selection: &Selection,
        preset: Option<MatchRel>,
    ) -> Result<SelectionSql, QueryError> {
        self.measuring = selection.measured;
        self.rank_index = selection.search.rank;
        let scope = RowContext::plain(selection.unit);
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
        let top = selection.top_level();
        let applied: Vec<(units::DefaultId, String)> = selection
            .defaults
            .iter()
            .map(|&id| (id, units::default_sql(id, &scope.a)))
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
            // Bound conditions keep their original spans, while the compiler's
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
        Ok(SelectionSql {
            scope,
            top,
            applied,
            defaults_only,
            where_sql,
            why,
            matched: preset,
            measure: None,
            explicit_order: None,
        })
    }

    fn apply_stages(
        &mut self,
        selection: &Selection,
        stages: &[Transform],
        parts: &mut SelectionSql,
    ) -> Result<(), QueryError> {
        for stage in stages {
            match &stage.kind {
                TransformKind::Sort { .. } => {
                    debug_assert!(super::search::recent_sort(
                        parts.scope.unit,
                        std::slice::from_ref(stage)
                    ));
                    let spec = units::spec(parts.scope.unit);
                    let time = parts.scope.a.expand(spec.time);
                    let mut order = vec![format!("{time} IS NULL"), format!("{time} DESC")];
                    order.extend(spec.identity.iter().map(|term| parts.scope.a.expand(term)));
                    parts.explicit_order = Some(order);
                }
                TransformKind::Scope {
                    unit,
                    condition: cond,
                } => {
                    self.scope_stage(unit, cond.as_ref(), stage.span, parts)?;
                }
                TransformKind::Extension { name, arguments }
                    if name == polylogue_query::catalog::MEASURE =>
                {
                    self.measure_stage(selection, arguments, stage.span, parts)?;
                }
                TransformKind::Matches => {
                    self.matches_stage(stage.span, parts)?;
                }
                TransformKind::AsOf { .. } => return Err(relations::as_of_unavailable()),
                _ => return Err(not_built("this shared query stage")),
            }
        }
        Ok(())
    }

    fn scope_stage(
        &mut self,
        unit: &str,
        cond: Option<&Condition>,
        span: Span,
        parts: &mut SelectionSql,
    ) -> Result<(), QueryError> {
        let SelectionSql {
            scope,
            top,
            where_sql,
            why,
            ..
        } = parts;
        if scope.unit != Unit::Sessions {
            return Err(QueryError::Invalid {
                message: "a `| UNIT where …` stage follows a session query".into(),
                position: span.start,
            });
        }
        let sessions = format!(
            "SELECT s.pk FROM {} WHERE {}",
            self.from(scope)?,
            and_all(where_sql)
        );
        // The session ranking selects sessions; children have their own order.
        self.near_request = None;
        self.rank = None;
        self.rank_source = None;
        self.text_search = None;
        self.rank_index = None;
        self.near_rank = None;
        self.near_join = None;
        self.fusion = None;
        self.shown.clear();
        let unit = bind_unit(unit)?;
        let next = RowContext::plain(unit);
        let ntop = cond.map(top_level).unwrap_or_default();
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
        self.fold_bound_sessions(&next, &ntop, &mut w)?;
        *scope = next;
        *top = ntop;
        *where_sql = w;
        why.clear();
        // A fused rank and the lists it names belong to the
        // sessions.
        self.fusion = None;
        self.shown.retain(|(name, ..)| name != "found_in");
        Ok(())
    }

    fn measure_stage(
        &mut self,
        selection: &Selection,
        arguments: &[ExtensionArgument],
        span: Span,
        parts: &mut SelectionSql,
    ) -> Result<(), QueryError> {
        let SelectionSql { scope, measure, .. } = parts;
        let [ExtensionArgument::Value(what), rest @ ..] = arguments else {
            return Err(not_built("this measure declaration"));
        };
        if rest.iter().any(|argument| !matches!(argument, ExtensionArgument::Optional(values) if values.is_empty())) {
                        return Err(not_built("measure by"));
                    }
        if scope.unit != Unit::Sessions {
            return Err(QueryError::Invalid {
                message: "`| measure WHAT` follows a session query and ends it".into(),
                position: span.start,
            });
        }
        *measure = Some(Measure {
            what: value_text(what)?.into(),
            population_sql: String::new(),
            selection_sql: String::new(),
            everything: selection.written.is_empty(),
            since: None,
            until: None,
        });
        Ok(())
    }

    fn matches_stage(&mut self, span: Span, parts: &mut SelectionSql) -> Result<(), QueryError> {
        let SelectionSql {
            scope,
            top,
            where_sql,
            matched,
            ..
        } = parts;
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
                position: span.start,
            });
        };
        let parent_rows = and_all(where_sql);
        *matched = Some(match &c.kind {
            ConditionKind::Exists { unit, condition } => self.exists_matches(
                bind_unit(unit)?,
                condition.as_deref(),
                scope,
                &parent_rows,
                c.span,
            )?,
            ConditionKind::Sequence {
                steps,
                gaps,
                all: false,
                ..
            } => self.seq_matches(steps, gaps, scope, &parent_rows, c.span)?,
            ConditionKind::Sequence { all: true, .. } => {
                return Err(not_built("overlapping sequence matches (seq all)"));
            }
            _ => {
                return Err(QueryError::Invalid {
                    message: "matches requires an exists or sequence qualifier".into(),
                    position: c.span.start,
                });
            }
        });
        self.notes.push("one row per step of each match".into());
        Ok(())
    }

    fn finish_conditions(
        &mut self,
        selection: &Selection,
        parts: &mut SelectionSql,
    ) -> Result<Option<(String, String)>, QueryError> {
        let SelectionSql {
            scope,
            matched,
            measure,
            where_sql,
            defaults_only,
            explicit_order,
            ..
        } = parts;
        let rows_answer = matched.is_none() && measure.is_none();
        if rows_answer {
            self.setup_near(true, scope, where_sql);
        }
        let count_relation = if rows_answer {
            Some((self.from(scope)?, and_all(where_sql)))
        } else {
            None
        };
        self.fuse_text_and_near(scope, where_sql)?;
        self.search_scope(selection, scope, defaults_only, explicit_order.as_deref())?;
        if let Some(search) = &self.text_search
            && let Some(scoped) = &search.scope
            && let Some(recent) = &scoped.recent
        {
            // A shared descendant can make other rows match. A recency
            // batch must still return only its own candidate rows.
            where_sql.push(format!(
                "(?{}<>2 OR {} IN (SELECT value FROM json_each(?{})))",
                scoped.mode,
                scope.a.expand(units::spec(scope.unit).key),
                recent.batch
            ));
        }
        Ok(count_relation)
    }

    fn row_statements(
        &mut self,
        selection: &Selection,
        parts: &SelectionSql,
        ctes: &str,
        count_relation: Option<(String, String)>,
    ) -> Result<RowStatements, QueryError> {
        let scope = &parts.scope;
        let from = self.from(scope)?;
        let wh = and_all(&parts.where_sql);
        if let Some(rel) = &parts.matched {
            return Ok(RowStatements {
                sql: format!("{ctes}{}", rel.sql),
                shape: Shape::Matches,
                columns: rel.columns.clone(),
                order: rel.order.clone(),
                count: None,
                members: None,
                keyed: None,
            });
        }
        let order = parts
            .explicit_order
            .clone()
            .unwrap_or_else(|| self.default_order(scope));
        let (pk, cols, mut columns) = self.shown_columns(selection, scope)?;
        let (row_select, mut select) = self.selected_columns(scope, &pk, &cols);
        if self.ranking().is_some() {
            columns.push(Column {
                name: "rank".into(),
                kind: ColKind::Number,
                reference: None,
            });
        }
        let without = self.near_rank.as_ref().map_or_else(
            || "0".to_owned(),
            |near| format!("count(*) FILTER (WHERE ({near}) IS NULL)"),
        );
        let (count_from, count_where) = count_relation
            .as_ref()
            .map(|(from, wh)| (from.as_str(), wh.as_str()))
            .unwrap_or((&from, &wh));
        let count = Some(format!(
            "{ctes}SELECT count(*), {without} FROM {count_from} WHERE {count_where}"
        ));
        let members = Some(format!(
            "{ctes}SELECT {pk} AS _pk,{} AS _near FROM {count_from} WHERE {count_where}",
            self.near_rank.as_deref().unwrap_or("1")
        ));
        let (hidden, keys) = self.selection_order(&order);
        let keyed = parts
            .measure
            .is_none()
            .then(|| self.keyed_statements(scope, ctes, (&from, &wh), &pk, &row_select, &hidden));
        select.extend(hidden);
        let sql = format!(
            "{ctes}SELECT {} FROM {from} WHERE {wh} ORDER BY {}",
            select.join(", "),
            order_by(&keys)
        );
        Ok(RowStatements {
            sql,
            shape: Shape::Rows,
            columns,
            order: keys,
            count,
            members,
            keyed,
        })
    }

    fn selected_columns(
        &self,
        scope: &RowContext,
        pk: &str,
        cols: &[RowColumn],
    ) -> (Vec<String>, Vec<String>) {
        let row_select: Vec<String> =
            [
                format!("{pk} AS _pk"),
                format!(
                    "{} AS _recency",
                    scope.a.expand(units::spec(scope.unit).time)
                ),
            ]
            .into_iter()
            .chain(
                cols.iter().map(|(name, expression, _, _)| {
                    format!("{expression} AS {}", column_sql(name))
                }),
            )
            .chain(units::spec(scope.unit).identity.iter().enumerate().map(
                |(index, expression)| format!("{} AS _identity{index}", scope.a.expand(expression)),
            ))
            .collect();
        let mut select = vec![
            format!("{pk} AS _pk"),
            format!(
                "{} AS _recency",
                scope.a.expand(units::spec(scope.unit).time)
            ),
        ];
        if let Some(rank) = self.ranking() {
            select.push(format!("{rank} AS _rank"));
        }
        select.extend(
            cols.iter()
                .map(|(name, expression, _, _)| format!("{expression} AS {}", column_sql(name))),
        );
        select.push(format!(
            "{} AS _near",
            self.near_rank.as_deref().unwrap_or("1")
        ));
        for (index, expression) in units::spec(scope.unit).identity.iter().enumerate() {
            select.push(format!(
                "{} AS _identity{index}",
                scope.a.expand(expression)
            ));
        }
        (row_select, select)
    }

    fn selection_order(&self, order: &[String]) -> (Vec<String>, Vec<OrderKey>) {
        let (hidden, mut keys) = hidden_order(order, "_o");
        let rank = self.ranking();
        let hidden = hidden
            .into_iter()
            .enumerate()
            .filter_map(|(index, column)| {
                if rank.as_deref() == Some(order_term(&order[index]).0) {
                    keys[index].column = "_rank".into();
                    None
                } else {
                    Some(column)
                }
            })
            .collect::<Vec<_>>();
        (hidden, keys)
    }

    fn keyed_statements(
        &self,
        scope: &RowContext,
        ctes: &str,
        condition: (&str, &str),
        pk: &str,
        row_select: &[String],
        hidden: &[String],
    ) -> (String, String) {
        let (from, wh) = condition;
        let mut key_select = vec![format!("{pk} AS _pk")];
        if let Some(rank) = self.ranking() {
            key_select.push(format!("{rank} AS _rank"));
        }
        if self.fusion.is_none()
            && self.near_rank.is_none()
            && let Some(source) = &self.rank_source
        {
            key_select.push(format!("{} AS _hit_block", source.block));
        }
        key_select.push(format!(
            "{} AS _near",
            self.near_rank.as_deref().unwrap_or("1")
        ));
        key_select.extend(hidden.iter().cloned());
        (
            format!(
                "{ctes}SELECT {} FROM {from} WHERE {wh}",
                key_select.join(", ")
            ),
            format!(
                "{ctes}SELECT {} FROM {}",
                row_select.join(", "),
                base_from(scope.unit, &scope.a)
            ),
        )
    }

    fn shown_columns(
        &mut self,
        selection: &Selection,
        scope: &RowContext,
    ) -> Result<(String, Vec<RowColumn>, Vec<Column>), QueryError> {
        let (pk, mut cols) =
            row_columns(scope.unit, &scope.a, self.opts.clock.now.as_millisecond());
        if let FieldSelection::Fields(fields) = &selection.field_selection {
            cols.clear();
            for field in fields {
                let (expression, kind) = self.expr(field, scope, 0)?;
                cols.push((
                    flat(field),
                    Self::display_value(&expression, kind),
                    kind,
                    Self::field_reference(field, scope),
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
        Ok((pk, cols, columns))
    }

    fn explanations(
        &self,
        parts: &SelectionSql,
        ctes: &str,
        from: &str,
    ) -> (Vec<WhyClause>, String, Vec<WhyClause>) {
        let SelectionSql {
            defaults_only,
            applied,
            ..
        } = parts;
        let why = parts
            .why
            .iter()
            .cloned()
            .map(|(clause, cond)| WhyClause {
                matches_sql: None,
                sql: format!(
                    "{ctes}SELECT count(*) FROM {from} WHERE {}",
                    and_all(&[defaults_only.clone(), vec![cond]].concat())
                ),
                clause,
            })
            .collect();
        let why_of = format!(
            "{ctes}SELECT count(*) FROM {from} WHERE {}",
            and_all(defaults_only)
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
                    matches_sql: (*describe == units::DefaultId::LogicalSessions).then(|| {
                        let ids: Vec<_> = applied.iter().map(|(id, _)| *id).collect();
                        let folded = folded_sessions(&parts.scope, from, defaults_only, &ids);
                        format!(
                            "{ctes}SELECT count(*) FROM {from} WHERE {}",
                            and_all(&folded)
                        )
                    }),
                    clause: describe.describe(),
                    sql: format!(
                        "{ctes}SELECT count(*) FROM {from} WHERE {}",
                        and_all(&others)
                    ),
                }
            })
            .collect();
        (why, why_of, why_defaults)
    }

    fn into_program(
        mut self,
        mut parts: SelectionSql,
        rows: RowStatements,
        attach: Vec<Attach>,
        why: (Vec<WhyClause>, String, Vec<WhyClause>),
    ) -> Program {
        self.measure_statements(&mut parts, &rows);
        let row_kind = rows.row_kind(parts.scope.unit);
        let SelectionSql { scope, measure, .. } = parts;
        let RowStatements {
            sql,
            shape,
            columns,
            order,
            count: row_count,
            members: row_membership,
            keyed,
        } = rows;
        let (why, why_of, why_defaults) = why;
        let unit = scope.unit;
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
        self.program_notes(&columns);
        let statements = self.program_statements(sql, keyed, row_count, row_membership);
        Program {
            row_kind,
            query: self.text.to_owned(),
            unit,
            shape,
            statements,
            params: self.params,
            columns,
            column_slots: if ranked && shape == Shape::Rows {
                std::collections::BTreeMap::from([("rank".into(), "_rank".into())])
            } else {
                Default::default()
            },
            row_sources: None,
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
        }
    }

    fn measure_statements(&self, parts: &mut SelectionSql, rows: &RowStatements) {
        if let Some(measure) = &mut parts.measure {
            (measure.since, measure.until) = self.session_window;
            measure.selection_sql = rows.sql.clone();
            let defaults: Vec<_> = parts.applied.iter().map(|(id, _)| *id).collect();
            let population = if defaults.contains(&units::DefaultId::LogicalSessions) {
                folded_sessions(
                    &parts.scope,
                    &base_from(parts.scope.unit, &parts.scope.a),
                    &parts.defaults_only,
                    &defaults,
                )
            } else {
                parts.defaults_only.clone()
            };
            measure.population_sql = format!(
                "SELECT {s}.pk AS _pk FROM session {s} WHERE {defaults}{parameters}",
                s = parts.scope.a.s,
                defaults = and_all(&population),
                parameters = if self.params.is_empty() {
                    String::new()
                } else {
                    format!(" AND (?{0} IS ?{0})", self.params.len())
                }
            );
        }
    }

    fn program_notes(&mut self, columns: &[Column]) {
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
    }

    fn program_statements(
        &mut self,
        sql: String,
        keyed: Option<(String, String)>,
        row_count: Option<String>,
        row_membership: Option<String>,
    ) -> Statements {
        // Only the answer's own rows are read key first; a stage or a
        // combination over them reads them whole.
        match (keyed, row_count) {
            (Some((keys, rows)), Some(count)) if self.answer => Statements::Keyed(KeyedRows {
                keys,
                rows,
                count,
                search: if self.fusion.is_none() && self.near_rank.is_none() {
                    self.text_search.take()
                } else {
                    None
                },
            }),
            (_, count) => Statements::Derived(DerivedRows {
                rows: sql,
                count,
                members: row_membership,
            }),
        }
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

    /// Match physical sessions first, then select each known logical root once.
    fn fold_into_logical_sessions(
        &self,
        scope: &RowContext,
        defaults: &[units::DefaultId],
        conditions: &mut Vec<String>,
    ) -> Result<(), QueryError> {
        *conditions = folded_sessions(scope, &self.from(scope)?, conditions, defaults);
        Ok(())
    }

    /// Nested sources bind their own defaults, just like the outer selection.
    pub(super) fn fold_bound_sessions(
        &self,
        scope: &RowContext,
        top: &[Condition],
        conditions: &mut Vec<String>,
    ) -> Result<(), QueryError> {
        let defaults =
            polylogue_query::defaults::bind(scope.unit, top, &self.opts.origins, self.measuring);
        if defaults.contains(&units::DefaultId::LogicalSessions) {
            self.fold_into_logical_sessions(scope, &defaults, conditions)?;
        }
        Ok(())
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
        let time = if self.rank.is_some() && self.fusion.is_none() && self.near_rank.is_none() {
            self.rank_source
                .as_ref()
                .map(|source| source.at.clone())
                .unwrap_or_else(|| a.expand(spec.time))
        } else {
            a.expand(spec.time)
        };
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

/// The inner row is a physical match; outer visibility also applies to its root.
fn folded_sessions(
    scope: &RowContext,
    from: &str,
    conditions: &[String],
    defaults: &[units::DefaultId],
) -> Vec<String> {
    // Drive the fold from the paged root. A global IN (SELECT root ...)
    // materializes every physical session before a keyset page can stop.
    let mut folded = vec![format!(
        "EXISTS (SELECT 1 FROM session logical_owner WHERE logical_owner.pk = {s}.pk \
         AND logical_owner.pk = {root} AND EXISTS (SELECT 1 FROM {from} WHERE \
         {s}.pk IN ({descendants}) AND {candidate_root} = logical_owner.pk AND {conditions}))",
        s = scope.a.s,
        root = fields::logical_root("logical_owner"),
        descendants = fields::descendants("logical_owner"),
        candidate_root = fields::logical_root(&scope.a.s),
        conditions = and_all(conditions),
    )];
    folded.extend(
        defaults
            .iter()
            .filter(|id| {
                matches!(
                    id,
                    units::DefaultId::NotStub | units::DefaultId::SessionsVisible
                )
            })
            .map(|&id| units::default_sql(id, &scope.a)),
    );
    folded
}

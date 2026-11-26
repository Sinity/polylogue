//! Assembling one program from a parsed query: the unit's rows, its
//! conditions, stages and the columns shown, ranked and ordered.

use super::*;
use polylogue_query::filter::{Filter, Special, Test};

/// One selection's SQL while it is lowered: its rows, conditions and stages.
struct SelectionSql<'s> {
    scope: RowContext,
    /// The top-level `exists` and `seq` conditions, which `| matches` reads.
    patterns: Vec<&'s Filter>,
    applied: Vec<(units::DefaultId, String)>,
    defaults_only: Vec<String>,
    where_sql: Vec<String>,
    why: Vec<(String, String)>,
    matched: Option<MatchRel>,
    measure: Option<Measure>,
    explicit_order: Option<Vec<String>>,
    reading: Reading,
    /// A `| measure` stage reads this selection: its population keeps the
    /// sessions a default would hide.
    measured: bool,
    /// The `similar:` text to rank by and the rows whose texts it scores,
    /// built once the rows are complete.
    similar: Option<(Near, RowContext)>,
    ranking: Option<Ranking>,
    /// The scope that can drive the ranked text search.
    search_scope: Option<super::search::SearchScope>,
}

struct RowStatements {
    sql: String,
    shape: Shape,
    columns: Vec<Column>,
    order: Vec<OrderKey>,
    count: Option<String>,
    members: Option<String>,
    keyed: Option<(String, String)>,
    search: Option<TextSearch>,
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
        reading: Reading,
    ) -> Result<Program, QueryError> {
        let mut parts = self.selection_sql(selection, preset, reading)?;
        let defaults: Vec<_> = parts.applied.iter().map(|(id, _)| *id).collect();
        if defaults.contains(&units::DefaultId::LogicalSessions) {
            let filtered = parts.where_sql.len() > parts.defaults_only.len();
            self.fold_into_logical_sessions(
                &parts.scope,
                parts.ranking.as_ref(),
                &defaults,
                &mut parts.where_sql,
                filtered,
            );
        }
        self.apply_stages(selection, stages, &mut parts)?;
        let count_relation = self.finish_conditions(selection, &mut parts)?;
        let recursive = parts.matched.as_ref().is_some_and(|m| m.recursive);
        let ctes = match super::search::with(&self.ctes) {
            with if recursive => with.replacen("WITH ", "WITH RECURSIVE ", 1),
            with => with,
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
                attach.push(self.attach(child, &parts.scope, parts.measured)?);
            }
        }

        let from = self.from(&parts.scope, parts.ranking.as_ref());
        let why = self.explanations(&parts, &ctes, &from);
        Ok(self.into_program(parts, rows, attach, why))
    }

    fn selection_sql<'s>(
        &mut self,
        selection: &'s Selection,
        preset: Option<MatchRel>,
        reading: Reading,
    ) -> Result<SelectionSql<'s>, QueryError> {
        let measured = selection.measured;
        let alternates = (selection.unit == Unit::Messages)
            .then(|| {
                let mut admitted = false;
                let mut languages = Vec::new();
                for clause in &selection.clauses {
                    if let Filter::Field(field) = &clause.filter {
                        match (field.field.declaration().name, &field.test) {
                            ("alternate", Test::Bool(true)) => admitted = true,
                            (
                                "lang",
                                Test::Special(
                                    value @ (Special::One { .. }
                                    | Special::Set(_)
                                    | Special::Parameter(_)
                                    | Special::Unknown
                                    | Special::Known),
                                ),
                            ) => {
                                admitted = true;
                                languages.extend(value.listed());
                            }
                            _ => {}
                        }
                    }
                }
                admitted.then_some(languages)
            })
            .flatten();
        let scope = RowContext::plain(selection.unit);
        let similar = match &selection.semantic {
            Some(semantic) if !reading.similar => {
                return Err(custom_fields::similar_misplaced(semantic.span));
            }
            Some(semantic) => Some((
                Near {
                    text: semantic.text.clone(),
                },
                RowContext {
                    unit: semantic.unit,
                    a: scope.a.clone(),
                },
            )),
            None => None,
        };
        let patterns = selection
            .conjuncts()
            .into_iter()
            .filter(|filter| matches!(filter, Filter::Exists { .. } | Filter::Sequence { .. }))
            .collect();
        let applied: Vec<(units::DefaultId, String)> = selection
            .defaults
            .get(measured)
            .iter()
            .map(|&id| (id, units::default_sql(id, &scope.a)))
            .collect();
        let mut where_sql: Vec<String> = applied.iter().map(|(_, sql)| sql.clone()).collect();
        let defaults_only = where_sql.clone();
        if let Some(input) = &reading.within {
            let key = scope.a.expand(units::spec(scope.unit).key);
            where_sql.push(format!("{key} IN (SELECT _pk FROM ({input}))"));
        }
        let top_level = ranking::TopLevel {
            index: selection.search.rank,
            plans: reading.answer && !measured && similar.is_none(),
            ranking: Default::default(),
        };
        let cx = Cx {
            measured,
            top: Some(&top_level),
            similar: similar.as_ref().map(|(near, _)| near.text.as_str()),
            alternates: alternates.as_deref(),
        };
        let mut why = vec![];
        // An order condition makes the source a relation boundary, read by
        // the relation that holds it; the selection then has no filter.
        if let Some(filter) = &selection.filter {
            where_sql.push(self.lower(filter, &scope, cx)?);
        }
        self.snippet_terms.extend(selection.search.terms.clone());
        for clause in &selection.clauses {
            let alone = self.lower(&clause.filter, &scope, cx.inner())?;
            why.push((clause.text.clone(), alone));
        }
        let ranking = top_level.take().map(Ranking::Text);
        Ok(SelectionSql {
            scope,
            patterns,
            applied,
            defaults_only,
            where_sql,
            why,
            matched: preset,
            measure: None,
            explicit_order: None,
            reading,
            measured,
            // A later stage orders the rows; no similarity is computed for them.
            similar: similar.filter(|_| selection.order != Order::Sorted),
            ranking,
            search_scope: None,
        })
    }

    fn apply_stages(
        &mut self,
        selection: &Selection,
        stages: &[Transform],
        parts: &mut SelectionSql<'_>,
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
                TransformKind::Extension { name, arguments }
                    if name == polylogue_query::catalog::MEASURE =>
                {
                    self.measure_stage(selection, arguments, stage.span, parts)?;
                }
                TransformKind::Matches => {
                    self.matches_stage(stage.span, parts)?;
                }
                TransformKind::AsOf { .. } => return Err(relations::as_of_unavailable()),
                _ => return Err(not_built("this shared query transform")),
            }
        }
        Ok(())
    }

    fn measure_stage(
        &mut self,
        selection: &Selection,
        arguments: &[ExtensionArgument],
        span: Span,
        parts: &mut SelectionSql<'_>,
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

    fn matches_stage(
        &mut self,
        span: Span,
        parts: &mut SelectionSql<'_>,
    ) -> Result<(), QueryError> {
        let SelectionSql {
            scope,
            patterns,
            where_sql,
            matched,
            measured,
            ranking,
            ..
        } = parts;
        let cx = Cx::plain(*measured);
        let [pattern] = patterns.as_slice() else {
            return Err(QueryError::Invalid {
                message: format!(
                    "`| matches` needs exactly one top-level exists(...) or seq(...); this query has {}",
                    patterns.len()
                ),
                position: span.start,
            });
        };
        let parent_rows = and_all(where_sql);
        *matched = Some(match pattern {
            Filter::Exists { rows, span } => self.exists_matches(
                rows,
                (
                    scope,
                    &parent_rows,
                    self.default_order(scope, ranking.as_ref()),
                ),
                *span,
                cx,
            )?,
            Filter::Sequence { steps, gaps, span } => {
                self.seq_matches(steps, gaps, scope, &parent_rows, *span, cx)?
            }
            _ => unreachable!("patterns are exists and sequence conditions"),
        });
        self.notes.push("one row per step of each match".into());
        Ok(())
    }

    /// The ranking once the rows are complete: words and `similar:` fuse,
    /// and the scope that can drive a ranked text search is built. Returns
    /// the relation that counts the rows, before fusion replaces their
    /// conditions.
    fn finish_conditions(
        &mut self,
        selection: &Selection,
        parts: &mut SelectionSql<'_>,
    ) -> Result<Option<(String, String)>, QueryError> {
        let rows_answer = parts.matched.is_none() && parts.measure.is_none();
        let similarity = match parts.similar.take().filter(|_| rows_answer) {
            Some(_) if !self.opts.embeddings => {
                let note = "semantic search is not enabled; rows are ordered by time and unranked";
                if !self.incomplete.iter().any(|n| n == note) {
                    self.incomplete.push(note.into());
                }
                None
            }
            Some((near, rows)) => {
                Some(self.similarity(&near, &rows, &parts.scope, &parts.where_sql))
            }
            None => None,
        };
        let scope = &parts.scope;
        let unfused = match (&similarity, &parts.ranking) {
            (Some(similarity), _) => Some(Ranking::Similar(similarity.clone())),
            (None, ranking) => ranking.clone(),
        };
        let count_relation = rows_answer.then(|| {
            (
                self.from(scope, unfused.as_ref()),
                and_all(&parts.where_sql),
            )
        });
        parts.ranking = match (parts.ranking.take(), similarity) {
            (Some(text), Some(similarity)) => Some(self.fuse_text_and_near(
                scope,
                &mut parts.where_sql,
                text.expression(),
                similarity,
            )),
            (None, Some(similarity)) => Some(Ranking::Similar(similarity)),
            (ranking, None) => ranking,
        };
        parts.search_scope = self.search_scope(
            selection,
            scope,
            parts.ranking.as_ref(),
            &parts.defaults_only,
            parts.explicit_order.as_deref(),
            parts.measured,
        )?;
        Ok(count_relation)
    }

    fn row_statements(
        &mut self,
        selection: &Selection,
        parts: &SelectionSql<'_>,
        ctes: &str,
        count_relation: Option<(String, String)>,
    ) -> Result<RowStatements, QueryError> {
        let scope = &parts.scope;
        let ranking = parts.ranking.as_ref();
        let from = self.from(scope, ranking);
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
                search: None,
            });
        }
        let order = parts
            .explicit_order
            .clone()
            .unwrap_or_else(|| self.default_order(scope, ranking));
        let (pk, cols, mut columns) = self.shown_columns(selection, scope)?;
        let (row_select, mut select) = self.selected_columns(scope, &pk, &cols, ranking);
        if ranking.is_some() {
            columns.push(Column {
                name: "rank".into(),
                kind: ColKind::Number,
                reference: None,
            });
        }
        let similarity = ranking.and_then(Ranking::similarity);
        let without = similarity.map_or_else(
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
            similarity.unwrap_or("1")
        ));
        let (hidden, keys) = self.selection_order(&order, ranking);
        let keyed = parts.measure.is_none().then(|| {
            (
                self.keys_statement(ctes, (&from, &wh), &pk, &hidden, ranking),
                format!(
                    "{ctes}SELECT {} FROM {}",
                    row_select.join(", "),
                    base_from(scope.unit, &scope.a)
                ),
            )
        });
        // Only the answer's own rows are read by search plans.
        let search = ranking
            .and_then(Ranking::leg)
            .filter(|_| parts.reading.answer && parts.measure.is_none())
            .map(|leg| {
                self.search_plans(scope, leg, parts.search_scope.as_ref(), |with, batch| {
                    let wh = match batch {
                        Some(batch) => format!("{wh} AND {batch}"),
                        None => wh.clone(),
                    };
                    self.keys_statement(with, (&from, &wh), &pk, &hidden, ranking)
                })
            });
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
            search,
        })
    }

    fn selected_columns(
        &self,
        scope: &RowContext,
        pk: &str,
        cols: &[RowColumn],
        ranking: Option<&Ranking>,
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
        if let Some(ranking) = ranking {
            select.push(format!("{} AS _rank", ranking.expression()));
        }
        select.extend(
            cols.iter()
                .map(|(name, expression, _, _)| format!("{expression} AS {}", column_sql(name))),
        );
        select.push(format!(
            "{} AS _near",
            ranking.and_then(Ranking::similarity).unwrap_or("1")
        ));
        for (index, expression) in units::spec(scope.unit).identity.iter().enumerate() {
            select.push(format!(
                "{} AS _identity{index}",
                scope.a.expand(expression)
            ));
        }
        (row_select, select)
    }

    fn selection_order(
        &self,
        order: &[String],
        ranking: Option<&Ranking>,
    ) -> (Vec<String>, Vec<OrderKey>) {
        let (hidden, mut keys) = hidden_order(order, "_o");
        let rank = ranking.map(Ranking::expression);
        let hidden = hidden
            .into_iter()
            .enumerate()
            .filter_map(|(index, column)| {
                if rank == Some(order_term(&order[index]).0) {
                    keys[index].column = "_rank".into();
                    None
                } else {
                    Some(column)
                }
            })
            .collect::<Vec<_>>();
        (hidden, keys)
    }

    /// Every selected row's key, rank, similarity and order values.
    fn keys_statement(
        &self,
        ctes: &str,
        condition: (&str, &str),
        pk: &str,
        hidden: &[String],
        ranking: Option<&Ranking>,
    ) -> String {
        let (from, wh) = condition;
        let mut key_select = vec![format!("{pk} AS _pk")];
        if let Some(ranking) = ranking {
            key_select.push(format!("{} AS _rank", ranking.expression()));
        }
        if let Some(source) = ranking.and_then(Ranking::hit) {
            key_select.push(format!("{} AS _hit_block", source.block));
        }
        key_select.push(format!(
            "{} AS _near",
            ranking.and_then(Ranking::similarity).unwrap_or("1")
        ));
        key_select.extend(hidden.iter().cloned());
        format!(
            "{ctes}SELECT {} FROM {from} WHERE {wh}",
            key_select.join(", ")
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
                    column_name(field),
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
        parts: &SelectionSql<'_>,
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
                        let folded =
                            folded_sessions(&parts.scope, from, defaults_only, &ids, false);
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
        mut parts: SelectionSql<'_>,
        rows: RowStatements,
        attach: Vec<Attach>,
        why: (Vec<WhyClause>, String, Vec<WhyClause>),
    ) -> Program {
        self.measure_statements(&mut parts, &rows);
        let row_kind = rows.row_kind(parts.scope.unit);
        let logical_sessions = parts.scope.unit == Unit::Sessions
            && parts
                .applied
                .iter()
                .any(|(id, _)| *id == units::DefaultId::LogicalSessions);
        let SelectionSql {
            scope,
            measure,
            ranking,
            reading,
            ..
        } = parts;
        let RowStatements {
            sql,
            shape,
            columns,
            order,
            count: row_count,
            members: row_membership,
            keyed,
            search,
        } = rows;
        let (why, why_of, why_defaults) = why;
        let unit = scope.unit;
        let ranked = ranking.is_some();
        let counts_near =
            shape == Shape::Rows && ranking.as_ref().and_then(Ranking::similarity).is_some();
        let page_size = self
            .opts
            .page_size
            .unwrap_or(default_page_size(unit, shape));
        self.program_notes(&columns);
        // Only the answer's own rows are read key first; a stage or a
        // combination over them reads them whole.
        let statements = match (keyed, row_count) {
            (Some((keys, rows)), Some(count)) if reading.answer => Statements::Keyed(KeyedRows {
                keys,
                rows,
                count,
                search,
            }),
            (_, count) => Statements::Derived(DerivedRows {
                rows: sql,
                count,
                members: row_membership,
            }),
        };
        Program {
            row_kind,
            session_rows: if logical_sessions {
                SessionRows::Logical
            } else {
                SessionRows::Physical
            },
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
            annotation_cutoffs: self.annotation_cutoffs,
            attach,
            notes: self.notes,
            incomplete: self.incomplete,
            why,
            why_of,
            why_defaults,
            page_size,
            counts_near,
            near: self.similar_text,
            definitions: self.definitions,
            clock: self.opts.clock.clone(),
            resolved_times: self.resolved_times.into_inner(),
            measure,
            order,
        }
    }

    fn measure_statements(&self, parts: &mut SelectionSql<'_>, rows: &RowStatements) {
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
                    false,
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
        if columns.iter().any(|c| c.name.ends_with("clustered_time")) {
            self.notes.push(format!(
                "clustered_time uses a {}-minute gap, extended across paired tools",
                polylogue_model::derived::CLUSTER_GAP_MS / 60_000
            ));
        }
    }

    /// Words and `similar:` together: the conditions and both orders are
    /// evaluated once, over the rows the conditions keep, and each row's
    /// fused rank joins it.
    fn fuse_text_and_near(
        &mut self,
        scope: &RowContext,
        where_sql: &mut Vec<String>,
        text: &str,
        similarity: ranking::Similarity,
    ) -> Ranking {
        let near = &similarity.expression;
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now.as_millisecond());
        let rows = format!("f{}", self.ctes.len() + 1);
        let from = self.from(scope, Some(&Ranking::Similar(similarity.clone())));
        self.ctes.push(format!(
            "{rows} AS MATERIALIZED (SELECT {pk} AS k, {text} AS t, {near} AS n FROM {from} WHERE {})",
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
        where_sql.clear();
        Ranking::Fused {
            join: format!("JOIN {fused} fused ON fused.k = {pk}"),
            expression: "fused.r".into(),
            similarity,
        }
    }

    /// The FROM clause of a unit, joining the ranking its rows read.
    pub(super) fn from(&self, scope: &RowContext, ranking: Option<&Ranking>) -> String {
        let mut from = base_from(scope.unit, &scope.a);
        for join in ranking.map(Ranking::joins).unwrap_or_default() {
            from.push(' ');
            from.push_str(join);
        }
        from
    }

    /// Match physical sessions first, then select each known logical root once.
    fn fold_into_logical_sessions(
        &self,
        scope: &RowContext,
        ranking: Option<&Ranking>,
        defaults: &[units::DefaultId],
        conditions: &mut Vec<String>,
        filtered: bool,
    ) {
        *conditions = folded_sessions(
            scope,
            &self.from(scope, ranking),
            conditions,
            defaults,
            filtered,
        );
    }

    /// Nested sources fold into logical sessions by the defaults they
    /// bound, just like the outer selection.
    pub(super) fn sessions_within_bounds(
        &self,
        scope: &RowContext,
        defaults: &[units::DefaultId],
        conditions: &mut Vec<String>,
        filtered: bool,
    ) {
        if defaults.contains(&units::DefaultId::LogicalSessions) {
            self.fold_into_logical_sessions(scope, None, defaults, conditions, filtered);
        }
    }

    /// The predicates of the defaults a selection bound.
    pub(super) fn defaults(
        &self,
        scope: &RowContext,
        defaults: &[units::DefaultId],
    ) -> Vec<String> {
        defaults
            .iter()
            .map(|&id| units::default_sql(id, &scope.a))
            .collect()
    }

    /// The order of `scope`'s rows: by `ranking`, lower first, ties going
    /// to the more recent row (a text ranking's chosen hit), then identity;
    /// without one, the unit's own order.
    pub(super) fn default_order(
        &self,
        scope: &RowContext,
        ranking: Option<&Ranking>,
    ) -> Vec<String> {
        let a = &scope.a;
        let spec = units::spec(scope.unit);
        let time = ranking
            .and_then(Ranking::hit)
            .map_or_else(|| a.expand(spec.time), |source| source.at.clone());
        if let Some(rank) = ranking.map(|ranking| ranking.expression().to_owned()) {
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
        if let Some(terms) = spec.order {
            return terms.iter().map(|term| a.expand(term)).collect();
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
    filtered: bool,
) -> Vec<String> {
    let selection = if filtered {
        // Keep the physical predicate's indexed access path (for example cwd)
        // before mapping its matches to roots.
        format!(
            "{}.pk IN (SELECT {} FROM {from} WHERE {})",
            scope.a.s,
            fields::logical_root(&scope.a.s),
            and_all(conditions)
        )
    } else {
        // With only defaults, every eligible root matches its own physical
        // row. Neither its descendants nor the complete archive need a scan.
        format!("{}.pk = {}", scope.a.s, fields::logical_root(&scope.a.s))
    };
    let mut folded = vec![selection];
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

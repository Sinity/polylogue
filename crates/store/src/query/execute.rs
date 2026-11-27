//! Executing the one compiled query and shaping declared public values.
use super::compile::{PageColumns, page_of};
use super::{ColKind, Program};
use crate::ResultExt as _;
use crate::search::{SearchDriver, SnippetIndex, SnippetQuery, Within};
use crate::selection::{sql_value, values};
use crate::{Reader, SqlValue, selection::SelectionError};
use serde_json::Value;

/// One transfer page, prepared independently of its population count.
pub struct QueryPage {
    pub rows: Vec<Value>,
    pub more: bool,
}
/// Exact counts are reused only within the read position that produced them.
#[derive(Default)]
pub(crate) struct CountCache {
    position: Option<polylogue_model::ChangePosition>,
    entries: std::collections::VecDeque<(blake3::Hash, (u64, u64))>,
}

#[derive(Clone, Copy)]
pub enum QueryPopulation {
    Answer,
    MeasureSelection,
    MeasurePopulation,
}

impl Program {
    /// Resolve semantic ranking once, with the same notes for listings and readings.
    pub fn ranking(
        &mut self,
        reader: &Reader<'_>,
        near: Option<&crate::selection::NearFn<'_>>,
    ) -> Result<Option<std::sync::Arc<crate::vectors::Similarity>>, SelectionError> {
        let Some(request) = &self.near else {
            return Ok(None);
        };
        let result = match near {
            Some(near) => near(&request.text),
            None => Err(polylogue_model::Reason::EmbeddingsOff {}.into()),
        };
        reader.check()?;
        match result {
            Ok(similarity) => Ok(Some(similarity)),
            Err(crate::StoreError::Refused(
                reason @ (polylogue_model::Reason::EmbeddingsOff {}
                | polylogue_model::Reason::EmbeddingStore { .. }
                | polylogue_model::Reason::EmbeddingConfiguration { .. }
                | polylogue_model::Reason::NoVectors { .. }
                | polylogue_model::Reason::EmbeddingProvider { .. }
                | polylogue_model::Reason::EmbeddingCallsOff {}),
            )) => {
                if matches!(reason, polylogue_model::Reason::EmbeddingsOff {})
                    && !request.text_ranked
                {
                    return Err(crate::StoreError::Refused(reason).into());
                }
                let order = if request.text_ranked {
                    "rows are ranked by text relevance alone"
                } else {
                    "rows are ordered by time and unranked"
                };
                self.incomplete.push(format!("{reason}; {order}"));
                Ok(Some(std::sync::Arc::new(
                    crate::vectors::Similarity::unknown(),
                )))
            }
            Err(error) => Err(error.into()),
        }
    }
}

impl Reader<'_> {
    /// Ranking and calendar inputs are injected; this never chooses a provider.
    pub fn with_query_context<T, E>(
        &self,
        program: &Program,
        near: Option<&crate::selection::NearFn<'_>>,
        run: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<SelectionError>,
    {
        self.with_clock(&program.clock, || {
            if let Some(request) = &program.near {
                let near = near.ok_or_else(|| {
                    E::from(SelectionError::from(crate::StoreError::from(
                        polylogue_model::Reason::EmbeddingsOff {},
                    )))
                })?;
                let similarity =
                    near(&request.text).map_err(|error| E::from(SelectionError::from(error)))?;
                crate::read::limit::ReadLimit::set_stage("similarity");
                self.rank_selection(
                    &similarity,
                    &program.statements.members(true),
                    &crate::selection::values(&program.params),
                )
                .map_err(|error| E::from(SelectionError::from(error)))?;
                self.with_similarity(similarity, run)
            } else {
                run()
            }
        })
    }
    /// Execute and format one result row at a time. SQLite ordering work may
    /// precede delivery; a callback break stops further result decoding.
    pub fn visit_query_values<E>(
        &self,
        program: &Program,
        near: Option<&crate::selection::NearFn<'_>>,
        mut visit: impl FnMut(Value) -> Result<std::ops::ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        self.validate_selection(program).map_err(E::from)?;
        let scope = self.scope_token();
        self.with_query_context(program, near, || {
            self.visit_page(program, None, false, -1, |mut row| {
                scope.check().map_err(SelectionError::from)?;
                present(program, &mut row)?;
                visit(row)
            })?;
            scope.check().map_err(SelectionError::from).map_err(E::from)
        })
    }

    pub fn query_page(
        &self,
        program: &Program,
        after: Option<&[Value]>,
    ) -> Result<QueryPage, SelectionError> {
        self.validate_selection(program)?;
        if program.page_size == 0 {
            return Err(SelectionError::invalid(
                polylogue_model::Code::BadRequest,
                "page size must be positive",
            ));
        }
        self.with_clock(&program.clock, || {
            crate::read::limit::ReadLimit::set_stage("page");
            let mut rows = Vec::new();
            self.visit_page(
                program,
                after,
                false,
                i64::try_from(program.page_size.saturating_add(1)).unwrap_or(i64::MAX),
                |row| {
                    rows.push(row);
                    Ok::<_, SelectionError>(std::ops::ControlFlow::Continue(()))
                },
            )?;
            let more = rows.len() as u64 > program.page_size;
            rows.truncate(usize::try_from(program.page_size).unwrap_or(usize::MAX));
            Ok(QueryPage { rows, more })
        })
    }
    /// Reuse the exact count only for the same bound question and read position.
    /// Called after all page presentation, under the request's original bound.
    pub fn query_count(&self, program: &Program) -> Result<Option<(u64, u64)>, SelectionError> {
        let count = || self.count_at_position(program);
        let result = match self.read_limit() {
            Some(limit) => limit.scope(count),
            None => count(),
        };
        match result {
            Err(SelectionError::Store(crate::StoreError::ReadDeadlineExceeded)) => {
                self.scope_token().check()?;
                if let Some(limit) = self.read_limit() {
                    limit.cancellation.check()?;
                }
                Ok(None)
            }
            Ok(counts) => {
                self.scope_token().check()?;
                Ok(Some(counts))
            }
            Err(error) => Err(error),
        }
    }

    fn count_at_position(&self, program: &Program) -> Result<(u64, u64), SelectionError> {
        let position = self.change_position()?;
        let sql = program.statements.count(program.counts_near);
        let key = blake3::hash(
            &serde_json::to_vec(&(
                &sql,
                &program.params,
                &program.clock,
                &program.near,
                &program.incomplete,
                crate::sql_functions::similarity_count_inputs(),
                self.index_identity()?,
            ))
            .map_err(crate::StoreError::from)?,
        );
        {
            let mut cache = self
                .count_cache()
                .lock()
                .or_internal("count cache lock poisoned")?;
            if cache.position.as_ref() == Some(&position) {
                if let Some((_, count)) = cache.entries.iter().find(|(entry, _)| *entry == key) {
                    return Ok(*count);
                }
            } else {
                cache.position = Some(position.clone());
                cache.entries.clear();
            }
        }
        crate::read::limit::ReadLimit::set_stage("count");
        let counted = self.with_clock(&program.clock, || {
            self.query_counts(&sql, &values(&program.params))
        })?;
        let mut cache = self
            .count_cache()
            .lock()
            .or_internal("count cache lock poisoned")?;
        if cache.position.as_ref() == Some(&position) {
            // This bounded cache evicts reusable answers, never query rows.
            if cache.entries.len() == 64 {
                cache.entries.pop_front();
            }
            cache.entries.push_back((key, counted));
        }
        Ok(counted)
    }

    /// Visit the rows of one page in order: at most `limit` rows (all for
    /// -1) after the row whose order values are `after`, or before it when
    /// `reverse`. A unit's rows are read key first: the page's keys in
    /// order, then the columns of those keys only, so no row off the page
    /// is read beyond its key. A derived relation's page holds its rows;
    /// read in reverse, it sorts only order values and reads each row at
    /// its own.
    pub(crate) fn visit_page<E>(
        &self,
        program: &Program,
        after: Option<&[Value]>,
        reverse: bool,
        limit: i64,
        mut visit: impl FnMut(Value) -> Result<std::ops::ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        // The statement numbers its parameters: the program's, one per
        // order key, then the limit. A position of another length would put
        // the limit in a key's place.
        if after.is_some_and(|after| after.len() != program.order.len()) {
            return Err(E::from(SelectionError::invalid(
                polylogue_model::Code::InvalidQuery,
                "selection order position has the wrong arity",
            )));
        }
        let mut params = values(&program.params);
        if let Some(after) = after {
            params.extend(after.iter().map(sql_value));
        }
        params.push(SqlValue::Int(limit));
        let row_at = reverse
            .then(|| {
                program
                    .statements
                    .row_at(&program.order, program.params.len())
            })
            .flatten();
        let page = program.statements.page(
            &program.order,
            program.params.len(),
            after.map(|after| super::compile::in_second_part(&program.order, after)),
            reverse,
            if row_at.is_some() {
                PageColumns::Order
            } else {
                PageColumns::All
            },
        );
        let result = match program.statements.hydrate(program.params.len()) {
            Some(hydrate) => {
                let search = match &program.statements {
                    super::compile::Statements::Keyed(rows) => rows.search.as_ref(),
                    _ => None,
                };
                let driver = if limit != 0
                    && let Some(search) = search
                    && let Some(scope) = &search.scope
                {
                    self.search_driver(
                        &scope.blocks,
                        &values(&program.params),
                        search.index,
                        &search.expression,
                        scope.recent.is_some() && !reverse && limit > 0,
                    )
                    .map_err(SelectionError::from)
                    .map_err(E::from)?
                } else {
                    SearchDriver::Index
                };
                let page = match (driver, search.and_then(|search| search.scope.as_ref())) {
                    (SearchDriver::Scope, Some(scope)) => page_of(
                        &scope.by_scope,
                        &program.order,
                        program.params.len(),
                        after.map(|after| super::compile::in_second_part(&program.order, after)),
                        reverse,
                        PageColumns::All,
                    ),
                    _ => page,
                };
                let selected = search.filter(|search| {
                    driver == SearchDriver::Recency
                        || (driver == SearchDriver::Index
                            && !reverse
                            && limit >= 0
                            && search
                                .scope
                                .as_ref()
                                .is_none_or(|scope| scope.recent.is_none()))
                });
                if let Some(search) = selected {
                    let keys = if driver == SearchDriver::Recency {
                        self.recent_keys(program, search, after, limit)
                    } else {
                        self.ranked_keys(program, search, after, limit)
                    };
                    keys.map_err(StreamError::Selection).and_then(|keys| {
                        let mut keys = keys.into_iter();
                        loop {
                            let chunk = keys.by_ref().take(HYDRATE_BATCH).collect::<Vec<_>>();
                            if chunk.is_empty() {
                                break;
                            }
                            self.visit_hydrated(program, &hydrate, chunk, &mut visit)?;
                        }
                        Ok(())
                    })
                } else {
                    let mut keys = Vec::new();
                    self.query_each(&page, &params, |key| {
                        keys.push(key);
                        if keys.len() < HYDRATE_BATCH {
                            return Ok(());
                        }
                        self.visit_hydrated(
                            program,
                            &hydrate,
                            std::mem::take(&mut keys),
                            &mut visit,
                        )
                    })
                    .and_then(|()| self.visit_hydrated(program, &hydrate, keys, &mut visit))
                }
            }
            None => match row_at {
                None => self.query_each(&page, &params, |row| flow(visit(row))),
                Some(row_at) => self.query_each(&page, &params, |order| {
                    let mut row_params = values(&program.params);
                    row_params.extend(
                        program
                            .order
                            .iter()
                            .map(|key| sql_value(&order[&key.column])),
                    );
                    let row = self.query(&row_at, &row_params)?.pop().ok_or_else(|| {
                        StreamError::Selection(SelectionError::invalid(
                            polylogue_model::Code::StaleCursor,
                            "the selected row position is unavailable",
                        ))
                    })?;
                    flow(visit(row))
                }),
            },
        };
        match result {
            Ok(()) | Err(StreamError::Stop) => Ok(()),
            Err(StreamError::Consumer(e)) => Err(e),
            Err(StreamError::Selection(e)) => Err(E::from(e)),
        }
    }

    /// One candidate cursor stays open while bounded batches of row keys are tested.
    /// Only the requested page and the current batch live in Rust memory.
    fn recent_keys(
        &self,
        program: &Program,
        search: &super::compile::TextSearch,
        after: Option<&[Value]>,
        limit: i64,
    ) -> Result<Vec<Value>, SelectionError> {
        let recent = search
            .scope
            .as_ref()
            .and_then(|scope| scope.recent.as_ref())
            .expect("recency candidate keys");
        let base = program.params.len();
        let paged = |source: &str, params: usize| {
            page_of(
                source,
                &program.order,
                params,
                after.map(|after| super::compile::in_second_part(&program.order, after)),
                false,
                PageColumns::All,
            )
        };
        let inputs = |batch: Option<String>, limit: i64| {
            let mut params = values(&program.params);
            params.extend(batch.map(SqlValue::Text));
            if let Some(after) = after {
                params.extend(after.iter().map(sql_value));
            }
            params.push(SqlValue::Int(limit));
            params
        };
        let candidates = paged(&recent.candidates, base);
        let batch_page = paged(&recent.batch, base + 1);
        let conn = self.conn()?;
        let mut statement = conn
            .prepare_cached(&candidates)
            .map_err(crate::StoreError::from)?;
        for (i, value) in inputs(None, -1).iter().enumerate() {
            statement
                .raw_bind_parameter(i + 1, value)
                .map_err(crate::StoreError::from)?;
        }
        let mut candidates = statement.raw_query();
        let mut rows = Vec::new();
        let mut batch_size = limit.clamp(1, CANDIDATE_BATCH as i64) as usize;
        loop {
            let mut batch = Vec::new();
            let mut exhausted = false;
            for _ in 0..batch_size {
                let Some(row) = candidates.next().map_err(crate::StoreError::from)? else {
                    exhausted = true;
                    break;
                };
                batch.push(row.get::<_, i64>("_pk").map_err(crate::StoreError::from)?);
            }
            if batch.is_empty() {
                break;
            }
            let batch = serde_json::to_string(&batch).expect("row keys serialize");
            let remaining = limit - rows.len() as i64;
            let matched = self.query(&batch_page, &inputs(Some(batch), remaining))?;
            let found = !matched.is_empty();
            rows.extend(matched);
            if rows.len() as i64 >= limit || exhausted {
                break;
            }
            batch_size = if found {
                (limit - rows.len() as i64).clamp(1, CANDIDATE_BATCH as i64) as usize
            } else {
                (batch_size * 2).min(CANDIDATE_BATCH)
            };
        }
        Ok(rows)
    }

    /// Enlarge the ranked text prefix until no unseen score can enter this
    /// page. Each row's best score is chosen in SQL before the cursor applies,
    /// so weaker duplicate hits never bring back an earlier row.
    fn ranked_keys(
        &self,
        program: &Program,
        search: &super::compile::TextSearch,
        after: Option<&[Value]>,
        limit: i64,
    ) -> Result<Vec<Value>, SelectionError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let page = page_of(
            &search.by_index,
            &program.order,
            program.params.len() + 1,
            after.map(|after| super::compile::in_second_part(&program.order, after)),
            false,
            PageColumns::All,
        );
        let boundary = format!(
            "SELECT rank AS score FROM ({}) ORDER BY rank LIMIT 1 OFFSET ?2",
            crate::search::ranked_texts(search.index, "?1", None, None)
        );
        let mut prefix = limit.max(1);
        loop {
            let mut params = values(&program.params);
            params.push(SqlValue::Int(prefix));
            if let Some(after) = after {
                params.extend(after.iter().map(sql_value));
            }
            params.push(SqlValue::Int(limit));
            let rows = self.query(&page, &params)?;
            if prefix < 0 {
                return Ok(rows);
            }
            let next = self.query(
                &boundary,
                &[
                    SqlValue::Text(search.expression.clone()),
                    SqlValue::Int(prefix),
                ],
            )?;
            let Some(next_score) = next.first().and_then(|row| row["score"].as_f64()) else {
                return Ok(rows);
            };
            if rows.len() as u64 >= limit as u64
                && rows
                    .last()
                    .and_then(|row| row["_rank"].as_f64())
                    .is_some_and(|score| next_score > score)
            {
                return Ok(rows);
            }
            // Overflow selects the complete index, rather than imposing a
            // correctness limit on a large valid result.
            prefix = prefix.checked_mul(2).unwrap_or(-1);
        }
    }

    /// The rows of `keys`, in their order, each with its key's order
    /// values, rank and similarity.
    fn visit_hydrated<E>(
        &self,
        program: &Program,
        hydrate: &str,
        keys: Vec<Value>,
        visit: &mut impl FnMut(Value) -> Result<std::ops::ControlFlow<()>, E>,
    ) -> Result<(), StreamError<E>> {
        if keys.is_empty() {
            return Ok(());
        }
        let missing = |what: &str| {
            StreamError::Selection(SelectionError::invalid(
                polylogue_model::Code::Store,
                format!("a selected row has no {what}"),
            ))
        };
        // A unit's key is an integer or, for changes, text: both are JSON
        // scalars, and json_each returns each with its SQL type.
        let pks = keys
            .iter()
            .map(|key| match &key["_pk"] {
                pk @ (Value::Number(_) | Value::String(_)) => Ok(pk.clone()),
                _ => Err(missing("key")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut params = values(&program.params);
        params.push(SqlValue::Text(Value::Array(pks.clone()).to_string()));
        let mut rows = std::collections::HashMap::with_capacity(pks.len());
        self.query_each(hydrate, &params, |row| {
            rows.insert(row["_pk"].to_string(), row);
            Ok::<_, StreamError<E>>(())
        })?;
        for (key, pk) in keys.into_iter().zip(pks) {
            let mut row = rows
                .get(&pk.to_string())
                .cloned()
                .ok_or_else(|| missing("columns"))?;
            if let Value::Object(key) = key {
                for (name, value) in key {
                    if name == "_rank" {
                        row["rank"] = value.clone();
                    }
                    row[name.as_str()] = value;
                }
            }
            flow(visit(row))?;
        }
        Ok(())
    }

    /// Every row of the answer in order, before presentation.
    pub fn query_rows(&self, program: &Program) -> Result<Vec<Value>, SelectionError> {
        self.validate_selection(program)?;
        let mut rows = Vec::new();
        self.with_clock(&program.clock, || {
            self.visit_page(program, None, false, -1, |row| {
                rows.push(row);
                Ok::<_, SelectionError>(std::ops::ControlFlow::Continue(()))
            })
        })?;
        Ok(rows)
    }
    /// Current columns of the accepted native population, in its original order.
    /// The keyed projection has no selection predicate to evaluate again.
    pub fn query_fixed_rows(
        &self,
        program: &Program,
        references: &[polylogue_model::Reference],
    ) -> Result<Vec<Value>, SelectionError> {
        self.validate_selection(program)?;
        let hydrate = program
            .statements
            .hydrate(program.params.len())
            .ok_or_else(|| {
                SelectionError::invalid(
                    polylogue_model::Code::Internal,
                    "a derived relation cannot hydrate native rows",
                )
            })?;
        let mut keys = Vec::with_capacity(references.len());
        for reference in references {
            let mut params = Vec::new();
            let sql = super::references::membership(program.unit, reference, &mut |value| {
                params.push(SqlValue::Text(value));
                format!("?{}", params.len())
            })
            .ok_or_else(|| {
                SelectionError::invalid(
                    polylogue_model::Code::BadRequest,
                    "reference has no identity in the kept unit",
                )
            })?;
            let rows = self.query(&format!("SELECT pk AS _pk FROM ({sql})"), &params)?;
            let [key] = rows.as_slice() else {
                return Err(SelectionError::invalid(
                    polylogue_model::Code::Conflict,
                    "a kept reference no longer names one row",
                ));
            };
            keys.push(key.clone());
        }
        let mut rows = Vec::new();
        match self.with_clock(&program.clock, || {
            self.visit_hydrated(program, &hydrate, keys, &mut |row| {
                rows.push(row);
                Ok::<_, SelectionError>(std::ops::ControlFlow::Continue(()))
            })
        }) {
            Ok(()) | Err(StreamError::Stop) => {}
            Err(StreamError::Selection(error) | StreamError::Consumer(error)) => return Err(error),
        }
        Ok(rows)
    }

    pub fn query_keys(
        &self,
        program: &Program,
        population: QueryPopulation,
    ) -> Result<Vec<i64>, SelectionError> {
        self.validate_selection(program)?;
        let sql = match population {
            QueryPopulation::Answer => program.statements.members(program.counts_near),
            QueryPopulation::MeasureSelection => format!(
                "SELECT _pk FROM ({})",
                program
                    .measure
                    .as_ref()
                    .ok_or_else(|| SelectionError::invalid(
                        polylogue_model::Code::BadRequest,
                        "query has no measure selection"
                    ))?
                    .selection_sql
            ),
            QueryPopulation::MeasurePopulation => program
                .measure
                .as_ref()
                .ok_or_else(|| {
                    SelectionError::invalid(
                        polylogue_model::Code::BadRequest,
                        "query has no measure population",
                    )
                })?
                .population_sql
                .clone(),
        };
        let mut keys = Vec::new();
        self.with_clock(&program.clock, || {
            self.query_each(&sql, &values(&program.params), |row| {
                if let Some(key) = row["_pk"].as_i64() {
                    keys.push(key);
                }
                Ok::<(), SelectionError>(())
            })
        })?;
        Ok(keys)
    }
}

/// One snippet per row when the query has positive text terms, ranked or
/// not (an `OR` of words selects without ranking).
pub fn snippets(
    reader: &Reader<'_>,
    program: &Program,
    rows: &mut [Value],
) -> Result<(), SelectionError> {
    let queries = [SnippetIndex::Prose, SnippetIndex::Output]
        .into_iter()
        .filter(|&index| {
            program
                .snippet_terms
                .iter()
                .any(|terms| terms.fts(index).is_some())
        })
        .map(|index| SnippetQuery {
            index,
            terms: &program.snippet_terms,
        })
        .collect::<Vec<_>>();
    if queries.is_empty() {
        return Ok(());
    }
    let scope: fn(i64) -> Within = match program.unit {
        polylogue_query::ast::Unit::Sessions => Within::Session,
        polylogue_query::ast::Unit::Messages => Within::Message,
        _ => return Ok(()),
    };
    let reuse = match &program.statements {
        super::compile::Statements::Keyed(keyed) => keyed.search.as_ref().is_some_and(|search| {
            queries[0].index == search.index
                && queries[0].fts().as_deref() == Some(search.expression.as_str())
        }),
        _ => false,
    };
    let mut supplied = vec![false; rows.len()];
    if reuse {
        let (indices, blocks): (Vec<_>, Vec<_>) = rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| row["_hit_block"].as_i64().map(|block| (index, block)))
            .unzip();
        for (index, snippet) in indices
            .into_iter()
            .zip(reader.snippets_from_blocks(&queries[0], &blocks)?)
        {
            if let Some(snippet) = snippet {
                rows[index]["snippet"] = serde_json::to_value(snippet).unwrap_or(Value::Null);
                supplied[index] = true;
            }
        }
    }
    let (indices, scopes): (Vec<_>, Vec<_>) = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            if supplied[index] {
                None
            } else {
                row["_pk"].as_i64().map(|pk| (index, scope(pk)))
            }
        })
        .unzip();
    for (index, snippet) in indices.into_iter().zip(reader.snippets(&queries, &scopes)?) {
        if let Some(snippet) = snippet {
            rows[index]["snippet"] = serde_json::to_value(snippet).unwrap_or(Value::Null);
        }
    }
    Ok(())
}

/// Each attachment's child rows, read for the whole page in one statement
/// per attachment.
pub fn attach(
    reader: &Reader<'_>,
    program: &Program,
    rows: &mut [Value],
) -> Result<(), SelectionError> {
    if program.attach.is_empty() {
        return Ok(());
    }
    let parents: Vec<i64> = rows.iter().filter_map(|row| row["_pk"].as_i64()).collect();
    reader.with_clock(&program.clock, || {
        for a in &program.attach {
            let mut children: std::collections::HashMap<i64, Vec<Value>> =
                std::collections::HashMap::with_capacity(parents.len());
            let mut params = values(&a.params);
            params.push(SqlValue::Text(Value::from(parents.clone()).to_string()));
            reader.query_each(&a.sql, &params, |mut child| {
                let parent = child["_parent"].as_i64();
                if let Value::Object(fields) = &mut child {
                    fields.remove("_parent");
                }
                for col in &a.columns {
                    shape_value(
                        col.kind,
                        program.clock.day.zone(),
                        &mut child[col.name.as_str()],
                    )?;
                }
                if let Some(parent) = parent {
                    children.entry(parent).or_default().push(child);
                }
                Ok::<_, SelectionError>(())
            })?;
            for row in rows.iter_mut() {
                let Some(pk) = row["_pk"].as_i64() else {
                    continue;
                };
                row[a.name.as_str()] = Value::Array(children.remove(&pk).unwrap_or_default());
            }
        }
        Ok(())
    })
}

/// Shape declared output columns and keep generated snippets.
pub fn present(program: &Program, row: &mut Value) -> Result<(), SelectionError> {
    let mut output = serde_json::Map::new();
    for col in &program.columns {
        let mut value = row[program.column_slot(&col.name)].clone();
        shape_value(col.kind, program.clock.day.zone(), &mut value)?;
        output.insert(col.name.clone(), value);
    }
    for attachment in &program.attach {
        output.insert(
            attachment.name.clone(),
            row[attachment.name.as_str()].clone(),
        );
    }
    if !program.snippet_terms.is_empty()
        && let Some(snippet) = row.get("snippet")
    {
        output.entry("snippet").or_insert_with(|| snippet.clone());
    }
    *row = Value::Object(output);
    Ok(())
}

/// Times leave as RFC 3339 in `zone` (the question's logical-day zone), with
/// its offset, so a person reads local clock times and a program still gets an
/// exact instant. A zero offset is written `Z`.
pub fn shape_value(
    kind: ColKind,
    zone: &jiff::tz::TimeZone,
    v: &mut Value,
) -> Result<(), SelectionError> {
    match kind {
        ColKind::Json => {
            if let Some(text) = v.as_str() {
                *v = serde_json::from_str(text).map_err(|_| {
                    SelectionError::invalid(
                        polylogue_model::Code::Store,
                        "invalid stored JSON value",
                    )
                })?;
            }
        }
        ColKind::Ids => {
            if v.is_null() {
                return Ok(());
            }
            let selection: Value = if let Some(text) = v.as_str() {
                serde_json::from_str(text).map_err(|e| {
                    SelectionError::invalid(
                        polylogue_model::Code::Store,
                        format!("invalid stored selection: {e}"),
                    )
                })?
            } else {
                v.take()
            };
            if selection.is_null() {
                *v = Value::Null;
                return Ok(());
            }
            if !selection.as_array().is_some_and(|ids| {
                ids.iter()
                    .all(|id| id.as_str().is_some_and(|id| !id.is_empty()))
            }) {
                return Err(SelectionError::invalid(
                    polylogue_model::Code::Store,
                    "a stored selection must be an array of IDs",
                ));
            }
            *v = selection;
        }
        ColKind::Time => {
            if let Some(ms) = v.as_i64() {
                let time = polylogue_model::Timestamp::from_unix_ms(ms).ok_or_else(|| {
                    SelectionError::invalid(polylogue_model::Code::Store, "invalid stored time")
                })?;
                *v = Value::String(polylogue_model::rfc3339_in(time, zone));
            }
        }
        ColKind::Bool => {
            if let Some(n) = v.as_i64() {
                *v = Value::Bool(n != 0);
            }
        }
        // Integer nano-USD leaves as exact decimal dollars.
        ColKind::Money => {
            if let Some(nusd) = v.as_i64() {
                *v = Value::Number(polylogue_model::money::usd_number(nusd));
            } else if !v.is_null() {
                return Err(SelectionError::invalid(
                    polylogue_model::Code::Store,
                    "a money value must be integer nano-USD",
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

/// How many rows one clause of an empty query matches, of the rows its
/// unit has.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct Explanation {
    pub clause: String,
    pub matches: u64,
    pub of: u64,
}

/// Each default with the rows it keeps of the rows without it, then each
/// top-level condition counted alone against the rows the defaults keep:
/// `repo:other → 0 of 5`.
pub fn why(
    reader: &Reader<'_>,
    program: &Program,
    params: &[SqlValue],
) -> Result<Vec<Explanation>, SelectionError> {
    reader.with_clock(&program.clock, || {
        let count = |sql: &str| -> Result<u64, SelectionError> {
            let r = reader.query(sql, params)?;
            Ok(r.first().and_then(|r| r["count(*)"].as_u64()).unwrap_or(0))
        };
        let of = count(&program.why_of)?;
        // Each default first: the rows it keeps of the rows without it.
        let defaults = program.why_defaults.iter().map(|w| {
            Ok(Explanation {
                clause: w.clause.clone(),
                matches: match &w.matches_sql {
                    Some(sql) => count(sql)?,
                    None => of,
                },
                of: count(&w.sql)?,
            })
        });
        let clauses = program.why.iter().map(|w| {
            Ok(Explanation {
                clause: w.clause.clone(),
                matches: count(&w.sql)?,
                of,
            })
        });
        defaults.chain(clauses).collect()
    })
}

/// Most candidate keys one recency batch reads.
const CANDIDATE_BATCH: usize = 256;

/// Rows read per statement while a key-first answer streams.
const HYDRATE_BATCH: usize = 256;

fn flow<E>(visited: Result<std::ops::ControlFlow<()>, E>) -> Result<(), StreamError<E>> {
    match visited {
        Ok(std::ops::ControlFlow::Continue(())) => Ok(()),
        Ok(std::ops::ControlFlow::Break(())) => Err(StreamError::Stop),
        Err(e) => Err(StreamError::Consumer(e)),
    }
}

enum StreamError<E> {
    Consumer(E),
    Selection(SelectionError),
    Stop,
}
impl<E> From<crate::StoreError> for StreamError<E> {
    fn from(error: crate::StoreError) -> Self {
        Self::Selection(error.into())
    }
}

/// Capture options that survive binding to the normal form. Transfer size and
/// current alias tables are deliberately outside the immutable execution input.
pub fn options_snapshot(
    options: &super::Options,
    why: bool,
    ranking_identity: Option<String>,
) -> Result<polylogue_model::query_objects::QueryExecutionOptions, SelectionError> {
    Ok(polylogue_model::query_objects::QueryExecutionOptions {
        fields: options.fields.clone(),
        self_session: options
            .self_session
            .as_deref()
            .map(polylogue_model::SessionId::parse)
            .transpose()
            .map_err(|error| {
                SelectionError::invalid(polylogue_model::Code::BadRequest, error.to_string())
            })?,
        here: options.here.clone(),
        seen_marks_of: options.seen_marks_of.clone(),
        embeddings: options.embeddings,
        why,
        ranking_identity,
        references: None,
    })
}

/// Lowering takes saved definitions and kept schemas from the validated saved
/// Plan. It never needs current names, completion values or command vocabulary.
pub fn options_from_snapshot(
    options: &polylogue_model::query_objects::QueryExecutionOptions,
    clock: query_language::time::Clock,
    random_seed: u64,
    page_size: Option<u64>,
) -> super::Options {
    super::Options {
        clock,
        origins: vec![],
        page_size,
        random_seed: Some(random_seed),
        here: options.here.clone(),
        seen_marks_of: options.seen_marks_of.clone(),
        embeddings: options.embeddings,
        self_session: options.self_session.as_ref().map(ToString::to_string),
        saved: Default::default(),
        fields: options.fields.clone(),
        kept: Default::default(),
        provider_catalog: None,
    }
}

/// Measure descriptors choose their transfer grain; Store alone changes the
/// executable relation and its total order.
pub fn prepare_measure(
    program: &mut Program,
    paging: Option<crate::measure::Paging>,
    options: &str,
    page_size: Option<u64>,
) -> Result<(), SelectionError> {
    let Some(measure) = &program.measure else {
        return Ok(());
    };
    // A measured selection is a derived relation: its sessions in order.
    let super::compile::Statements::Derived(derived) = &mut program.statements else {
        return Err(SelectionError::invalid(
            polylogue_model::Code::BadRequest,
            "a measured query reads a derived relation",
        ));
    };
    match paging {
        Some(crate::measure::Paging::Sessions) => {
            derived.rows = crate::measure::comparison::page_query(&derived.rows)
        }
        Some(crate::measure::Paging::Facets) => {
            *derived = super::compile::DerivedRows {
                rows: crate::measure::facets::query(
                    &measure.selection_sql,
                    &measure.population_sql,
                    options,
                )?,
                count: None,
                members: None,
            };
            program.counts_near = false;
            program.page_size = page_size.unwrap_or(100);
            program.order = vec![
                super::OrderKey {
                    column: "family".into(),
                    descending: false,
                    ..Default::default()
                },
                super::OrderKey {
                    column: "value".into(),
                    descending: false,
                    ..Default::default()
                },
            ];
        }
        Some(crate::measure::Paging::Responses) => {
            *derived = super::compile::DerivedRows {
                rows: crate::measure::responses_query(
                    &measure.selection_sql,
                    measure.since,
                    measure.until,
                    options,
                ),
                count: None,
                members: None,
            };
            program.counts_near = false;
            program.page_size = page_size.unwrap_or(50);
            program.order = ["time_order", "id"]
                .into_iter()
                .map(|column| super::OrderKey {
                    column: column.into(),
                    descending: false,
                    ..Default::default()
                })
                .collect();
        }
        None => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;

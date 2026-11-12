//! Executing the one compiled query and shaping declared public values.
use super::compile::PageColumns;
use super::{ColKind, Program};
use crate::search::{SnippetIndex, SnippetQuery, Within};
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
                    E::from(SelectionError::Invalid {
                        kind: "embed",
                        message: "semantic search unavailable".into(),
                    })
                })?;
                let similarity = near(&request.text).map_err(|message| {
                    E::from(SelectionError::Invalid {
                        kind: "embed",
                        message,
                    })
                })?;
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
            return Err(SelectionError::Invalid {
                kind: "bad_request",
                message: "page size must be positive".into(),
            });
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
                if self
                    .read_limit()
                    .is_some_and(|limit| limit.cancellation.is_cancelled())
                {
                    return Err(crate::StoreError::ReadCancelled.into());
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
                crate::sql_functions::similarity_count_basis(),
                self.index_identity()?,
                self.query_scope_fingerprint()?,
            ))
            .map_err(crate::StoreError::from)?,
        );
        {
            let mut cache = self
                .count_cache()
                .lock()
                .map_err(|_| crate::StoreError::Invalid("count cache lock poisoned".into()))?;
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
            .map_err(|_| crate::StoreError::Invalid("count cache lock poisoned".into()))?;
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
            after.is_some(),
            reverse,
            if row_at.is_some() {
                PageColumns::Order
            } else {
                PageColumns::All
            },
        );
        let result = match program.statements.hydrate(program.params.len()) {
            Some(hydrate) => {
                let mut keys = Vec::new();
                self.query_each(&page, &params, |key| {
                    keys.push(key);
                    if keys.len() < HYDRATE_BATCH {
                        return Ok(());
                    }
                    self.visit_hydrated(program, &hydrate, std::mem::take(&mut keys), &mut visit)
                })
                .and_then(|()| self.visit_hydrated(program, &hydrate, keys, &mut visit))
            }
            None => match row_at {
                None => self.query_each(&page, &params, |row| flow(visit(row))),
                Some(row_at) => self.query_each(&page, &params, |order| {
                    let mut exact = values(&program.params);
                    exact.extend(
                        program
                            .order
                            .iter()
                            .map(|key| sql_value(&order[&key.column])),
                    );
                    let row = self.query(&row_at, &exact)?.pop().ok_or_else(|| {
                        StreamError::Selection(SelectionError::Invalid {
                            kind: "stale_cursor",
                            message: "the selected row position is unavailable".into(),
                        })
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
            StreamError::Selection(SelectionError::Invalid {
                kind: "store",
                message: format!("a selected row has no {what}"),
            })
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
                    .ok_or_else(|| invalid("query has no measure selection"))?
                    .selection_sql
            ),
            QueryPopulation::MeasurePopulation => program
                .measure
                .as_ref()
                .ok_or_else(|| invalid("query has no measure population"))?
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
fn invalid(message: &str) -> SelectionError {
    SelectionError::Invalid {
        kind: "bad_request",
        message: message.into(),
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
    let (indices, scopes): (Vec<_>, Vec<_>) = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row["_pk"].as_i64().map(|pk| (index, scope(pk))))
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

/// Shape declared output columns and retain generated snippets.
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
                *v = serde_json::from_str(text).map_err(|_| SelectionError::Invalid {
                    kind: "store",
                    message: "invalid stored JSON value".into(),
                })?;
            }
        }
        ColKind::Ids => {
            if v.is_null() {
                return Ok(());
            }
            let selection: Value = if let Some(text) = v.as_str() {
                serde_json::from_str(text).map_err(|e| SelectionError::Invalid {
                    kind: "store",
                    message: format!("invalid stored selection: {e}"),
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
                return Err(SelectionError::Invalid {
                    kind: "store",
                    message: "a stored selection must be an array of IDs".into(),
                });
            }
            *v = selection;
        }
        ColKind::Time => {
            if let Some(ms) = v.as_i64() {
                *v = polylogue_model::Timestamp::from_unix_ms(ms).map_or(Value::Null, |t| {
                    Value::String(polylogue_model::rfc3339_in(t, zone))
                });
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
                return Err(SelectionError::Invalid {
                    kind: "store",
                    message: "a money value must be integer nano-USD".into(),
                });
            }
        }
        _ => {}
    }
    Ok(())
}

/// Each default with the rows it keeps of the rows without it, then each
/// top-level condition counted alone against the rows the defaults keep:
/// `repo:other → 0 of 5`.
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
                matches: of,
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

/// Capture options that survive canonical semantic binding. Transfer size and
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
            .map_err(|error| invalid(&error.to_string()))?,
        embeddings: options.embeddings,
        why,
        ranking_identity,
        direct_reference: None,
    })
}

/// Lowering takes saved definitions and kept schemas from the validated frozen
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
        embeddings: options.embeddings,
        self_session: options.self_session.as_ref().map(ToString::to_string),
        saved: Default::default(),
        fields: options.fields.clone(),
        kept: Default::default(),
        chat_pages: vec![],
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
        return Err(invalid("a measured query reads a derived relation"));
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
                },
                super::OrderKey {
                    column: "value".into(),
                    descending: false,
                },
            ];
        }
        None => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;

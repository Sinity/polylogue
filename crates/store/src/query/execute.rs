//! Executing the one compiled query and shaping declared public values.
use super::{ColKind, Program};
use crate::search::{SnippetIndex, SnippetQuery, Within};
use crate::selection::{sql_value, values};
use crate::{Reader, SqlValue, selection::SelectionError};
use serde_json::Value;

/// A transfer page; cardinality describes the complete selected answer.
pub struct QueryPage {
    pub rows: Vec<Value>,
    pub count: u64,
    pub without: u64,
    pub more: bool,
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
            let result = self.query_each(&program.sql, &values(&program.params), |mut row| {
                scope
                    .check()
                    .map_err(SelectionError::from)
                    .map_err(StreamError::Selection)?;
                present(program, &mut row).map_err(StreamError::Selection)?;
                match visit(row).map_err(StreamError::Consumer)? {
                    std::ops::ControlFlow::Continue(()) => Ok(()),
                    std::ops::ControlFlow::Break(()) => Err(StreamError::Stop),
                }
            });
            scope
                .check()
                .map_err(SelectionError::from)
                .map_err(E::from)?;
            match result {
                Ok(()) | Err(StreamError::Stop) => Ok(()),
                Err(StreamError::Consumer(e)) => Err(e),
                Err(StreamError::Selection(e)) => Err(E::from(e)),
            }
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
            let params = values(&program.params);
            crate::read::limit::ReadLimit::set_stage("count");
            let (count, without) = self.query_counts(&program.count_sql(), &params)?;
            let mut page_params = params;
            if let Some(after) = after {
                page_params.extend(after.iter().map(sql_value));
            }
            page_params.push(SqlValue::Int(
                i64::try_from(program.page_size.saturating_add(1)).unwrap_or(i64::MAX),
            ));
            crate::read::limit::ReadLimit::set_stage("page");
            let mut rows = self.query(&program.page_sql(after.is_some()), &page_params)?;
            let more = rows.len() as u64 > program.page_size;
            rows.truncate(usize::try_from(program.page_size).unwrap_or(usize::MAX));
            Ok(QueryPage {
                rows,
                count,
                without,
                more,
            })
        })
    }
    pub fn query_rows(&self, program: &Program) -> Result<Vec<Value>, SelectionError> {
        self.validate_selection(program)?;
        Ok(self.with_clock(&program.clock, || {
            self.query(&program.sql, &values(&program.params))
        })?)
    }
    pub fn query_keys(
        &self,
        program: &Program,
        population: QueryPopulation,
    ) -> Result<Vec<i64>, SelectionError> {
        self.validate_selection(program)?;
        let sql = match population {
            QueryPopulation::Answer => format!("SELECT _pk FROM ({})", program.sql),
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
    let queries: Vec<SnippetQuery<'_>> = [SnippetIndex::Prose, SnippetIndex::Output]
        .into_iter()
        .filter_map(|index| {
            let fts: Vec<String> = program
                .snippet_terms
                .iter()
                .filter_map(|terms| terms.fts(index))
                .collect();
            if fts.is_empty() {
                return None;
            }
            let mut words = vec![];
            for terms in &program.snippet_terms {
                terms.words(index, &mut words);
            }
            Some(SnippetQuery {
                index,
                fts: fts.join(" AND "),
                words,
            })
        })
        .collect();
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

pub fn attach(
    reader: &Reader<'_>,
    program: &Program,
    rows: &mut [Value],
) -> Result<(), SelectionError> {
    reader.with_clock(&program.clock, || {
        for a in &program.attach {
            for row in rows.iter_mut() {
                let Some(pk) = row["_pk"].as_i64() else {
                    continue;
                };
                let mut params = values(&a.params);
                params.push(SqlValue::Int(pk));
                let mut children = reader.query(&a.sql, &params)?;
                if let Some(n) = a.first {
                    children.truncate(usize::try_from(n).unwrap_or(usize::MAX));
                }
                if let Some(n) = a.last {
                    let n = usize::try_from(n).unwrap_or(usize::MAX);
                    let skip = children.len().saturating_sub(n);
                    children.drain(..skip);
                }
                for c in &mut children {
                    for col in &a.columns {
                        shape_value(
                            col.kind,
                            program.clock.day.zone(),
                            &mut c[col.name.as_str()],
                        )?;
                    }
                }
                row[a.name.as_str()] = Value::Array(children);
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
    match paging {
        Some(crate::measure::Paging::Sessions) => {
            program.sql = crate::measure::comparison::page_query(&program.sql)
        }
        Some(crate::measure::Paging::Facets) => {
            program.counts_near = false;
            program.row_count_sql = None;
            program.row_membership_sql = None;
            program.page_size = page_size.unwrap_or(100);
            program.sql = crate::measure::facets::query(
                &measure.selection_sql,
                &measure.population_sql,
                options,
            )?;
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

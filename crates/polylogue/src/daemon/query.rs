//! Running a query: compile it, check pasted IDs, count the whole answer,
//! read one page, cut snippets for text matches, fill attached rows, and
//! explain an empty result when asked. Every step reads through one reader, so the count, the page and
//! the attachments agree.

use polylogue_store::Reader;
use polylogue_store::query::{Options, Program, QueryError, Shape};
#[cfg(test)]
use polylogue_store::{SqlValue, query::ColKind};
use query_language::time::DaySetting;

use serde_json::{Value, json};

/// A failed query, as the response's error and its detail.
#[derive(Debug)]
pub struct Failure {
    pub kind: String,
    pub message: String,
    pub detail: Option<Value>,
}

impl From<polylogue_store::StoreError> for Failure {
    fn from(e: polylogue_store::StoreError) -> Self {
        Failure {
            kind: crate::protocol::store_code(&e).into(),
            message: e.to_string(),
            detail: None,
        }
    }
}

impl From<polylogue_store::selection::SelectionError> for Failure {
    fn from(error: polylogue_store::selection::SelectionError) -> Self {
        if let polylogue_store::selection::SelectionError::Store(error) = error {
            return Self::from(error);
        }
        Self {
            kind: error.kind().to_owned(),
            message: error.to_string(),
            detail: error.detail(),
        }
    }
}

impl From<QueryError> for Failure {
    fn from(e: QueryError) -> Self {
        Failure {
            kind: e.kind().into(),
            message: e.to_string(),
            detail: e.detail(),
        }
    }
}

pub(crate) mod assertions;
pub(crate) mod feed;
mod page;
use query_language::cursor::CursorSeal;

/// What a request asks besides the text.
pub struct Ask<'a> {
    pub text: &'a str,
    pub cursor: Option<&'a str>,
    pub page_size: Option<u64>,
    pub why: bool,
    pub session: Option<String>,
    /// One captured invocation and owner policy for the complete operation.
    pub context: std::sync::Arc<crate::operation::Context>,
    pub embeddings: bool,
    /// Captured provider/vector basis; never inferred from availability.
    pub ranking_identity: Option<String>,
    /// For `similar:`: every stored text's similarity to a text.
    pub near: Option<&'a NearFn<'a>>,
    /// The columns to show instead of the unit's defaults.
    pub fields: &'a [String],
}

/// Every stored text's similarity to a text, for `similar:`.
pub type NearFn<'a> =
    dyn Fn(&str) -> Result<std::sync::Arc<polylogue_store::vectors::Similarity>, String> + 'a;

/// Uses the configured provider and warm vectors for every query client.
pub(crate) fn with_ranking<T>(
    reader: &Reader<'_>,
    warm: &super::warm::Warm,
    config: &crate::config::Config,
    run: impl FnOnce(bool, Option<String>, &NearFn<'_>) -> T,
) -> T {
    let settings = reader.settings_snapshot(config.requests.machine()).ok();
    let provider = settings.as_ref().and_then(|settings| {
        crate::embeddings::Provider::from_settings(settings, &config.embedding_credentials)
            .ok()
            .flatten()
    });
    let identity = crate::embeddings::ranking_key(reader, provider.as_ref());
    let near = |text: &str| match &provider {
        Some(provider) => warm.similarity(reader, provider, text),
        None => Err("embeddings are off".into()),
    };
    run(identity.is_some(), identity, &near)
}

fn with_similarity<T>(
    reader: &Reader<'_>,
    program: &Program,
    near: Option<&NearFn<'_>>,
    read: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, Failure> {
    reader.with_query_context(program, near, read)
}

/// Saved definitions come from the same Store snapshot as selection preparation.
pub(crate) fn saved_queries(
    reader: &Reader<'_>,
) -> Result<std::collections::BTreeMap<String, Vec<polylogue_query::SavedQuery>>, Failure> {
    reader.selection_saved_queries().map_err(Failure::from)
}
fn kept_results(
    reader: &Reader<'_>,
) -> Result<std::collections::BTreeMap<String, polylogue_query::Kept>, Failure> {
    reader.selection_kept_results().map_err(Failure::from)
}

pub(crate) fn options(reader: &Reader<'_>, ask: &Ask<'_>, seed: u64) -> Result<Options, Failure> {
    Ok(Options {
        clock: ask.context.clock.clone(),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|o| o.name.to_owned())
            .collect(),
        command_words: crate::commands::command_words(),
        page_size: ask.page_size,
        random_seed: Some(seed),
        embeddings: ask.embeddings,
        self_session: ask.session.clone(),
        saved: saved_queries(reader)?,
        kept: kept_results(reader)?,
        fields: ask.fields.to_vec(),
        chat_pages: chat_pages(),
    })
}

/// Every origin's web chat pages, for pasted conversation URLs.
fn chat_pages() -> Vec<polylogue_query::ChatPage> {
    polylogue_sources::origin::ORIGINS
        .iter()
        .filter_map(|o| {
            let web = o.web.as_ref()?;
            Some(polylogue_query::ChatPage {
                origin: o.name.to_owned(),
                hosts: web.hosts.iter().map(|h| (*h).to_owned()).collect(),
                path: web.path.to_owned(),
            })
        })
        .collect()
}

/// Runs a query in full for keeping: its rows, and, when it has one
/// top-level exists or seq and no stages, its match rows too. The values
/// are what was shown; a kept result is read back, never re-run.
pub fn keep(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    let mut opts = options(reader, &ask, uuid::Uuid::now_v7().as_u128() as u64)?;
    opts.page_size = None;
    let program = polylogue_store::query::compile(ask.text, &opts)?;
    validate(reader, &program)?;
    let mut rows = with_similarity(reader, &program, ask.near, || {
        Ok(reader.query_rows(&program)?)
    })?;
    for row in &mut rows {
        present(&program, row)?;
    }
    let mut kept = json!({
        "unit": if program.shape == Shape::Matches { "matches" } else { program.unit.name() },
        "shape": program.shape,
        "columns": program.columns,
        "rows": rows,
        "query": ask.text,
        "now": program.clock.now,
        "definitions": program.definitions,
    });
    if polylogue_query::parse(ask.text)?.stages.is_empty()
        && let Ok(m) = polylogue_store::query::compile(&format!("{} | matches", ask.text), &opts)
    {
        validate(reader, &m)?;
        let mut matches = with_similarity(reader, &m, ask.near, || Ok(reader.query_rows(&m)?))?;
        for row in &mut matches {
            present(&m, row)?;
        }
        kept["match_columns"] = json!(m.columns);
        kept["matches"] = json!(matches);
    }
    Ok(kept)
}

pub fn run(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    polylogue_store::read::limit::ReadLimit::set_stage("query preparation");
    let seal = CursorSeal::new(reader.cursor_key()?);
    let position = reader.change_position()?;
    let index_file = reader.index_identity()?;
    let mut opts = options(reader, &ask, 0)?;
    let plan = reader.prepare_query_plan(ask.text, &opts)?;
    let question = plan.question_id();
    let cursor = ask
        .cursor
        .map(|text| page::open(text, &seal, question, &position, index_file, &opts))
        .transpose()?;
    let mut operation_versions = ask.context.cursor_versions();
    operation_versions.insert(
        "catalog".into(),
        polylogue_query::plan::catalog_fingerprint(),
    );
    if let Some(identity) = &ask.ranking_identity {
        operation_versions.insert(
            "ranking_identity".into(),
            blake3::hash(identity.as_bytes()).to_hex().to_string(),
        );
    }
    if let Some(scope) = reader.query_scope_fingerprint()? {
        operation_versions.insert("scope".into(), scope);
    }
    if let Some(cursor) = &cursor {
        for key in ["operation_context", "catalog", "scope", "ranking_identity"] {
            if cursor.context.versions.get(key) != operation_versions.get(key) {
                return Err(Failure {
                    kind: "stale_cursor".into(),
                    message: "the captured query context is no longer available".into(),
                    detail: None,
                });
            }
        }
    }
    let seed = cursor.as_ref().map_or_else(
        || uuid::Uuid::now_v7().as_u128() as u64,
        |cursor| cursor.context.random_seed,
    );
    if let Some(cursor) = &cursor {
        opts.clock = cursor.context.clock.clone();
    }
    opts.random_seed = Some(seed);
    let mut program = polylogue_store::query::lower(&plan, &opts)?;
    validate(reader, &program)?;
    prepare_measure(&mut program, ask.page_size)?;
    let after = cursor
        .as_ref()
        .map(|cursor| page::check(cursor, &program, &opts, ask.why))
        .transpose()?;
    let mut context = match cursor {
        Some(cursor) => cursor.context,
        None => page::context(
            question, &position, index_file, &opts, &program, seed, ask.why,
        )?,
    };
    context.versions.extend(operation_versions);
    let paging = page::Paging {
        seal: &seal,
        context,
    };
    let evaluation = ask
        .context
        .replay_clock(program.clock.clone())
        .map_err(context_failure)?;
    let mut output = with_similarity(reader, &program, ask.near, || {
        answer(
            reader,
            &program,
            &program.clock.day,
            after,
            &paging,
            ask.why,
        )
    })?;
    output["resolved"] = json!({"context":evaluation.provenance(),"times":program.resolved_times});
    Ok(output)
}

fn context_failure(response: crate::daemon::Response) -> Failure {
    match response {
        crate::daemon::Response::Error {
            kind,
            message,
            detail,
        } => Failure {
            kind,
            message,
            detail,
        },
        _ => Failure {
            kind: "internal".into(),
            message: "operation context replay failed".into(),
            detail: None,
        },
    }
}

/// The answer to a compiled program: its count and one page after `after`.
/// `day` is the logical day its times resolved in.
fn answer(
    reader: &Reader<'_>,
    program: &Program,
    day: &DaySetting,
    after: Option<Vec<Value>>,
    paging: &page::Paging<'_>,
    why_asked: bool,
) -> Result<Value, Failure> {
    let params = values(&program.params);
    let read_at = read_at(reader)?;
    if let Some(m) = &program.measure {
        polylogue_store::read::limit::ReadLimit::set_stage("measure");
        let (definition, _) = super::measure::find_checked(&m.what)?;
        if let Some(measure_paging) = definition.paging {
            let (rows, count, next, _) = read_page(reader, program, &after, paging)?;
            let session_order: std::collections::BTreeMap<_, _> = rows
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r["_session_id"].as_str().map(|id| (id.to_owned(), i)))
                .collect();
            let keys = rows.iter().filter_map(|r| r["_pk"].as_i64()).collect();
            let selection = polylogue_store::measure::Selection {
                sessions: Some(keys),
                ..Default::default()
            };
            let data = match measure_paging {
                super::measure::Paging::Facets => {
                    let selected = reader.query_keys(
                        program,
                        polylogue_store::query::execute::QueryPopulation::MeasureSelection,
                    )?;
                    let population = reader.query_keys(
                        program,
                        polylogue_store::query::execute::QueryPopulation::MeasurePopulation,
                    )?;
                    let scope = polylogue_store::measure::Selection {
                        sessions: Some(selected),
                        population: Some(population),
                        ..Default::default()
                    };
                    let summary = polylogue_store::measure::facets::summary(reader, &scope)?;
                    let mut facets = rows;
                    polylogue_store::measure::facets::weights(&mut facets, &summary);
                    json!({"summary":summary,"values":facets})
                }
                super::measure::Paging::Sessions => {
                    let mut data = super::measure::run(
                        reader,
                        &m.what,
                        selection.sessions,
                        program.clock.now.as_millisecond(),
                        day,
                        None,
                    )?;
                    let sessions = if data.is_array() {
                        data.as_array_mut()
                    } else {
                        data["sessions"].as_array_mut()
                    };
                    if let Some(sessions) = sessions {
                        sessions.sort_by_key(|p| {
                            session_order
                                .get(p["session"].as_str().unwrap_or(""))
                                .copied()
                        });
                    }
                    if definition.name == "time" {
                        data = json!({"sessions":data});
                    }
                    let selected = reader.query_keys(
                        program,
                        polylogue_store::query::execute::QueryPopulation::MeasureSelection,
                    )?;
                    let all = polylogue_store::measure::Selection {
                        sessions: Some(selected),
                        ..Default::default()
                    };
                    data["pending"] = json!(polylogue_store::measure::time::pending(reader, &all)?);
                    if definition.name == "compare" {
                        let (_, options) = super::measure::find_checked(&m.what)?;
                        let (varies, unknown, pending) =
                            polylogue_store::measure::comparison::summary(reader, &all, options)?;
                        data["varies"] = json!(varies);
                        data["unknown"] = json!(unknown);
                        data["pending"] = json!(pending);
                    }
                    data
                }
            };
            return Ok(
                json!({"outcome":"measure","measure":m.what,"data":data,"count":count,"next":next,"read_at":read_at}),
            );
        }
        let sessions = if m.everything {
            None
        } else {
            Some(reader.query_keys(
                program,
                polylogue_store::query::execute::QueryPopulation::Answer,
            )?)
        };
        let population = if definition.name == "facets" {
            Some(reader.query_keys(
                program,
                polylogue_store::query::execute::QueryPopulation::MeasurePopulation,
            )?)
        } else {
            None
        };
        let data = super::measure::run(
            reader,
            &m.what,
            sessions,
            program.clock.now.as_millisecond(),
            day,
            population,
        )?;
        return Ok(
            json!({ "outcome": "measure", "measure": m.what, "data": data, "read_at": read_at }),
        );
    }
    let (mut rows, count, next, without) = read_page(reader, program, &after, paging)?;

    polylogue_store::read::limit::ReadLimit::set_stage("attached rows");
    if program.shape == Shape::Rows {
        snippets(reader, program, &mut rows)?;
        attach(reader, program, &mut rows)?;
    }
    for row in &mut rows {
        present(program, row)?;
    }
    let mut notes = program.notes.clone();
    let mut unknown = serde_json::Map::new();
    if program.counts_near {
        if without > 0 {
            notes.push(format!(
                "{without} of {count} rows have no vector; their semantic similarity is unknown"
            ));
        }
        unknown.insert("near".into(), json!(without));
    }
    let outcome = if !program.incomplete.is_empty() {
        "incomplete"
    } else if count == 0 {
        "empty"
    } else {
        "rows"
    };
    let mut out = json!({
        "outcome": outcome,
        "unit": if program.shape == Shape::Matches { "matches" } else { program.unit.name() },
        "shape": program.shape,
        "columns": program.columns,
        "rows": rows,
        "count": count,
        "next": next,
        "ranked": program.ranked,
        "notes": notes,
        "incomplete": program.incomplete,
        "read_at": read_at,
    });
    if !program.definitions.is_empty() {
        out["definitions"] = json!(program.definitions);
    }
    if !unknown.is_empty() {
        out["unknown"] = Value::Object(unknown);
    }
    if why_asked && count == 0 {
        polylogue_store::read::limit::ReadLimit::set_stage("explanation counts");
        out["why"] = why(reader, program, &params)?;
    }
    Ok(out)
}

fn prepare_measure(program: &mut Program, page_size: Option<u64>) -> Result<(), Failure> {
    let Some(m) = program.measure.clone() else {
        return Ok(());
    };
    let (definition, options) = super::measure::find_checked(&m.what)?;
    polylogue_store::query::execute::prepare_measure(
        program,
        definition.paging,
        options,
        page_size,
    )?;
    Ok(())
}

/// Row selection for ordinary queries and paged measures uses the same SQL and cursor.
fn read_page(
    reader: &Reader<'_>,
    program: &Program,
    after: &Option<Vec<Value>>,
    paging: &page::Paging<'_>,
) -> Result<(Vec<Value>, u64, Option<String>, u64), Failure> {
    let page = reader.query_page(program, after.as_deref())?;
    let next = match page.rows.last() {
        Some(last) if page.more => Some(page::after_row(program, last, paging)?),
        _ => None,
    };
    Ok((page.rows, page.count, next, page.without))
}

/// What one read transaction read: the index generation and the
/// counters, scoped to the current index.
fn read_at(reader: &Reader<'_>) -> Result<Value, Failure> {
    let position = reader.change_position()?;
    Ok(
        json!({ "generation": position.generation, "index": position.index, "archive": position.archive, "cursor": position.to_string() }),
    )
}

pub(crate) use polylogue_store::selection::values;

/// Validation shares the Store authority with prepared reads and paging.
fn validate(reader: &Reader<'_>, program: &Program) -> Result<(), Failure> {
    Ok(reader.validate_selection(program)?)
}

#[cfg(test)]
pub(crate) use polylogue_store::query::execute::shape_value;
pub(crate) use polylogue_store::query::execute::{attach, present, snippets, why};

/// Session keys for usage and its response drill-down, from the same compiler
/// path as a session measure. Presentation limits never limit the population.
pub(crate) fn prepare_usage(
    reader: &Reader<'_>,
    text: &str,
    context: std::sync::Arc<crate::operation::Context>,
    embeddings: bool,
    ranking_identity: Option<String>,
) -> Result<(polylogue_query::plan::Plan, Options), Failure> {
    let text = format!("{text} | measure usage");
    let ask = Ask {
        text: &text,
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        context,
        embeddings,
        ranking_identity,
        near: None,
        fields: &[],
    };
    let opts = options(reader, &ask, 0)?;
    let plan = polylogue_query::compile(&text, &opts)?;
    Ok((plan, opts))
}

pub(crate) fn select_usage(
    reader: &Reader<'_>,
    program: &Program,
    near: Option<&NearFn<'_>>,
) -> Result<polylogue_store::measure::Selection, Failure> {
    validate(reader, program)?;
    let m = program.measure.as_ref().ok_or_else(|| Failure {
        kind: "bad_request".into(),
        message: "cost requires a sessions query".into(),
        detail: None,
    })?;
    let select = || -> Result<Option<Vec<i64>>, Failure> {
        Ok(if m.everything {
            None
        } else {
            Some(reader.query_keys(
                program,
                polylogue_store::query::execute::QueryPopulation::Answer,
            )?)
        })
    };
    let sessions = with_similarity(reader, program, near, select)?;
    Ok(polylogue_store::measure::Selection {
        sessions,
        ..Default::default()
    })
}

#[cfg(test)]
pub(crate) fn usage_selection(
    reader: &Reader<'_>,
    text: &str,
    context: std::sync::Arc<crate::operation::Context>,
    embeddings: bool,
    ranking_identity: Option<String>,
    near: Option<&NearFn<'_>>,
) -> Result<
    (
        polylogue_store::measure::Selection,
        query_language::identity::QuestionId,
    ),
    Failure,
> {
    let (plan, options) = prepare_usage(reader, text, context, embeddings, ranking_identity)?;
    let program = polylogue_store::query::lower(&plan, &options)?;
    Ok((select_usage(reader, &program, near)?, plan.question_id()))
}

/// Keep declared column metadata and source values as the compiled answer supplies them.
pub(crate) fn response(data: Value) -> super::Response {
    super::Response::Ok { data }
}

#[cfg(test)]
mod tests;

/// Compile with the same archive settings and calling session as execution.
pub fn explain(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    let options = options(reader, &ask, uuid::Uuid::now_v7().as_u128() as u64)?;
    Ok(
        json!({"subject": "query", "explanation": polylogue_store::query::explain::query(ask.text, &options)?}),
    )
}

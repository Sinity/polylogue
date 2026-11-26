//! Prepare a query's page, snippets, attachments and requested explanations,
//! then count its population within the same read transaction. A slow optional
//! count leaves the completed rows available.

use polylogue_store::Reader;
use polylogue_store::query::{Options, Program, RowKind, Shape};
#[cfg(test)]
use polylogue_store::{SqlValue, query::ColKind};
use query_language::time::DaySetting;

use serde_json::{Value, json};

use super::Failure;

pub(crate) mod annotations;
pub(crate) mod feed;
mod page;
use query_language::cursor::CursorSeal;

/// What a request asks besides the text.
#[derive(Clone)]
pub struct Ask<'a> {
    pub text: &'a str,
    pub cursor: Option<&'a str>,
    pub page_size: Option<u64>,
    pub why: bool,
    pub session: Option<String>,
    /// One captured invocation and its policy for the complete operation.
    pub context: std::sync::Arc<polylogue_reading::operation::Context>,
    pub embeddings: bool,
    /// Captured provider and vector identity; never inferred from availability.
    pub ranking_identity: Option<String>,
    /// For `similar:`: texts' similarity to a text.
    pub near: Option<&'a NearFn<'a>>,
    /// The columns to show instead of the unit's defaults.
    pub fields: &'a [String],
}

/// Texts' similarity to a text, for `similar:`.
pub use polylogue_store::selection::NearFn;

/// Uses the configured provider and warm vectors for every query client.
pub(crate) fn with_ranking<T>(
    reader: &Reader<'_>,
    warm: &super::warm::Warm,
    config: &crate::config::Config,
    run: impl FnOnce(bool, Option<String>, &NearFn<'_>) -> T,
) -> T {
    // A settings error stays the reason a `similar:` query cannot rank.
    let provider = reader
        .settings_snapshot(config.requests.machine())
        .map_err(|error| error.to_string())
        .and_then(|settings| {
            crate::embeddings::Provider::from_settings(&settings, &config.embedding_credentials)
        });
    let identity =
        crate::embeddings::ranking_key(reader, provider.as_ref().ok().and_then(Option::as_ref));
    let near = |text: &str| ranking_similarity(reader, warm, &provider, text);
    run(true, identity, &near)
}

/// Classify unavailable semantic rankings without losing read control failures.
pub(crate) fn ranking_similarity(
    reader: &Reader<'_>,
    warm: &super::warm::Warm,
    provider: &Result<Option<crate::embeddings::Provider>, String>,
    text: &str,
) -> polylogue_store::Result<std::sync::Arc<polylogue_store::vectors::Similarity>> {
    use polylogue_model::Reason;
    reader.check()?;
    let provider = match provider {
        Ok(Some(provider)) => provider,
        Ok(None) => return Err(Reason::EmbeddingsOff {}.into()),
        Err(problem) => {
            return Err(Reason::EmbeddingConfiguration {
                problem: problem.clone(),
            }
            .into());
        }
    };
    warm.similarity(reader, provider, text)
}

/// Asks the provider for the query vector the validated `query` ranks by,
/// before the request holds a reader for its run, so the provider wait
/// holds neither a reader slot nor the request's reader; the ranking then
/// finds the vector kept. The query is prepared and validated first, as its
/// run prepares it, so a query that is refused, or that ranks by nothing,
/// asks for nothing. Failures are left to the run, which reports them.
pub(crate) fn prefetch_ranking(call: &super::handler::Call<'_>, query: &str) {
    let daemon = call.daemon;
    let wanted = (|| {
        let reader = call.reader().ok()?;
        let context = call.context().ok()?;
        let settings = reader
            .settings_snapshot(daemon.config.requests.machine())
            .ok()?;
        let provider = crate::embeddings::Provider::from_settings(
            &settings,
            &daemon.config.embedding_credentials,
        )
        .ok()
        .flatten()?;
        let ranking_identity = Some(crate::embeddings::ranking_key(&reader, Some(&provider))?);
        let ask = Ask {
            text: query,
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            context,
            embeddings: ranking_identity.is_some(),
            ranking_identity,
            near: None,
            fields: &[],
        };
        let text = ranking_text(&reader, &ask).ok().flatten()?;
        Some((provider, text))
    })();
    if let Some((provider, text)) = wanted {
        let _ = daemon.warm.query_vector(&provider, &text);
    }
}

/// The text a query's semantic ranking asks a vector for, once the query
/// is prepared and validated as its run prepares it.
fn ranking_text(reader: &Reader<'_>, ask: &Ask<'_>) -> Result<Option<String>, Failure> {
    let opts = execution_options(reader, ask, 0)?;
    let plan = reader.prepare_query_plan(ask.text, &opts)?;
    let program = reader.lower_plan(&plan, &opts)?;
    validate(reader, &program)?;
    Ok(program.near.map(|near| near.text))
}

/// Whether a query's rows are archive rows: not groups, a summary, the
/// matches of a pattern or a measurement.
pub(crate) fn selects_rows(reader: &Reader<'_>, ask: &Ask<'_>) -> Result<bool, Failure> {
    let opts = execution_options(reader, ask, 0)?;
    let plan = reader.prepare_query_plan(ask.text, &opts)?;
    let program = reader.lower_plan(&plan, &opts)?;
    Ok(program.shape == Shape::Rows && program.measure.is_none())
}

fn with_similarity<T>(
    reader: &Reader<'_>,
    program: &Program,
    near: Option<&NearFn<'_>>,
    read: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, Failure> {
    reader.with_query_context(program, near, read)
}

/// Options for the compiler entry points that take text and definitions
/// separately. Prepared Store queries load their own definitions.
pub(crate) fn options(reader: &Reader<'_>, ask: &Ask<'_>, seed: u64) -> Result<Options, Failure> {
    let mut options = execution_options(reader, ask, seed)?;
    let named = reader.selection_definitions(ask.text, &options.origins)?;
    options.saved = named.saved;
    options.kept = named.kept;
    Ok(options)
}

/// Invocation settings for a query whose names are bound by the Store.
pub(crate) fn execution_options(
    reader: &Reader<'_>,
    ask: &Ask<'_>,
    seed: u64,
) -> Result<Options, Failure> {
    Ok(Options {
        clock: ask.context.clock.clone(),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|o| o.name.to_owned())
            .collect(),
        page_size: ask.page_size,
        random_seed: Some(seed),
        embeddings: ask.embeddings,
        self_session: ask
            .session
            .clone()
            .or_else(|| ask.context.calling_session().map(str::to_owned)),
        here: ask
            .context
            .working_directory()
            .map(|cwd| super::here::scope(cwd, reader))
            .transpose()?,
        seen_marks_of: Some(ask.context.authorship.author.to_string()),
        saved: Default::default(),
        kept: Default::default(),
        fields: ask.fields.to_vec(),
        provider_catalog: Some(polylogue_sources::origin::conversation_catalog()),
    })
}

/// Runs a query in full for keeping: its rows, and, when it has one
/// top-level exists or seq and no stages, its match rows too. The values
/// are what was shown; a kept result is read back, never re-run.
pub(crate) fn save_selected(
    reader: &Reader<'_>,
    ask: Ask<'_>,
    retained: &polylogue_model::user::LineSelection,
    references: Option<&[polylogue_model::Reference]>,
) -> Result<Value, Failure> {
    save_with_rows(reader, ask, retained, references, None)
}

/// Already visited relation values feed the same kept-result producer.
pub(crate) fn save_captured(
    reader: &Reader<'_>,
    ask: Ask<'_>,
    retained: &polylogue_model::user::LineSelection,
    rows: Vec<Value>,
    identities: Option<Vec<Vec<Value>>>,
) -> Result<Value, Failure> {
    save_with_rows(
        reader,
        ask,
        retained,
        None,
        Some(CapturedRows { rows, identities }),
    )
}

struct CapturedRows {
    rows: Vec<Value>,
    identities: Option<Vec<Vec<Value>>>,
}

fn save_with_rows(
    reader: &Reader<'_>,
    ask: Ask<'_>,
    retained: &polylogue_model::user::LineSelection,
    references: Option<&[polylogue_model::Reference]>,
    captured: Option<CapturedRows>,
) -> Result<Value, Failure> {
    // A line accepted under a read scope (primary mode's grant) is from a
    // scope this archive does not have.
    if retained.binding.replay.scope_fingerprint.is_some() {
        return Err(Failure::new(
            polylogue_model::Code::ScopeChanged,
            "the accepted line's read scope changed",
        ));
    }
    if ask.ranking_identity != retained.binding.query_options.ranking_identity {
        return Err(Failure::stale_cursor(
            "the accepted line's ranking provider changed",
        ));
    }
    if let Some(relation) = retained.relation.as_ref() {
        return Ok(relation.clone());
    }
    let binding = &retained.binding;
    let clock = binding
        .replay
        .context
        .clock
        .clone()
        .resolve(ask.context.clock.day.zone().clone())
        .map_err(Failure::bad_value)?;
    let opts = polylogue_store::query::execute::options_from_snapshot(
        &binding.query_options,
        clock,
        binding.replay.context.random_seed,
        None,
    );
    let plan = polylogue_query::plan::replay_binding(&binding.replay)?;
    let program = reader.lower_plan(&plan, &opts)?;
    if program.measure.is_some() {
        return save_measure(reader, &ask, &opts, &plan, &program);
    }
    let schema = plan.output_schema_for_columns(&program.columns)?;
    let mut match_schema = None;
    let mut match_columns = vec![];
    validate(reader, &program)?;
    let captured_values = captured.is_some();
    let (mut rows, identities) = match captured {
        Some(CapturedRows { rows, identities }) => (rows, identities),
        None => {
            let rows = with_similarity(reader, &program, ask.near, || {
                Ok(match references {
                    Some(references) => reader.query_fixed_rows(&program, references)?,
                    None => reader.query_rows(&program)?,
                })
            })?;
            let identities = match &program.row_kind {
                RowKind::UnitRows { columns, .. } => Some(
                    rows.iter()
                        .map(|row| columns.iter().map(|column| row[column].clone()).collect())
                        .collect(),
                ),
                _ => None,
            };
            (rows, identities)
        }
    };
    if !captured_values {
        for row in &mut rows {
            present(&program, row)?;
        }
    }
    let mut kept = json!({
        "unit": if program.shape == Shape::Matches { "matches" } else { program.unit.name() },
        "shape": program.shape,
        "columns": program.columns,
        "rows": rows,
        "query": ask.text,
        "now": program.clock.now,
        "read_at": read_at(reader)?,
        "definitions": program.definitions,
        "ranked":program.ranked,"notes":program.notes,"incomplete":program.incomplete,
    });
    if let Some(identities) = identities {
        kept["identities"] = json!(identities);
    }
    if let Some(matches) = retained.matches.as_ref() {
        kept["match_columns"] = matches["columns"].clone();
        kept["matches"] = matches["rows"].clone();
        match_columns =
            serde_json::from_value(matches["columns"].clone()).map_err(Failure::internal)?;
        match_schema =
            serde_json::from_value(matches["schema"].clone()).map_err(Failure::internal)?;
    } else if captured_values
        && polylogue_query::parse(ask.text)?.transforms.is_empty()
        && let Ok(match_plan) = {
            let syntax =
                query_language::format::format_query(plan.expanded_query().query(), plan.catalog())
                    .map_err(Failure::bad_value)?;
            let mut match_options = opts.clone();
            match_options.saved = plan.bindings().saved.clone();
            match_options.kept = plan.bindings().kept.clone();
            polylogue_query::compile(&format!("{syntax} | matches"), &match_options)
        }
    {
        let matches_program = polylogue_store::query::lower(&match_plan, &opts)?;
        match_schema = Some(match_plan.output_schema_for_columns(&matches_program.columns)?);
        validate(reader, &matches_program)?;
        let mut matches = with_similarity(reader, &matches_program, ask.near, || {
            Ok(reader.query_rows(&matches_program)?)
        })?;
        for row in &mut matches {
            present(&matches_program, row)?;
        }
        match_columns = matches_program.columns.clone();
        kept["match_columns"] = json!(matches_program.columns);
        kept["matches"] = json!(matches);
    }
    let output_metadata = polylogue_model::kept::OutputMetadata::captured(schema, match_schema);
    polylogue_query::Kept {
        output_metadata: output_metadata.clone(),
        shape: Some(program.shape),
        row_identity: program.shape == Shape::Rows
            && matches!(&program.row_kind, RowKind::UnitRows { .. }),
        unit: kept["unit"].as_str().expect("producer unit").into(),
        columns: program.columns.clone(),
        match_columns,
    }
    .validate_output_metadata()?;
    kept["output_metadata"] =
        serde_json::to_value(output_metadata).expect("kept metadata serializes");
    Ok(kept)
}

/// Resolve a line population and capture only the relation values that cannot
/// be hydrated later from native identities.
pub(crate) fn capture_selection(
    reader: &Reader<'_>,
    ask: &Ask<'_>,
) -> Result<
    (
        polylogue_model::user::LineSelection,
        Option<Vec<polylogue_model::Reference>>,
    ),
    Failure,
> {
    use polylogue_model::query_objects::ExecutionPresentation;
    use polylogue_store::query::Statements;
    let mut options = options(reader, ask, uuid::Uuid::now_v7().as_u128() as u64)?;
    options.page_size = None;
    let prepared = reader.prepare_query_selection(ask.text, &options)?;
    let measure = prepared.program().measure.is_some();
    let derived = measure || matches!(prepared.program().statements, Statements::Derived(_));
    // The Store owns which rows have readable references. A row-shaped feed
    // or query history is still a retainable relation, not a readable object.
    let mut references = (prepared.program().shape == Shape::Rows && !measure).then(Vec::new);
    let mut rows = Vec::new();
    let mut identities =
        matches!(prepared.program().row_kind, RowKind::UnitRows { .. }).then(Vec::new);
    if !measure {
        reader.visit_selection(
            &prepared,
            polylogue_store::read::VisitFrom::First,
            ask.near,
            |row| {
                match (&mut references, row.target) {
                    (Some(references), Some(target)) => references.push(target.reference()),
                    _ => references = None,
                }
                if let Some(identities) = &mut identities {
                    let Some(polylogue_store::selection::RowIdentity::Entity { key, .. }) =
                        row.identity
                    else {
                        return Err(Failure::internal("a native result has no identity"));
                    };
                    identities.push(key);
                }
                rows.push(row.values);
                Ok::<_, Failure>(std::ops::ControlFlow::Continue(()))
            },
        )?;
    }
    let mut retained = polylogue_model::user::LineSelection {
        binding: prepared.execution_binding(
            polylogue_store::query::execute::options_snapshot(
                &options,
                false,
                ask.ranking_identity.clone(),
            )?,
            ExecutionPresentation::Query,
            None,
        )?,
        relation: None,
        matches: None,
    };
    let value = save_captured(reader, ask.clone(), &retained, rows, identities)?;
    if derived || references.as_ref().is_none_or(Vec::is_empty) {
        retained.relation = Some(value);
    } else if value.get("matches").is_some() {
        retained.matches = Some(
            serde_json::json!({"rows":value["matches"], "columns":value["match_columns"], "schema":value["output_metadata"]["matches"]}),
        );
    }
    Ok((retained, references))
}

/// The accepted population uses the same current hydration or retained
/// relation values as Save, with native identities carried outside its columns.
pub(crate) fn accepted_population(
    reader: &Reader<'_>,
    ask: Ask<'_>,
    selected: &crate::daemon::verbs::Selection,
) -> Result<command_surface::Answer<crate::protocol::query::Answer>, Failure> {
    let retained = selected
        .retained
        .as_deref()
        .ok_or_else(|| Failure::internal("the accepted selection has no binding"))?;
    let context = ask.context.clone();
    let value = save_selected(reader, ask, retained, selected.resolved.as_deref())?;
    let rows: Vec<serde_json::Map<String, Value>> =
        serde_json::from_value(value["rows"].clone()).map_err(Failure::internal)?;
    let empty = rows.is_empty();
    let incomplete: Vec<String> =
        serde_json::from_value(value["incomplete"].clone()).map_err(Failure::internal)?;
    let answer = crate::protocol::query::Answer::Rows(Box::new(crate::protocol::query::Rows {
        read_at: serde_json::from_value(value["read_at"].clone()).map_err(Failure::internal)?,
        unit: value["unit"]
            .as_str()
            .ok_or_else(|| Failure::internal("the accepted population has no unit"))?
            .into(),
        shape: serde_json::from_value(value["shape"].clone()).map_err(Failure::internal)?,
        columns: serde_json::from_value(value["columns"].clone()).map_err(Failure::internal)?,
        count: Some(rows.len() as u64),
        rows,
        references: selected.resolved.clone(),
        next: None,
        ranked: value["ranked"].as_bool().unwrap_or(false),
        notes: serde_json::from_value(value["notes"].clone()).map_err(Failure::internal)?,
        incomplete: incomplete.clone(),
        resolved: crate::protocol::query::Resolved {
            context: context.run_context(),
            times: retained.binding.replay.context.resolved_times.clone(),
        },
        definitions: serde_json::from_value(value["definitions"].clone())
            .map_err(Failure::internal)?,
        unknown: Default::default(),
        why: None,
    }));
    Ok(command_surface::Answer {
        outcome: if !incomplete.is_empty() {
            command_surface::Outcome::Degraded {
                value: answer,
                missing: incomplete,
            }
        } else if empty {
            command_surface::Outcome::Empty(answer)
        } else {
            command_surface::Outcome::Ok(answer)
        },
        envelope: Default::default(),
    })
}

/// Keep the actual measurement answer, following the normal authenticated
/// cursor until its declared paging owner has returned every page.
fn save_measure(
    reader: &Reader<'_>,
    ask: &Ask<'_>,
    options: &Options,
    plan: &polylogue_query::plan::Plan,
    program: &Program,
) -> Result<Value, Failure> {
    use crate::protocol::query::Answer;
    let mut program = program.clone();
    prepare_measure(&mut program, options.page_size)?;
    let what = program
        .measure
        .as_ref()
        .expect("measurement program")
        .what
        .clone();
    let (definition, _) = super::measure::find_checked(&what)?;
    let position = reader.change_position()?;
    let index = reader.index_identity()?;
    let seal = CursorSeal::new(reader.cursor_key()?);
    let context = page::context(
        plan.question_id(),
        &position,
        index,
        options,
        &program,
        options.random_seed.unwrap_or(0),
        false,
    )?;
    let paging = page::Paging {
        seal: &seal,
        context,
    };
    let resolved = crate::protocol::query::Resolved {
        context: ask
            .context
            .replay_clock(program.clock.clone())?
            .run_context(),
        times: program.resolved_times.clone(),
    };
    let mut after = None;
    let mut data = None;
    loop {
        let page = with_similarity(reader, &program, ask.near, || {
            answer(
                reader,
                &program,
                &program.clock.day,
                after.clone(),
                &paging,
                false,
                resolved.clone(),
            )
        })?;
        let (value, next) = match page.value {
            Answer::Measure(value) => (value.data, None),
            Answer::PagedMeasure(value) => (value.data, value.next),
            Answer::Rows(_) => return Err(Failure::internal("a measurement returned entity rows")),
        };
        match &mut data {
            Some(data) => super::measure::append_page(
                definition.paging.ok_or_else(|| {
                    Failure::internal("an unpaged measure returned a continuation")
                })?,
                data,
                value,
            )?,
            None => data = Some(value),
        }
        let Some(next) = next else {
            break;
        };
        let cursor = page::open(&next, &seal, plan.question_id(), &position, index, options)?;
        after = Some(page::check(&cursor, &program, options, false)?);
    }
    let columns = vec![
        polylogue_query::Column {
            name: "measure".into(),
            kind: polylogue_query::ColKind::Text,
            reference: None,
        },
        polylogue_query::Column {
            name: "data".into(),
            kind: polylogue_query::ColKind::Json,
            reference: None,
        },
    ];
    let output_metadata = polylogue_model::kept::OutputMetadata::captured(
        polylogue_query::Kept::summary_output(&columns),
        None,
    );
    polylogue_query::Kept {
        output_metadata: output_metadata.clone(),
        shape: Some(Shape::Summary),
        row_identity: false,
        unit: program.unit.name().into(),
        columns: columns.clone(),
        match_columns: vec![],
    }
    .validate_output_metadata()?;
    Ok(
        json!({"unit":program.unit.name(),"shape":Shape::Summary,"columns":columns,
        "rows":[{"measure":what,"data":data.expect("a measure returns at least one page")}],
        "query":ask.text,"now":program.clock.now,"read_at":read_at(reader)?,"definitions":program.definitions,
        "output_metadata":output_metadata}),
    )
}

/// How a query answered, decided where the answer is assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Rows,
    Empty,
    /// Assembled from parts of which at least one failed.
    Incomplete,
    Measure,
}

/// A query's answer with its outcome and the notes that explain it.
#[derive(Debug)]
pub struct QueryAnswer {
    pub value: crate::protocol::query::Answer,
    pub logical_sessions: Vec<bool>,
    pub outcome: Outcome,
    pub notes: Vec<String>,
}

impl QueryAnswer {
    fn measure(value: crate::protocol::query::Answer) -> Self {
        Self {
            value,
            logical_sessions: Vec::new(),
            outcome: Outcome::Measure,
            notes: Vec::new(),
        }
    }

    fn count_deferred(&mut self, note: String) {
        if let crate::protocol::query::Answer::Rows(rows) = &mut self.value {
            rows.notes.push(note.clone());
        }
        self.notes.push(note);
    }
}

/// An answer whose rows are ready before its optional exact count runs.
pub struct PreparedQuery<'read, 'pool> {
    reader: &'read Reader<'pool>,
    answer: QueryAnswer,
    count_program: Option<Program>,
    similarity: Option<std::sync::Arc<polylogue_store::vectors::Similarity>>,
}
impl PreparedQuery<'_, '_> {
    pub(super) fn is_empty(&self) -> bool {
        self.answer.outcome == Outcome::Empty
    }

    /// Completes a listing's page with its optional exact count, run on this
    /// reader after the page under the request's deadline less the time its
    /// reply needs. A count that runs out answers "count deferred" with the
    /// command that runs it without a deadline.
    pub(super) fn complete_within(
        self,
        limit: &polylogue_store::read::limit::ReadLimit,
    ) -> Result<QueryAnswer, Failure> {
        if let (Some(deadline), Some(left)) = (limit.deadline(), limit.remaining()) {
            limit.stop_by(deadline - super::service::answer_reserve(left));
        }
        self.complete()
    }

    pub fn complete(self) -> Result<QueryAnswer, Failure> {
        let Self {
            reader,
            answer,
            count_program,
            similarity,
        } = self;
        let Some(program) = count_program else {
            return Ok(answer);
        };
        let query_count = || reader.query_count(&program);
        let counts = match similarity {
            Some(similarity) => reader.with_similarity(similarity, query_count),
            None => query_count(),
        }?;
        let mut answer = answer;
        if let Some((count, without)) = counts {
            match &mut answer.value {
                crate::protocol::query::Answer::Rows(rows) => {
                    rows.count = Some(count);
                    if program.counts_near {
                        rows.unknown.insert("similar".into(), without);
                        if without > 0 {
                            let note = format!(
                                "{without} of {count} rows have no vector; their semantic ranking is unknown"
                            );
                            rows.notes.push(note.clone());
                            answer.notes.push(note);
                        }
                    }
                }
                crate::protocol::query::Answer::PagedMeasure(measure) => {
                    measure.count = Some(count)
                }
                crate::protocol::query::Answer::Measure(_) => {}
            }
        } else {
            answer.count_deferred(format!(
                "count deferred; run the exact count without a deadline: polylogue {} --deadline none",
                polylogue_render::shell_quote(&if program.measure.is_some() {
                    program.query.clone()
                } else {
                    format!("{} | count", program.query)
                })
            ));
        }
        Ok(answer)
    }
}

/// Internal query consumers keep their enclosing operation's read phase.
pub fn run(reader: &Reader<'_>, ask: Ask<'_>) -> Result<QueryAnswer, Failure> {
    prepare(reader, ask)?.complete()
}

fn cursor_failure(mut failure: Failure, text: &str) -> Failure {
    if failure.code() == "stale_cursor" {
        if !failure
            .message
            .ends_with("; run the query again without its cursor")
        {
            failure
                .error
                .message
                .push_str("; run the query again without its cursor");
        }
        failure.error = failure
            .error
            .with_suggestion(command_surface::Suggestion::Query(text.to_owned()));
    }
    failure
}

pub fn prepare<'read, 'pool>(
    reader: &'read Reader<'pool>,
    ask: Ask<'_>,
) -> Result<PreparedQuery<'read, 'pool>, Failure> {
    polylogue_store::read::limit::ReadLimit::set_stage("query preparation");
    let seal = CursorSeal::new(reader.cursor_key()?);
    let position = reader.change_position()?;
    let index_file = reader.index_identity()?;
    let mut opts = execution_options(reader, &ask, 0)?;
    let plan = reader.prepare_query_plan(ask.text, &opts)?;
    let question = plan.question_id();
    let cursor = ask
        .cursor
        .map(|text| {
            page::open(text, &seal, question, &position, index_file, &opts)
                .map_err(|failure| cursor_failure(failure, ask.text))
        })
        .transpose()?;
    let mut operation_versions = ask.context.cursor_versions();
    operation_versions.insert(
        "catalog".into(),
        polylogue_query::plan::catalog_fingerprint().to_owned(),
    );
    if let Some(identity) = &ask.ranking_identity {
        operation_versions.insert(
            "ranking_identity".into(),
            blake3::hash(identity.as_bytes()).to_hex().to_string(),
        );
    }
    if let Some(cursor) = &cursor {
        for key in ["operation_context", "catalog", "scope", "ranking_identity"] {
            if cursor.context.versions.get(key) != operation_versions.get(key) {
                return Err(cursor_failure(
                    Failure::stale_cursor("the settings or data used by this query changed"),
                    ask.text,
                ));
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
    let mut program = reader.lower_plan(&plan, &opts)?;
    validate(reader, &program)?;
    prepare_measure(&mut program, ask.page_size)?;
    let similarity = program.ranking(reader, ask.near)?;
    let ready = |_: &str| {
        similarity.clone().ok_or_else(|| {
            polylogue_store::StoreError::from(polylogue_model::Reason::EmbeddingsOff {})
        })
    };
    let after = cursor
        .as_ref()
        .map(|cursor| {
            page::check(cursor, &program, &opts, ask.why)
                .map_err(|failure| cursor_failure(failure, ask.text))
        })
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
    let evaluation = ask.context.replay_clock(program.clock.clone())?;
    let resolved = crate::protocol::query::Resolved {
        context: evaluation.run_context(),
        times: program.resolved_times.clone(),
    };
    let answer = with_similarity(
        reader,
        &program,
        ask.near.map(|_| &ready as &NearFn<'_>),
        || {
            answer(
                reader,
                &program,
                &program.clock.day,
                after,
                &paging,
                ask.why,
                resolved,
            )
        },
    )?;
    let count_program = match &answer.value {
        crate::protocol::query::Answer::Measure(_) => None,
        _ => Some(program),
    };
    Ok(PreparedQuery {
        reader,
        answer,
        count_program,
        similarity,
    })
}

/// Prepare one page after `after`, before its optional population count.
/// `day` is the logical day its times resolved in.
fn answer(
    reader: &Reader<'_>,
    program: &Program,
    day: &DaySetting,
    after: Option<Vec<Value>>,
    paging: &page::Paging<'_>,
    why_asked: bool,
    resolved: crate::protocol::query::Resolved,
) -> Result<QueryAnswer, Failure> {
    use crate::protocol::query::{Answer, Measure, PagedMeasure, Rows};
    let params = values(&program.params);
    let read_at = read_at(reader)?;
    if let Some(measure) = &program.measure {
        polylogue_store::read::limit::ReadLimit::set_stage("measure");
        let (definition, _) = super::measure::find_checked(&measure.what)?;
        if let Some(measure_paging) = definition.paging {
            let (rows, next) = read_page(reader, program, &after, paging)?;
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
                super::measure::Paging::Responses => {
                    let responses: Vec<Value> = rows
                        .into_iter()
                        .map(|mut row| {
                            if let Some(row) = row.as_object_mut() {
                                row.remove("time_order");
                                // Amounts travel as nano-USD and exact dollars.
                                for (stored, shown) in [
                                    ("cost_nusd", "cost"),
                                    ("provider_cost_nusd", "provider_cost"),
                                ] {
                                    let amount = row
                                        .remove(stored)
                                        .and_then(|v| v.as_i64())
                                        .map(polylogue_model::money::Amount);
                                    row.insert(shown.into(), json!(amount));
                                }
                            }
                            row
                        })
                        .collect();
                    json!({ "responses": responses })
                }
                super::measure::Paging::Sessions => {
                    let mut data = super::measure::run(
                        reader,
                        &measure.what,
                        selection.sessions,
                        (measure.since, measure.until),
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
                        let (_, options) = super::measure::find_checked(&measure.what)?;
                        let (varies, unknown, pending) =
                            polylogue_store::measure::comparison::summary(reader, &all, options)?;
                        data["varies"] = json!(varies);
                        data["unknown"] = json!(unknown);
                        data["pending"] = json!(pending);
                    }
                    data
                }
            };
            return Ok(QueryAnswer::measure(Answer::PagedMeasure(PagedMeasure {
                measure: measure.what.clone(),
                data,
                read_at,
                resolved,
                count: None,
                next,
            })));
        }
        let sessions = if measure.everything {
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
            &measure.what,
            sessions,
            (measure.since, measure.until),
            program.clock.now.as_millisecond(),
            day,
            population,
        )?;
        return Ok(QueryAnswer::measure(Answer::Measure(Measure {
            measure: measure.what.clone(),
            data,
            read_at,
            resolved,
        })));
    }
    let (mut rows, next) = read_page(reader, program, &after, paging)?;

    polylogue_store::read::limit::ReadLimit::set_stage("attached rows");
    if program.shape == Shape::Rows {
        snippets(reader, program, &mut rows)?;
        attach(reader, program, &mut rows)?;
    }
    let references = if matches!(program.row_kind, RowKind::UnitRows { .. }) {
        rows.iter()
            .map(|row| polylogue_store::selection::row_reference(program, row))
            .collect::<Option<Vec<_>>>()
    } else {
        None
    };
    let logical_sessions =
        if program.unit == polylogue_query::ast::Unit::Sessions && program.shape == Shape::Rows {
            rows.iter()
                .map(|row| {
                    program.session_rows.logical(row).ok_or_else(|| {
                        Failure::new(
                            polylogue_model::Code::Internal,
                            "a selected session has no physical/logical coordinate",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
    for row in &mut rows {
        present(program, row)?;
    }
    let why = if why_asked {
        polylogue_store::read::limit::ReadLimit::set_stage("explanation counts");
        Some(why(reader, program, &params)?)
    } else {
        None
    };
    let rows: Vec<_> = rows
        .into_iter()
        .map(|row| match row {
            Value::Object(fields) => Ok(fields),
            _ => Err(Failure::new(
                polylogue_model::Code::Internal,
                "a query row is not a record",
            )),
        })
        .collect::<Result<_, _>>()?;
    let notes = program.notes.clone();
    let unknown = std::collections::BTreeMap::new();
    let outcome = if !program.incomplete.is_empty() {
        Outcome::Incomplete
    } else if rows.is_empty() && next.is_none() {
        Outcome::Empty
    } else {
        Outcome::Rows
    };
    let mut answer_notes = notes.clone();
    answer_notes.extend(program.incomplete.iter().cloned());
    Ok(QueryAnswer {
        logical_sessions,
        value: Answer::Rows(Box::new(Rows {
            read_at,
            unit: if program.shape == Shape::Matches {
                "matches".to_owned()
            } else {
                program.unit.name().to_owned()
            },
            shape: program.shape,
            columns: program.columns.clone(),
            rows,
            references,
            count: None,
            next,
            ranked: program.ranked,
            notes,
            incomplete: program.incomplete.clone(),
            resolved,
            definitions: program.definitions.clone(),
            unknown,
            why,
        })),
        outcome,
        notes: answer_notes,
    })
}

fn prepare_measure(program: &mut Program, page_size: Option<u64>) -> Result<(), Failure> {
    let Some(measure) = program.measure.clone() else {
        return Ok(());
    };
    let (definition, options) = super::measure::find_checked(&measure.what)?;
    polylogue_store::query::execute::prepare_measure(
        program,
        definition.paging,
        options,
        page_size,
    )?;
    Ok(())
}

/// Row selection for queries and paged measures uses the same SQL and cursor.
fn read_page(
    reader: &Reader<'_>,
    program: &Program,
    after: &Option<Vec<Value>>,
    paging: &page::Paging<'_>,
) -> Result<(Vec<Value>, Option<String>), Failure> {
    let page = reader.query_page(program, after.as_deref())?;
    let next = match page.rows.last() {
        Some(last) if page.more => Some(page::after_row(program, last, paging)?),
        _ => None,
    };
    Ok((page.rows, next))
}

/// What one read transaction read: the index generation and the
/// counters, scoped to the current index.
fn read_at(reader: &Reader<'_>) -> Result<crate::protocol::query::Position, Failure> {
    let position = reader.change_position()?;
    let counter = |value: i64| {
        u64::try_from(value)
            .map_err(|_| Failure::new(polylogue_model::Code::Store, "a change counter is negative"))
    };
    Ok(crate::protocol::query::Position {
        freshness: None,
        cursor: position.to_string(),
        index: counter(position.index)?,
        archive: counter(position.archive)?,
        generation: position.generation,
    })
}

pub(crate) use polylogue_store::selection::values;

/// Validation uses the same store check as prepared reads and paging.
fn validate(reader: &Reader<'_>, program: &Program) -> Result<(), Failure> {
    Ok(reader.validate_selection(program)?)
}

#[cfg(test)]
pub(crate) use polylogue_store::query::execute::shape_value;
pub(crate) use polylogue_store::query::execute::{attach, present, snippets, why};

#[cfg(test)]
mod tests;

/// Compile with the same archive settings and calling session as execution.
pub fn explain(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    let options = options(reader, &ask, uuid::Uuid::now_v7().as_u128() as u64)?;
    Ok(
        json!({"subject": "query", "explanation": polylogue_store::query::explain::query(ask.text, &options)?}),
    )
}

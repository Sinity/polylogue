//! Running a query: compile it, check pasted IDs, count the whole answer,
//! read one page, cut snippets for text matches, fill attached rows, and
//! explain an empty result when asked. Every step reads through one reader, so the count, the page and
//! the attachments agree.

use polylogue_model::LogicalDay;
use polylogue_query::{ColKind, Options, Param, Program, QueryError, Shape};
use polylogue_store::search::Within;
use polylogue_store::{Reader, SqlValue};

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
            kind: match &e {
                polylogue_store::StoreError::ReadDeadlineExceeded => "deadline_exceeded",
                polylogue_store::StoreError::ReadCancelled => "cancelled",
                _ => "store",
            }
            .into(),
            message: e.to_string(),
            detail: None,
        }
    }
}

impl From<QueryError> for Failure {
    fn from(e: QueryError) -> Self {
        Failure {
            kind: e.kind().into(),
            message: e.to_string(),
            detail: serde_json::to_value(&e).ok(),
        }
    }
}

pub(crate) mod feed;
pub(crate) mod notes;
mod page;
use page::Cursor;

/// What a request asks besides the text.
pub struct Ask<'a> {
    pub text: &'a str,
    pub cursor: Option<&'a str>,
    pub page_size: Option<u64>,
    pub why: bool,
    pub session: Option<String>,
    /// Unix ms; the reader's `now` unless a cursor fixes it.
    pub now: i64,
    /// The logical day times resolve in; `None` reads the operator's
    /// settings ([`logical_day`]).
    pub day: Option<LogicalDay>,
    pub embeddings: bool,
    /// For `near:`: every stored text's similarity to a text.
    pub near: Option<&'a NearFn<'a>>,
    /// The columns to show instead of the unit's defaults.
    pub fields: &'a [String],
}

/// Every stored text's similarity to a text, for `near:`.
pub type NearFn<'a> =
    dyn Fn(&str) -> Result<std::sync::Arc<polylogue_store::vectors::Similarity>, String> + 'a;

/// Uses the configured provider and warm vectors for every query client.
pub(crate) fn with_ranking<T>(
    reader: &Reader<'_>,
    warm: &super::warm::Warm,
    run: impl FnOnce(bool, &NearFn<'_>) -> T,
) -> T {
    let provider = crate::embeddings::Provider::from_settings(
        reader,
        &polylogue_store::user_writes::machine_id(),
    )
    .ok()
    .flatten();
    let available = crate::embeddings::ranking_key(reader, provider.as_ref()).is_some();
    let near = |text: &str| match &provider {
        Some(provider) => warm.similarity(reader, provider, text),
        None => Err("embeddings are off".into()),
    };
    run(available, &near)
}

fn with_similarity<T>(
    reader: &Reader<'_>,
    program: &Program,
    near: Option<&NearFn<'_>>,
    read: impl FnOnce() -> Result<T, Failure>,
) -> Result<T, Failure> {
    match &program.near {
        Some(request) => {
            let near = near.ok_or_else(|| Failure {
                kind: "embed".into(),
                message: "semantic search unavailable".into(),
                detail: None,
            })?;
            let similarity = near(&request.text).map_err(|message| Failure {
                kind: "embed".into(),
                message,
                detail: None,
            })?;
            reader.with_similarity(similarity, read)
        }
        None => read(),
    }
}

/// The current saved queries by name (`saved_query` notes, newest
/// revision of each family, not withdrawn), for `query:NAME`. Two families
/// with one name both stay listed, so the name is refused as ambiguous.
pub(crate) fn saved_queries(
    reader: &Reader<'_>,
) -> Result<std::collections::BTreeMap<String, Vec<polylogue_query::SavedQuery>>, Failure> {
    let sql = format!(
        "SELECT n.key AS k, n.value AS v, n.id AS id, n.first_note AS first \
         FROM source_note n WHERE n.kind = 'saved_query' \
         AND n.key IS NOT NULL AND n.value IS NOT NULL AND {} AND NOT {} ORDER BY n.first_note",
        polylogue_query::fields::current_note("n"),
        polylogue_query::fields::withdrawn_note("n"),
    );
    let mut saved: std::collections::BTreeMap<String, Vec<polylogue_query::SavedQuery>> =
        Default::default();
    for r in reader.query(&sql, &[])? {
        let (Some(name), Some(text), Some(id), Some(first)) = (
            r["k"].as_str(),
            r["v"].as_str(),
            r["id"].as_str(),
            r["first"].as_str(),
        ) else {
            continue;
        };
        saved
            .entry(name.to_owned())
            .or_default()
            .push(polylogue_query::SavedQuery {
                text: text.to_owned(),
                revision: id.to_owned(),
                first_note: first.to_owned(),
            });
    }
    Ok(saved)
}

/// Kept results by ID, with the shape of their stored rows, for
/// `result:<id>`.
fn kept_results(
    reader: &Reader<'_>,
) -> Result<std::collections::BTreeMap<String, polylogue_query::Kept>, Failure> {
    reader
        .query(
            "SELECT id, json_extract(rows_json, '$.unit') AS unit, \
             json_extract(rows_json, '$.columns') AS columns, \
             json_extract(rows_json, '$.match_columns') AS match_columns, \
             json_type(rows_json, '$.match_columns') AS match_columns_type FROM source_kept_result",
            &[],
        )?
        .iter()
        .map(|r| {
            let invalid = |field: &str| Failure {
                kind: "store".into(),
                message: format!("kept result {} has invalid {field}", r["id"]),
                detail: None,
            };
            let parse = |field: &str| -> Result<Vec<polylogue_query::Column>, Failure> {
                let text = r[field].as_str().ok_or_else(|| invalid(field))?;
                serde_json::from_str(text).map_err(|_| invalid(field))
            };
            Ok((
                r["id"].as_str().ok_or_else(|| invalid("id"))?.to_owned(),
                polylogue_query::Kept {
                    unit: r["unit"]
                        .as_str()
                        .ok_or_else(|| invalid("unit"))?
                        .to_owned(),
                    columns: parse("columns")?,
                    match_columns: if r["match_columns_type"].is_null() {
                        Vec::new()
                    } else {
                        parse("match_columns")?
                    },
                },
            ))
        })
        .collect()
}

/// The operator's logical day: `day.zone` (an IANA zone; the machine's
/// unless set) and `day.logical_start` (`HH:MM`; 04:00 unless set).
pub fn logical_day(reader: &Reader<'_>) -> Result<LogicalDay, Failure> {
    let machine = polylogue_store::user_writes::machine_id();
    let invalid = |message: String| Failure {
        kind: "invalid".into(),
        message,
        detail: None,
    };
    let start = match reader.setting("day.logical_start", &machine)? {
        None => LogicalDay::DEFAULT_START_MINUTES,
        Some(text) => text
            .split_once(':')
            .and_then(|(h, m)| Some((h.parse::<u16>().ok()?, m.parse::<u16>().ok()?)))
            .filter(|(h, m)| *h < 24 && *m < 60)
            .map(|(h, m)| h * 60 + m)
            .ok_or_else(|| {
                invalid(format!(
                    "day.logical_start is {text:?}; it is a time of day such as 04:00"
                ))
            })?,
    };
    match reader.setting("day.zone", &machine)? {
        None => Ok(LogicalDay {
            start_minutes: start,
            ..LogicalDay::system()
        }),
        Some(zone) => LogicalDay::named(&zone, start).ok_or_else(|| {
            invalid(format!(
                "day.zone is {zone:?}; it is a time zone such as Europe/Warsaw"
            ))
        }),
    }
}

fn options(reader: &Reader<'_>, ask: &Ask<'_>, now: i64, seed: u64) -> Result<Options, Failure> {
    let day = match &ask.day {
        Some(day) => day.clone(),
        None => logical_day(reader)?,
    };
    Ok(Options {
        clock: polylogue_query::Clock::new(now, day),
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
    let mut opts = options(reader, &ask, ask.now, uuid::Uuid::now_v7().as_u128() as u64)?;
    opts.page_size = None;
    let program = polylogue_query::compile(ask.text, &opts)?;
    validate(reader, &program)?;
    let mut rows = with_similarity(reader, &program, ask.near, || {
        Ok(reader.query(&program.sql, &values(&program.params))?)
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
        "now": program.now,
        "definitions": program.definitions,
    });
    if polylogue_query::parse(ask.text)?.stages.is_empty()
        && let Ok(m) = polylogue_query::compile(&format!("{} | matches", ask.text), &opts)
    {
        validate(reader, &m)?;
        let mut matches = with_similarity(reader, &m, ask.near, || {
            Ok(reader.query(&m.sql, &values(&m.params))?)
        })?;
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
    let cursor = ask.cursor.map(Cursor::decode).transpose()?;
    let seed = cursor
        .as_ref()
        .map_or_else(|| uuid::Uuid::now_v7().as_u128() as u64, |c| c.seed);
    let opts = options(
        reader,
        &ask,
        cursor.as_ref().map_or(ask.now, |c| c.now),
        seed,
    )?;
    let mut program = polylogue_query::compile(ask.text, &opts)?;
    validate(reader, &program)?;
    prepare_measure(&mut program, ask.page_size)?;
    let question = page::question(&program, &opts, ask.why)?;
    let after = cursor
        .as_ref()
        .map(|cursor| cursor.check(&program, &question).map(<[Value]>::to_vec))
        .transpose()?;
    let day = opts.clock.day.clone();
    with_similarity(reader, &program, ask.near, || {
        answer(reader, &program, &day, after, seed, question, ask.why)
    })
}

/// The answer to a compiled program: its count and one page after `after`.
/// `day` is the logical day its times resolved in.
fn answer(
    reader: &Reader<'_>,
    program: &Program,
    day: &LogicalDay,
    after: Option<Vec<Value>>,
    seed: u64,
    question: String,
    why_asked: bool,
) -> Result<Value, Failure> {
    let params = values(&program.params);
    let read_at = read_at(reader)?;
    if let Some(m) = &program.measure {
        polylogue_store::read::limit::ReadLimit::set_stage("measure");
        let (definition, _) = super::measure::find_checked(&m.what)?;
        if let Some(paging) = definition.paging {
            let (rows, count, next, _) = read_page(reader, program, &after, seed, question)?;
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
            let data = match paging {
                super::measure::Paging::Facets => {
                    let selected =
                        reader.query(&format!("SELECT _pk FROM ({})", m.selection_sql), &params)?;
                    let population = reader.query(&m.population_sql, &params)?;
                    let scope = polylogue_store::measure::Selection {
                        sessions: Some(selected.iter().filter_map(|r| r["_pk"].as_i64()).collect()),
                        population: Some(
                            population
                                .iter()
                                .filter_map(|r| r["_pk"].as_i64())
                                .collect(),
                        ),
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
                        program.now,
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
                    let selected =
                        reader.query(&format!("SELECT _pk FROM ({})", m.selection_sql), &params)?;
                    let all = polylogue_store::measure::Selection {
                        sessions: Some(selected.iter().filter_map(|r| r["_pk"].as_i64()).collect()),
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
            let rows = reader.query(&format!("SELECT _pk FROM ({})", program.sql), &params)?;
            Some(rows.iter().filter_map(|r| r["_pk"].as_i64()).collect())
        };
        let population = if definition.name == "facets" {
            Some(
                reader
                    .query(&m.population_sql, &params)?
                    .iter()
                    .filter_map(|r| r["_pk"].as_i64())
                    .collect(),
            )
        } else {
            None
        };
        let data = super::measure::run(reader, &m.what, sessions, program.now, day, population)?;
        return Ok(
            json!({ "outcome": "measure", "measure": m.what, "data": data, "read_at": read_at }),
        );
    }
    let (mut rows, count, next, without) = read_page(reader, program, &after, seed, question)?;

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
        "resolved": {
            "now": polylogue_model::Timestamp::from_unix_ms(program.now),
            "zone": program.zone,
            "day_start_minutes": program.day_start_minutes,
        },
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
    match definition.paging {
        Some(super::measure::Paging::Sessions) => {
            program.sql = polylogue_store::measure::comparison::page_query(&program.sql);
        }
        Some(super::measure::Paging::Facets) => {
            program.counts_near = false;
            program.row_count_sql = None;
            program.row_membership_sql = None;
            program.page_size = page_size.unwrap_or(100);
            program.sql = polylogue_store::measure::facets::query(
                &m.selection_sql,
                &m.population_sql,
                options,
                &program.zone,
                program.day_start_minutes,
            )?;
            program.order = vec![
                polylogue_query::OrderKey {
                    column: "family".into(),
                    descending: false,
                },
                polylogue_query::OrderKey {
                    column: "value".into(),
                    descending: false,
                },
            ];
        }
        None => {}
    }
    Ok(())
}

/// Row selection for ordinary queries and paged measures uses the same SQL and cursor.
fn read_page(
    reader: &Reader<'_>,
    program: &Program,
    after: &Option<Vec<Value>>,
    seed: u64,
    question: String,
) -> Result<(Vec<Value>, u64, Option<String>, u64), Failure> {
    let params = values(&program.params);
    polylogue_store::read::limit::ReadLimit::set_stage("count");
    let (count, without) = reader.query_counts(&program.count_sql(), &params)?;
    let page_size = program.page_size.max(1);
    let mut page_params = params;
    if let Some(after) = after {
        page_params.extend(after.iter().map(key_value));
    }
    // One row past the page says whether another page follows.
    page_params.push(SqlValue::Int(
        i64::try_from(page_size.saturating_add(1)).unwrap_or(i64::MAX),
    ));
    polylogue_store::read::limit::ReadLimit::set_stage("page");
    let mut rows = reader.query(&program.page_sql(after.is_some()), &page_params)?;
    let more = rows.len() as u64 > page_size;
    rows.truncate(usize::try_from(page_size).unwrap_or(usize::MAX));
    let next = match rows.last() {
        Some(last) if more => Some(Cursor::after_row(program, last, seed, question).encode()?),
        _ => None,
    };

    Ok((rows, count, next, without))
}

/// What one read transaction read: the index generation and the
/// counters, scoped to the current index.
fn read_at(reader: &Reader<'_>) -> Result<Value, Failure> {
    let position = reader.change_position()?;
    Ok(
        json!({ "generation": position.generation, "index": position.index, "archive": position.archive, "cursor": position.to_string() }),
    )
}

/// An order-key value from a cursor, bound as the type it was read as.
fn key_value(value: &Value) -> SqlValue {
    match value {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Int(i64::from(*b)),
        Value::Number(n) => n.as_i64().map_or_else(
            || SqlValue::Real(n.as_f64().unwrap_or_default()),
            SqlValue::Int,
        ),
        Value::String(s) => SqlValue::Text(s.clone()),
        other => SqlValue::Text(other.to_string()),
    }
}

pub(crate) fn values(params: &[Param]) -> Vec<SqlValue> {
    params
        .iter()
        .map(|p| match p {
            Param::Int(n) => SqlValue::Int(*n),
            Param::Real(f) => SqlValue::Real(*f),
            Param::Text(s) => SqlValue::Text(s.clone()),
        })
        .collect()
}

/// Every execution path validates its compiled references before reading rows.
fn validate(reader: &Reader<'_>, program: &Program) -> Result<(), Failure> {
    let position = reader.change_position()?;
    for expected in &program.change_generations {
        if expected != &position.generation {
            return Err(Failure {
                kind: "resync_required".into(),
                message: "the index generation changed; read the current state again".into(),
                detail: Some(
                    json!({"generation": position.generation, "reason": "generation_changed"}),
                ),
            });
        }
    }
    check_ids(reader, program)
}

/// A pasted ID is a reference only when it names exactly one session.
fn check_ids(reader: &Reader<'_>, program: &Program) -> Result<(), Failure> {
    for check in &program.id_checks {
        let found = reader.query(&check.sql, &values(&check.params))?;
        let ids: Vec<String> = found
            .iter()
            .filter_map(|r| r["id"].as_str().map(str::to_owned))
            .collect();
        match ids.len() {
            1 => {}
            0 => {
                return Err(Failure {
                    kind: "not_found".into(),
                    message: format!("no session or message has an ID starting {:?}", check.word),
                    detail: Some(json!({ "word": check.word })),
                });
            }
            _ => {
                return Err(Failure {
                    kind: "ambiguous".into(),
                    message: format!("{:?} starts several IDs: {}", check.word, ids.join(", ")),
                    detail: Some(json!({ "word": check.word, "ids": ids })),
                });
            }
        }
    }
    Ok(())
}

/// One snippet per row when text terms ranked the rows.
fn snippets(reader: &Reader<'_>, program: &Program, rows: &mut [Value]) -> Result<(), Failure> {
    if !program.ranked {
        return Ok(());
    }
    let fts: Vec<String> = program
        .text_terms
        .iter()
        .filter_map(|t| polylogue_model::text::fts_phrase(t))
        .collect();
    if fts.is_empty() {
        return Ok(());
    }
    let fts = fts.join(" AND ");
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
    for (index, snippet) in
        indices
            .into_iter()
            .zip(reader.snippets(&fts, &scopes, &program.text_terms)?)
    {
        if let Some(snippet) = snippet {
            rows[index]["snippet"] = serde_json::to_value(snippet).unwrap_or(Value::Null);
        }
    }
    Ok(())
}

fn attach(reader: &Reader<'_>, program: &Program, rows: &mut [Value]) -> Result<(), Failure> {
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
                    shape_value(col.kind, &mut c[col.name.as_str()])?;
                }
            }
            row[a.name.as_str()] = Value::Array(children);
        }
    }
    Ok(())
}

/// Times as RFC 3339, booleans as booleans. Columns starting with `_`
/// (the row key and order keys) are the reader's own.
fn present(program: &Program, row: &mut Value) -> Result<(), Failure> {
    if let Some(obj) = row.as_object_mut() {
        obj.retain(|name, _| !name.starts_with('_'));
    }
    for col in &program.columns {
        shape_value(col.kind, &mut row[col.name.as_str()])?;
    }
    Ok(())
}

fn shape_value(kind: ColKind, v: &mut Value) -> Result<(), Failure> {
    match kind {
        ColKind::Json => {
            if let Some(text) = v.as_str() {
                *v = serde_json::from_str(text).map_err(|_| Failure {
                    kind: "store".into(),
                    message: "invalid stored JSON value".into(),
                    detail: None,
                })?;
            }
        }
        ColKind::Ids => {
            if v.is_null() {
                return Ok(());
            }
            let selection: Value = if let Some(text) = v.as_str() {
                serde_json::from_str(text).map_err(|e| Failure {
                    kind: "store".into(),
                    message: format!("invalid stored selection: {e}"),
                    detail: None,
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
                return Err(Failure {
                    kind: "store".into(),
                    message: "a stored selection must be an array of IDs".into(),
                    detail: None,
                });
            }
            *v = selection;
        }
        ColKind::Time => {
            if let Some(ms) = v.as_i64() {
                *v = polylogue_model::Timestamp::from_unix_ms(ms)
                    .map_or(Value::Null, |t| Value::String(t.to_string()));
            }
        }
        ColKind::Bool => {
            if let Some(n) = v.as_i64() {
                *v = Value::Bool(n != 0);
            }
        }
        _ => {}
    }
    Ok(())
}

/// Each top-level condition counted alone against the rows the defaults
/// keep: `repo:other → 0 of 5`.
fn why(reader: &Reader<'_>, program: &Program, params: &[SqlValue]) -> Result<Value, Failure> {
    let count = |sql: &str| -> Result<u64, Failure> {
        let r = reader.query(sql, params)?;
        Ok(r.first().and_then(|r| r["count(*)"].as_u64()).unwrap_or(0))
    };
    let of = count(&program.why_of)?;
    let mut out = vec![];
    for w in &program.why {
        out.push(json!({ "clause": w.clause, "matches": count(&w.sql)?, "of": of }));
    }
    Ok(Value::Array(out))
}

/// Session keys for usage and its response drill-down, from the same compiler
/// path as a session measure. Presentation limits never limit the population.
pub(crate) fn usage_selection(
    reader: &Reader<'_>,
    text: &str,
    now: i64,
    embeddings: bool,
    near: Option<&NearFn<'_>>,
) -> Result<(polylogue_store::measure::Selection, String), Failure> {
    let text = format!("{text} | measure usage");
    let ask = Ask {
        text: &text,
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        now,
        day: None,
        embeddings,
        near,
        fields: &[],
    };
    let opts = options(reader, &ask, now, 0)?;
    let program = polylogue_query::compile(&text, &opts)?;
    validate(reader, &program)?;
    let question = page::question(&program, &opts, false)?;
    let m = program.measure.as_ref().ok_or_else(|| Failure {
        kind: "bad_request".into(),
        message: "cost requires a sessions query".into(),
        detail: None,
    })?;
    let select = || -> Result<Option<Vec<i64>>, Failure> {
        if m.everything {
            Ok(None)
        } else {
            Ok(Some(
                reader
                    .query(
                        &format!("SELECT _pk FROM ({})", program.sql),
                        &values(&program.params),
                    )?
                    .iter()
                    .filter_map(|r| r["_pk"].as_i64())
                    .collect(),
            ))
        }
    };
    let sessions = with_similarity(reader, &program, near, select)?;
    Ok((
        polylogue_store::measure::Selection {
            sessions,
            ..Default::default()
        },
        question,
    ))
}

/// Keep declared column metadata and source values as the compiled answer supplies them.
pub(crate) fn response(data: Value) -> super::Response {
    super::Response::Ok { data }
}

#[cfg(test)]
mod tests;

/// Compile with the same archive settings and calling session as execution.
pub fn explain(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    let options = options(reader, &ask, ask.now, uuid::Uuid::now_v7().as_u128() as u64)?;
    Ok(
        json!({"subject": "query", "explanation": polylogue_query::explain::query(ask.text, &options)?}),
    )
}

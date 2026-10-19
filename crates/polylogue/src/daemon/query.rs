//! Running a query (architecture 7; spec query.md, Results): compile it,
//! check pasted IDs, count the whole answer, read one page, cut snippets
//! for text matches, fill attached rows, and explain an empty result when
//! asked. Every step reads through one reader, so the count, the page and
//! the attachments agree.

use base64::Engine as _;
use polylogue_model::LogicalDay;
use polylogue_query::{ColKind, Options, Param, Program, QueryError, Shape};
use polylogue_store::search::Within;
use polylogue_store::{Reader, SqlValue};
use serde::{Deserialize, Serialize};
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
            kind: "store".into(),
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

/// What a page carries forward: where the next page starts, and the
/// instant the query's times were resolved against, so every page answers
/// the same question.
#[derive(Debug, Serialize, Deserialize)]
struct Cursor {
    offset: u64,
    now: i64,
    /// BLAKE3 of the query text: a cursor continues only its own query.
    query: String,
}

fn encode(c: &Cursor) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(c).unwrap_or_default())
}

fn decode(text: &str) -> Option<Cursor> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(text)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn query_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex()[..16].to_owned()
}

/// What a request asks besides the text.
pub struct Ask<'a> {
    pub text: &'a str,
    pub cursor: Option<&'a str>,
    pub page_size: Option<u64>,
    pub why: bool,
    pub session: Option<String>,
    /// Unix ms; the reader's `now` unless a cursor fixes it.
    pub now: i64,
    pub day: LogicalDay,
    pub embeddings: bool,
    /// For `near:`: the nearest texts to a text, as the JSON the query
    /// binds.
    pub near: Option<&'a NearFn<'a>>,
}

/// Looks up the nearest texts to a text, as the JSON the query binds.
pub type NearFn<'a> = dyn Fn(&str) -> Result<String, String> + 'a;

/// The current saved queries by name (`saved_query` notes, newest
/// revision, not withdrawn), for `query:NAME`.
fn saved_queries(reader: &Reader<'_>) -> std::collections::BTreeMap<String, String> {
    let sql = format!(
        "SELECT n.key AS k, n.value AS v FROM a.note n WHERE n.kind = 'saved_query' \
         AND n.key IS NOT NULL AND n.value IS NOT NULL AND {} AND NOT {}",
        polylogue_query::fields::current_note("n"),
        polylogue_query::fields::withdrawn_note("n"),
    );
    reader
        .query(&sql, &[])
        .unwrap_or_default()
        .iter()
        .filter_map(|r| Some((r["k"].as_str()?.to_owned(), r["v"].as_str()?.to_owned())))
        .collect()
}

/// Kept results by ID, with the shape of their stored rows, for
/// `result:<id>`.
fn kept_results(reader: &Reader<'_>) -> std::collections::BTreeMap<String, polylogue_query::Kept> {
    reader
        .query(
            "SELECT id, json_extract(rows_json, '$.unit') AS unit, \
             json_extract(rows_json, '$.columns') AS columns, \
             json_extract(rows_json, '$.match_columns') AS match_columns FROM a.kept_result",
            &[],
        )
        .unwrap_or_default()
        .iter()
        .filter_map(|r| {
            let parse = |k: &str| -> Vec<polylogue_query::Column> {
                r[k].as_str()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default()
            };
            Some((
                r["id"].as_str()?.to_owned(),
                polylogue_query::Kept {
                    unit: r["unit"].as_str().unwrap_or("sessions").to_owned(),
                    columns: parse("columns"),
                    match_columns: parse("match_columns"),
                },
            ))
        })
        .collect()
}

/// Runs a query in full for keeping: its rows, and, when it has one
/// top-level exists or seq and no stages, its match rows too. The values
/// are what was shown; a kept result is read back, never re-run.
pub fn keep(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    let opts = Options {
        clock: polylogue_query::Clock::new(ask.now, ask.day.clone()),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|o| o.name.to_owned())
            .collect(),
        page_size: None,
        embeddings: false,
        self_session: ask.session,
        saved: saved_queries(reader),
        kept: kept_results(reader),
    };
    let program = polylogue_query::compile(ask.text, &opts)?;
    check_ids(reader, &program)?;
    let mut rows = reader.query(&program.sql, &values(&program.params))?;
    for row in &mut rows {
        present(&program, row);
    }
    let mut kept = json!({
        "unit": if program.shape == Shape::Matches { "matches" } else { program.unit.name() },
        "shape": program.shape,
        "columns": program.columns,
        "rows": rows,
        "query": ask.text,
        "now": program.now,
    });
    if !ask.text.contains('|')
        && let Ok(m) = polylogue_query::compile(&format!("{} | matches", ask.text), &opts)
    {
        let mut matches = reader.query(&m.sql, &values(&m.params))?;
        for row in &mut matches {
            present(&m, row);
        }
        kept["match_columns"] = json!(m.columns);
        kept["matches"] = json!(matches);
    }
    Ok(kept)
}

pub fn run(reader: &Reader<'_>, ask: Ask<'_>) -> Result<Value, Failure> {
    let cursor = match ask.cursor {
        Some(c) => {
            let c = decode(c).ok_or_else(|| Failure {
                kind: "bad_request".into(),
                message: "the cursor is not one this daemon issued".into(),
                detail: None,
            })?;
            if c.query != query_hash(ask.text) {
                return Err(Failure {
                    kind: "bad_request".into(),
                    message: "the cursor belongs to another query".into(),
                    detail: None,
                });
            }
            Some(c)
        }
        None => None,
    };
    let opts = Options {
        clock: polylogue_query::Clock::new(cursor.as_ref().map_or(ask.now, |c| c.now), ask.day),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|o| o.name.to_owned())
            .collect(),
        page_size: ask.page_size,
        embeddings: ask.embeddings,
        self_session: ask.session,
        saved: saved_queries(reader),
        kept: kept_results(reader),
    };
    let mut program = polylogue_query::compile(ask.text, &opts)?;
    if let (Some(near), Some(f)) = (program.near.clone(), ask.near) {
        let hits = f(&near.text).map_err(|message| Failure {
            kind: "embed".into(),
            message,
            detail: None,
        })?;
        if let Some(p) = program.params.get_mut(near.param - 1) {
            *p = Param::Text(hits);
        }
    }
    check_ids(reader, &program)?;
    let params = values(&program.params);
    let count = reader.query(&program.count_sql(), &params)?;
    let count = count
        .first()
        .and_then(|r| r["count(*)"].as_u64())
        .unwrap_or(0);
    let offset = cursor.as_ref().map_or(0, |c| c.offset);
    let page_size = program.page_size.max(1);
    let mut page_params = params.clone();
    page_params.push(SqlValue::Int(i64::try_from(page_size).unwrap_or(i64::MAX)));
    page_params.push(SqlValue::Int(i64::try_from(offset).unwrap_or(i64::MAX)));
    let mut rows = reader.query(&program.page_sql(), &page_params)?;

    if program.shape == Shape::Rows {
        snippets(reader, &program, &mut rows)?;
        attach(reader, &program, &mut rows)?;
    }
    for row in &mut rows {
        present(&program, row);
    }
    let next = (offset + page_size < count).then(|| {
        encode(&Cursor {
            offset: offset + page_size,
            now: program.now,
            query: query_hash(ask.text),
        })
    });
    let outcome = if !program.degraded.is_empty() {
        "degraded"
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
        "notes": program.notes,
        "degraded": program.degraded,
        "resolved": {
            "now": polylogue_model::Timestamp::from_unix_ms(program.now),
            "zone": program.zone,
            "day_start_minutes": program.day_start_minutes,
        },
    });
    if ask.why && count == 0 {
        out["why"] = why(reader, &program, &params)?;
    }
    Ok(out)
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
                    detail: Some(json!({ "word": check.word, "candidates": ids })),
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
    for row in rows {
        let Some(pk) = row["_pk"].as_i64() else {
            continue;
        };
        let within = match program.unit {
            polylogue_query::ast::Unit::Sessions => Within::Session(pk),
            polylogue_query::ast::Unit::Messages => Within::Message(pk),
            _ => continue,
        };
        if let Some(s) = reader.snippet(&fts, within, &program.text_terms)? {
            row["snippet"] = serde_json::to_value(s).unwrap_or(Value::Null);
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
                    shape_value(col.kind, &mut c[col.name.as_str()]);
                }
            }
            row[a.name.as_str()] = Value::Array(children);
        }
    }
    Ok(())
}

/// Times as RFC 3339, booleans as booleans; `_pk` is the reader's own.
fn present(program: &Program, row: &mut Value) {
    if let Some(obj) = row.as_object_mut() {
        obj.remove("_pk");
    }
    for col in &program.columns {
        shape_value(col.kind, &mut row[col.name.as_str()]);
    }
}

fn shape_value(kind: ColKind, v: &mut Value) {
    match kind {
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod morrow;

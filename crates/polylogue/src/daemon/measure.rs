//! `Measure` requests: token lanes and cost, the subscription outlook and
//! the analyses, over the whole archive or the sessions a query selects.

use super::Response;
use polylogue_model::LogicalDay;
use polylogue_query::ast::Unit;
use polylogue_query::{Options, Shape};
use polylogue_store::Reader;
use polylogue_store::measure::{self, Scope, analysis, outlook};
use serde::Serialize;

/// The measurements a request can name, for errors and help.
pub const WHAT: &str = "usage, cost, sessions, quota, outcomes, survival, survival-turns, \
    daily:<tokens|api_cost|sessions|prompts|tool_errors>, process, rhythm";

pub fn handle(reader: &Reader<'_>, what: &str, query: Option<&str>, now: i64) -> Response {
    let day = LogicalDay::system();
    let scope = match query.filter(|q| !q.trim().is_empty()) {
        None => Scope::default(),
        Some(q) => match sessions(reader, q, now, &day) {
            Ok(pks) => Scope {
                sessions: Some(pks),
                ..Scope::default()
            },
            Err(r) => return r,
        },
    };
    let out = match what {
        "usage" | "cost" => json(measure::usage(reader, &scope)),
        "sessions" => json(measure::sessions(reader, &scope)),
        "quota" | "outlook" => {
            if query.is_some() {
                return error(
                    "bad_request",
                    "the subscription outlook covers each tool's whole windows; it takes no query",
                );
            }
            json(outlook::outlook(reader, now))
        }
        "outcomes" => json(analysis::outcomes(reader, &scope)),
        "survival" => json(analysis::first_failure(reader, &scope, false)),
        "survival-turns" => json(analysis::first_failure(reader, &scope, true)),
        "process" => json(analysis::process(reader, &scope, 20)),
        "rhythm" => json(analysis::rhythm(reader, &scope, &day)),
        w => match w.strip_prefix("daily:") {
            Some(metric) => json(analysis::daily(reader, &scope, metric, &day)),
            None => return error("bad_request", format!("no measurement `{w}`: {WHAT}")),
        },
    };
    match out {
        Ok(data) => Response::Ok { data },
        Err(e) => error("store", e),
    }
}

fn json<T: Serialize>(r: polylogue_store::Result<T>) -> Result<serde_json::Value, String> {
    r.map_err(|e| e.to_string())
        .and_then(|v| serde_json::to_value(v).map_err(|e| e.to_string()))
}

fn error(kind: &str, message: impl ToString) -> Response {
    Response::Error {
        kind: kind.into(),
        message: message.to_string(),
        detail: None,
    }
}

/// The session keys a sessions query selects.
fn sessions(
    reader: &Reader<'_>,
    text: &str,
    now: i64,
    day: &LogicalDay,
) -> Result<Vec<i64>, Response> {
    let opts = Options {
        clock: polylogue_query::Clock::new(now, day.clone()),
        origins: polylogue_sources::origin::ORIGINS
            .iter()
            .map(|o| o.name.to_owned())
            .collect(),
        page_size: None,
        embeddings: false,
        self_session: None,
        saved: Default::default(),
    };
    let program = polylogue_query::compile(text, &opts).map_err(|e| error(e.kind(), e))?;
    if program.unit != Unit::Sessions || program.shape != Shape::Rows {
        return Err(error(
            "bad_request",
            "a measurement's query selects sessions (e.g. `repo:polylogue since:7d`)",
        ));
    }
    let params = super::query::values(&program.params);
    let rows = reader
        .query(&format!("SELECT _pk FROM ({})", program.sql), &params)
        .map_err(|e| error("store", e))?;
    Ok(rows.iter().filter_map(|r| r["_pk"].as_i64()).collect())
}

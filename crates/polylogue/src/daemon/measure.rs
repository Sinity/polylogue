//! `| measure WHAT`: tokens by kind and cost, subscription quota and the
//! analyses, over the sessions a query selects (every session when it names
//! no conditions). The query program chooses the sessions; this runs the
//! measurement over them.

use super::query::Failure;
use polylogue_model::LogicalDay;
use polylogue_store::Reader;
use polylogue_store::measure::{self, Scope, analysis, quota};
use serde::Serialize;
use serde_json::Value;

/// The measurements a query can name, for errors and help.
pub const WHAT: &str = "usage, cost, sessions, quota, outcomes, failing, drift, survival, \
    survival-turns, unanswered, \
    daily:<tokens|api_cost|sessions|prompts|tool_errors>, process, rhythm";

pub fn run(
    reader: &Reader<'_>,
    what: &str,
    sessions: Option<Vec<i64>>,
    now: i64,
) -> Result<Value, Failure> {
    let day = LogicalDay::system();
    let whole = |what: &str| -> Result<(), Failure> {
        match sessions {
            Some(_) => Err(bad_request(format!(
                "{what} covers every session; measure it without conditions (`sessions | measure {what}`)"
            ))),
            None => Ok(()),
        }
    };
    let scope = Scope {
        sessions: sessions.clone(),
        ..Scope::default()
    };
    match what {
        "usage" | "cost" => json(measure::usage(reader, &scope)),
        "sessions" => json(measure::sessions(reader, &scope)),
        "quota" => {
            whole("quota")?;
            let known: Vec<_> = polylogue_sources::ORIGINS
                .iter()
                .filter_map(|o| Some((o.name, o.quota?)))
                .collect();
            json(quota::quota(reader, now, &known))
        }
        "outcomes" => json(analysis::outcomes(reader, &scope)),
        "survival" => json(analysis::first_failure(reader, &scope, false)),
        "survival-turns" => json(analysis::first_failure(reader, &scope, true)),
        "failing" => json(analysis::failing_commands(reader, &scope)),
        "drift" => {
            whole("format drift")?;
            json(analysis::drift(reader))
        }
        "unanswered" => json(analysis::unanswered_end(reader, &scope, now)),
        "process" => json(analysis::process(reader, &scope, 20)),
        "rhythm" => json(analysis::rhythm(reader, &scope, &day)),
        w => match w.strip_prefix("daily:") {
            Some(metric) => json(analysis::daily(reader, &scope, metric, &day)),
            None => Err(bad_request(format!("no measurement `{w}`: {WHAT}"))),
        },
    }
}

fn json<T: Serialize>(r: polylogue_store::Result<T>) -> Result<Value, Failure> {
    let v = r?;
    serde_json::to_value(v).map_err(|e| Failure {
        kind: "store".into(),
        message: e.to_string(),
        detail: None,
    })
}

fn bad_request(message: String) -> Failure {
    Failure {
        kind: "bad_request".into(),
        message,
        detail: None,
    }
}

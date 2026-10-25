//! `| measure WHAT`: tokens by kind and cost, subscription quota and the
//! analyses, over the sessions a query selects (every session when it names
//! no conditions). The query program chooses the sessions; this runs the
//! measurement over them. [`MEASURES`] is the one list: the `measure *`
//! commands, the `query` help, the plain-text view and the error for an
//! unknown name all read it.

use super::query::Failure;
use polylogue_model::LogicalDay;
use polylogue_render::measures as text;
use polylogue_store::Reader;
use polylogue_store::measure::{self, Scope, daily, drift, quota, survival, tool_calls};
use serde::Serialize;
use serde_json::Value;

pub struct Measure {
    /// The name after `| measure` and `polylogue measure`.
    pub name: &'static str,
    pub help: &'static str,
    /// Covers every session: refused after conditions.
    pub whole_archive: bool,
    /// The series it takes after `:` (`daily:tokens`); empty when none.
    pub metrics: &'static [&'static str],
    pub run: fn(&Reader<'_>, &Run<'_>) -> polylogue_store::Result<Value>,
    /// The answer as text for a person.
    pub text: fn(&Value) -> String,
}

/// What a measurement runs over.
pub struct Run<'a> {
    pub scope: Scope,
    pub now: i64,
    pub day: &'a LogicalDay,
    /// The series, for a measure that takes one.
    pub metric: &'a str,
}

fn json<T: Serialize>(r: polylogue_store::Result<T>) -> polylogue_store::Result<Value> {
    Ok(serde_json::to_value(r?).unwrap_or_default())
}

pub static MEASURES: &[Measure] = &[
    Measure {
        name: "usage",
        help: "Tokens by kind and cost by model, over a sessions query or everything",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(measure::usage(r, &x.scope)),
        text: text::usage,
    },
    Measure {
        name: "sessions",
        help: "Each session's tokens and cost, physical (its own responses) and logical (what it shows, inherited responses included)",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(measure::sessions(r, &x.scope)),
        text: text::json,
    },
    Measure {
        name: "quota",
        help: "The 5-hour and weekly subscription windows: burn and forecast",
        whole_archive: true,
        metrics: &[],
        run: |r, x| {
            let known: Vec<_> = polylogue_sources::ORIGINS
                .iter()
                .filter_map(|origin| Some((origin.name, origin.quota?)))
                .collect();
            json(quota::quota(r, x.now, &known))
        },
        text: text::quota,
    },
    Measure {
        name: "outcomes",
        help: "Tool outcomes: ok, error, no result and unknown, never one rate",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(tool_calls::outcomes(r, &x.scope)),
        text: text::json,
    },
    Measure {
        name: "failing",
        help: "Per repository, the commands whose most recent run failed",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(tool_calls::failing_commands(r, &x.scope)),
        text: text::failing,
    },
    Measure {
        name: "drift",
        help: "Record types and fields each format's parser did not know, newest first: where the tools' formats moved",
        whole_archive: true,
        metrics: &[],
        run: |r, _| json(drift::drift(r)),
        text: text::drift,
    },
    Measure {
        name: "unanswered",
        help: "Time to a session ending on an unanswered prompt (Kaplan-Meier)",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(survival::unanswered_end(r, &x.scope, x.now)),
        text: text::json,
    },
    Measure {
        name: "survival",
        help: "Time to a session's first failed tool call, in minutes (Kaplan-Meier)",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(survival::first_failure(r, &x.scope, false)),
        text: text::json,
    },
    Measure {
        name: "survival-turns",
        help: "Time to a session's first failed tool call, in human prompts (Kaplan-Meier)",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(survival::first_failure(r, &x.scope, true)),
        text: text::json,
    },
    Measure {
        name: "daily",
        help: "A daily series with its 7-day baseline and level changes",
        whole_archive: false,
        metrics: daily::DAILY_METRICS,
        run: |r, x| json(daily::daily(r, &x.scope, x.metric, x.day)),
        text: text::json,
    },
    Measure {
        name: "process",
        help: "Which tool call follows which, and what follows a failure",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(tool_calls::process(r, &x.scope, 20)),
        text: text::json,
    },
    Measure {
        name: "rhythm",
        help: "Prompts by weekday and hour",
        whole_archive: false,
        metrics: &[],
        run: |r, x| json(daily::rhythm(r, &x.scope, x.day)),
        text: text::json,
    },
];

/// The measure `what` names (`daily:tokens` names `daily`), and its series.
pub fn find(what: &str) -> Option<(&'static Measure, &str)> {
    let (name, metric) = what.split_once(':').unwrap_or((what, ""));
    MEASURES
        .iter()
        .find(|m| m.name == name)
        .map(|m| (m, metric))
}

/// Every measure as `| measure` takes it, for help and errors.
pub fn names() -> String {
    MEASURES
        .iter()
        .map(|m| match m.metrics {
            [] => m.name.to_owned(),
            metrics => format!("{}:<{}>", m.name, metrics.join("|")),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn run(
    reader: &Reader<'_>,
    what: &str,
    sessions: Option<Vec<i64>>,
    now: i64,
) -> Result<Value, Failure> {
    let Some((m, metric)) = find(what) else {
        return Err(bad_request(format!("no measurement `{what}`: {}", names())));
    };
    if m.whole_archive && sessions.is_some() {
        return Err(bad_request(format!(
            "{0} covers every session; measure it without conditions (`sessions | measure {0}`)",
            m.name
        )));
    }
    let day = LogicalDay::system();
    let run = Run {
        scope: Scope {
            sessions,
            ..Scope::default()
        },
        now,
        day: &day,
        metric,
    };
    Ok((m.run)(reader, &run)?)
}

fn bad_request(message: String) -> Failure {
    Failure {
        kind: "bad_request".into(),
        message,
        detail: None,
    }
}

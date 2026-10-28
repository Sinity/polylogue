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
use polylogue_store::measure::{self, Selection, daily, drift, quota, survival, tool_calls};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy)]
pub enum Paging {
    Sessions,
    Facets,
}

pub struct Measure {
    /// The name after `| measure` and `polylogue measure`.
    pub name: &'static str,
    pub help: &'static str,
    /// Covers every session: refused after conditions.
    pub whole_archive: bool,
    pub paging: Option<Paging>,
    /// Optional colon-separated arguments, also exposed as --option.
    pub argument: Option<&'static str>,
    /// Closed series choices, for measures such as daily:tokens.
    pub metrics: &'static [&'static str],
    pub run: fn(&Reader<'_>, &Run<'_>) -> polylogue_store::Result<Value>,
    /// The answer as text for a person.
    pub text: fn(&Value) -> String,
}

/// What a measurement runs over.
pub struct Run<'a> {
    pub selection: Selection,
    pub now: i64,
    pub day: &'a LogicalDay,
    /// The measure's series or named options.
    pub metric: &'a str,
}

fn json<T: Serialize>(r: polylogue_store::Result<T>) -> polylogue_store::Result<Value> {
    Ok(serde_json::to_value(r?)?)
}

pub static MEASURES: &[Measure] = &[
    Measure {
        name: "facets",
        help: "Paged read-time session counts by family and value, with global counts, IDF and lift; basic, deep or one family",
        whole_archive: false,
        paging: Some(Paging::Facets),
        argument: Some("family"),
        metrics: &[],
        run: |_, _| {
            Err(polylogue_store::StoreError::Invalid(
                "facets requires query paging".into(),
            ))
        },
        text: text::json,
    },
    Measure {
        name: "outlook",
        help: "Billing estimate from dated plan.<provider> settings: provider:method (linear, trailing-7d-mean); money in nano-USD",
        whole_archive: false,
        paging: None,
        argument: Some("provider:method"),
        metrics: &[],
        run: |r, x| {
            json(measure::outlook::outlook(
                r,
                &x.selection,
                x.now,
                x.metric,
                &polylogue_store::user_writes::machine_id(),
            ))
        },
        text: text::outlook,
    },
    Measure {
        name: "time",
        help: "Recorded session clocks and wall-time lanes; clustering gap 5 minutes; query cursor pages sessions",
        whole_archive: false,
        paging: Some(Paging::Sessions),
        argument: None,
        metrics: &[],
        run: |r, x| json(measure::time::times(r, &x.selection)),
        text: text::json,
    },
    Measure {
        name: "compare",
        help: "Recorded session fields side by side, with varying fields and unknown counts; optional comma-separated fields",
        whole_archive: false,
        paging: Some(Paging::Sessions),
        argument: Some("fields"),
        metrics: &[],
        run: |r, x| json(measure::comparison::compare(r, &x.selection, x.metric)),
        text: text::json,
    },
    Measure {
        name: "usage",
        help: "Tokens by kind and cost by model, over a sessions query or everything",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(measure::usage(r, &x.selection)),
        text: text::usage,
    },
    Measure {
        name: "sessions",
        help: "Each session's tokens and cost, own (its own responses) and with_parents (what it shows, inherited responses included)",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(measure::sessions(r, &x.selection)),
        text: text::json,
    },
    Measure {
        name: "quota",
        help: "The 5-hour and weekly subscription windows: burn and forecast",
        whole_archive: true,
        paging: None,
        argument: None,
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
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::outcomes(r, &x.selection)),
        text: text::json,
    },
    Measure {
        name: "failing",
        help: "Per repository, the commands whose most recent run failed",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::failing_commands(r, &x.selection)),
        text: text::failing,
    },
    Measure {
        name: "drift",
        help: "Record types and fields each format's parser did not know, newest first: where the tools' formats moved",
        whole_archive: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, _| json(drift::drift(r)),
        text: text::drift,
    },
    Measure {
        name: "unanswered",
        help: "Time to a session ending on an unanswered prompt after 6 quiet hours (Kaplan-Meier)",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(survival::unanswered_end(r, &x.selection, x.now)),
        text: text::json,
    },
    Measure {
        name: "survival",
        help: "Time to a session's first failed tool call, in minutes (Kaplan-Meier)",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(survival::first_failure(r, &x.selection, false)),
        text: text::json,
    },
    Measure {
        name: "survival-turns",
        help: "Time to a session's first failed tool call, in human prompts (Kaplan-Meier)",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(survival::first_failure(r, &x.selection, true)),
        text: text::json,
    },
    Measure {
        name: "daily",
        help: "A daily series with its 7-day baseline and level changes",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: daily::DAILY_METRICS,
        run: |r, x| json(daily::daily(r, &x.selection, x.metric, x.day)),
        text: text::json,
    },
    Measure {
        name: "process",
        help: "Which tool call follows which, and what follows a failure",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::process(r, &x.selection, 20)),
        text: text::json,
    },
    Measure {
        name: "rhythm",
        help: "Prompts by weekday and hour",
        whole_archive: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(daily::rhythm(r, &x.selection, x.day)),
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
        .map(|m| match (m.argument, m.metrics) {
            (Some(argument), _) => format!("{}[:{}]", m.name, argument),
            (None, []) => m.name.to_owned(),
            (None, metrics) => format!("{}:<{}>", m.name, metrics.join("|")),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn run(
    reader: &Reader<'_>,
    what: &str,
    sessions: Option<Vec<i64>>,
    now: i64,
    day: &LogicalDay,
    population: Option<Vec<i64>>,
) -> Result<Value, Failure> {
    let (m, metric) = find_checked(what)?;
    if m.whole_archive && sessions.is_some() {
        return Err(bad_request(format!(
            "{0} covers every session; measure it without conditions (`sessions | measure {0}`)",
            m.name
        )));
    }
    let run = Run {
        selection: Selection {
            sessions,
            population,
            ..Selection::default()
        },
        now,
        day,
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

pub fn find_checked(what: &str) -> Result<(&'static Measure, &str), Failure> {
    let Some((m, metric)) = find(what) else {
        return Err(bad_request(format!("no measurement `{what}`: {}", names())));
    };
    if m.argument.is_none() && m.metrics.is_empty() && !metric.is_empty() {
        return Err(bad_request(format!("{} takes no arguments", m.name)));
    }
    Ok((m, metric))
}

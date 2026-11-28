//! `| measure WHAT`: tokens by kind and cost, subscription quota and the
//! analyses, over the sessions a query selects (every session when it names
//! no conditions). The query program chooses the sessions; this runs the
//! measurement over them. [`MEASURES`] is the one list: the `query` help,
//! the plain-text view and the error for an unknown name all read it.

use super::Failure;
use polylogue_render::measures as text;
use polylogue_store::Reader;
use polylogue_store::measure::{self, Selection, daily, quota, survival, tool_calls};
use query_language::time::DaySetting;
use serde::Serialize;
use serde_json::Value;

pub use polylogue_store::measure::Paging;

pub struct Measure {
    /// The name after `| measure` and `polylogue measure`.
    pub name: &'static str,
    pub help: &'static str,
    /// Covers every session: refused after conditions.
    pub whole_archive: bool,
    /// Counts messages: a selection bounded by `since:`/`until:` counts only
    /// the messages inside the bounds, and a session's figures include the
    /// whole logical session.
    pub messages: bool,
    pub paging: Option<Paging>,
    /// Optional colon-separated arguments, also exposed as --option.
    pub argument: Option<&'static str>,
    /// Closed series choices, for measures such as daily:tokens.
    pub metrics: &'static [&'static str],
    pub run: fn(&Reader<'_>, &Run<'_>) -> polylogue_store::Result<Value>,
    /// The answer as text for a person.
    pub text: fn(&Value, &jiff::tz::TimeZone) -> String,
}

/// Join pages produced by the measure's existing declared paging mode.
/// Global summary values come from the same pinned read on every page.
pub(crate) fn append_page(paging: Paging, all: &mut Value, mut page: Value) -> Result<(), Failure> {
    let field = match paging {
        Paging::Sessions => "sessions",
        Paging::Facets => "values",
        Paging::Responses => "responses",
    };
    let values = page
        .get_mut(field)
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .ok_or_else(|| Failure::internal("a measure page has no declared row array"))?;
    let extra = page
        .as_object()
        .ok_or_else(|| Failure::internal("a measure page has no object"))?;
    for (name, value) in extra {
        if name != field && all.get(name) != Some(value) {
            return Err(Failure::internal(
                "measure page summaries differ within one read",
            ));
        }
    }
    all.get_mut(field)
        .and_then(Value::as_array_mut)
        .ok_or_else(|| Failure::internal("the first measure page has no declared row array"))?
        .extend(values);
    Ok(())
}

/// What a measurement runs over.
pub struct Run<'a> {
    pub selection: Selection,
    pub now: i64,
    pub day: &'a DaySetting,
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
        messages: false,
        paging: Some(Paging::Facets),
        argument: Some("family"),
        metrics: &[],
        run: |_, _| {
            Err(polylogue_store::StoreError::internal(
                "facets runs through query paging",
            ))
        },
        text: |data, _| text::json(data),
    },
    Measure {
        name: "outlook",
        help: "Billing estimate from dated plan.<provider> settings: provider:method (linear, trailing-7d-mean); money in nano-USD",
        whole_archive: false,
        messages: false,
        paging: None,
        argument: Some("provider:method"),
        metrics: &[],
        run: |r, x| json(measure::outlook::outlook(r, &x.selection, x.now, x.metric)),
        text: |data, _| text::outlook(data),
    },
    Measure {
        name: "time",
        help: "Recorded session clocks and wall-time per party; clustering gap 5 minutes; query cursor pages sessions",
        whole_archive: false,
        messages: false,
        paging: Some(Paging::Sessions),
        argument: None,
        metrics: &[],
        run: |r, x| json(measure::time::times(r, &x.selection)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "compare",
        help: "Recorded session fields side by side, with varying fields and unknown counts; optional comma-separated fields",
        whole_archive: false,
        messages: false,
        paging: Some(Paging::Sessions),
        argument: Some("fields"),
        metrics: &[],
        run: |r, x| json(measure::comparison::compare(r, &x.selection, x.metric)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "usage",
        help: "Tokens by kind and cost by model, over a sessions query or everything",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(measure::usage(r, &x.selection)),
        text: |data, _| text::usage(data),
    },
    Measure {
        name: "responses",
        help: "The responses usage counts, one row each with time, model, tokens by kind and cost, in time order; an optional model, or unknown for none recorded",
        whole_archive: false,
        messages: true,
        paging: Some(Paging::Responses),
        argument: Some("model"),
        metrics: &[],
        run: |_, _| {
            Err(polylogue_store::StoreError::internal(
                "responses runs through query paging",
            ))
        },
        text: |data, zone| text::responses(data, zone),
    },
    Measure {
        name: "sessions",
        help: "Each session's tokens and cost, own (its own responses) and with_parents (what it shows, inherited responses included)",
        whole_archive: false,
        messages: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(measure::sessions(r, &x.selection)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "quota",
        help: "The 5-hour and weekly subscription windows: burn and forecast",
        whole_archive: true,
        messages: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(quota::quota(r, x.now, &quota_limits())),
        text: |data, _| text::quota(data),
    },
    Measure {
        name: "outcomes",
        help: "Tool outcomes: ok, error, no result and unknown, never one rate",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::outcomes(r, &x.selection)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "shell-stages",
        help: "Normalized stages extracted from submitted shell text, with occurrence, distinct call and distinct session counts",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::shell_stages(r, &x.selection)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "failing",
        help: "Per repository, the commands whose most recent run failed",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::failing_commands(r, &x.selection)),
        text: text::failing,
    },
    Measure {
        name: "unanswered",
        help: "Time to a session ending on an unanswered prompt after 6 quiet hours (Kaplan-Meier)",
        whole_archive: false,
        messages: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(survival::unanswered_end(r, &x.selection, x.now)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "survival",
        help: "Time to a session's first failed tool call, in minutes (Kaplan-Meier)",
        whole_archive: false,
        messages: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(survival::first_failure(r, &x.selection, false)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "survival-turns",
        help: "Time to a session's first failed tool call, in human prompts (Kaplan-Meier)",
        whole_archive: false,
        messages: false,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(survival::first_failure(r, &x.selection, true)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "daily",
        help: "A daily series with its 7-day baseline and level changes",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: daily::DAILY_METRICS,
        run: |r, x| json(daily::daily(r, &x.selection, x.metric, x.day)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "process",
        help: "Which tool call follows which, and what follows a failure",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(tool_calls::process(r, &x.selection, 20)),
        text: |data, _| text::json(data),
    },
    Measure {
        name: "rhythm",
        help: "Prompts by weekday and hour",
        whole_archive: false,
        messages: true,
        paging: None,
        argument: None,
        metrics: &[],
        run: |r, x| json(daily::rhythm(r, &x.selection, x.day)),
        text: |data, _| text::json(data),
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

pub fn run(
    reader: &Reader<'_>,
    what: &str,
    sessions: Option<Vec<i64>>,
    window: (Option<i64>, Option<i64>),
    now: i64,
    day: &DaySetting,
    population: Option<Vec<i64>>,
) -> Result<Value, Failure> {
    let (m, metric) = find_checked(what)?;
    if m.whole_archive && sessions.is_some() {
        return Err(bad_request(format!(
            "{0} covers every session; measure it without conditions (`sessions | measure {0}`)",
            m.name
        )));
    }
    let (since, until) = if m.messages { window } else { (None, None) };
    let run = Run {
        selection: Selection {
            sessions,
            population,
            since,
            until,
            logical: m.messages,
            ..Selection::default()
        },
        now,
        day,
        metric,
    };
    let mut answer = (m.run)(reader, &run)?;
    if let Some(object) = answer.as_object_mut() {
        object
            .entry("zone")
            .or_insert_with(|| Value::from(day.zone_name()));
    }
    Ok(answer)
}

fn bad_request(message: String) -> Failure {
    Failure::bad_request(message)
}

pub fn find_checked(what: &str) -> Result<(&'static Measure, &str), Failure> {
    let Some((m, metric)) = find(what) else {
        let mut known = MEASURES
            .iter()
            .map(|measure| measure.name)
            .collect::<Vec<_>>();
        known.sort_unstable();
        return Err(Failure::refusal(polylogue_model::Reason::UnknownMeasure {
            value: what.to_owned(),
            known: known.into_iter().collect(),
        }));
    };
    if m.argument.is_none() && m.metrics.is_empty() && !metric.is_empty() {
        return Err(bad_request(format!("{} takes no arguments", m.name)));
    }
    Ok((m, metric))
}

/// The origins that run under a subscription quota, with the limits each
/// always has.
pub(crate) fn quota_limits() -> Vec<(&'static str, &'static [polylogue_model::QuotaLimit])> {
    polylogue_sources::ORIGINS
        .iter()
        .filter_map(|origin| Some((origin.name, origin.quota?)))
        .collect()
}

#[cfg(test)]
mod tests;

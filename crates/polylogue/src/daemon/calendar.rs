//! Shared logical-day composition for timeline, day and cost clients.
use super::{Failure, query};
use polylogue_model::context::RunContext;
use polylogue_reading::operation::Context;
use polylogue_store::{Reader, timeline};
use query_language::cursor::{
    Continuation, CursorContext, CursorSeal, OrderPosition, SealedCursor,
};
use query_language::identity::QuestionId;
use query_language::time::{Clock, DaySettingWire};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One page of records in time order, as the store reads them.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Activity {
    pub rows: Vec<Map<String, Value>>,
    /// The cursor of the next page, when there is one.
    pub next: Option<String>,
}

/// How many records one day of the shown month holds.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CalendarDay {
    pub date: String,
    pub records: u64,
}

/// A part of the answer the store could not read; the rest still answers.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StoreFailure {
    /// Always `store`.
    pub error: String,
    pub message: String,
}

impl StoreFailure {
    fn new(error: &impl ToString) -> Self {
        Self {
            error: "store".into(),
            message: error.to_string(),
        }
    }
}

/// A measurement in the answer, or why the store could not make it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Measured {
    Ready { data: Value },
    Failed(StoreFailure),
}

/// The quota measurement, or the failure the measure returned.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Quota {
    Ready {
        data: Value,
    },
    Failed {
        error: command_surface::FailureKind,
        message: String,
    },
}

/// How many selected responses have no representable time.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Unplaced {
    Counted { count: u64 },
    Failed(StoreFailure),
}

/// `timeline`: one logical day's records, its month's calendar and its usage.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimelineAnswer {
    pub context: RunContext,
    pub date: String,
    pub previous: Option<String>,
    pub following: Option<String>,
    pub previous_month: Option<String>,
    pub following_month: Option<String>,
    pub day: DaySettingWire,
    pub since: i64,
    pub until: i64,
    pub activity: Activity,
    pub cursor: Option<String>,
    pub calendar: Vec<CalendarDay>,
    /// Absent for undated records and across years.
    pub usage: Option<Measured>,
    pub across_years: bool,
    pub undated: bool,
}

/// `cost`: usage of the selected sessions, its days in a month and the
/// responses behind it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CostAnswer {
    pub context: RunContext,
    pub unplaced: Unplaced,
    pub quota: Quota,
    pub usage: Measured,
    pub responses: Activity,
    /// Absent when the store could not read them; `days_error` says why.
    pub days: Option<Vec<Map<String, Value>>>,
    pub days_error: Option<StoreFailure>,
    pub month: String,
    pub previous_month: Option<String>,
    pub following_month: Option<String>,
    pub date: Option<String>,
    pub model: Option<String>,
    pub day: DaySettingWire,
    pub since: Option<i64>,
    pub until: Option<i64>,
}

/// Store query rows as objects; the store reads every row as one.
fn objects(rows: Vec<Value>) -> Vec<Map<String, Value>> {
    rows.into_iter()
        .filter_map(|row| match row {
            Value::Object(row) => Some(row),
            _ => None,
        })
        .collect()
}

fn date(text: Option<&str>, clock: &Clock) -> Result<jiff::civil::Date, Failure> {
    text.map_or_else(
        || {
            clock
                .day
                .logical_day(clock.now)
                .map_err(|error| Failure::bad_request(error.to_string()))
        },
        |text| {
            text.parse()
                .map_err(|_| Failure::bad_request("date must be YYYY-MM-DD"))
        },
    )
}

fn fingerprint(domain: &str, value: &impl Serialize) -> Result<blake3::Hash, Failure> {
    let mut digest = blake3::Hasher::new();
    digest.update(b"polylogue.calendar.cursor-context.v1\0");
    digest.update(domain.as_bytes());
    digest.update(b"\0");
    serde_json::to_writer(&mut digest, value)
        .map_err(|error| Failure::bad_request(error.to_string()))?;
    Ok(digest.finalize())
}
struct Paging {
    seal: CursorSeal,
    context: CursorContext,
    operation: Arc<Context>,
    after: Option<timeline::Position>,
    resuming: bool,
}
fn paging(
    reader: &Reader<'_>,
    request_context: &Arc<Context>,
    cursor: Option<&str>,
    question_parts: &Value,
    binding: &Value,
) -> Result<Paging, Failure> {
    let request_versions = request_context.cursor_versions();
    let question = QuestionId(*fingerprint("question", question_parts)?.as_bytes());
    let binding = fingerprint("binding", binding)?.to_hex().to_string();
    let position = reader.change_position()?;
    let file = reader.index_identity()?;
    let file = format!("{}:{}", file.0, file.1);
    let seal = CursorSeal::new(reader.cursor_key()?);
    let continuation = cursor
        .map(|text| {
            polylogue_store::read::open_cursor(
                &seal,
                &position.generation,
                &SealedCursor::from_wire(text),
                question,
                |name| {
                    if name == request_context.clock.day.zone_name() {
                        Ok(request_context.clock.day.zone().clone())
                    } else {
                        Err(query_language::Error::new(
                            query_language::ErrorKind::StaleCursor,
                            "the request day changed",
                            0,
                        ))
                    }
                },
                |context| {
                    context.clock.day == request_context.clock.day
                        && request_versions
                            .iter()
                            .all(|(key, value)| context.versions.get(key) == Some(value))
                        && context.versions.get("domain").map(String::as_str) == Some("calendar-v1")
                        && context.versions.get("binding") == Some(&binding)
                        && context.read_positions.get("generation") == Some(&position.generation)
                        && context.read_positions.get("index_file") == Some(&file)
                        && context
                            .read_positions
                            .get("index")
                            .and_then(|value| value.parse::<i64>().ok())
                            .is_some_and(|value| value <= position.index)
                        && context
                            .read_positions
                            .get("archive")
                            .and_then(|value| value.parse::<i64>().ok())
                            .is_some_and(|value| value <= position.archive)
                },
            )
        })
        .transpose()?;
    let resuming = continuation.is_some();
    let (context, after) = match continuation {
        Some(cursor) => {
            let [position] = cursor.after.values.as_slice() else {
                return Err(Failure::bad_request(
                    "the calendar cursor position is invalid",
                ));
            };
            let identity = serde_json::to_string(position).map_err(|_| {
                Failure::bad_request("the calendar cursor position cannot be encoded")
            })?;
            if cursor.after.row_identity != identity {
                return Err(Failure::bad_request(
                    "the calendar cursor row identity is invalid",
                ));
            }
            (
                cursor.context,
                Some(
                    serde_json::from_value(position.clone())
                        .map_err(|error| Failure::bad_request(error.to_string()))?,
                ),
            )
        }
        None => (
            CursorContext {
                question,
                clock: request_context.clock.clone(),
                resolved_times: vec![],
                random_seed: 0,
                read_positions: BTreeMap::from([
                    ("generation".into(), position.generation),
                    ("index_file".into(), file),
                    ("index".into(), position.index.to_string()),
                    ("archive".into(), position.archive.to_string()),
                ]),
                versions: request_versions
                    .into_iter()
                    .chain([
                        ("domain".into(), "calendar-v1".into()),
                        ("binding".into(), binding),
                    ])
                    .collect(),
            },
            None,
        ),
    };
    let operation = Arc::new(
        request_context
            .replay_clock(context.clock.clone())
            .map_err(|_| {
                Failure::stale_cursor("the calendar context changed; run without the cursor")
            })?,
    );
    Ok(Paging {
        operation,
        seal,
        context,
        after,
        resuming,
    })
}
fn resolved(
    paging: &mut Paging,
    values: Vec<query_language::time::ResolvedTime>,
) -> Result<(), Failure> {
    if paging.resuming && paging.context.resolved_times != values {
        return Err(Failure::stale_cursor(
            "the resolved calendar times changed; run without the cursor",
        ));
    }
    paging.context.resolved_times = values;
    Ok(())
}
fn resolve(text: &str, clock: &Clock) -> Result<query_language::time::ResolvedTime, Failure> {
    query_language::time::resolve(text, clock).map_err(Failure::from)
}
fn ranking(paging: &mut Paging, program: &polylogue_store::query::Program) -> Result<(), Failure> {
    let current = program
        .near
        .as_ref()
        .map(|_| polylogue_store::vectors::writes().to_string());
    if paging.resuming && paging.context.versions.get("ranking") != current.as_ref() {
        return Err(Failure::stale_cursor(
            "the ranking changed; run without the cursor",
        ));
    }
    if let Some(current) = current {
        paging.context.versions.insert("ranking".into(), current);
    }
    Ok(())
}
fn activity(
    reader: &Reader<'_>,
    selection: &timeline::Selection<'_>,
    paging: &Paging,
    size: Option<u64>,
) -> Result<Activity, Failure> {
    let size = size.unwrap_or(50);
    if size == 0 {
        return Err(Failure::bad_request("page size must be positive"));
    }
    let mut page = timeline::page(
        reader,
        &paging.context.clock,
        selection,
        paging.after.as_ref(),
        size,
    )?;
    let next = match page.get("after").filter(|value| !value.is_null()) {
        Some(after) => {
            let continuation = Continuation {
                context: paging.context.clone(),
                after: OrderPosition {
                    values: vec![after.clone()],
                    row_identity: after.to_string(),
                },
            };
            Some(paging.seal.seal(&continuation)?.as_str().to_owned())
        }
        None => None,
    };
    let rows = match page.get_mut("rows").map(Value::take) {
        Some(Value::Array(rows)) => objects(rows),
        _ => Vec::new(),
    };
    Ok(Activity { rows, next })
}

pub struct Timeline<'a> {
    pub date: Option<&'a str>,
    pub across_years: bool,
    pub undated: bool,
    pub cursor: Option<&'a str>,
    pub page_size: Option<u64>,
}

pub fn timeline(
    reader: &Reader<'_>,
    options: Timeline<'_>,
    request_context: &Arc<Context>,
) -> Result<TimelineAnswer, Failure> {
    let Timeline {
        date: text,
        across_years,
        undated,
        cursor,
        page_size: size,
    } = options;
    let mut paging = paging(
        reader,
        request_context,
        cursor,
        &json!(["timeline", text, across_years, undated]),
        &json!(["timeline"]),
    )?;
    let clock = paging.operation.clock.clone();
    let day = &clock.day;
    let date = date(text, &clock)?;
    resolved(&mut paging, vec![resolve(text.unwrap_or("today"), &clock)?])?;
    let selected = timeline::Selection {
        date: Some(date),
        across_years,
        undated,
        responses: None,
        model: None,
        unknown_model: false,
    };
    let activity = activity(reader, &selected, &paging, size)?;
    let counts = timeline::calendar(reader, &clock, date)?;
    let mut calendar = Vec::new();
    let mut cell = date.first_of_month();
    loop {
        let label = cell.to_string();
        let records = counts
            .iter()
            .find(|row| row["date"].as_str() == Some(&label))
            .map_or(0, |row| row["records"].as_u64().unwrap_or(0));
        calendar.push(CalendarDay {
            date: label,
            records,
        });
        if cell == date.last_of_month() {
            break;
        }
        cell = cell
            .tomorrow()
            .map_err(|e| Failure::bad_request(e.to_string()))?;
    }
    let (since, until) = polylogue_store::calendar::range(day, date)?;
    let usage = if undated || across_years {
        None
    } else {
        Some(measured(polylogue_store::measure::usage(
            reader,
            &polylogue_store::measure::Selection {
                since: Some(since),
                until: Some(until),
                ..Default::default()
            },
        )))
    };
    Ok(TimelineAnswer {
        context: paging.operation.run_context(),
        date: date.to_string(),
        previous: date.yesterday().ok().map(|d| d.to_string()),
        following: date.tomorrow().ok().map(|d| d.to_string()),
        previous_month: date
            .first_of_month()
            .yesterday()
            .ok()
            .map(|d| d.first_of_month().to_string()),
        following_month: date.last_of_month().tomorrow().ok().map(|d| d.to_string()),
        day: day.wire(),
        since,
        until,
        activity,
        cursor: cursor.map(str::to_owned),
        calendar,
        usage,
        across_years,
        undated,
    })
}

/// A usage report as measure data, or the store's failure to read it.
fn measured(
    report: Result<polylogue_store::measure::UsageReport, polylogue_store::StoreError>,
) -> Measured {
    match report.map(serde_json::to_value) {
        Ok(Ok(data)) => Measured::Ready { data },
        Ok(Err(error)) => Measured::Failed(StoreFailure::new(&error)),
        Err(error) => Measured::Failed(StoreFailure::new(&error)),
    }
}

pub struct Cost<'a> {
    pub embeddings: bool,
    pub ranking_identity: Option<String>,
    pub near: Option<&'a query::NearFn<'a>>,
    pub month: Option<&'a str>,
    pub query: &'a str,
    pub date: Option<&'a str>,
    pub model: Option<&'a str>,
    pub undated: bool,
    pub unknown_model: bool,
    pub cursor: Option<&'a str>,
    pub page_size: Option<u64>,
}

pub fn cost(
    reader: &Reader<'_>,
    options: Cost<'_>,
    request_context: &Arc<Context>,
) -> Result<CostAnswer, Failure> {
    let Cost {
        embeddings,
        ranking_identity,
        near,
        month,
        query: scope,
        date: text,
        model,
        undated,
        unknown_model,
        cursor,
        page_size: size,
    } = options;
    let (plan, mut query_options) = query::prepare_usage(
        reader,
        scope,
        Arc::clone(request_context),
        embeddings,
        ranking_identity.clone(),
    )?;
    let mut paging = paging(
        reader,
        request_context,
        cursor,
        &json!([
            "cost",
            plan.question_id(),
            text,
            model,
            undated,
            unknown_model
        ]),
        &json!([
            "cost",
            month,
            embeddings,
            ranking_identity,
            plan.definitions()
        ]),
    )?;
    let clock = paging.operation.clock.clone();
    query_options.clock = clock.clone();
    let day = &clock.day;
    let now = clock.now.as_millisecond();
    if undated && text.is_some() {
        return Err(Failure::bad_request(
            "undated records cannot be placed within a date",
        ));
    }
    if unknown_model && model.is_some() {
        return Err(Failure::bad_request(
            "choose a recorded model or unknown model",
        ));
    }
    if size == Some(0) {
        return Err(Failure::bad_request("page size must be positive"));
    }
    let chosen = text.map(|s| date(Some(s), &clock)).transpose()?;
    let bounds = chosen
        .map(|date| polylogue_store::calendar::range(day, date))
        .transpose()?;
    let month = date(month.or(text), &clock)?.first_of_month();
    let month_bounds = polylogue_store::calendar::month(day, month)?;
    if month_bounds.1 <= month_bounds.0 {
        return Err(Failure::bad_request(
            "calendar month has no representable interval",
        ));
    }
    let program = polylogue_store::query::lower(&plan, &query_options)?;
    let mut times = program.resolved_times.clone();
    if let Some(date) = chosen {
        times.push(resolve(&date.to_string(), &clock)?);
    }
    resolved(&mut paging, times)?;
    ranking(&mut paging, &program)?;
    let mut selection = query::select_usage(reader, &program, near)?;
    selection.undated = undated;
    if let Some((since, until)) = bounds {
        selection.since = Some(since);
        selection.until = Some(until);
    }
    let selected = timeline::Selection {
        date: None,
        across_years: false,
        undated,
        responses: Some(&selection),
        model,
        unknown_model,
    };
    let responses = activity(reader, &selected, &paging, size)?;
    let (days, days_error) = match timeline::costs_by_day(reader, &clock, month, &selection) {
        Ok(days) => (Some(objects(days)), None),
        Err(error) => (None, Some(StoreFailure::new(&error))),
    };
    let previous_month = month
        .yesterday()
        .ok()
        .map(|d| d.first_of_month().to_string());
    let following_month = month.last_of_month().tomorrow().ok().map(|d| d.to_string());
    let usage = measured(polylogue_store::measure::usage(reader, &selection));
    let unplaced = match timeline::undated_responses(reader, &selection) {
        Ok(count) => Unplaced::Counted { count },
        Err(error) => Unplaced::Failed(StoreFailure::new(&error)),
    };
    let quota = match super::measure::run(reader, "quota", None, (None, None), now, day, None) {
        Ok(data) => Quota::Ready { data },
        Err(error) => Quota::Failed {
            error: error.kind,
            message: error.message.clone(),
        },
    };
    Ok(CostAnswer {
        context: paging.operation.run_context(),
        unplaced,
        quota,
        usage,
        responses,
        days,
        days_error,
        month: month.to_string(),
        previous_month,
        following_month,
        date: text.map(str::to_owned),
        model: model.map(str::to_owned),
        day: day.wire(),
        since: selection.since,
        until: selection.until,
    })
}

#[cfg(test)]
mod tests;

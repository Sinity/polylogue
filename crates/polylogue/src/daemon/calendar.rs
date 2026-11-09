//! Shared logical-day composition for timeline, day and cost clients.
use super::{Failure, query};
use crate::operation::Context;
use polylogue_store::{Reader, timeline};
use query_language::cursor::{
    Continuation, CursorContext, CursorSeal, OrderPosition, SealedCursor,
};
use query_language::identity::QuestionId;
use query_language::time::Clock;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

fn invalid(message: impl ToString) -> Failure {
    Failure::new("bad_request", message)
}

fn date(text: Option<&str>, clock: &Clock) -> Result<jiff::civil::Date, Failure> {
    text.map_or_else(
        || {
            clock
                .day
                .logical_day(clock.now)
                .map_err(|error| invalid(error.to_string()))
        },
        |text| text.parse().map_err(|_| invalid("date must be YYYY-MM-DD")),
    )
}

fn fingerprint(domain: &str, value: &impl Serialize) -> Result<blake3::Hash, Failure> {
    let mut digest = blake3::Hasher::new();
    digest.update(b"polylogue.calendar.cursor-context.v1\0");
    digest.update(domain.as_bytes());
    digest.update(b"\0");
    serde_json::to_writer(&mut digest, value).map_err(|error| invalid(error.to_string()))?;
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
    owner: &Arc<Context>,
    cursor: Option<&str>,
    semantic: &Value,
    binding: &Value,
) -> Result<Paging, Failure> {
    let owner_versions = owner.cursor_versions();
    let question = QuestionId(*fingerprint("question", semantic)?.as_bytes());
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
                    if name == owner.clock.day.zone_name() {
                        Ok(owner.clock.day.zone().clone())
                    } else {
                        Err(query_language::Error::new(
                            query_language::ErrorKind::StaleCursor,
                            "the owner day changed",
                            0,
                        ))
                    }
                },
                |context| {
                    context.clock.day == owner.clock.day
                        && owner_versions
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
                return Err(invalid("the calendar cursor position is invalid"));
            };
            let identity = serde_json::to_string(position)
                .map_err(|_| invalid("the calendar cursor position cannot be encoded"))?;
            if cursor.after.row_identity != identity {
                return Err(invalid("the calendar cursor row identity is invalid"));
            }
            (
                cursor.context,
                Some(
                    serde_json::from_value(position.clone())
                        .map_err(|error| invalid(error.to_string()))?,
                ),
            )
        }
        None => (
            CursorContext {
                question,
                clock: owner.clock.clone(),
                resolved_times: vec![],
                random_seed: 0,
                read_positions: BTreeMap::from([
                    ("generation".into(), position.generation),
                    ("index_file".into(), file),
                    ("index".into(), position.index.to_string()),
                    ("archive".into(), position.archive.to_string()),
                ]),
                versions: owner_versions
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
    let operation = Arc::new(owner.replay_clock(context.clock.clone()).map_err(|_| {
        Failure::new(
            "stale_cursor",
            "the canonical calendar context changed; run without the cursor",
        )
    })?);
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
        return Err(Failure::new(
            "stale_cursor",
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
        return Err(Failure::new(
            "stale_cursor",
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
) -> Result<Value, Failure> {
    let size = size.unwrap_or(50);
    if size == 0 {
        return Err(invalid("page size must be positive"));
    }
    let mut result = timeline::page(
        reader,
        &paging.context.clock,
        selection,
        paging.after.as_ref(),
        size,
    )?;
    result["next"] = match result.get("after").filter(|value| !value.is_null()) {
        Some(after) => {
            let continuation = Continuation {
                context: paging.context.clone(),
                after: OrderPosition {
                    values: vec![after.clone()],
                    row_identity: after.to_string(),
                },
            };
            json!(paging.seal.seal(&continuation)?.as_str())
        }
        None => Value::Null,
    };
    result.as_object_mut().unwrap().remove("after");
    Ok(result)
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
    owner: &Arc<Context>,
) -> Result<Value, Failure> {
    let Timeline {
        date: text,
        across_years,
        undated,
        cursor,
        page_size: size,
    } = options;
    let mut paging = paging(
        reader,
        owner,
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
        calendar.push(json!({"date":label,"records":records}));
        if cell == date.last_of_month() {
            break;
        }
        cell = cell.tomorrow().map_err(|e| invalid(e.to_string()))?;
    }
    let (since, until) = polylogue_store::calendar::range(day, date)?;
    let usage = if undated || across_years {
        Value::Null
    } else {
        match polylogue_store::measure::usage(
            reader,
            &polylogue_store::measure::Selection {
                since: Some(since),
                until: Some(until),
                ..Default::default()
            },
        ) {
            Ok(report) => json!({"data":report}),
            Err(e) => json!({"error":"store","message":e.to_string()}),
        }
    };
    Ok(
        json!({"context":paging.operation.provenance(),"date":date.to_string(),"previous":date.yesterday().ok().map(|d|d.to_string()),"following":date.tomorrow().ok().map(|d|d.to_string()),"previous_month":date.first_of_month().yesterday().ok().map(|d|d.first_of_month().to_string()),"following_month":date.last_of_month().tomorrow().ok().map(|d|d.to_string()),"day":day,"since":since,"until":until,"activity":activity,"cursor":cursor,"calendar":calendar,"usage":usage,"across_years":across_years,"undated":undated}),
    )
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
    owner: &Arc<Context>,
) -> Result<Value, Failure> {
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
        Arc::clone(owner),
        embeddings,
        ranking_identity.clone(),
    )?;
    let mut paging = paging(
        reader,
        owner,
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
        return Err(invalid("undated records cannot be placed within a date"));
    }
    if unknown_model && model.is_some() {
        return Err(invalid("choose a recorded model or unknown model"));
    }
    if size == Some(0) {
        return Err(invalid("page size must be positive"));
    }
    let chosen = text.map(|s| date(Some(s), &clock)).transpose()?;
    let bounds = chosen
        .map(|date| polylogue_store::calendar::range(day, date))
        .transpose()?;
    let month = date(month.or(text), &clock)?.first_of_month();
    let month_bounds = polylogue_store::calendar::month(day, month)?;
    if month_bounds.1 <= month_bounds.0 {
        return Err(invalid("calendar month has no representable interval"));
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
        Ok(days) => (json!(days), Value::Null),
        Err(error) => (
            Value::Null,
            json!({"error":"store","message":error.to_string()}),
        ),
    };
    let previous_month = month
        .yesterday()
        .ok()
        .map(|d| d.first_of_month().to_string());
    let following_month = month.last_of_month().tomorrow().ok().map(|d| d.to_string());
    let report = match polylogue_store::measure::usage(reader, &selection) {
        Ok(report) => json!({"data":report}),
        Err(e) => json!({"error":"store","message":e.to_string()}),
    };
    let unplaced = match timeline::undated_responses(reader, &selection) {
        Ok(count) => json!({"count":count}),
        Err(error) => json!({"error":"store","message":error.to_string()}),
    };
    let quota = match super::measure::run(reader, "quota", None, (None, None), now, day, None) {
        Ok(data) => json!({"data":data}),
        Err(error) => json!({"error":error.kind,"message":error.message}),
    };
    Ok(
        json!({"context":paging.operation.provenance(),"unplaced":unplaced,"quota":quota,"usage":report,"responses":responses,"days":days,"days_error":days_error,"month":month.to_string(),"previous_month":previous_month,"following_month":following_month,"date":text,"model":model,"day":day,"since":selection.since,"until":selection.until}),
    )
}

#[cfg(test)]
mod tests;

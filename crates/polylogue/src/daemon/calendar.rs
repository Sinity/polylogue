//! Shared logical-day composition for timeline, day and cost clients.
use super::query::{self, Failure};
use base64::Engine as _;
use polylogue_store::{Reader, timeline};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn invalid(message: impl Into<String>) -> Failure {
    Failure {
        kind: "bad_request".into(),
        message: message.into(),
        detail: None,
    }
}

fn date(
    text: Option<&str>,
    day: &polylogue_model::LogicalDay,
    now: i64,
) -> Result<jiff::civil::Date, Failure> {
    text.map_or_else(
        || Ok(day.date_of(now)),
        |text| text.parse().map_err(|_| invalid("date must be YYYY-MM-DD")),
    )
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    now: i64,
    context: Value,
    after: timeline::Position,
}

fn cursor(text: &str) -> Result<Cursor, Failure> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(text)
        .map_err(|_| invalid("invalid timeline cursor"))?;
    serde_json::from_slice(&bytes).map_err(|_| invalid("invalid timeline cursor"))
}

fn clock(text: Option<&str>, now: i64) -> Result<i64, Failure> {
    text.map(cursor)
        .transpose()
        .map(|c| c.map_or(now, |c| c.now))
}

fn activity(
    reader: &Reader<'_>,
    day: &polylogue_model::LogicalDay,
    selection: &timeline::Selection<'_>,
    context: Value,
    cursor: Option<&str>,
    size: Option<u64>,
    now: i64,
) -> Result<Value, Failure> {
    let context = json!(
        blake3::hash(context.to_string().as_bytes())
            .to_hex()
            .to_string()
    );
    let size = size.unwrap_or(50);
    if size == 0 {
        return Err(invalid("page size must be positive"));
    }
    let cursor: Option<Cursor> = cursor.map(self::cursor).transpose()?;
    if cursor.as_ref().is_some_and(|c| c.context != context) {
        return Err(invalid(
            "cursor belongs to a different selection, logical day or index generation",
        ));
    }
    let mut result = timeline::page(
        reader,
        day,
        selection,
        cursor.as_ref().map(|c| &c.after),
        size,
    )?;
    result["next"] = match result.get("after").filter(|a| !a.is_null()) {
        Some(after) => {
            let cursor = Cursor {
                now,
                context,
                after: serde_json::from_value(after.clone()).map_err(|e| invalid(e.to_string()))?,
            };
            json!(
                base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .encode(serde_json::to_vec(&cursor).map_err(|e| invalid(e.to_string()))?)
            )
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

pub fn timeline(reader: &Reader<'_>, options: Timeline<'_>, now: i64) -> Result<Value, Failure> {
    let Timeline {
        date: text,
        across_years,
        undated,
        cursor,
        page_size: size,
    } = options;
    let now = clock(cursor, now)?;
    let day = query::logical_day(reader)?;
    let date = date(text, &day, now)?;
    let context = json!([
        "timeline",
        date.to_string(),
        across_years,
        undated,
        format!("{:?}", day.zone),
        day.start_minutes,
        reader.change_position()?.generation
    ]);
    let selected = timeline::Selection {
        date: Some(date),
        across_years,
        undated,
        responses: None,
        model: None,
        unknown_model: false,
    };
    let activity = activity(reader, &day, &selected, context, cursor, size, now)?;
    let counts = timeline::calendar(reader, &day, date)?;
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
    let (since, until) = day.day(date);
    if until <= since {
        return Err(invalid(
            "this date has no representable logical-day interval",
        ));
    }
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
        json!({"date":date.to_string(),"previous":date.yesterday().ok().map(|d|d.to_string()),"following":date.tomorrow().ok().map(|d|d.to_string()),"previous_month":date.first_of_month().yesterday().ok().map(|d|d.first_of_month().to_string()),"following_month":date.last_of_month().tomorrow().ok().map(|d|d.to_string()),"zone":day.zone_name().map(str::to_owned).unwrap_or_else(||format!("{:?}",day.zone)),"day_start_minutes":day.start_minutes,"since":since,"until":until,"activity":activity,"cursor":cursor,"calendar":calendar,"usage":usage,"across_years":across_years,"undated":undated}),
    )
}

pub struct Cost<'a> {
    pub embeddings: bool,
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

pub fn cost(reader: &Reader<'_>, options: Cost<'_>, now: i64) -> Result<Value, Failure> {
    let Cost {
        embeddings,
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
    let now = clock(cursor, now)?;
    let day = query::logical_day(reader)?;
    if undated && text.is_some() {
        return Err(invalid("undated records cannot be placed within a date"));
    }
    if unknown_model && model.is_some() {
        return Err(invalid("choose a recorded model or unknown model"));
    }
    if size == Some(0) {
        return Err(invalid("page size must be positive"));
    }
    let chosen = text.map(|s| date(Some(s), &day, now)).transpose()?;
    let bounds = chosen.map(|date| day.day(date));
    if bounds.is_some_and(|(since, until)| until <= since) {
        return Err(invalid(
            "this date has no representable logical-day interval",
        ));
    }
    let month = date(month.or(text), &day, now)?.first_of_month();
    let month_bounds = day.month(month);
    if month_bounds.1 <= month_bounds.0 {
        return Err(invalid("calendar month has no representable interval"));
    }
    let (mut selection, question) = query::usage_selection(reader, scope, now, embeddings, near)?;
    selection.undated = undated;
    if let Some((since, until)) = bounds {
        selection.since = Some(since);
        selection.until = Some(until);
    }
    let context = json!([
        "cost",
        scope,
        text,
        model,
        undated,
        unknown_model,
        question,
        format!("{:?}", day.zone),
        day.start_minutes,
        reader.change_position()?.generation
    ]);
    let selected = timeline::Selection {
        date: None,
        across_years: false,
        undated,
        responses: Some(&selection),
        model,
        unknown_model,
    };
    let responses = activity(reader, &day, &selected, context, cursor, size, now)?;
    let (days, days_error) = match timeline::costs_by_day(reader, &day, month, &selection) {
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
    let quota = match super::measure::run(reader, "quota", None, now, &day) {
        Ok(data) => json!({"data":data}),
        Err(error) => json!({"error":error.kind,"message":error.message}),
    };
    Ok(
        json!({"unplaced":unplaced,"quota":quota,"usage":report,"responses":responses,"days":days,"days_error":days_error,"month":month.to_string(),"previous_month":previous_month,"following_month":following_month,"date":text,"model":model,"zone":day.zone_name(),"day_start_minutes":day.start_minutes,"since":selection.since,"until":selection.until}),
    )
}

#[cfg(test)]
mod tests;

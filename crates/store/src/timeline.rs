//! Chronological metadata and calendar counts, without loading message bodies.
use crate::{Reader, Result, SqlValue};
use polylogue_model::LogicalDay;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub undated: bool,
    pub at: Option<i64>,
    pub kind: String,
    pub id: String,
}

pub struct Selection<'a> {
    pub date: Option<jiff::civil::Date>,
    pub across_years: bool,
    pub undated: bool,
    pub responses: Option<&'a crate::measure::Selection>,
    pub model: Option<&'a str>,
    pub unknown_model: bool,
}

fn bind(params: &mut Vec<SqlValue>, value: SqlValue) -> String {
    params.push(value);
    format!("?{}", params.len())
}

fn sources(selection: &Selection<'_>, session: Option<i64>) -> (Vec<String>, Vec<SqlValue>) {
    if let Some(responses) = selection.responses {
        let (condition, values) = responses.condition();
        let mut params = values
            .into_iter()
            .map(|value| match value {
                rusqlite::types::Value::Integer(n) => SqlValue::Int(n),
                rusqlite::types::Value::Text(s) => SqlValue::Text(s),
                _ => unreachable!("measure selection uses integers and text"),
            })
            .collect::<Vec<_>>();
        let mut condition = format!("{} AND {condition}", crate::measure::RESPONSE_ROWS);
        if let Some(session) = session {
            condition.push_str(&format!(
                " AND m.session_pk = {}",
                bind(&mut params, SqlValue::Int(session))
            ));
        }
        // A session range supplies the ordering even for a common model.
        let model_column = if session.is_some() { "+m.model" } else { "m.model" };
        if selection.unknown_model {
            condition.push_str(&format!(" AND {model_column} IS NULL"));
        }
        if let Some(model) = selection.model {
            condition.push_str(&format!(
                " AND {model_column} = {}",
                bind(&mut params, SqlValue::Text(model.to_owned()))
            ));
        }
        return (
            vec![format!(
                "SELECT m.at, 'message' AS kind, m.id, s.id AS session, m.role AS label, 'Recorded message time' AS time_source, m.cost_nusd FROM message m CROSS JOIN session s ON s.pk=m.session_pk WHERE {condition}"
            )],
            params,
        );
    }
    (vec![
        "SELECT m.at, 'message' AS kind, m.id, s.id AS session, m.role AS label, 'Recorded message time' AS time_source FROM message m CROSS JOIN session s ON s.pk=m.session_pk WHERE 1".into(),
        "SELECT e.at, 'event' AS kind, e.id, s.id AS session, e.kind AS label, 'Recorded event time' AS time_source FROM event e LEFT JOIN session s ON s.pk=e.session_pk WHERE 1".into(),
        "SELECT n.created_at AS at, 'note' AS kind, n.id, NULL AS session, n.kind AS label, 'Note writer clock' AS time_source FROM a.note n WHERE 1".into(),
    ], vec![])
}

fn representable(column: &str, undated: bool) -> String {
    let min = jiff::Timestamp::MIN.as_millisecond();
    let max = jiff::Timestamp::MAX.as_millisecond();
    if undated {
        format!("({column} IS NULL OR {column} < {min} OR {column} > {max})")
    } else {
        format!("{column} >= {min} AND {column} <= {max}")
    }
}

/// Finds recorded logical years by seeking to the first timestamp after each
/// year. Empty years and individual records are never walked.
fn years(reader: &Reader<'_>, day: &LogicalDay, through: jiff::civil::Date) -> Result<Vec<i16>> {
    let mut years = std::collections::BTreeSet::new();
    for column in ["message.at", "event.at", "a.note.created_at"] {
        let (table, field) = column.rsplit_once('.').unwrap();
        let mut since = jiff::Timestamp::MIN.as_millisecond();
        loop {
            let rows = reader.query(
                &format!(
                    "SELECT min({field}) AS at FROM {table} WHERE {field} >= ?1 AND {field} < ?2"
                ),
                &[SqlValue::Int(since), SqlValue::Int(day.day(through).1)],
            )?;
            let Some(at) = rows.first().and_then(|row| row["at"].as_i64()) else {
                break;
            };
            let year = day.date_of(at).year();
            years.insert(year);
            let Some(next) = year
                .checked_add(1)
                .and_then(|y| jiff::civil::Date::new(y, 1, 1).ok())
            else {
                break;
            };
            let next = day.start_of(next);
            if next <= since || next <= at {
                break;
            }
            since = next;
        }
    }
    Ok(years.into_iter().collect())
}

/// Session selections read one ordered range per session and merge their
/// bounded pages. Selecting every session uses the global time order instead.
fn page_sources(
    reader: &Reader<'_>,
    selection: &Selection<'_>,
) -> Result<Vec<(&'static str, String, Vec<SqlValue>)>> {
    if let Some(responses) = selection.responses
        && (responses.sessions.is_some() || responses.origins.is_some())
    {
        let mut clauses = Vec::new();
        let mut params = Vec::new();
        if let Some(sessions) = &responses.sessions {
            clauses.push(format!(
                "pk IN (SELECT value FROM json_each({}))",
                bind(&mut params, SqlValue::Text(json!(sessions).to_string()))
            ));
        }
        if let Some(origins) = &responses.origins {
            clauses.push(format!(
                "origin IN (SELECT value FROM json_each({}))",
                bind(&mut params, SqlValue::Text(json!(origins).to_string()))
            ));
        }
        let sessions = reader.query(
            &format!("SELECT pk FROM session WHERE {}", clauses.join(" AND ")),
            &params,
        )?;
        let total = reader.query("SELECT count(*) AS n FROM session", &[])?;
        let keys = if sessions.len() as u64 == total[0]["n"].as_u64().unwrap() {
            vec![None]
        } else {
            sessions.into_iter().map(|session| Some(session["pk"].as_i64().unwrap())).collect()
        };
        let responses = crate::measure::Selection {
            sessions: None,
            origins: None,
            ..responses.clone()
        };
        let selection = Selection {
            responses: Some(&responses),
            ..*selection
        };
        return Ok(keys.into_iter().map(|key| {
            let (sources, params) = sources(&selection, key);
            ("message", sources.into_iter().next().unwrap(), params)
        }).collect());
    }

    let (sources, params) = sources(selection, None);
    Ok(sources.into_iter().enumerate().map(|(index, sql)| {
        (["message", "event", "note"][index], sql, params.clone())
    }).collect())
}

fn row_queries(
    reader: &Reader<'_>,
    selection: &Selection<'_>,
    bounds: Option<(i64, i64)>,
    after: Option<&Position>,
    limit: i64,
    undated: bool,
) -> Result<Vec<(String, Vec<SqlValue>)>> {
    let sources = page_sources(reader, selection)?;
    let mut queries = Vec::new();
    for (kind, source, mut params) in sources {
        let mut conditions = Vec::new();
        if undated {
            conditions.push(representable("at", true));
            if let Some(after) = after.filter(|p| p.undated) {
                match kind.cmp(after.kind.as_str()) {
                    std::cmp::Ordering::Less => continue,
                    std::cmp::Ordering::Equal => conditions.push(format!(
                        "id > {}",
                        bind(&mut params, SqlValue::Text(after.id.clone()))
                    )),
                    std::cmp::Ordering::Greater => {}
                }
            }
        } else {
            // One lower bound lets SQLite seek directly to the requested day
            // or continuation, rather than choosing the representable minimum.
            let min = jiff::Timestamp::MIN.as_millisecond();
            let max = jiff::Timestamp::MAX.as_millisecond();
            let (mut since, mut through) = bounds.map_or((min, max), |(since, until)| {
                (since.max(min), until.saturating_sub(1).min(max))
            });
            if let Some(responses) = selection.responses {
                since = since.max(responses.since.unwrap_or(min));
                through = through.min(responses.until.map_or(max, |until| until.saturating_sub(1)));
            }
            if since > through || after.is_some_and(|p| p.undated || p.at.is_none()) {
                continue;
            }
            if let Some(after) = after.filter(|p| p.at.unwrap() >= since) {
                let at = bind(&mut params, SqlValue::Int(after.at.unwrap()));
                conditions.push(match kind.cmp(after.kind.as_str()) {
                    std::cmp::Ordering::Equal => format!(
                        "(at,id) > ({at},{})",
                        bind(&mut params, SqlValue::Text(after.id.clone()))
                    ),
                    std::cmp::Ordering::Less => format!("at > {at}"),
                    std::cmp::Ordering::Greater => format!("at >= {at}"),
                });
            } else {
                conditions.push(format!("at >= {}", bind(&mut params, SqlValue::Int(since))));
            }
            conditions.push(format!(
                "at <= {}",
                bind(&mut params, SqlValue::Int(through))
            ));
        }
        let order = if undated { "kind,id" } else { "at,id" };
        let limit = bind(&mut params, SqlValue::Int(limit));
        queries.push((
            format!(
                "SELECT * FROM ({source}) WHERE {} ORDER BY {order} LIMIT {limit}",
                conditions.join(" AND ")
            ),
            params,
        ));
    }
    Ok(queries)
}

fn compare_rows(a: &Value, b: &Value, undated: bool) -> std::cmp::Ordering {
        let time = if undated {
            std::cmp::Ordering::Equal
        } else {
            a["at"].as_i64().cmp(&b["at"].as_i64())
        };
        time.then_with(|| a["kind"].as_str().cmp(&b["kind"].as_str()))
            .then_with(|| a["id"].as_str().cmp(&b["id"].as_str()))
 }

fn rows(
    reader: &Reader<'_>,
    selection: &Selection<'_>,
    bounds: Option<(i64, i64)>,
    after: Option<&Position>,
    limit: i64,
    undated: bool,
) -> Result<Vec<Value>> {
    let mut found = Vec::new();
    for (sql, params) in row_queries(reader, selection, bounds, after, limit, undated)? {
        found.extend(reader.query(&sql, &params)?);
        found.sort_by(|a, b| compare_rows(a, b, undated));
        found.truncate(limit as usize);
    }
    Ok(found)
}

/// One page across activity kinds. Dated reads seek by time and ID;
/// undated reads follow separately, ordered by kind and ID.
pub fn page(
    reader: &Reader<'_>,
    day: &LogicalDay,
    selection: &Selection<'_>,
    after: Option<&Position>,
    size: u64,
) -> Result<Value> {
    let limit = i64::try_from(
        size.checked_add(1)
            .ok_or_else(|| crate::StoreError::Invalid("page size exceeds i64".into()))?,
    )
    .map_err(|_| crate::StoreError::Invalid("page size exceeds i64".into()))?;
    let mut found = Vec::new();
    if !selection.undated && !after.is_some_and(|p| p.undated) {
        let ranges = match selection.date {
            Some(date) if selection.across_years => years(reader, day, date)?
                .into_iter()
                .filter_map(|year| jiff::civil::Date::new(year, date.month(), date.day()).ok())
                .map(|date| Some(day.day(date)))
                .collect(),
            Some(date) => vec![Some(day.day(date))],
            None => vec![None],
        };
        for bounds in ranges {
            found.extend(rows(
                reader,
                selection,
                bounds,
                after,
                limit - found.len() as i64,
                false,
            )?);
            if found.len() >= limit as usize {
                break;
            }
        }
    }
    let dated = found.len();
    if found.len() < limit as usize
        && (selection.undated || (selection.responses.is_some() && selection.date.is_none()))
    {
        found.extend(rows(
            reader,
            selection,
            None,
            after,
            limit - found.len() as i64,
            true,
        )?);
    }
    for (i, row) in found.iter_mut().enumerate() {
        row["date"] = if i < dated {
            json!(day.day_label(row["at"].as_i64().unwrap()))
        } else {
            Value::Null
        };
        row.as_object_mut().unwrap().remove("cost_nusd");
    }
    let more = found.len() as u64 > size;
    if more {
        found.pop();
    }
    let next = if more {
        found.last().map(|row| json!({"undated":row["date"].is_null(),"at":row["at"],"kind":row["kind"],"id":row["id"]}))
    } else {
        None
    };
    Ok(json!({"rows":found,"after":next}))
}

/// A single calendar month. Each source is bounded before date conversion.
pub fn calendar(
    reader: &Reader<'_>,
    day: &LogicalDay,
    date: jiff::civil::Date,
) -> Result<Vec<Value>> {
    dates(reader, day)?;
    let (since, until) = day.month(date);
    if until <= since {
        return Err(crate::StoreError::Invalid(
            "calendar month has no representable interval".into(),
        ));
    }
    reader.query(
        "SELECT pl_timeline_date(at) AS date,count(*) AS records FROM (
        SELECT at FROM message WHERE at >= ?1 AND at < ?2
        UNION ALL SELECT at FROM event WHERE at >= ?1 AND at < ?2
        UNION ALL SELECT created_at AS at FROM a.note WHERE created_at >= ?1 AND created_at < ?2
    ) GROUP BY date ORDER BY date",
        &[SqlValue::Int(since), SqlValue::Int(until)],
    )
}

fn dates(reader: &Reader<'_>, day: &LogicalDay) -> Result<()> {
    let day = day.clone();
    reader.conn()?.create_scalar_function(
        "pl_timeline_date",
        1,
        rusqlite::functions::FunctionFlags::SQLITE_UTF8
            | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            let at = ctx.get::<Option<i64>>(0)?;
            Ok(at
                .filter(|at| jiff::Timestamp::from_millisecond(*at).is_ok())
                .map(|at| day.day_label(at)))
        },
    )?;
    Ok(())
}

/// Known physical response subtotals in a calendar month. Independent usage
/// observations remain separate in the usage report, as they do by model.
pub fn costs_by_day(
    reader: &Reader<'_>,
    day: &LogicalDay,
    month: jiff::civil::Date,
    selected: &crate::measure::Selection,
) -> Result<Vec<Value>> {
    dates(reader, day)?;
    let (sources, mut params) = sources(&Selection {
        date: None,
        across_years: false,
        undated: false,
        responses: Some(selected),
        model: None,
        unknown_model: false,
    }, None);
    let source = &sources[0];
    let (since, until) = day.month(month);
    if until <= since {
        return Err(crate::StoreError::Invalid(
            "calendar month has no representable interval".into(),
        ));
    }
    params.push(SqlValue::Int(since));
    let first = params.len();
    params.push(SqlValue::Int(until));
    let last = params.len();
    reader.query(&format!("WITH activity AS ({source}) SELECT pl_timeline_date(at) AS date,count(*) AS records,sum(cost_nusd)/1e9 AS known_usd,sum(cost_nusd IS NULL) AS unpriced_records FROM activity WHERE at >= ?{first} AND at < ?{last} GROUP BY date ORDER BY date"),&params)
}

/// Known responses with unrepresentable source times in the selected sessions,
/// counted apart because no date interval can include them.
pub fn undated_responses(
    reader: &Reader<'_>,
    selection: &crate::measure::Selection,
) -> Result<u64> {
    let selection = crate::measure::Selection {
        since: None,
        until: None,
        undated: true,
        ..selection.clone()
    };
    let (sources, params) = sources(&Selection {
        date: None,
        across_years: false,
        undated: true,
        responses: Some(&selection),
        model: None,
        unknown_model: false,
    }, None);
    let source = &sources[0];
    let rows = reader.query(
        &format!("SELECT count(*) AS records FROM ({source})"),
        &params,
    )?;
    rows.first()
        .and_then(|row| row["records"].as_u64())
        .ok_or_else(|| {
            crate::StoreError::Invalid("response count is not an unsigned integer".into())
        })
}

#[cfg(test)]
mod tests;

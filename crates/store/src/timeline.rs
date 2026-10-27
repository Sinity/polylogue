//! Chronological metadata and calendar counts, without loading message bodies.
use crate::{Reader, Result, SqlValue};
use polylogue_model::LogicalDay;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
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

fn source(selection: &Selection<'_>) -> (String, Vec<SqlValue>) {
    if let Some(responses) = selection.responses {
        let (condition, params) = responses.condition();
        let params = params
            .into_iter()
            .map(|p| match p {
                rusqlite::types::Value::Integer(n) => SqlValue::Int(n),
                rusqlite::types::Value::Text(s) => SqlValue::Text(s),
                _ => unreachable!("measure selection uses integers and text"),
            })
            .collect();
        return (
            format!(
                "SELECT m.at, 'message' AS kind, m.id, s.id AS session, m.role AS label, 'Provider message time' AS time_source, m.cost_nusd AS cost FROM message m JOIN session s ON s.pk=m.session_pk WHERE {} AND {condition}",
                crate::measure::usage::RESPONSE_ROWS
            ),
            params,
        );
    }
    ("SELECT m.at, 'message' AS kind, m.id, s.id AS session, m.role AS label, 'Provider message time' AS time_source FROM message m JOIN session s ON s.pk=m.session_pk
       UNION ALL SELECT e.at, 'event', e.id, s.id, e.kind, 'Provider event time' FROM event e LEFT JOIN session s ON s.pk=e.session_pk
       UNION ALL SELECT n.created_at, 'note', n.id, NULL, n.kind, 'Note writer clock' FROM a.note n".into(), vec![])
}

/// One page across all activity kinds, ordered by recorded time, kind and ID.
/// Unknown and unrepresentable timestamps remain in the undated population.
pub fn page(
    reader: &Reader<'_>,
    day: &LogicalDay,
    selection: &Selection<'_>,
    after: Option<&Position>,
    size: u64,
) -> Result<Value> {
    let (source, mut params) = source(selection);
    dates(reader, day)?;
    let mut bind = |v: SqlValue| {
        params.push(v);
        format!("?{}", params.len())
    };
    let date_expr = "pl_timeline_date(at)";
    let mut conditions = vec![if selection.undated {
        "date IS NULL".to_owned()
    } else {
        "date IS NOT NULL".to_owned()
    }];
    if !selection.undated
        && let Some(date) = selection.date
    {
        if selection.across_years {
            conditions.push(format!(
                "substr(date,-5) = {} AND at < {}",
                bind(SqlValue::Text(format!(
                    "{:02}-{:02}",
                    date.month(),
                    date.day()
                ))),
                bind(SqlValue::Int(day.day(date).1))
            ));
        } else {
            let (since, until) = day.day(date);
            conditions.push(format!(
                "at >= {} AND at < {}",
                bind(SqlValue::Int(since)),
                bind(SqlValue::Int(until))
            ));
            conditions.push(format!("date = {}", bind(SqlValue::Text(date.to_string()))));
        }
    }
    if selection.unknown_model {
        conditions.push("id IN (SELECT id FROM message WHERE model IS NULL)".into());
    }
    if let Some(model) = selection.model {
        conditions.push(format!(
            "id IN (SELECT id FROM message WHERE model = {})",
            bind(SqlValue::Text(model.to_owned()))
        ));
    }
    if let Some(after) = after {
        if selection.undated {
            conditions.push(format!(
                "(kind,id) > ({},{})",
                bind(SqlValue::Text(after.kind.clone())),
                bind(SqlValue::Text(after.id.clone()))
            ));
        } else {
            conditions.push(format!(
                "(at,kind,id) > ({},{},{})",
                bind(after.at.map_or(SqlValue::Null, SqlValue::Int)),
                bind(SqlValue::Text(after.kind.clone())),
                bind(SqlValue::Text(after.id.clone()))
            ));
        }
    }
    let limit = bind(SqlValue::Int(
        i64::try_from(size.saturating_add(1))
            .map_err(|_| crate::StoreError::Invalid("page size exceeds i64".into()))?,
    ));
    let order = if selection.undated {
        "kind,id"
    } else {
        "at,kind,id"
    };
    let mut rows = reader.query(&format!("WITH activity AS ({source}), dated AS (SELECT *, {date_expr} AS date FROM activity) SELECT * FROM dated WHERE {} ORDER BY {order} LIMIT {limit}",conditions.join(" AND ")), &params)?;
    let more = rows.len() as u64 > size;
    if more {
        rows.pop();
    }
    let next = if more {
        rows.last()
            .map(|row| json!({"at":row["at"],"kind":row["kind"],"id":row["id"]}))
    } else {
        None
    };
    Ok(json!({"rows":rows,"after":next}))
}

/// A single calendar month. Every cell counts messages, events and note revisions.
pub fn calendar(
    reader: &Reader<'_>,
    day: &LogicalDay,
    date: jiff::civil::Date,
) -> Result<Vec<Value>> {
    dates(reader, day)?;
    let (source, _) = source(&Selection {
        date: None,
        across_years: false,
        undated: false,
        responses: None,
        model: None,
        unknown_model: false,
    });
    let (since, until) = day.month(date);
    if until <= since {
        return Err(crate::StoreError::Invalid(
            "calendar month has no representable interval".into(),
        ));
    }
    reader.query(&format!("WITH activity AS ({source}) SELECT pl_timeline_date(at) AS date, count(*) AS records FROM activity WHERE at >= ?1 AND at < ?2 GROUP BY date ORDER BY date"), &[SqlValue::Int(since),SqlValue::Int(until)])
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
    let (source, mut params) = source(&Selection {
        date: None,
        across_years: false,
        undated: false,
        responses: Some(selected),
        model: None,
        unknown_model: false,
    });
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
    reader.query(&format!("WITH activity AS ({source}) SELECT pl_timeline_date(at) AS date,count(*) AS records,sum(cost)/1e9 AS known_usd,sum(cost IS NULL) AS unpriced_records FROM activity WHERE at >= ?{first} AND at < ?{last} GROUP BY date ORDER BY date"),&params)
}

#[cfg(test)]
mod tests;

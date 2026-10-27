//! Gaps in source message order. These describe recorded timing, not attention.
use super::Selection;
use crate::{Reader, Result, StoreError};
use serde::Serialize;

struct Turn {
    role: String,
    at: Option<i64>,
}

fn turns(conn: &rusqlite::Connection, pk: i64) -> Result<Vec<Turn>> {
    let mut statement=conn.prepare_cached("SELECT role,at FROM message WHERE session_pk=?1 AND duplicate_of IS NULL
        AND written_by<>'harness' AND role IN ('user','assistant') ORDER BY order_input_id,seq,record_position,id")?;
    Ok(statement
        .query_map([pk], |r| {
            Ok(Turn {
                role: r.get(0)?,
                at: r.get(1)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
}

#[derive(Debug, Serialize)]
pub struct SessionPace {
    pub session: String,
    pub turns: usize,
    pub timestamped_turns: usize,
    pub ignored_negative_gaps: u64,
    pub active_ms: i64,
    pub model_work_ms: i64,
    pub user_idle_ms: i64,
    pub user_gaps: usize,
    pub burstiness: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Pace {
    pub threshold_ms: i64,
    pub sessions: Vec<SessionPace>,
    pub total_gaps: usize,
}

/// Population standard deviation of nonnegative gaps. Ten gaps are required.
pub fn burstiness(gaps: &[f64]) -> Option<f64> {
    if gaps.len() < 10 {
        return None;
    }
    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let sigma = (gaps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / gaps.len() as f64).sqrt();
    (sigma + mean > 0.0).then(|| (sigma - mean) / (sigma + mean))
}

fn threshold(options: &str) -> Result<i64> {
    let threshold_ms = if options.is_empty() {
        60_000
    } else {
        options
            .parse::<i64>()
            .ok()
            .filter(|n| *n >= 0)
            .and_then(|n| n.checked_mul(1000))
            .ok_or_else(|| {
                StoreError::Invalid("pace takes a nonnegative threshold in seconds".into())
            })?
    };
    Ok(threshold_ms)
}

/// A gap statement with stable source keys for the ordinary query cursor.
pub fn gap_query(selection_sql: &str, options: &str) -> Result<String> {
    let threshold = threshold(options)?;
    Ok(format!("WITH selected AS ({selection_sql}),
        ordered AS (
            SELECT s.pk AS _pk,s.id AS session,m.id,m.role,m.at,m.order_input_id,m.seq,m.record_position,
              row_number() OVER (PARTITION BY s.pk ORDER BY m.order_input_id,m.seq,m.record_position,m.id)-1 AS position
            FROM message m JOIN session s ON s.pk=m.session_pk
            WHERE s.pk IN (SELECT _pk FROM selected) AND m.duplicate_of IS NULL
              AND m.written_by<>'harness' AND m.role IN ('user','assistant')),
        dated AS (
            SELECT *,lag(id) OVER w AS from_message,lag(position) OVER w AS from_position,
              lag(at) OVER w AS _previous_at,lag(order_input_id) OVER w AS _from_input,
              lag(seq) OVER w AS _from_seq,lag(record_position) OVER w AS _from_record
            FROM ordered WHERE at IS NOT NULL WINDOW w AS (PARTITION BY _pk ORDER BY position)),
        gaps AS (
            SELECT _pk,session,from_message,id AS to_message,from_position,position AS to_position,
              at-_previous_at AS milliseconds,role,_from_input,_from_seq,_from_record
            FROM dated WHERE _previous_at IS NOT NULL AND at>=_previous_at)
        SELECT *,CASE WHEN milliseconds<={threshold} THEN 'active' WHEN role='assistant' THEN 'model_work' ELSE 'user_idle' END AS kind FROM gaps"))
}

/// Session summaries cover the full selection; Query reads the gap page.
pub fn pace(reader: &Reader<'_>, selection: &Selection, options: &str) -> Result<Pace> {
    let threshold_ms = threshold(options)?;
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let mut statement = conn.prepare(&format!(
        "SELECT s.pk,s.id FROM session s WHERE {condition} ORDER BY s.id"
    ))?;
    let sessions = statement
        .query_map(rusqlite::params_from_iter(params.iter()), |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut reports = vec![];
    let mut total_gaps = 0;
    for (pk, session) in sessions {
        let turns = turns(conn, pk)?;
        let dated: Vec<_> = turns.iter().filter(|t| t.at.is_some()).collect();
        let mut report = SessionPace {
            session: session.clone(),
            turns: turns.len(),
            timestamped_turns: dated.len(),
            ignored_negative_gaps: 0,
            active_ms: 0,
            model_work_ms: 0,
            user_idle_ms: 0,
            user_gaps: 0,
            burstiness: None,
        };
        for pair in dated.windows(2) {
            let [a, b] = pair else { unreachable!() };
            let ms = b.at.unwrap() - a.at.unwrap();
            if ms < 0 {
                report.ignored_negative_gaps += 1;
                continue;
            }
            if ms <= threshold_ms {
                report.active_ms += ms;
            } else if b.role == "assistant" {
                report.model_work_ms += ms;
            } else {
                report.user_idle_ms += ms;
            }
            total_gaps += 1;
        }
        // Missing user times break the consecutive-user population.
        let users: Vec<_> = turns.iter().filter(|t| t.role == "user").collect();
        let user_gaps: Vec<_> = users
            .windows(2)
            .filter_map(|p| Some(p[1].at? - p[0].at?))
            .filter(|x| *x >= 0)
            .map(|x| x as f64)
            .collect();
        report.user_gaps = user_gaps.len();
        report.burstiness = burstiness(&user_gaps);
        reports.push(report);
    }
    Ok(Pace {
        threshold_ms,
        sessions: reports,
        total_gaps,
    })
}

#[cfg(test)]
mod tests;

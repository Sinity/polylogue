//! Gaps in source message order. These describe recorded timing, not attention.
use super::Selection;
use crate::{Reader, Result, StoreError};
use serde::Serialize;

pub(crate) struct Turn {
    pub position: u64,
    pub id: String,
    pub role: String,
    pub written_by: String,
    pub at: Option<i64>,
}

pub(crate) fn turns(conn: &rusqlite::Connection, pk: i64) -> Result<Vec<Turn>> {
    let mut statement = conn.prepare_cached(
        "SELECT id,role,at,written_by FROM message WHERE session_pk=?1 AND duplicate_of IS NULL
        AND written_by<>'harness' AND role IN ('user','assistant') ORDER BY order_input_id,seq,record_position,id",
    )?;
    Ok(statement
        .query_map([pk], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<i64>>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?
        .enumerate()
        .map(|(i, row)| {
            row.map(|(id, role, at, written_by)| Turn {
                position: i as u64,
                id,
                role,
                at,
                written_by,
            })
        })
        .collect::<rusqlite::Result<_>>()?)
}

#[derive(Debug, Serialize)]
pub struct Gap {
    pub session: String,
    pub from_position: u64,
    pub to_position: u64,
    pub from_message: String,
    pub to_message: String,
    pub milliseconds: i64,
    pub kind: &'static str,
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
    pub gaps: Vec<Gap>,
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

/// The query cursor pages sessions; every gap of each returned session is shown.
pub fn pace(reader: &Reader<'_>, selection: &Selection, options: &str) -> Result<Pace> {
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
    let mut gaps = vec![];
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
            let kind = if ms <= threshold_ms {
                report.active_ms += ms;
                "active"
            } else if b.role == "assistant" {
                report.model_work_ms += ms;
                "model_work"
            } else {
                report.user_idle_ms += ms;
                "user_idle"
            };
            gaps.push(Gap {
                session: session.clone(),
                from_position: a.position,
                to_position: b.position,
                from_message: a.id.clone(),
                to_message: b.id.clone(),
                milliseconds: ms,
                kind,
            });
        }
        // Missing user times break the consecutive-user population.
        let users: Vec<_> = turns
            .iter()
            .filter(|t| t.role == "user" && t.written_by != "harness")
            .collect();
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
    gaps.sort_by(|a, b| {
        b.milliseconds
            .cmp(&a.milliseconds)
            .then_with(|| a.session.cmp(&b.session))
            .then(a.from_position.cmp(&b.from_position))
    });
    let total_gaps = gaps.len();
    Ok(Pace {
        threshold_ms,
        sessions: reports,
        total_gaps,
        gaps,
    })
}

#[cfg(test)]
mod tests;

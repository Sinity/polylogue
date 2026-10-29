//! Backfill jobs: the operator asks the browser extension to capture a web
//! chat's older conversations. The job is stored once; what happened to it
//! is a list of events, and its state is the latest one. The writer applies
//! every change ([`crate::user_writes`]), so two extension instances never
//! run the same job.
//!
//! The extension holds a job by a claim that expires, reports where in the
//! provider's conversation list it got to with running totals, and reports
//! the end. A restarted extension claims its job again and continues from
//! the reported position.

use crate::{Reader, Result, StoreError};
use polylogue_model::user::{BackfillCounts, BackfillReport, NewBackfillJob};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

/// How long a claim lasts without a report from the extension instance
/// holding it; an expired claim lets the job be claimed again.
pub const CLAIM_MS: i64 = 180_000;

/// One thing that happened to a backfill job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackfillEvent {
    pub at: i64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    pub counts: BackfillCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Where a backfill job stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackfillStatus {
    /// Started and not held by an extension instance.
    Waiting,
    Running,
    Finished,
    Failed,
    Cancelled,
}

impl BackfillStatus {
    pub fn ended(self) -> bool {
        matches!(self, Self::Finished | Self::Failed | Self::Cancelled)
    }
}

/// What the extension instance holding a job may report, and the state
/// each report leaves; `progress` renews the claim.
pub fn report_status(kind: &str) -> Option<BackfillStatus> {
    Some(match kind {
        "progress" => BackfillStatus::Running,
        "finished" => BackfillStatus::Finished,
        "failed" => BackfillStatus::Failed,
        _ => return None,
    })
}

/// A stored backfill job with everything that happened to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackfillJob {
    pub id: String,
    pub provider: String,
    pub since: Option<i64>,
    pub created_at: i64,
    pub events: Vec<BackfillEvent>,
}

impl BackfillJob {
    /// The state at `now`, and the instance holding an unexpired claim.
    pub fn status(&self, now: i64) -> (BackfillStatus, Option<&str>) {
        let mut status = BackfillStatus::Waiting;
        let mut claim: Option<(&str, i64)> = None;
        for e in &self.events {
            match e.kind.as_str() {
                "claimed" | "progress" => {
                    status = BackfillStatus::Running;
                    claim = e.by.as_deref().map(|by| (by, e.at + CLAIM_MS));
                }
                "cancelled" => {
                    status = BackfillStatus::Cancelled;
                    claim = None;
                }
                kind => {
                    if let Some(next) = report_status(kind) {
                        status = next;
                        claim = None;
                    }
                }
            }
        }
        match (status, claim) {
            (BackfillStatus::Running, Some((by, until))) if until >= now => (status, Some(by)),
            (BackfillStatus::Running, _) => (BackfillStatus::Waiting, None),
            _ => (status, None),
        }
    }

    /// The totals and list position of the latest report.
    pub fn progress(&self) -> (BackfillCounts, Option<&str>) {
        let counts = self.events.last().map(|e| e.counts).unwrap_or_default();
        let position = self.events.iter().rev().find_map(|e| e.position.as_deref());
        (counts, position)
    }
}

impl Reader<'_> {
    /// Every backfill job, oldest first.
    pub fn backfill_jobs(&self) -> Result<Vec<BackfillJob>> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn
            .prepare_cached("SELECT id FROM a.backfill_job ORDER BY created_at, id")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        ids.iter()
            .map(|id| {
                read(conn, "a.", id)?
                    .ok_or_else(|| StoreError::NotFound(format!("backfill job {id}")))
            })
            .collect()
    }

    pub fn backfill_job(&self, id: &str) -> Result<Option<BackfillJob>> {
        read(self.conn()?, "a.", id)
    }
}

/// One backfill job from the archive tables under `schema` (`a.` when the
/// archive is attached, empty on the archive connection).
pub(crate) fn read(conn: &Connection, schema: &str, id: &str) -> Result<Option<BackfillJob>> {
    let row = conn
        .prepare_cached(&format!(
            "SELECT provider, since, created_at FROM {schema}backfill_job WHERE id = ?1"
        ))?
        .query_row([id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })
        .optional()?;
    let Some((provider, since, created_at)) = row else {
        return Ok(None);
    };
    let count = |n: i64| u64::try_from(n).unwrap_or(0);
    let events = conn
        .prepare_cached(&format!(
            "SELECT at, kind, by, position, listed, captured, unchanged, failed, detail
             FROM {schema}backfill_job_event WHERE job = ?1 ORDER BY n"
        ))?
        .query_map([id], |r| {
            Ok(BackfillEvent {
                at: r.get(0)?,
                kind: r.get(1)?,
                by: r.get(2)?,
                position: r.get(3)?,
                counts: BackfillCounts {
                    listed: count(r.get(4)?),
                    captured: count(r.get(5)?),
                    unchanged: count(r.get(6)?),
                    failed: count(r.get(7)?),
                },
                detail: r.get(8)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(Some(BackfillJob {
        id: id.to_owned(),
        provider,
        since,
        created_at,
        events,
    }))
}

/// A refused change to a backfill job, by a stable code.
fn refused(code: &'static str) -> StoreError {
    StoreError::Backfill(code)
}

/// Starts a job, or finds the provider's job that has not ended: one runs
/// per provider at a time.
pub(crate) fn start(tx: &Connection, new: &NewBackfillJob, at: i64, seq: i64) -> Result<String> {
    let ids: Vec<String> = tx
        .prepare_cached("SELECT id FROM backfill_job WHERE provider = ?1 ORDER BY created_at, id")?
        .query_map([&new.provider], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        if let Some(job) = read(tx, "", &id)?
            && !job.status(at).0.ended()
        {
            return Ok(id);
        }
    }
    tx.execute(
        "INSERT INTO backfill_job(id, provider, since, created_at, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![new.id, new.provider, new.since, at, seq],
    )?;
    let event = BackfillEvent {
        at,
        kind: "started".into(),
        by: None,
        position: None,
        counts: BackfillCounts::default(),
        detail: None,
    };
    add_event(tx, &new.id, &event, seq)?;
    Ok(new.id.clone())
}

/// Hands `by` the job it already holds, or the oldest waiting one.
pub(crate) fn claim(tx: &Connection, by: &str, at: i64, seq: i64) -> Result<String> {
    let ids: Vec<String> = tx
        .prepare_cached("SELECT id FROM backfill_job ORDER BY created_at, id")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut next = None;
    for id in ids {
        let Some(job) = read(tx, "", &id)? else {
            continue;
        };
        match job.status(at) {
            (BackfillStatus::Running, Some(holder)) if holder == by => return Ok(id),
            (BackfillStatus::Waiting, _) if next.is_none() => next = Some(job),
            _ => {}
        }
    }
    let job = next.ok_or_else(|| StoreError::NotFound("a waiting backfill job".into()))?;
    let (counts, position) = job.progress();
    let event = BackfillEvent {
        at,
        kind: "claimed".into(),
        by: Some(by.to_owned()),
        position: position.map(str::to_owned),
        counts,
        detail: None,
    };
    add_event(tx, &job.id, &event, seq)?;
    Ok(job.id)
}

/// A report from the extension instance holding the claim.
pub(crate) fn report(
    tx: &Connection,
    id: &str,
    by: &str,
    report: &BackfillReport,
    at: i64,
    seq: i64,
) -> Result<()> {
    let job = read(tx, "", id)?.ok_or_else(|| refused("unknown_backfill_job"))?;
    if report_status(&report.kind).is_none() {
        return Err(refused("unknown_report"));
    }
    let (status, holder) = job.status(at);
    if status.ended() {
        return Err(refused("job_ended"));
    }
    if holder != Some(by) {
        return Err(refused("not_the_claimant"));
    }
    let event = BackfillEvent {
        at,
        kind: report.kind.clone(),
        by: Some(by.to_owned()),
        position: report.position.clone(),
        counts: report.counts,
        detail: report.detail.clone(),
    };
    add_event(tx, id, &event, seq)
}

/// The operator stops a job.
pub(crate) fn cancel(tx: &Connection, id: &str, at: i64, seq: i64) -> Result<()> {
    let job = read(tx, "", id)?.ok_or_else(|| refused("unknown_backfill_job"))?;
    if job.status(at).0.ended() {
        return Err(refused("job_ended"));
    }
    let (counts, position) = job.progress();
    let event = BackfillEvent {
        at,
        kind: "cancelled".into(),
        by: Some("operator".into()),
        position: position.map(str::to_owned),
        counts,
        detail: None,
    };
    add_event(tx, id, &event, seq)
}

fn add_event(tx: &Connection, id: &str, e: &BackfillEvent, seq: i64) -> Result<()> {
    let count = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
    tx.execute(
        "INSERT INTO backfill_job_event(job, n, at, kind, by, position, listed, captured,
             unchanged, failed, detail, commit_seq)
         VALUES (?1, (SELECT COALESCE(max(n), 0) + 1 FROM backfill_job_event WHERE job = ?1),
                 ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            id,
            e.at,
            e.kind,
            e.by,
            e.position,
            count(e.counts.listed),
            count(e.counts.captured),
            count(e.counts.unchanged),
            count(e.counts.failed),
            e.detail,
            seq
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;

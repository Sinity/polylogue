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
use rusqlite::{Connection, OptionalExtension, params};

use polylogue_model::user::{
    BackfillCounts, BackfillEvent, BackfillJob, BackfillReport, BackfillStatus, NewBackfillJob,
    backfill_report_status,
};

impl Reader<'_> {
    /// Every backfill job, oldest first.
    pub fn backfill_jobs(&self) -> Result<Vec<BackfillJob>> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn
            .prepare_cached("SELECT id FROM source_backfill_job ORDER BY created_at, id")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        ids.iter()
            .map(|id| {
                read(conn, "source_", id)?
                    .ok_or_else(|| StoreError::NotFound(format!("backfill job {id}")))
            })
            .collect()
    }

    pub fn backfill_job(&self, id: &str) -> Result<Option<BackfillJob>> {
        read(self.conn()?, "source_", id)
    }
}

/// One backfill job from the archive tables under `schema` (`a.` when the
/// archive is attached, empty on the archive connection).
pub(crate) fn read(conn: &Connection, schema: &str, id: &str) -> Result<Option<BackfillJob>> {
    let row = conn
        .prepare_cached(&format!(
            "SELECT provider, since, created_at, execution FROM {schema}backfill_job WHERE id = ?1"
        ))?
        .query_row([id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .optional()?;
    let Some((provider, since, created_at, execution)) = row else {
        return Ok(None);
    };
    let count = |n: i64| u64::try_from(n).unwrap_or(0);
    let events = conn
        .prepare_cached(&format!(
            "SELECT at, kind, by, position, listed, captured, unchanged, failed, detail, request
             FROM {schema}backfill_job_event WHERE job = ?1 ORDER BY n"
        ))?
        .query_map([id], |r| {
            Ok(BackfillEvent {
                request: r.get(9)?,
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
        execution: polylogue_model::authored::ExecutionEligibility::from_name(&execution)
            .ok_or_else(|| StoreError::Invalid("unknown execution eligibility".into()))?,
        events,
    }))
}

/// A refused change to a backfill job, by a stable code.
fn refused(code: &'static str) -> StoreError {
    StoreError::Backfill(code)
}

/// Starts a job, or finds the provider's job that has not ended: one runs
/// per provider at a time.
pub(crate) fn start(
    tx: &Connection,
    new: &NewBackfillJob,
    at: i64,
    seq: i64,
    request: &str,
) -> Result<String> {
    let source = crate::source::Source::of(tx)?;
    let ids: Vec<String> = tx
        .prepare_cached(&source.sql(
            "SELECT id FROM @source.backfill_job WHERE provider = ?1 AND execution='live' ORDER BY created_at, id",
        ))?
        .query_map([&new.provider], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        if let Some(job) = read(tx, source.prefix(), &id)?
            && !job.status(at).0.ended()
        {
            return Ok(id);
        }
    }
    tx.execute(
        &source.sql(
            "INSERT INTO @source.backfill_job(id, provider, since, created_at, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        ),
        params![new.id, new.provider, new.since, at, seq],
    )?;
    let event = BackfillEvent {
        request: request.into(),
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
pub(crate) fn claim(tx: &Connection, by: &str, at: i64, seq: i64, request: &str) -> Result<String> {
    let source = crate::source::Source::of(tx)?;
    let ids: Vec<String> = tx
        .prepare_cached(&source.sql(
            "SELECT id FROM @source.backfill_job WHERE execution='live' ORDER BY created_at, id",
        ))?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut next = None;
    for id in ids {
        let Some(job) = read(tx, source.prefix(), &id)? else {
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
        request: request.into(),
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
    request: &str,
) -> Result<()> {
    let source = crate::source::Source::of(tx)?;
    let job = read(tx, source.prefix(), id)?.ok_or_else(|| refused("unknown_backfill_job"))?;
    if backfill_report_status(&report.kind).is_none() {
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
        request: request.into(),
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
pub(crate) fn cancel(tx: &Connection, id: &str, at: i64, seq: i64, request: &str) -> Result<()> {
    let source = crate::source::Source::of(tx)?;
    let job = read(tx, source.prefix(), id)?.ok_or_else(|| refused("unknown_backfill_job"))?;
    if job.status(at).0.ended() {
        return Err(refused("job_ended"));
    }
    let (counts, position) = job.progress();
    let event = BackfillEvent {
        request: request.into(),
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
    let source = crate::source::Source::of(tx)?;
    let count = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
    tx.execute(
        &source.sql("INSERT INTO @source.backfill_job_event(job, n, at, kind, by, position, listed, captured,
             unchanged, failed, detail, request, commit_seq)
         VALUES (?1, (SELECT COALESCE(max(n), 0) + 1 FROM @source.backfill_job_event WHERE job = ?1),
                 ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"),
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
            e.request,
            seq
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;

//! Import jobs: the operator asks the browser extension to capture a web
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
    ImportJob, ImportJobCounts, ImportJobEvent, ImportJobReport, ImportJobStatus, NewImportJob,
    import_job_report_status,
};

impl Reader<'_> {
    /// Every import job, oldest first.
    pub fn import_jobs(&self) -> Result<Vec<ImportJob>> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn
            .prepare_cached("SELECT id FROM source_import_job ORDER BY created_at, id")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        ids.iter()
            .map(|id| {
                read(conn, "source_", id)?
                    .ok_or_else(|| StoreError::NotFound(format!("import job {id}")))
            })
            .collect()
    }

    pub fn import_job(&self, id: &str) -> Result<Option<ImportJob>> {
        read(self.conn()?, "source_", id)
    }
}

/// One import job from the archive tables, whose names start with `table_prefix`
/// (`source_` on the cache connection, empty on a standalone archive).
pub(crate) fn read(conn: &Connection, table_prefix: &str, id: &str) -> Result<Option<ImportJob>> {
    let row = conn
        .prepare_cached(&format!(
            "SELECT provider, since, created_at, execution FROM {table_prefix}import_job WHERE id = ?1"
        ))?
        .query_row([id], |r| <(String, Option<i64>, i64, String)>::try_from(r))
        .optional()?;
    let Some((provider, since, created_at, execution)) = row else {
        return Ok(None);
    };
    let events = conn
        .prepare_cached(&format!(
            "SELECT at, kind, by, provider_position, listed, captured, unchanged, failed, detail, request
             FROM {table_prefix}import_job_event WHERE job = ?1 ORDER BY position"
        ))?
        .query_map([id], |r| {
            Ok(ImportJobEvent {
                request: r.get(9)?,
                at: r.get(0)?,
                kind: r.get(1)?,
                by: r.get(2)?,
                position: r.get(3)?,
                counts: ImportJobCounts {
                    listed: r.get(4)?,
                    captured: r.get(5)?,
                    unchanged: r.get(6)?,
                    failed: r.get(7)?,
                },
                detail: r.get(8)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(Some(ImportJob {
        id: id.to_owned(),
        provider,
        since,
        created_at,
        execution: polylogue_model::authored::RunsHere::from_name(&execution)
            .ok_or_else(|| StoreError::stored(3, "unknown execution eligibility"))?,
        events,
    }))
}

/// A refused change to an import job, by a stable code.
fn refused(code: &'static str) -> StoreError {
    StoreError::ImportJob(code)
}

/// Starts a job, or finds the provider's job that has not ended: one runs
/// per provider at a time. Finding a paused job resumes it from its saved
/// position and totals.
pub(crate) fn start(
    tx: &Connection,
    new: &NewImportJob,
    at: i64,
    seq: i64,
    request: &str,
) -> Result<String> {
    let source = crate::source::Source::of(tx)?;
    let ids: Vec<String> = tx
        .prepare_cached(&source.sql(
            "SELECT id FROM @source.import_job WHERE provider = ?1 AND execution='live' ORDER BY created_at, id",
        ))?
        .query_map([&new.provider], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        let Some(job) = read(tx, source.prefix(), &id)? else {
            continue;
        };
        match job.status(at).0 {
            status if status.ended() => {}
            ImportJobStatus::Paused => {
                let (counts, position) = job.progress();
                let event = ImportJobEvent {
                    request: request.into(),
                    at,
                    kind: "resumed".into(),
                    by: None,
                    position: position.map(str::to_owned),
                    counts,
                    detail: None,
                };
                add_event(tx, &id, &event, seq)?;
                return Ok(id);
            }
            _ => return Ok(id),
        }
    }
    tx.execute(
        &source.sql(
            "INSERT INTO @source.import_job(id, provider, since, created_at, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        ),
        params![new.id, new.provider, new.since, at, seq],
    )?;
    let event = ImportJobEvent {
        request: request.into(),
        at,
        kind: "started".into(),
        by: None,
        position: None,
        counts: ImportJobCounts::default(),
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
            "SELECT id FROM @source.import_job WHERE execution='live' ORDER BY created_at, id",
        ))?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut next = None;
    for id in ids {
        let Some(job) = read(tx, source.prefix(), &id)? else {
            continue;
        };
        match job.status(at) {
            (ImportJobStatus::Running, Some(holder)) if holder == by => return Ok(id),
            (ImportJobStatus::Waiting, _) if next.is_none() => next = Some(job),
            _ => {}
        }
    }
    let job = next.ok_or_else(|| StoreError::NotFound("a waiting import job".into()))?;
    let (counts, position) = job.progress();
    let event = ImportJobEvent {
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
    report: &ImportJobReport,
    at: i64,
    seq: i64,
    request: &str,
) -> Result<()> {
    let source = crate::source::Source::of(tx)?;
    let job = read(tx, source.prefix(), id)?.ok_or_else(|| refused("unknown_import_job"))?;
    if import_job_report_status(&report.kind).is_none() {
        return Err(refused("unknown_report"));
    }
    let (status, holder) = job.status(at);
    if status.ended() {
        return Err(refused("job_ended"));
    }
    if holder != Some(by) {
        return Err(refused("not_the_claimant"));
    }
    let event = ImportJobEvent {
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
    let job = read(tx, source.prefix(), id)?.ok_or_else(|| refused("unknown_import_job"))?;
    if job.status(at).0.ended() {
        return Err(refused("job_ended"));
    }
    let (counts, position) = job.progress();
    let event = ImportJobEvent {
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

fn add_event(tx: &Connection, id: &str, e: &ImportJobEvent, seq: i64) -> Result<()> {
    let source = crate::source::Source::of(tx)?;
    tx.execute(
        &source.sql(
            "INSERT INTO @source.import_job_event(job, position, at, kind, by, provider_position, listed, captured,
             unchanged, failed, detail, request, commit_seq)
         VALUES (?1, (SELECT COALESCE(max(position), 0) + 1 FROM @source.import_job_event WHERE job = ?1),
                 ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        ),
        params![
            id,
            e.at,
            e.kind,
            e.by,
            e.position,
            e.counts.listed,
            e.counts.captured,
            e.counts.unchanged,
            e.counts.failed,
            e.detail,
            e.request,
            seq
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;

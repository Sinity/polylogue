use super::*;
use crate::{Archive, ReadPool, Writer};
use polylogue_model::user::IMPORT_JOB_CLAIM_MS;
use polylogue_model::user::UserWrite;

fn setup() -> (
    tempfile::TempDir,
    Archive,
    crate::ArchiveLock,
    Writer,
    ReadPool,
) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let reads = ReadPool::open(&archive).unwrap();
    (dir, archive, lock, writer, reads)
}

fn write(writer: &Writer, write: UserWrite) -> Result<polylogue_model::user::UserOutcome> {
    writer
        .archive()
        .user(crate::fixture::authored::operator_request(
            crate::user_writes::new_request_id(),
            write,
        ))
}

fn start(writer: &Writer, id: &str, provider: &str) -> polylogue_model::user::UserOutcome {
    write(
        writer,
        UserWrite::ImportJobStart(NewImportJob {
            id: id.into(),
            provider: provider.into(),
            since: None,
        }),
    )
    .unwrap()
}

fn progress(position: &str, captured: u64) -> ImportJobReport {
    ImportJobReport {
        kind: "progress".into(),
        position: Some(position.into()),
        counts: ImportJobCounts {
            listed: captured + 1,
            captured,
            unchanged: 1,
            failed: 0,
        },
        detail: None,
    }
}

#[test]
fn one_job_runs_per_provider_and_a_restarted_instance_continues_where_it_reported() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    assert_eq!(start(&writer, "j1", "chatgpt").id, "j1");
    let again = start(&writer, "j9", "chatgpt");
    assert_eq!((again.id.as_str(), again.repeat), ("j1", true));
    start(&writer, "j2", "claude-ai");

    let claimed = write(&writer, UserWrite::ImportJobClaim { by: "ext-1".into() }).unwrap();
    assert_eq!(claimed.id, "j1");
    write(
        &writer,
        UserWrite::ImportJobReport {
            job: "j1".into(),
            by: "ext-1".into(),
            report: progress("0:56", 3),
        },
    )
    .unwrap();
    // Another instance cannot report on a job it does not hold.
    assert!(matches!(
        write(
            &writer,
            UserWrite::ImportJobReport {
                job: "j1".into(),
                by: "ext-2".into(),
                report: progress("0:84", 9),
            }
        ),
        Err(StoreError::ImportJob("not_the_claimant"))
    ));
    // The instance that holds a job gets the same job back after a restart.
    let reclaimed = write(&writer, UserWrite::ImportJobClaim { by: "ext-1".into() }).unwrap();
    assert_eq!(reclaimed.id, "j1");
    let job = reads.reader().unwrap().import_job("j1").unwrap().unwrap();
    let (counts, position) = job.progress();
    assert_eq!((counts.captured, position), (3, Some("0:56")));
    assert_eq!(
        job.status(job.events.last().unwrap().at).0,
        ImportJobStatus::Running
    );

    // A second instance takes the other provider's job.
    let other = write(&writer, UserWrite::ImportJobClaim { by: "ext-2".into() }).unwrap();
    assert_eq!(other.id, "j2");
    assert!(matches!(
        write(&writer, UserWrite::ImportJobClaim { by: "ext-3".into() }),
        Err(StoreError::NotFound(message)) if message == "a waiting import job"
    ));

    let mut finished = progress("done", 7);
    finished.kind = "finished".into();
    write(
        &writer,
        UserWrite::ImportJobReport {
            job: "j1".into(),
            by: "ext-1".into(),
            report: finished,
        },
    )
    .unwrap();
    let job = reads.reader().unwrap().import_job("j1").unwrap().unwrap();
    assert_eq!(job.status(i64::MAX).0, ImportJobStatus::Finished);
    // A finished job lets the provider start a new one.
    assert_eq!(start(&writer, "j3", "chatgpt").id, "j3");
}

#[test]
fn an_expired_claim_lets_the_job_wait_and_the_operator_can_cancel_it() {
    let job = ImportJob {
        id: "j".into(),
        provider: "grok".into(),
        since: None,
        created_at: 0,
        execution: polylogue_model::authored::RunsHere::Live,
        events: vec![
            ImportJobEvent {
                request: "event-request".into(),
                at: 0,
                kind: "started".into(),
                by: None,
                position: None,
                counts: ImportJobCounts::default(),
                detail: None,
            },
            ImportJobEvent {
                request: "event-request".into(),
                at: 10,
                kind: "claimed".into(),
                by: Some("ext".into()),
                position: None,
                counts: ImportJobCounts::default(),
                detail: None,
            },
        ],
    };
    assert_eq!(
        job.status(10 + IMPORT_JOB_CLAIM_MS),
        (ImportJobStatus::Running, Some("ext"))
    );
    assert_eq!(
        job.status(11 + IMPORT_JOB_CLAIM_MS),
        (ImportJobStatus::Waiting, None)
    );

    let (_dir, _archive, _lock, writer, reads) = setup();
    start(&writer, "j1", "grok");
    write(&writer, UserWrite::ImportJobCancel { job: "j1".into() }).unwrap();
    let job = reads.reader().unwrap().import_job("j1").unwrap().unwrap();
    assert_eq!(job.status(i64::MAX).0, ImportJobStatus::Cancelled);
    assert!(matches!(
        write(&writer, UserWrite::ImportJobCancel { job: "j1".into() }),
        Err(StoreError::ImportJob("job_ended"))
    ));
    assert!(matches!(
        write(&writer, UserWrite::ImportJobClaim { by: "ext".into() }),
        Err(StoreError::NotFound(message)) if message == "a waiting import job"
    ));
}

#[test]
fn a_paused_job_is_not_claimed_until_the_import_is_started_again_and_then_resumes_where_it_stopped()
{
    let (_dir, _archive, _lock, writer, reads) = setup();
    start(&writer, "j1", "claude-ai");
    write(&writer, UserWrite::ImportJobClaim { by: "ext-1".into() }).unwrap();
    let mut paused = progress("200", 150);
    paused.kind = "paused".into();
    paused.detail = Some("Sign in to claude-ai in this browser".into());
    write(
        &writer,
        UserWrite::ImportJobReport {
            job: "j1".into(),
            by: "ext-1".into(),
            report: paused,
        },
    )
    .unwrap();
    let job = reads.reader().unwrap().import_job("j1").unwrap().unwrap();
    assert_eq!(job.status(i64::MAX), (ImportJobStatus::Paused, None));
    assert!(matches!(
        write(&writer, UserWrite::ImportJobClaim { by: "ext-1".into() }),
        Err(StoreError::NotFound(message)) if message == "a waiting import job"
    ));

    // Starting the provider's import again finds the paused job and resumes it.
    let again = start(&writer, "j2", "claude-ai");
    assert_eq!((again.id.as_str(), again.repeat), ("j1", true));
    let job = reads.reader().unwrap().import_job("j1").unwrap().unwrap();
    assert_eq!(job.status(i64::MAX), (ImportJobStatus::Waiting, None));
    let claimed = write(&writer, UserWrite::ImportJobClaim { by: "ext-2".into() }).unwrap();
    assert_eq!(claimed.id, "j1");
    let job = reads.reader().unwrap().import_job("j1").unwrap().unwrap();
    let (counts, position) = job.progress();
    assert_eq!(
        (counts.listed, counts.captured, counts.unchanged, position),
        (151, 150, 1, Some("200"))
    );
}

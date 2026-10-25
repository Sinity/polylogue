use super::*;
use crate::user_writes::{UserRequest, UserWrite};
use crate::{Archive, ReadPool, Writer};

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

fn write(writer: &Writer, write: UserWrite) -> Result<crate::UserOutcome> {
    writer.user(UserRequest::now(
        crate::user_writes::new_request_id(),
        write,
    ))
}

fn start(writer: &Writer, id: &str, provider: &str) -> crate::UserOutcome {
    write(
        writer,
        UserWrite::BackfillStart(NewBackfillJob {
            id: id.into(),
            provider: provider.into(),
            since: None,
        }),
    )
    .unwrap()
}

fn progress(position: &str, captured: u64) -> BackfillReport {
    BackfillReport {
        kind: "progress".into(),
        position: Some(position.into()),
        counts: BackfillCounts {
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

    let claimed = write(&writer, UserWrite::BackfillClaim { by: "ext-1".into() }).unwrap();
    assert_eq!(claimed.id, "j1");
    write(
        &writer,
        UserWrite::BackfillReport {
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
            UserWrite::BackfillReport {
                job: "j1".into(),
                by: "ext-2".into(),
                report: progress("0:84", 9),
            }
        ),
        Err(StoreError::Backfill("not_the_claimant"))
    ));
    // The instance that holds a job gets the same job back after a restart.
    let reclaimed = write(&writer, UserWrite::BackfillClaim { by: "ext-1".into() }).unwrap();
    assert_eq!(reclaimed.id, "j1");
    let job = reads.reader().unwrap().backfill_job("j1").unwrap().unwrap();
    let (counts, position) = job.progress();
    assert_eq!((counts.captured, position), (3, Some("0:56")));
    assert_eq!(job.status(job.events.last().unwrap().at).0, BackfillStatus::Running);

    // A second instance takes the other provider's job.
    let other = write(&writer, UserWrite::BackfillClaim { by: "ext-2".into() }).unwrap();
    assert_eq!(other.id, "j2");
    assert!(matches!(
        write(&writer, UserWrite::BackfillClaim { by: "ext-3".into() }),
        Err(StoreError::NotFound(_))
    ));

    let mut finished = progress("done", 7);
    finished.kind = "finished".into();
    write(
        &writer,
        UserWrite::BackfillReport {
            job: "j1".into(),
            by: "ext-1".into(),
            report: finished,
        },
    )
    .unwrap();
    let job = reads.reader().unwrap().backfill_job("j1").unwrap().unwrap();
    assert_eq!(job.status(i64::MAX).0, BackfillStatus::Finished);
    // A finished job lets the provider start a new one.
    assert_eq!(start(&writer, "j3", "chatgpt").id, "j3");
}

#[test]
fn an_expired_claim_lets_the_job_wait_and_the_operator_can_cancel_it() {
    let job = BackfillJob {
        id: "j".into(),
        provider: "grok".into(),
        since: None,
        created_at: 0,
        events: vec![
            BackfillEvent {
                at: 0,
                kind: "started".into(),
                by: None,
                position: None,
                counts: BackfillCounts::default(),
                detail: None,
            },
            BackfillEvent {
                at: 10,
                kind: "claimed".into(),
                by: Some("ext".into()),
                position: None,
                counts: BackfillCounts::default(),
                detail: None,
            },
        ],
    };
    assert_eq!(job.status(10 + CLAIM_MS), (BackfillStatus::Running, Some("ext")));
    assert_eq!(job.status(11 + CLAIM_MS), (BackfillStatus::Waiting, None));

    let (_dir, _archive, _lock, writer, reads) = setup();
    start(&writer, "j1", "grok");
    write(&writer, UserWrite::BackfillCancel { job: "j1".into() }).unwrap();
    let job = reads.reader().unwrap().backfill_job("j1").unwrap().unwrap();
    assert_eq!(job.status(i64::MAX).0, BackfillStatus::Cancelled);
    assert!(matches!(
        write(&writer, UserWrite::BackfillCancel { job: "j1".into() }),
        Err(StoreError::Backfill("job_ended"))
    ));
    assert!(matches!(
        write(&writer, UserWrite::BackfillClaim { by: "ext".into() }),
        Err(StoreError::NotFound(_))
    ));
}

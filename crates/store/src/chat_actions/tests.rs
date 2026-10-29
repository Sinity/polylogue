use super::*;
use crate::{Archive, ReadPool, Writer};
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
    writer.user(crate::user_writes::request_now(
        crate::user_writes::new_request_id(),
        write,
    ))
}

fn new(id: &str, key: &str, needs_approval: bool) -> NewChatAction {
    NewChatAction {
        id: id.into(),
        request_key: key.into(),
        request_hash: BlobHash::of(key.as_bytes()),
        request: serde_json::json!({"text": key}),
        stage_only: !needs_approval,
        needs_approval,
        attachments: vec![],
    }
}

#[test]
fn one_action_is_out_at_a_time_and_a_resent_request_is_the_same_action() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    let first = write(&writer, UserWrite::ChatEnqueue(new("a1", "k1", false))).unwrap();
    assert_eq!((first.id.as_str(), first.repeat), ("a1", false));
    let again = write(&writer, UserWrite::ChatEnqueue(new("a9", "k1", false))).unwrap();
    assert_eq!((again.id.as_str(), again.repeat), ("a1", true));
    let mut other = new("a9", "k1", false);
    other.request_hash = BlobHash::of(b"other content");
    assert!(matches!(
        write(&writer, UserWrite::ChatEnqueue(other)),
        Err(StoreError::ChatAction("idempotency_key_reused"))
    ));
    write(&writer, UserWrite::ChatEnqueue(new("a2", "k2", false))).unwrap();

    let claimed = write(&writer, UserWrite::ChatClaim { by: "tab-1".into() }).unwrap();
    assert_eq!(claimed.id, "a1");
    assert!(matches!(
        write(&writer, UserWrite::ChatClaim { by: "tab-2".into() }),
        Err(StoreError::NotFound(_))
    ));
    let report = |by: &str, outcome: &str| {
        write(
            &writer,
            UserWrite::ChatReport {
                action: "a1".into(),
                by: by.into(),
                outcome: outcome.into(),
                phase: "composer".into(),
                detail: None,
                receipt: None,
            },
        )
    };
    assert!(matches!(
        report("tab-2", "progress"),
        Err(StoreError::ChatAction("not_the_claimant"))
    ));
    assert!(matches!(
        report("tab-1", "submitted"),
        Err(StoreError::ChatAction("stage_only_action_cannot_submit"))
    ));
    report("tab-1", "drafted").unwrap();
    assert_eq!(
        write(&writer, UserWrite::ChatClaim { by: "tab-2".into() })
            .unwrap()
            .id,
        "a2"
    );
    let r = reads.reader().unwrap();
    let a1 = r.chat_action("a1").unwrap().unwrap();
    assert_eq!(a1.status(i64::MAX / 2).0, ChatStatus::Drafted);
}

#[test]
fn an_expired_claim_returns_the_action_to_the_queue() {
    let action = ChatAction {
        id: "a".into(),
        request_key: "k".into(),
        request_hash: BlobHash::of(b"k"),
        request: serde_json::Value::Null,
        stage_only: true,
        needs_approval: false,
        created_at: 0,
        attachments: vec![],
        events: vec![ChatEvent {
            request: "event-request".into(),
            at: 1_000,
            kind: "claimed".into(),
            phase: "claimed".into(),
            detail: None,
            by: Some("tab".into()),
            receipt: None,
        }],
    };
    assert_eq!(
        action.status(1_000 + CLAIM_MS),
        (ChatStatus::Claimed, Some("tab"))
    );
    assert_eq!(action.status(1_001 + CLAIM_MS), (ChatStatus::Queued, None));
}

#[test]
fn a_held_submit_waits_for_the_operator() {
    let (_dir, _archive, _lock, writer, reads) = setup();
    write(&writer, UserWrite::ChatEnqueue(new("a1", "k1", true))).unwrap();
    assert!(matches!(
        write(&writer, UserWrite::ChatClaim { by: "tab".into() }),
        Err(StoreError::NotFound(_))
    ));
    let decide = |approve| {
        write(
            &writer,
            UserWrite::ChatDecide {
                action: "a1".into(),
                approve,
                by: "operator".into(),
                detail: None,
            },
        )
    };
    decide(true).unwrap();
    assert!(matches!(
        decide(false),
        Err(StoreError::ChatAction("not_awaiting_approval"))
    ));
    assert_eq!(
        write(&writer, UserWrite::ChatClaim { by: "tab".into() })
            .unwrap()
            .id,
        "a1"
    );
    let r = reads.reader().unwrap();
    let kinds: Vec<String> = r
        .chat_action("a1")
        .unwrap()
        .unwrap()
        .events
        .into_iter()
        .map(|e| e.kind)
        .collect();
    assert_eq!(kinds, ["awaiting_approval", "approved", "claimed"]);
}

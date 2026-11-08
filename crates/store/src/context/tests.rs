use super::*;
use crate::fixture::{authored::operator_request, writer::WriterArchive};
use polylogue_model::user::{Delivery, UserWrite};
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn old_unacknowledged_notices_are_scoped_metadata_until_selected() {
    let archive = WriterArchive::new();
    let deliver = |id: &str, cwd: Option<&str>| {
        archive
            .writer
            .user(operator_request(
                id.into(),
                UserWrite::Deliver(Delivery {
                    recipient_kind: "operator".into(),
                    recipient_id: None,
                    sender: "signal".into(),
                    event: "signal".into(),
                    cwd: cwd.map(str::to_owned),
                    text: format!("{id} {}", "body".repeat(10000)),
                    refs: vec![],
                    left_out: vec![],
                }),
            ))
            .unwrap();
    };
    deliver("old-pending", Some("/work/proj/child"));
    deliver("sibling", Some("/work/project"));
    deliver("acknowledged", Some("/work/proj"));
    deliver("unknown-scope", None);
    archive
        .writer
        .user(operator_request(
            "ack-old".into(),
            UserWrite::DeliveryEvent {
                delivery: "acknowledged".into(),
                what: polylogue_model::DeliveryOutcome::Delivered,
                detail: None,
            },
        ))
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let (candidates, inspect_count) = reader
        .pending_notice_candidates(&["/work/proj".into()], None)
        .unwrap();
    assert_eq!(inspect_count, 1);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<Vec<_>>(),
        ["old-pending"]
    );
    assert_eq!(candidates[0].at, Timestamp::from_unix_ms(1).unwrap());
    assert!(
        reader
            .delivery(&candidates[0].id)
            .unwrap()
            .unwrap()
            .1
            .starts_with("old-pending body")
    );
}

#[test]
fn latest_request_keeps_unknown_authorship_and_skips_injection_commands_and_old_branches() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:requests",
            ..Default::default()
        })
        .unwrap();
    for (id, by, kind, active, text) in [
        ("test:human", "human", "prose", true, "original human text"),
        (
            "test:unknown-user",
            "unknown",
            "prose",
            true,
            "current observed user text\nPreserve these spaces.  ",
        ),
        (
            "test:injected",
            "harness",
            "injected_context",
            true,
            "injected context is not a request",
        ),
        (
            "test:branch",
            "human",
            "prose",
            false,
            "off-branch request is not current",
        ),
        (
            "test:command",
            "human",
            "command",
            true,
            "/an-observed-command",
        ),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: by,
                    prompt_kind: Some(kind),
                    on_active_path: Some(active),
                    text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prompt = reader.latest_prompt("test:requests").unwrap().unwrap();
    assert_eq!(prompt.message_id, "test:unknown-user");
    assert_eq!(prompt.written_by, "unknown");
    assert_eq!(
        prompt.text,
        "current observed user text\nPreserve these spaces.  "
    );
}

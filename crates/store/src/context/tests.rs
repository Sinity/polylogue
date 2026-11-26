use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use crate::fixture::{authored::operator_request, writer::WriterArchive};
use polylogue_model::user::{Delivery, UserWrite};

#[test]
fn pending_notices_are_selected_by_directory_scope_and_acknowledgement() {
    let archive = WriterArchive::new();
    let deliver = |id: &str, cwd: Option<&str>| {
        archive
            .writer
            .archive()
            .user(operator_request(
                id.into(),
                UserWrite::Deliver(Delivery {
                    recipient_kind: "operator".into(),
                    recipient_id: None,
                    sender: polylogue_model::DeliverySender::Signal,
                    event: polylogue_model::DeliveryEvent::Signal,
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
        .archive()
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
    assert_eq!(
        reader.delivery(&candidates[0].id).unwrap().unwrap().1,
        format!("old-pending {}", "body".repeat(10000))
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
            "harness",
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

#[test]
fn open_items_are_worded_once_in_flag_order() {
    let open = OpenItems {
        unanswered_prompt: true,
        open_tool_call: true,
        todo: Some("write docs".into()),
        waiting_on: Some("permission".into()),
    };
    assert_eq!(
        open.phrases(),
        [
            "the last prompt has no answer",
            "a tool call never finished",
            "todo in progress: write docs",
            "waiting on permission"
        ]
    );
    assert!(OpenItems::default().phrases().is_empty());
}

/// A stored delivery whose sender or event is outside its closed set is
/// refused as unreadable stored data, never read as a free string.
#[test]
fn delivery_sender_and_event_refuse_unknown_values() {
    let archive = WriterArchive::new();
    archive
        .writer
        .archive()
        .user(operator_request(
            "to-session".into(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "session".into(),
                recipient_id: Some("claude-code:s1".into()),
                sender: polylogue_model::DeliverySender::Send,
                event: polylogue_model::DeliveryEvent::Message,
                cwd: None,
                text: "hello".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ))
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let row = pool
        .reader()
        .unwrap()
        .deliveries_to("claude-code:s1")
        .unwrap();
    assert_eq!(row[0].sender, polylogue_model::DeliverySender::Send);
    assert_eq!(row[0].event, polylogue_model::DeliveryEvent::Message);
    let register = rusqlite::Connection::open(archive.archive.root.join("archive.db")).unwrap();
    for (column, value) in [("sender", "pager"), ("event", "takeover")] {
        register
            .execute(
                "UPDATE delivery SET sender='send', event='message' WHERE id='to-session'",
                [],
            )
            .unwrap();
        register
            .execute(
                &format!("UPDATE delivery SET {column}=?1 WHERE id='to-session'"),
                [value],
            )
            .unwrap();
        let error = pool
            .reader()
            .unwrap()
            .deliveries_to("claude-code:s1")
            .unwrap_err();
        assert!(
            matches!(error, crate::StoreError::StoredValue { .. }),
            "{column}: {error:?}"
        );
    }
}

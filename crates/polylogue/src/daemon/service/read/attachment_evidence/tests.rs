use super::*;
use crate::test_daemon;
use polylogue_model::SessionId;
use std::io::{Read, Seek, SeekFrom, Write};

fn item(reader: &Reader<'_>, message: &MessageId) -> ReadItem {
    let resolution = reader.attachment(message, 0, 0).unwrap();
    ReadItem {
        source: Source::block(
            message,
            &SessionId::parse("chatgpt:transport-session").unwrap(),
            0,
            false,
        ),
        kind: ItemKind::Attachment {
            attachment: Box::new(resolution.source),
            supplied_by: ValueState::default(),
            availability: ValueState::default(),
            link: None,
            inline_bytes: None,
            verified_hash: None,
            open: None,
        },
    }
}
fn ready(
    daemon: &test_daemon::TestDaemon,
    message: &MessageId,
    block: u64,
    index: u64,
    input: Option<&str>,
) -> Result<ReadyCall, crate::daemon::Failure> {
    let arguments = crate::protocol::attachment::Arguments {
        message: message.to_string(),
        block,
        attachment: index,
        input: input.map(str::to_owned),
        length: crate::daemon::attachment::PAGE_BYTES as u64,
    };
    daemon
        .daemon
        .ready_call(&command_surface::RunnableRequest {
            operation: "attachment".into(),
            arguments: serde_json::to_value(arguments).unwrap(),
            request_id: "synthetic-attachment-evidence".into(),
            deadline: command_surface::Deadline::Default,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .map_err(crate::daemon::Failure::from)
}

#[test]
fn stored_attachment_metadata_keeps_the_original_descriptor() {
    for bytes in [Vec::new(), vec![b'Q'; 64 * 1024 + 17]] {
        let daemon = test_daemon::start("synthetic attachment evidence");
        let message = MessageId::parse(test_daemon::assets::seed_bytes(&daemon, &bytes)).unwrap();
        let reader = daemon.daemon.reads.reader().unwrap();
        let original = item(&reader, &message);
        let prepared = prepare(&reader, original.clone(), 0, &|m, b, a, i| {
            ready(&daemon, m, b, a, i)
        })
        .unwrap();
        assert_eq!(prepared.source, original.source);
        let (
            ItemKind::Attachment {
                attachment,
                availability,
                verified_hash,
                open,
                inline_bytes,
                ..
            },
            ItemKind::Attachment {
                attachment: before, ..
            },
        ) = (&prepared.kind, &original.kind)
        else {
            unreachable!()
        };
        assert_eq!(attachment, before);
        assert!(attachment.blob.is_none());
        assert_eq!(availability.value().map(String::as_str), Some("present"));
        assert!(verified_hash.is_none());
        assert!(inline_bytes.is_none());
        let call = open
            .as_ref()
            .expect("stored attachment bytes must be reachable");
        let request: command_surface::RunnableRequest = call.request.clone();
        assert_eq!(request.operation, "attachment");
        let arguments: crate::protocol::attachment::Arguments =
            serde_json::from_value(request.arguments).unwrap();
        assert_eq!(arguments.message, message.to_string());
        assert_eq!((arguments.block, arguments.attachment), (0, 0));
        let resolution = reader.attachment(&message, 0, 0).unwrap();
        assert_eq!(
            arguments.input.as_deref(),
            Some(resolution.matches[0].input.as_str())
        );
    }
}

#[test]
fn another_slot_or_descriptor_cannot_select_a_stored_file() {
    let daemon = test_daemon::start("synthetic exact attachment slot");
    let message = MessageId::parse(test_daemon::assets::seed(&daemon)).unwrap();
    let reader = daemon.daemon.reads.reader().unwrap();
    let original = item(&reader, &message);
    let mut changed = original.clone();
    if let ItemKind::Attachment { attachment, .. } = &mut changed.kind {
        attachment.source_id = Some("file-Unrelated".into());
    }
    for (item, index) in [(original, 1), (changed, 0)] {
        let prepared = prepare(&reader, item, index, &|_, _, _, _| {
            panic!("an unrelated slot must not receive an open action")
        })
        .unwrap();
        let ItemKind::Attachment {
            availability,
            verified_hash,
            open,
            ..
        } = prepared.kind
        else {
            unreachable!()
        };
        assert!(matches!(availability, ValueState::Unavailable { .. }));
        assert!(verified_hash.is_none());
        assert!(open.is_none());
    }
}

#[test]
fn page_read_never_touches_blob_bodies() {
    let daemon = test_daemon::start("synthetic corrupt attachment evidence");
    let message = MessageId::parse(test_daemon::assets::seed(&daemon)).unwrap();
    let reader = daemon.daemon.reads.reader().unwrap();
    let original = item(&reader, &message);
    let resolution = reader.attachment(&message, 0, 0).unwrap();
    assert_eq!(resolution.state, AttachmentState::Available);
    let hash = reader.attachment_hash(&resolution.matches[0]).unwrap();
    let blobs = reader.blob_reader().unwrap();
    let extent = blobs.extent(&hashed_blobs::Hash(hash.0)).unwrap();
    let path = blobs.root().join(&extent.segment);
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let offset = extent.offset + extent.len - 1;
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut byte = [0];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[byte[0] ^ 1]).unwrap();
    file.sync_all().unwrap();
    let response = test_daemon::Call::new(
        "read",
        serde_json::json!({"target": format!("message:{message}")}),
    )
    .answer(&daemon.daemon);
    let refused = crate::daemon::attachment::read(
        crate::daemon::attachment::Range {
            message: message.clone(),
            block: 0,
            attachment: 0,
            input: None,
            length: crate::daemon::attachment::PAGE_BYTES,
            cursor: None,
        },
        &daemon.daemon,
    );
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();
    let page: ReadPage =
        serde_json::from_value(response.value().expect("metadata page").clone()).unwrap();
    let attachment = page
        .document
        .items
        .iter()
        .find(|item| matches!(item.kind, ItemKind::Attachment { .. }))
        .expect("attachment in page");
    let ItemKind::Attachment {
        attachment: descriptor,
        availability,
        verified_hash,
        open,
        ..
    } = &attachment.kind
    else {
        unreachable!()
    };
    let ItemKind::Attachment {
        attachment: before, ..
    } = &original.kind
    else {
        unreachable!()
    };
    assert_eq!(descriptor, before);
    assert_eq!(availability.value().map(String::as_str), Some("present"));
    assert!(verified_hash.is_none());
    assert!(open.is_some());
    assert!(
        matches!(refused.outcome, command_surface::Outcome::Error(_)),
        "{refused:?}"
    );
}

#[test]
fn attachment_metadata_preserves_parent_cancellation() {
    use polylogue_store::read::limit::{ReadCancellation, ReadLimit};
    let daemon = test_daemon::start("synthetic cancelled attachment evidence");
    let bytes = vec![b'X'; 64 * 1024 + 1];
    let message = MessageId::parse(test_daemon::assets::seed_bytes(&daemon, &bytes)).unwrap();
    let cancelled = ReadCancellation::default();
    ReadLimit::without_deadline(cancelled.clone()).scope(|| {
        let reader = daemon.daemon.reads.reader().unwrap();
        let original = item(&reader, &message);
        cancelled.cancel();
        let result = prepare(&reader, original, 0, &|m, b, a, i| {
            ready(&daemon, m, b, a, i)
        });
        let Err(failure) = &result else {
            panic!("cancelled attachment preparation did not refuse: {result:?}");
        };
        let kind = failure.code().to_owned();
        assert_eq!(kind, StoreError::ReadCancelled.code());
    });
}

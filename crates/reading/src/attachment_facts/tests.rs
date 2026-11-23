use super::*;
use polylogue_model::{Block, Role};
use polylogue_store::{
    ReadPool,
    fixture::writer::{WriterArchive, message, sid},
};

fn fixture() -> (WriterArchive, MessageId) {
    let mut writer = WriterArchive::new();
    let message = message(
        "file",
        "attachments",
        Role::User,
        1,
        vec![Block::File(Attachment {
            name: Some("sample.txt".into()),
            source_id: Some("file-Sample".into()),
            ..Attachment::default()
        })],
    );
    let id = message.id.clone();
    writer.write("attachments", vec![message]);
    (writer, id)
}

fn item(reader: &Reader<'_>, message: &MessageId) -> ReadItem {
    ReadItem {
        source: Source::block(message, &sid("attachments"), 0, false),
        kind: ItemKind::Attachment {
            attachment: Box::new(reader.attachment(message, 0, 0).unwrap().source),
            supplied_by: ValueState::default(),
            availability: ValueState::default(),
            link: None,
            inline_bytes: None,
            verified_hash: None,
            open: None,
        },
    }
}

#[test]
fn changed_attachment_slot_is_unavailable() {
    let (writer, message) = fixture();
    let pool = ReadPool::open(&writer.archive).unwrap();
    let reader = pool.reader().unwrap();
    let original = item(&reader, &message);
    let mut changed = original.clone();
    if let ItemKind::Attachment { attachment, .. } = &mut changed.kind {
        attachment.source_id = Some("file-Unrelated".into());
    }
    for (item, index, reason) in [
        (original, 1, "selected block has no such attachment"),
        (
            changed,
            0,
            "request attachment descriptor differs from the selected source was already used for different content",
        ),
    ] {
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
        assert_eq!(
            availability,
            ValueState::Unavailable {
                reason: reason.into()
            }
        );
        assert_eq!(verified_hash, None);
        assert_eq!(open, None);
    }
}

#[test]
fn attachment_metadata_obeys_cancellation() {
    use polylogue_store::{cancellation::Cancellation, read::limit::ReadLimit};
    let (writer, message) = fixture();
    let cancelled = Cancellation::default();
    ReadLimit::without_deadline(cancelled.clone()).scope(|| {
        let pool = ReadPool::open(&writer.archive).unwrap();
        let reader = pool.reader().unwrap();
        let original = item(&reader, &message);
        cancelled.cancel();
        let result = prepare(&reader, original, 0, &|_, _, _, _| {
            panic!("a cancelled read cannot prepare an attachment action")
        });
        assert!(matches!(
            result,
            Err(Error::Store(StoreError::ReadCancelled))
        ));
    });
}

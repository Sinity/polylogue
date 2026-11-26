use super::*;
use polylogue_model::{
    Author, ChangePosition, MessageRow, Output, Role, Session, SessionKind, Timestamp,
};
use polylogue_store::{
    ReadPool,
    fixture::writer::{WriterArchive, message, sid},
};

fn begin() -> ReadBegin {
    ReadBegin {
        format: polylogue_render::Format::Json,
        snapshot: ReadSnapshot {
            query: format!("query:{}", "a".repeat(64)),
            query_text: "sessions".into(),
            ran_at: Timestamp::from_unix_ms(1).unwrap(),
            read_at: ChangePosition {
                generation: "synthetic".into(),
                index: 1,
                archive: 1,
            },
            freshness: polylogue_model::freshness::ReadFreshness {
                local: polylogue_model::freshness::ReadCompleteness::empty(),
                sinex: polylogue_model::freshness::IndexFreshness::Standalone,
            },
            enabled_alternates: vec![],
            author: Author::Operator,
            notes: vec![],
            incomplete: vec![],
        },
        header: ReadHeader {
            target: "query:synthetic".into(),
            view: View::Transcript,
            title: ValueState::known("Synthetic".into()),
            kind: HeaderKind::Query {
                query: "sessions".into(),
            },
            figures: vec![],
            zone: None,
            snapshot: None,
            options: None,
        },
        options: ReadOptions::default(),
    }
}

/// The streamed events, by name, with their data.
fn events(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
    String::from_utf8(bytes.to_vec())
        .unwrap()
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let mut name = String::new();
            let mut data = String::new();
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    name = rest.to_owned();
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data.push_str(rest);
                }
            }
            (name, serde_json::from_str(&data).unwrap())
        })
        .collect()
}

#[test]
fn an_attachment_whose_size_differs_is_unavailable_and_the_export_continues() {
    // The fixture stores each record as one byte and a newline ends the
    // input: the input's bytes are a stored blob the attachment can name.
    let bytes = b"r\n";
    let hash = BlobHash::of(bytes);
    let image = |size| {
        Block::Image(Attachment {
            media_type: Some("image/png".into()),
            size: Some(size),
            blob: Some(hash),
            ..Attachment::default()
        })
    };
    let mut writer = WriterArchive::new();
    writer.write_output(
        "attachments",
        "claude-code",
        "/p/attachments",
        Output {
            sessions: vec![Session::new(sid("attachments"), SessionKind::Main)],
            messages: vec![MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 0,
                end: 1,
                message: message("m", "attachments", Role::User, 1, vec![image(2)]),
            }],
            ..Output::default()
        },
        1,
    );
    let pool = ReadPool::open(&writer.archive).unwrap();
    let reader = pool.reader().unwrap();
    let check = || Ok(());
    let mut out = Vec::new();
    {
        let mut stream = Stream::begin(&mut out, &check, &begin()).unwrap();
        let mut blobs = Blobs::new(&reader).unwrap();
        // A recorded size of 9 for two stored bytes: the bytes are not sent,
        // the read says why, and the next attachment is still delivered.
        blobs
            .blocks(&[image(9), image(2)], &mut stream, &check)
            .unwrap();
    }
    let events = events(&out);
    let names: Vec<_> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["begin", "record", "record", "blob"]);
    let omission: ReadRecord = serde_json::from_value(events[1].1.clone()).unwrap();
    assert_eq!(
        omission,
        ReadRecord::Presentation {
            parts: vec![ReadPart::Item(ReadItem {
                source: Source::record(format!("blob:{hash}")),
                kind: ItemKind::Omission {
                    count: ValueState::known(1),
                    unit: "attachment".into(),
                    why: "unavailable: the attachment records 9 bytes and its stored bytes are 2"
                        .into(),
                    continue_call: None,
                },
            })],
        }
    );
    let descriptor: ReadRecord = serde_json::from_value(events[2].1.clone()).unwrap();
    assert!(matches!(
        descriptor.record(),
        Some(Record::Blob(blob)) if blob.file.size == 2 && blob.file.hash == hash
    ));
    assert_eq!(events[3].1["offset"], 0);
    assert_eq!(events[3].1["base64"], "cgo=");
}

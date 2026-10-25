use crate::parts::{RECORDS, held};
use crate::records::request_event;
use polylogue_model::{Author, NewNote, NoteKind, Target, Timestamp};
use polylogue_store::input_record::{Extent, InputRecord};
use polylogue_store::primary::{Held, Record};
use polylogue_store::user_writes::{UserRequest, UserWrite};
use sinex_model::{InputRef, MaterialId};
use sinex_protocol::{HeldMaterial, MaterialFrame};

fn request(write: UserWrite) -> UserRequest {
    UserRequest {
        id: "0199d8a0-0000-7000-8000-00000000b001".into(),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(1_790_000_000_000).unwrap(),
        write,
    }
}

fn material(parser: &str, blob: &str, manifest: serde_json::Value) -> HeldMaterial {
    HeldMaterial {
        number: 7,
        frame: "synthetic".into(),
        material: MaterialFrame {
            id: MaterialId::new(),
            input: InputRef {
                kind: "claude-code".into(),
                locator: "/home/op/.claude/projects/p/s.jsonl".into(),
                host: "laptop".into(),
            },
            blob: blob.into(),
            length: 12,
            parser: parser.into(),
            parser_version: 1,
            part_index: 2,
            previous_part: None,
            manifest,
        },
    }
}

#[test]
fn a_note_request_becomes_a_note_event_in_its_family() {
    let note = request(UserWrite::AddNote {
        note: NewNote {
            kind: NoteKind::Note,
            key: None,
            value: Some("synthetic".into()),
            author: Author::Operator,
            targets: vec![Target::Note("earlier".into())],
            relations: vec![],
            revises: Some("earlier".into()),
            valid_until: None,
        },
    });
    let (kind, payload) = request_event(&note, Some("first")).unwrap().unwrap();
    assert_eq!(kind, "polylogue.note");
    assert_eq!(payload["root"], "first");
    assert_eq!(payload["relations"][0]["kind"], "revises");
    let event: sinex_model::event_types::polylogue::Note = serde_json::from_value(payload).unwrap();
    assert_eq!(event.note, note.id);
    assert!(
        request_event(&note, None).is_err(),
        "a note needs its family"
    );
}

#[test]
fn a_chat_action_request_makes_no_event() {
    let claim = request(UserWrite::ChatClaim { by: "x".into() });
    assert!(request_event(&claim, None).unwrap().is_none());
}

#[test]
fn a_held_part_carries_the_rows_that_place_it() {
    let record = InputRecord {
        input: "input".into(),
        format: "claude-code".into(),
        revision: None,
        extent: Some(Extent {
            chain: 1,
            offset: 40,
            length: 12,
            blob: polylogue_model::BlobHash::of(b"twelve bytes"),
        }),
        observation: None,
    };
    let blob = format!("blake3:{}", polylogue_model::BlobHash::of(b"twelve bytes"));
    let store = hashed_blobs::Reader::new(tempfile::tempdir().unwrap().path(), vec![]);
    let Held::Part {
        number,
        index,
        record: back,
        ..
    } = held(
        &store,
        material(
            "polylogue-claude-code",
            &blob,
            serde_json::json!({ "polylogue": record }),
        ),
    )
    .unwrap()
    else {
        panic!("a part")
    };
    assert_eq!((number, index), (7, 2));
    assert_eq!(*back, record);
}

#[test]
fn a_held_record_is_read_from_the_content_store_and_checked() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let record = Record::Request {
        request: Box::new(request(UserWrite::ChatClaim { by: "x".into() })),
        root: None,
    };
    let bytes = serde_json::to_vec(&record).unwrap();
    let hash = store.put(bytes.as_slice(), 1).unwrap();
    let reader = hashed_blobs::Reader::new(dir.path(), vec![]);
    let held = held(
        &reader,
        material(
            RECORDS,
            &format!("blake3:{}", hash.hex()),
            serde_json::json!({}),
        ),
    )
    .unwrap();
    let Held::Record {
        number,
        record: back,
        ..
    } = held
    else {
        panic!("a record")
    };
    assert_eq!((number, back), (7, record));
}

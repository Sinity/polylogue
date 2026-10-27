use crate::parts::{REGISTER, held};
use crate::register::request_event;
use polylogue_model::{Author, NewNote, NoteKind, Target, Timestamp};
use polylogue_store::input_rows::{Extent, InputRows};
use polylogue_store::primary::{Held, RegisterChange};
use polylogue_store::user_writes::{UserRequest, UserWrite};
use sinex_model::{InputRef, MaterialId};
use sinex_protocol::{MaterialFrame, ProducerMaterial};

fn request(write: UserWrite) -> UserRequest {
    UserRequest {
        id: "0199d8a0-0000-7000-8000-00000000b001".into(),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(1_790_000_000_000).unwrap(),
        write,
    }
}

fn material(parser: &str, blob: &str, manifest: serde_json::Value) -> ProducerMaterial {
    ProducerMaterial {
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
            manifest: sinex_model::Manifest::from_json(manifest),
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
    assert_eq!(payload["first_note"], "first");
    assert_eq!(payload["relations"][0]["kind"], "revises");
    let event: crate::kinds::Note = serde_json::from_value(payload).unwrap();
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
    let rows = InputRows {
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
        rows: back,
        ..
    } = held(
        &store,
        material(
            "polylogue-claude-code",
            &blob,
            serde_json::json!({ "polylogue": rows }),
        ),
    )
    .unwrap()
    else {
        panic!("a part")
    };
    assert_eq!((number, index), (7, 2));
    assert_eq!(*back, rows);
}

#[test]
fn a_held_register_material_is_read_from_the_content_store_and_checked() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let change = RegisterChange::Request {
        request: Box::new(request(UserWrite::ChatClaim { by: "x".into() })),
        first_note: None,
    };
    let bytes = serde_json::to_vec(&change).unwrap();
    let hash = store.put(bytes.as_slice(), 1).unwrap();
    let reader = hashed_blobs::Reader::new(dir.path(), vec![]);
    let held = held(
        &reader,
        material(
            REGISTER,
            &format!("blake3:{}", hash.hex()),
            serde_json::json!({}),
        ),
    )
    .unwrap();
    let Held::Change {
        number,
        change: back,
        ..
    } = held
    else {
        panic!("a register change")
    };
    assert_eq!((number, back), (7, change));
}

#[test]
fn credentials_in_the_nats_url_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sinex.toml");
    std::fs::write(
        &path,
        "mode = 'primary'\nnats = 'nats://polylogue:secret@127.0.0.1:4222'\nstore = '/unused'\n",
    )
    .unwrap();
    let error = crate::Config::load_from(&path).unwrap_err().to_string();
    assert!(error.contains("password_file"), "{error}");
    assert!(!error.contains("secret"), "{error}");

    std::fs::write(
        &path,
        "mode = 'primary'\nnats = 'nats://127.0.0.1:4222'\npassword_file = '/run/secret'\nstore = '/unused'\n",
    )
    .unwrap();
    let config = crate::Config::load_from(&path).unwrap().unwrap();
    assert_eq!(
        config.password_file.unwrap(),
        std::path::Path::new("/run/secret")
    );
}

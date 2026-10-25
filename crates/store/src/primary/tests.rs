use super::*;
use crate::user_writes::UserWrite;
use crate::{Archive, Batch, Change, InputKey, Observation, Writer};
use polylogue_model::{Author, Timestamp};
use std::sync::{Arc, Mutex};

/// A stand-in for Sinex: it keeps what it is sent, in order, with the bytes
/// in a content store, and can refuse everything.
struct Recorder {
    held: Mutex<Vec<Held>>,
    store: hashed_blobs::Store,
    refuse: bool,
}

impl Recorder {
    fn new(store: &std::path::Path, refuse: bool) -> Arc<Self> {
        Arc::new(Self {
            held: Mutex::new(Vec::new()),
            store: hashed_blobs::Store::open(store, vec![], 0o600).unwrap(),
            refuse,
        })
    }

    fn next(&self) -> i64 {
        self.held.lock().unwrap().len() as i64 + 1
    }
}

impl Sinex for Recorder {
    fn send_part(&self, part: &Part, bytes: &mut dyn std::io::Read) -> Result<String> {
        if self.refuse {
            return Err(StoreError::Sinex("synthetic outage".into()));
        }
        let hash = self.store.put(bytes, 1)?;
        assert_eq!(hash.0, part.hash.0);
        let material = uuid::Uuid::now_v7().to_string();
        let number = self.next();
        self.held.lock().unwrap().push(Held::Part {
            number,
            material: material.clone(),
            hash: part.hash,
            length: part.length,
            index: part.index,
            rows: Box::new(part.rows.clone()),
        });
        Ok(material)
    }

    fn send_change(&self, change: &RegisterChange) -> Result<String> {
        if self.refuse {
            return Err(StoreError::Sinex("synthetic outage".into()));
        }
        let material = uuid::Uuid::now_v7().to_string();
        let number = self.next();
        self.held.lock().unwrap().push(Held::Change {
            number,
            material: material.clone(),
            change: change.clone(),
        });
        Ok(material)
    }

    fn excise(&self, materials: &[String]) -> Result<()> {
        self.held.lock().unwrap().retain(|held| match held {
            Held::Part { material, .. } | Held::Change { material, .. } => {
                !materials.contains(material)
            }
        });
        Ok(())
    }

    fn held(&self, after: i64) -> Result<Vec<Held>> {
        Ok(self
            .held
            .lock()
            .unwrap()
            .iter()
            .filter(|held| held.number() > after)
            .cloned()
            .collect())
    }

    fn send_sessions(&self, _: &ReadPool) -> Result<()> {
        Ok(())
    }
}

fn observed(path: &str, change: Change, length: u64) -> Batch {
    Batch {
        input: InputKey {
            id: "synthetic-input".into(),
            format: "claude-code".into(),
            arrival_rank: 0,
        },
        change,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: path.into(),
            facts: None,
            seen_at_ms: 1,
            import: None,
        }),
        continues: false,
        new_blob: None,
        blobs: Vec::new(),
        length,
        parse: Vec::new(),
        release: Default::default(),
    }
}

fn register(archive: &Archive) -> Vec<String> {
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    let mut rows = Vec::new();
    for sql in [
        "SELECT input, number, produced, chain, length FROM revision ORDER BY input, number",
        "SELECT input, chain, offset, length, hex(blob) FROM extent ORDER BY input, chain, offset",
        "SELECT id, input, class, revision, path FROM observation ORDER BY id",
        "SELECT input, chain, offset, part, id FROM material ORDER BY input, chain, offset",
        "SELECT hex(hash), size, place, state FROM blob ORDER BY hash",
        "SELECT id, kind, value FROM note ORDER BY id",
    ] {
        let mut statement = conn.prepare(sql).unwrap();
        let columns = statement.column_count();
        let mut query = statement.query([]).unwrap();
        while let Some(row) = query.next().unwrap() {
            rows.push(
                (0..columns)
                    .map(|i| format!("{:?}", row.get_ref(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }
    rows
}

fn note() -> crate::user_writes::UserRequest {
    crate::user_writes::UserRequest {
        id: "0199d8a0-0000-7000-8000-00000000c001".into(),
        machine: "synthetic-machine".into(),
        at: Timestamp::from_unix_ms(1_790_000_000_000).unwrap(),
        write: UserWrite::SetSetting {
            key: "day.zone".into(),
            value: "UTC".into(),
            author: Author::Operator,
            machine: None,
        },
    }
}

#[test]
fn every_register_row_is_sent_first_and_a_lost_copy_is_rebuilt_from_sinex() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, false);
    let path = "/home/op/.claude/projects/p/s.jsonl";
    let before = {
        let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
        let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
        for (bytes, change) in [
            (b"{\"first\":1}\n".as_slice(), Change::First),
            (
                b"{\"second\":2}\n".as_slice(),
                Change::Appended { from: 12 },
            ),
        ] {
            let prepared = archive.prepare_bytes(bytes).unwrap();
            let length = if matches!(change, Change::First) {
                12
            } else {
                25
            };
            writer
                .store_bytes(prepared, observed(path, change, length))
                .unwrap();
        }
        writer
            .send(observed(path, Change::Disappeared, 25))
            .unwrap();
        writer.user(note()).unwrap();
        writer.flush().unwrap();
        drop(writer);
        assert!(!root.join("archive.db").exists());
        assert!(!root.join("blobs").exists());
        register(&archive)
    };
    {
        let held = sinex.held.lock().unwrap();
        let parts: Vec<(u32, &str)> = held
            .iter()
            .filter_map(|h| match h {
                Held::Part { index, rows, .. } => Some((*index, rows.input.as_str())),
                Held::Change { .. } => None,
            })
            .collect();
        assert_eq!(parts, [(0, "synthetic-input"), (1, "synthetic-input")]);
        assert_eq!(
            held.len(),
            4,
            "two parts, the disappearance and the request"
        );
    }
    for name in [
        "sinex.db",
        "sinex.db-wal",
        "sinex.db-shm",
        "index.db",
        "index.db-wal",
        "index.db-shm",
    ] {
        let _ = std::fs::remove_file(root.join(name));
    }
    let (archive, lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
    assert_eq!(register(&archive), before);
    drop(lock);
    // A copy with another schema is rebuilt the same way.
    rusqlite::Connection::open(root.join("sinex.db"))
        .unwrap()
        .execute_batch("CREATE TABLE synthetic_older(value)")
        .unwrap();
    let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
    assert_eq!(register(&archive), before);
    let register = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert!(crate::schema::same_schema(&register, &crate::schema::SINEX).unwrap());
}

#[test]
fn nothing_is_written_locally_that_sinex_did_not_take() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, true);
    let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
    let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
    let prepared = archive
        .prepare_bytes(b"{\"first\":1}\n".as_slice())
        .unwrap();
    assert!(
        writer
            .store_bytes(prepared, observed("/p/s.jsonl", Change::First, 12))
            .is_err()
    );
    assert!(matches!(writer.user(note()), Err(StoreError::Sinex(_))));
    writer.flush().unwrap();
    drop(writer);
    assert!(register(&archive).is_empty());
}

#[test]
fn an_excision_in_primary_mode_is_sinex_excising_the_input_and_survives_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, false);
    let path = "/home/op/.claude/projects/p/s.jsonl";
    let before = {
        let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
        let writer = Writer::start_on_sinex(&archive, sinex.clone()).unwrap();
        let prepared = archive
            .prepare_bytes(b"{\"first\":1}\n".as_slice())
            .unwrap();
        writer
            .store_bytes(prepared, observed(path, Change::First, 12))
            .unwrap();
        writer.user(note()).unwrap();
        let excised = writer
            .excise(
                crate::excise::ExciseRequest {
                    id: "0199d8a0-0000-7000-8000-00000000d001".into(),
                    target: crate::excise::ExciseTarget::Input("synthetic-input".into()),
                    reason: None,
                    dry_run: false,
                },
                Arc::new(archive.blobs().unwrap()),
            )
            .unwrap();
        assert_eq!(excised.inputs_removed, ["synthetic-input"]);
        drop(writer);
        register(&archive)
    };
    {
        let held = sinex.held.lock().unwrap();
        assert!(
            !held.iter().any(|h| matches!(h, Held::Part { .. })),
            "Sinex excised the input's part"
        );
        assert!(held.iter().any(|h| matches!(
            h,
            Held::Change {
                change: RegisterChange::Excision(_),
                ..
            }
        )));
    }
    assert!(
        store.join("blobs").exists(),
        "Polylogue never deletes from Sinex's content store"
    );
    for name in [
        "sinex.db",
        "sinex.db-wal",
        "sinex.db-shm",
        "index.db",
        "index.db-wal",
        "index.db-shm",
    ] {
        let _ = std::fs::remove_file(root.join(name));
    }
    let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
    assert_eq!(register(&archive), before);
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    let excised: i64 = conn
        .query_row(
            "SELECT count(*) FROM excised WHERE kind = 'input'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(excised, 1, "the rebuilt copy still refuses the input");
}

#[test]
fn a_part_sent_by_a_write_that_never_committed_leaves_nothing_after_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, false);
    let part = |id: &str, bytes: &[u8]| {
        let hash = sinex.store.put(bytes, 1).unwrap();
        let blob = polylogue_model::BlobHash(hash.0);
        let mut b = observed("/p/s.jsonl", Change::First, bytes.len() as u64);
        b.observation.as_mut().unwrap().seen_at_ms = 5;
        Held::Part {
            number: sinex.next(),
            material: uuid::Uuid::now_v7().to_string(),
            hash: blob,
            length: bytes.len() as u64,
            index: 0,
            rows: Box::new(crate::input_rows::InputRows {
                input: "synthetic-input".into(),
                format: "claude-code".into(),
                revision: Some(crate::input_rows::Revision {
                    number: 1,
                    produced: "first".into(),
                    chain: 1,
                    length: bytes.len() as i64,
                }),
                extent: Some(crate::input_rows::Extent {
                    chain: 1,
                    offset: 0,
                    length: bytes.len() as i64,
                    blob,
                }),
                observation: Some(crate::input_rows::ObservationRow {
                    id: id.into(),
                    machine: "synthetic-machine".into(),
                    host: "synthetic-host".into(),
                    path: "/p/s.jsonl".into(),
                    class: "first".into(),
                    revision: Some(1),
                    length: Some(bytes.len() as i64),
                    device: None,
                    inode: None,
                    size: None,
                    mtime_ns: None,
                    ctime_ns: None,
                    polylogue_version: "synthetic".into(),
                    seen_at: b.observation.unwrap().seen_at_ms,
                    details_json: None,
                    excision: None,
                    zip: None,
                }),
            }),
        }
    };
    let stopped = part("stopped", b"{\"a\":1}\n");
    sinex.held.lock().unwrap().push(stopped);
    let committed = part("committed", b"{\"a\":1}\n{\"b\":2}\n");
    sinex.held.lock().unwrap().push(committed);
    let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    let observations: Vec<String> = conn
        .prepare("SELECT id FROM observation")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(observations, ["committed"]);
    let (blobs, length): (i64, i64) = conn
        .query_row(
            "SELECT (SELECT count(*) FROM blob), (SELECT length FROM extent)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!((blobs, length), (1, 16));
}

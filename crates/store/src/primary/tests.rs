use super::*;
use crate::{Archive, Batch, Change, InputKey, Observation, Writer};
use polylogue_model::user::UserWrite;
use polylogue_model::{Author, Timestamp};
use std::sync::{Arc, Mutex};

/// A stand-in for Sinex: it keeps what it is sent, in order, with the bytes
/// in a content store, and can refuse everything.
struct Recorder {
    held: Mutex<Vec<Held>>,
    store: Mutex<hashed_blobs::SegmentWriter>,
    root: std::path::PathBuf,
    blobs: Mutex<std::collections::HashMap<hashed_blobs::Hash, hashed_blobs::Extent>>,
    refuse: bool,
}

impl Recorder {
    fn new(store: &std::path::Path, refuse: bool) -> Arc<Self> {
        Arc::new(Self {
            held: Mutex::new(Vec::new()),
            store: Mutex::new(
                hashed_blobs::SegmentWriter::open(store, "segments/primary.zst", 0, &[], 0o600)
                    .unwrap(),
            ),
            root: store.to_owned(),
            blobs: Mutex::new(Default::default()),
            refuse,
        })
    }

    fn put(&self, bytes: impl std::io::Read) -> Result<hashed_blobs::Extent> {
        let prepared =
            hashed_blobs::Preparation::new(self.root.join("scratch"))?.plain(bytes, 1)?;
        let mut writer = self.store.lock().unwrap();
        let mut blobs = self.blobs.lock().unwrap();
        let group = writer
            .append_group(vec![prepared], |hash| Ok(blobs.get(hash).cloned()))?
            .wait()?;
        let extent = group.extents[0].clone();
        blobs.insert(extent.hash, extent.clone());
        writer.acknowledge(&group)?;
        Ok(extent)
    }

    fn next(&self) -> i64 {
        self.held.lock().unwrap().len() as i64 + 1
    }
}

impl Sinex for Recorder {
    fn send_part(&self, part: &Part, bytes: &mut dyn std::io::Read) -> Result<ConfirmedBlob> {
        if self.refuse {
            return Err(StoreError::Sinex("synthetic outage".into()));
        }
        let extent = self.put(bytes)?;
        assert_eq!(extent.hash.0, part.hash.0);
        let material = uuid::Uuid::now_v7().to_string();
        let number = self.next();
        self.held.lock().unwrap().push(Held::Part {
            number,
            material: material.clone(),
            hash: part.hash,
            length: part.length,
            extents: vec![extent.clone()],
            index: part.index,
            rows: Box::new(part.rows.clone()),
        });
        Ok(ConfirmedBlob {
            id: material,
            extents: vec![extent],
        })
    }

    fn send_blob(
        &self,
        hash: BlobHash,
        length: u64,
        bytes: &mut dyn std::io::Read,
    ) -> Result<ConfirmedBlob> {
        if self.refuse {
            return Err(StoreError::Sinex("synthetic outage".into()));
        }
        let extent = self.put(bytes)?;
        let material = uuid::Uuid::now_v7().to_string();
        let number = self.next();
        self.held.lock().unwrap().push(Held::Blob {
            number,
            material: material.clone(),
            hash,
            length,
            extents: vec![extent.clone()],
        });
        Ok(ConfirmedBlob {
            id: material,
            extents: vec![extent],
        })
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
        "SELECT hex(hash), size, stored_size, hex(storage) FROM blob ORDER BY hash",
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

fn note() -> polylogue_model::user::UserRequest {
    polylogue_model::user::UserRequest {
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
                Held::Change { .. } | Held::Blob { .. } => None,
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
fn a_part_sent_by_a_write_that_never_committed_leaves_nothing_after_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, false);
    let part = |id: &str, bytes: &[u8]| {
        let extent = sinex.put(bytes).unwrap();
        let blob = polylogue_model::BlobHash(extent.hash.0);
        let mut b = observed("/p/s.jsonl", Change::First, bytes.len() as u64);
        b.observation.as_mut().unwrap().seen_at_ms = 5;
        Held::Part {
            number: sinex.next(),
            material: uuid::Uuid::now_v7().to_string(),
            hash: blob,
            length: bytes.len() as u64,
            extents: vec![extent],
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

#[test]
fn decoded_attachment_bytes_rebuild_from_sinex_descriptors_without_input_rows() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, false);
    let bytes = b"decoded attachment bytes";
    let hash = BlobHash::of(bytes);
    let confirmed = sinex
        .send_blob(hash, bytes.len() as u64, &mut bytes.as_slice())
        .unwrap();
    let root = dir.path().join("archive");
    let (archive, _lock) = Archive::open_primary(&root, &store, sinex.as_ref()).unwrap();
    let conn = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM input", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let storage: Vec<u8> = conn
        .query_row(
            "SELECT storage FROM blob WHERE hash=?1",
            [&hash.0[..]],
            |r| r.get(0),
        )
        .unwrap();
    let extent = hashed_blobs::Extent::decode(&storage).unwrap();
    assert_eq!(&extent, confirmed.extent(hash, bytes.len() as u64).unwrap());
    assert_eq!(
        hashed_blobs::Reader::new(&store)
            .read(&[extent], None)
            .unwrap(),
        bytes
    );
    assert!(!root.join("segments").exists());
}

#[test]
fn mismatched_attachment_descriptor_is_refused_before_register_commit() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("sinex-store");
    let sinex = Recorder::new(&store, false);
    let extent = sinex.put(b"stored attachment".as_slice()).unwrap();
    sinex.held.lock().unwrap().push(Held::Blob {
        number: 1,
        material: uuid::Uuid::now_v7().to_string(),
        hash: BlobHash::of(b"different attachment"),
        length: 17,
        extents: vec![extent],
    });
    let root = dir.path().join("archive");
    assert!(Archive::open_primary(&root, &store, sinex.as_ref()).is_err());
    let conn = crate::archive::open_read(&root.join("sinex.db")).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

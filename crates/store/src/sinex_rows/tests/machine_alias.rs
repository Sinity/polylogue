use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::writer::{Batch as WriteBatch, InFlight, InputKey, Observation};
use crate::{Change, Writer};
use std::io::Cursor;

#[test]
fn machine_aliases_leave_old_observation_bytes_exact_after_restore_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut original = None;
    for (host, at, change) in [
        ("host-a", 900, Change::First),
        ("host-b", 1, Change::Unchanged),
    ] {
        writer
            .send(WriteBatch {
                input: InputKey {
                    id: "input".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change,
                observation: Some(Observation {
                    machine: "machine".into(),
                    host: host.into(),
                    path: "synthetic".into(),
                    facts: None,
                    seen_at_ms: at,
                }),
                continues: false,
                piece: None,
                blobs: vec![],
                length: 0,
                parse: vec![],
                release: InFlight::default(),
            })
            .unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
        let published = Published::open(&archive.root).unwrap();
        let inventory = published.inventory().unwrap();
        if let Some((unit, bytes, hash)) = &original {
            let actual = published.read(unit).unwrap().contents();
            assert_eq!(&actual, bytes);
            assert_eq!(&blake3::hash(&actual), hash);
        } else {
            let unit = (1..=inventory.len())
                .map(|p| inventory.unit(p).unwrap())
                .find(|u| u.kind == "observation")
                .unwrap();
            let bytes = published.read(&unit).unwrap().contents();
            let hash = blake3::hash(&bytes);
            original = Some((unit, bytes, hash));
        }
    }
    drop(writer);
    drop(lock);
    drop(archive);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let published = Published::open(&archive.root).unwrap();
    let inventory = published.inventory().unwrap();
    let mut units = Vec::new();
    let mut payloads = Vec::new();
    for p in 1..=inventory.len() {
        let unit = inventory.unit(p).unwrap();
        let batch = published.read(&unit).unwrap();
        let bytes = batch.contents();
        units.push(RestoreUnit {
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: vec![],
            excision: None,
        });
        payloads.push(bytes);
    }
    let target_dir = tempfile::tempdir().unwrap();
    let (target, target_lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut restore = Restoring::open(&target).unwrap();
    let selected = select(&units).unwrap();
    restore
        .install(&published.archive_id().unwrap(), &selected, |u| {
            Ok(Some(Cursor::new(
                payloads[units.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
        .unwrap();
    drop(restore);
    drop(target_lock);
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let restored = Published::open(&target.root).unwrap();
    let (unit, bytes, hash) = original.unwrap();
    let actual = restored.read(&unit).unwrap().contents();
    assert_eq!(actual, bytes);
    assert_eq!(blake3::hash(&actual), hash);
    for (unit, bytes) in units.iter().zip(&payloads) {
        assert_eq!(restored.read(&unit.unit).unwrap().contents(), *bytes);
    }
    assert!(
        payloads
            .iter()
            .any(|bytes| String::from_utf8_lossy(bytes).contains("host-b"))
    );
}

#[test]
fn legacy_alias_migration_keeps_current_batches_and_excludes_a_future_alias() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let observe = |writer: &Writer, host: &str, at, change| {
        writer
            .send(WriteBatch {
                input: InputKey {
                    id: "legacy-input".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change,
                observation: Some(Observation {
                    machine: "legacy-machine".into(),
                    host: host.into(),
                    path: "synthetic".into(),
                    facts: None,
                    seen_at_ms: at,
                }),
                continues: false,
                piece: None,
                blobs: vec![],
                length: 0,
                parse: vec![],
                release: InFlight::default(),
            })
            .unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
    };
    observe(&writer, "host-a", 900, Change::First);
    observe(&writer, "host-b", 1, Change::Unchanged);
    drop(writer);
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    // Archive8 exposed every current alias to every observation. Capture that
    // closure, then remove the new local-only column to exercise its migration.
    db.execute("UPDATE machine SET first_commit_seq=0", [])
        .unwrap();
    let source = Published::open(&archive.root).unwrap();
    let before = super::append_facts::descriptors(&source);
    drop(source);
    db.execute_batch("ALTER TABLE machine DROP COLUMN first_commit_seq; PRAGMA user_version=8;")
        .unwrap();
    drop(db);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let source = Published::open(&archive.root).unwrap();
    assert_eq!(super::append_facts::descriptors(&source), before);
    drop(source);
    let writer = Writer::start(&archive).unwrap();
    observe(&writer, "host-c", 0, Change::Unchanged);
    drop(writer);
    let source = Published::open(&archive.root).unwrap();
    for (unit, bytes) in before.0.iter().zip(before.1.iter()) {
        assert_eq!(source.read(&unit.unit).unwrap().contents(), *bytes);
    }
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM machine", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        3
    );
    assert_eq!(
        db.query_row(
            "SELECT COUNT(*) FROM machine WHERE first_commit_seq=0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
}

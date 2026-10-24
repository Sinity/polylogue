use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::writer::{Batch as WriteBatch, InFlight, InputKey, Observation};
use crate::{Change, Writer};
use std::io::Cursor;

fn observe(writer: &Writer, host: &str, at: i64, change: Change) {
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
                import: None,
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
}

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
        observe(&writer, host, at, change);
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
    let (units, payloads) = super::append_facts::descriptors(&published);
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

mod history;

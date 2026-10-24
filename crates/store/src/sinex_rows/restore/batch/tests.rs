use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::sinex_rows::{Published, RestoreUnit};
use crate::writer::{Batch, InFlight, InputKey, Observation};
use crate::{Change, Writer};
use std::io::Cursor;

#[test]
fn reversed_batches_reconstruct_alias_cutoffs_for_legacy_embedded_observations() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let observe = |host: &str, change| {
        let writer = Writer::start(&source).unwrap();
        writer
            .send(Batch {
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
                    seen_at_ms: 1,
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
    observe("host-a", Change::First);
    let db = open_write(&source.archive_db(), cache::SETUP).unwrap();
    db.execute("UPDATE observation SET commit_seq=0", [])
        .unwrap();
    db.execute("UPDATE machine SET first_commit_seq=0", [])
        .unwrap();
    observe("host-b", Change::Unchanged);
    let published = Published::open(&source.root).unwrap();
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
    assert_eq!(units.len(), 2);
    let selected = super::super::super::restore::select(&units).unwrap();
    let (target, _lock) = Archive::open_or_create(&dir.path().join("target")).unwrap();
    let target_db = open_write(&target.archive_db(), cache::SETUP).unwrap();
    // Exercise payload installation itself backwards, beyond the catalog's
    // ordinary sorting of selected descriptors.
    for (unit, bytes) in units.iter().zip(&payloads).rev() {
        install(
            unit,
            Some(Cursor::new(bytes)),
            Some(&target_db),
            &selected.catalog.db,
        )
        .unwrap();
    }
    verify_machines(&target_db, &selected.catalog.db).unwrap();
    let restored = Published::open(&target.root).unwrap();
    for (unit, bytes) in units.iter().zip(&payloads) {
        assert_eq!(restored.read(&unit.unit).unwrap().contents(), *bytes);
    }
    let cutoffs: Vec<(String, i64)> = target_db
        .prepare("SELECT host,first_commit_seq FROM machine ORDER BY host")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(cutoffs, vec![("host-a".into(), 1), ("host-b".into(), 2)]);
}

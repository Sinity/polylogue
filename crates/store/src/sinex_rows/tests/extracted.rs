//! Canonical decoded ownership is checked against actual retained source bytes.
use super::*;
use crate::retention::{Extraction, PreparedBatch};
use crate::writer::{Batch as WriteBatch, InputKey};
use crate::{Archive, Change, Writer};
use polylogue_model::BlobHash;
use std::{io::Cursor, sync::Arc};

#[test]
fn decoded_units_restore_atomically_and_refuse_forged_source_or_blob_claims() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let raw = br#"{"data":"QUJD"}"#;
    let decoded = b"ABC";
    let (piece, raw_file) = source.prepare_piece(&[], raw).unwrap();
    let (extra, decoded_file) = source.prepare_piece(&[], decoded).unwrap();
    let writer = Writer::start(&source).unwrap();
    writer
        .sender()
        .unwrap()
        .retain_group(
            vec![PreparedBatch {
                batch: WriteBatch {
                    input: InputKey {
                        id: "synthetic-input".into(),
                        format: "synthetic-text".into(),
                        route_rank: 0,
                    },
                    change: Change::First,
                    observation: None,
                    continues: false,
                    piece: Some(piece),
                    blobs: vec![extra],
                    length: raw.len() as u64,
                    parse: vec![],
                    release: Default::default(),
                },
                files: raw_file.into_iter().chain(decoded_file).collect(),
                extracted: vec![Extraction {
                    range: 0..raw.len() as u64,
                    source_hash: BlobHash::of(raw),
                    parser_version: 1,
                    blob: BlobHash::of(decoded),
                }],
            }],
            Arc::new(hashed_blobs::Store::open(source.root.clone(), vec![], 0o600).unwrap()),
        )
        .unwrap()
        .pop()
        .unwrap()
        .unwrap();
    let published = Published::open(&source.root).unwrap();
    let inventory = published.inventory().unwrap();
    let material_dir = dir.path().join("material");
    let material = hashed_blobs::Store::open(material_dir.clone(), vec![], 0o600).unwrap();
    let mut descriptors = Vec::new();
    let mut payloads = Vec::new();
    for position in 1..=inventory.len() {
        let unit = inventory.unit(position).unwrap();
        let batch = published.read(&unit).unwrap();
        for hash in &batch.blobs {
            let mut snapshot = published.blob_snapshot(hash).unwrap();
            assert_eq!(
                material
                    .put_plain_checked::<crate::StoreError>(&mut snapshot.file, 3, || Ok(()))
                    .unwrap()
                    .0,
                *hash
            );
        }
        let bytes = batch.contents();
        descriptors.push(RestoreUnit {
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: batch
                .blobs
                .iter()
                .map(|h| format!("blake3:{}", BlobHash(*h)))
                .collect(),
            excision: None,
        });
        payloads.push(bytes);
    }
    let extracted = descriptors
        .iter()
        .position(|d| d.unit.kind == "extracted_blob")
        .unwrap();
    // Same transaction, with extracted_blob ordered before its raw revision.
    assert!(
        extracted
            < descriptors
                .iter()
                .position(|d| d.unit.kind == "revision")
                .unwrap()
    );
    let id = published.archive_id().unwrap();
    for alteration in ["source", "descriptor", "foreign", "none"] {
        let (target, _lock) =
            Archive::open_or_create(&dir.path().join(format!("target-{alteration}"))).unwrap();
        let mut units = descriptors.clone();
        let mut bodies = payloads.clone();
        if alteration == "descriptor" {
            units[extracted].blobs.clear();
        }
        if matches!(alteration, "source" | "foreign") {
            let mut lines: Vec<Value> = bodies[extracted]
                .split(|b| *b == b'\n')
                .filter(|b| !b.is_empty())
                .map(|b| serde_json::from_slice(b).unwrap())
                .collect();
            if alteration == "source" {
                let wrong = BlobHash::of(b"different retained bytes").to_string();
                for line in &mut lines {
                    if line["table"] == "extracted_blob" {
                        line["row"]["source_hash"]["hex"] = json!(wrong);
                    }
                }
                let mut key: Value = serde_json::from_str(&units[extracted].unit.key).unwrap();
                key[4] = json!(wrong);
                units[extracted].unit.key = serde_json::to_string(&key).unwrap();
            } else {
                lines[0]["table"] = json!("note");
            }
            bodies[extracted] = lines
                .iter()
                .flat_map(|line| {
                    let mut bytes = serde_json::to_vec(line).unwrap();
                    bytes.push(b'\n');
                    bytes
                })
                .collect();
            units[extracted].batch = Some(format!("blake3:{}", blake3::hash(&bodies[extracted])));
            units[extracted].bytes = Some(bodies[extracted].len() as u64);
        }
        let selected = select(&units).unwrap();
        selected
            .verify_source_blobs(|name| {
                let hash = BlobHash::parse(name.strip_prefix("blake3:").unwrap()).unwrap();
                let mut reader =
                    material.read_plain(&hashed_blobs::Hash(hash.0), None, || Ok(()))?;
                Ok(std::io::copy(&mut reader, &mut std::io::sink())?)
            })
            .unwrap();
        let result = Restoring::with_roots(&target, vec![material_dir.clone()])
            .unwrap()
            .install(&id, &selected, |u| {
                Ok(Some(Cursor::new(
                    bodies[units.iter().position(|d| d == u).unwrap()].clone(),
                )))
            });
        let db = crate::archive::open_read(&target.archive_db()).unwrap();
        if alteration == "none" {
            assert!(result.unwrap() > 0);
        } else {
            assert!(result.is_err(), "accepted {alteration}");
        }
        let count: i64 = db
            .query_row("SELECT count(*) FROM extracted_blob", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, i64::from(alteration == "none"));
        let revisions: i64 = db
            .query_row("SELECT count(*) FROM revision", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            revisions,
            i64::from(alteration == "none"),
            "partial restore after {alteration}"
        );
    }
    assert_eq!(
        crate::archive::open_read(&source.archive_db())
            .unwrap()
            .query_row("SELECT count(*) FROM extracted_blob", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

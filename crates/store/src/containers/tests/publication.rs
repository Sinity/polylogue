use super::*;
use crate::sinex_rows::{Catalog, Published, RestoreUnit, Restoring};
use std::io::{Cursor, Read, Seek};

pub(super) fn contents(batch: &crate::sinex_rows::Batch) -> Vec<u8> {
    let mut file = batch.file.try_clone().unwrap();
    file.rewind().unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

#[test]
fn container_relationships_restore_in_reverse_order_with_exact_unit_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    let outcomes = writer
        .sender()
        .unwrap()
        .retain_group(linked(&archive, "published", true), store)
        .unwrap();
    assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
    let published = Published::open(&archive.root).unwrap();
    let inventory = published.inventory().unwrap();
    let material_dir = dir.path().join("material");
    let material = hashed_blobs::Store::open(&material_dir, vec![], 0o600).unwrap();
    let mut data = Vec::new();
    for position in 1..=inventory.len() {
        let unit = inventory.unit(position).unwrap();
        let batch = published.read(&unit).unwrap();
        for hash in &batch.blobs {
            let mut snapshot = published.blob_snapshot(hash).unwrap();
            let installed = material
                .put_plain_checked::<crate::StoreError>(&mut snapshot.file, 3, || Ok(()))
                .unwrap();
            assert_eq!(installed.0, *hash);
        }
        let bytes = contents(&batch);
        data.push((
            RestoreUnit {
                unit,
                batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
                bytes: Some(bytes.len() as u64),
                rows: batch.rows,
                blobs: batch
                    .blobs
                    .iter()
                    .map(|hash| format!("blake3:{}", BlobHash(*hash)))
                    .collect(),
                excision: None,
            },
            bytes,
        ));
    }
    let mut catalog = Catalog::new().unwrap();
    for (position, (unit, _)) in data.iter().enumerate().rev() {
        catalog.push(position as u64 + 1, unit).unwrap();
    }
    let selected = catalog.finish().unwrap();
    selected
        .verify_source_blobs(|name| {
            let hash = BlobHash::parse(name.strip_prefix("blake3:").unwrap()).unwrap();
            let mut input = material.read_plain(&hashed_blobs::Hash(hash.0), None, || Ok(()))?;
            Ok(std::io::copy(&mut input, &mut std::io::sink())?)
        })
        .unwrap();
    let (target, _target_lock) = Archive::open_or_create(&dir.path().join("target")).unwrap();
    Restoring::with_roots(&target, vec![material_dir])
        .unwrap()
        .install(&published.archive_id().unwrap(), &selected, |unit| {
            Ok(Some(Cursor::new(
                data.iter()
                    .find(|(candidate, _)| candidate == unit)
                    .unwrap()
                    .1
                    .clone(),
            )))
        })
        .unwrap();
    let restored = Published::open(&target.root).unwrap();
    for (unit, expected) in &data {
        assert_eq!(contents(&restored.read(&unit.unit).unwrap()), *expected);
    }
    let writer = Writer::start(&target).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&target.root, vec![], 0o600).unwrap());
    let removed = remove(&writer, store, "restored-member-removal").unwrap();
    assert_eq!(
        removed.containers_retired,
        [BlobHash::of(ORIGINAL).to_string()]
    );
}

#[test]
fn confirmed_before_images_do_not_resurrect_retired_originals_or_member_links() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    let mut group = linked(&archive, "confirmed", false);
    group.push(stage(
        &archive,
        b"surviving sibling",
        "confirmed",
        "sibling.dat",
        Some(ImportContainer {
            input: BlobHash::of(ORIGINAL).to_string(),
            member_index: 1,
        }),
    ));
    assert!(
        writer
            .sender()
            .unwrap()
            .retain_group(group, Arc::clone(&store))
            .unwrap()
            .iter()
            .all(Result::is_ok)
    );
    let material_dir = dir.path().join("material");
    let material = hashed_blobs::Store::open(&material_dir, vec![], 0o600).unwrap();
    let published = Published::open(&archive.root).unwrap();
    let id = published.archive_id().unwrap();
    let mut confirmations = Vec::new();
    for position in 1..=published.inventory().unwrap().len() {
        let unit = published.inventory().unwrap().unit(position).unwrap();
        let batch = published.read(&unit).unwrap();
        for hash in &batch.blobs {
            let mut snapshot = published.blob_snapshot(hash).unwrap();
            assert_eq!(
                material
                    .put_plain_checked::<StoreError>(&mut snapshot.file, 3, || Ok(()))
                    .unwrap()
                    .0,
                *hash
            );
        }
        let bytes = contents(&batch);
        confirmations.push(RestoreUnit {
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: batch
                .blobs
                .iter()
                .map(|hash| format!("blake3:{}", BlobHash(*hash)))
                .collect(),
            excision: None,
        });
    }
    let mut old_catalog = Catalog::new().unwrap();
    for (i, unit) in confirmations.iter().enumerate() {
        old_catalog.push(i as u64 + 1, unit).unwrap();
    }
    let old = old_catalog.finish().unwrap();
    drop(published);
    remove(&writer, Arc::clone(&store), "confirmed-removal").unwrap();
    let published = Published::open(&archive.root).unwrap();
    for position in confirmations.len() as u64 + 1..=published.inventory().unwrap().len() {
        let unit = published.inventory().unwrap().unit(position).unwrap();
        let batch = published.read(&unit).unwrap();
        let bytes = contents(&batch);
        confirmations.push(RestoreUnit {
            excision: published.publication(&unit).unwrap(),
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: batch
                .blobs
                .iter()
                .map(|hash| format!("blake3:{}", BlobHash(*hash)))
                .collect(),
        });
    }
    let mut catalog = Catalog::new().unwrap();
    for (i, unit) in confirmations.iter().enumerate().rev() {
        catalog.push(i as u64 + 1, unit).unwrap();
    }
    let selected = catalog.finish().unwrap();
    selected
        .verify_source_blobs(|name| {
            let hash = BlobHash::parse(name.strip_prefix("blake3:").unwrap()).unwrap();
            let mut file = material.read_plain(&hashed_blobs::Hash(hash.0), None, || Ok(()))?;
            Ok(std::io::copy(&mut file, &mut std::io::sink())?)
        })
        .unwrap();
    let (target, _target_lock) = Archive::open_or_create(&dir.path().join("target")).unwrap();
    let mut restore = Restoring::with_roots(&target, vec![material_dir]).unwrap();
    restore
        .install(&id, &selected, |unit| {
            Ok(unit
                .batch
                .as_ref()
                .map(|_| Cursor::new(contents(&published.read(&unit.unit).unwrap()))))
        })
        .unwrap();
    assert!(
        restore
            .install::<std::io::Empty>(&id, &old, |_| panic!("stale view fetched historical bytes"))
            .is_err()
    );
    drop(restore);
    let db = crate::archive::open_read(&target.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM observation_container", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let inputs: Vec<String> = db
        .prepare("SELECT input FROM revision ORDER BY input")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(inputs, [BlobHash::of(b"surviving sibling").to_string()]);
    let refused = crate::excise::Refused::load(&db).unwrap();
    let group = linked(&target, "reintroduced", false);
    assert!(
        refused
            .raw_import(
                &group[0].batch.input.id,
                &BlobHash::of(ORIGINAL),
                group[0].batch.observation.as_ref().unwrap()
            )
            .unwrap()
    );
    assert!(
        refused
            .raw_import(
                &group[1].batch.input.id,
                &BlobHash::of(MEMBER),
                group[1].batch.observation.as_ref().unwrap()
            )
            .unwrap()
    );
}

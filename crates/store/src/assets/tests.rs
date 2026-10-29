//! Range delivery reads the current chain, including bytes in several new blobs.
use super::*;
use crate::{Archive, Batch, Change, InputKey, Observation, ReadPool, Writer};
use polylogue_model::BlobHash;
use std::sync::Arc;

#[test]
fn segmented_asset_ranges_and_hashes_are_exact_and_old_selections_refuse_rewrites() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let input = InputKey {
        id: "synthetic-segmented-asset".into(),
        format: "export-file".into(),
        arrival_rank: 0,
    };
    let sender = writer.sender().unwrap();
    let store = Arc::new(archive.blobs().unwrap());
    let observation = Observation {
        machine: "synthetic".into(),
        host: "synthetic".into(),
        path: "/synthetic/asset.bin".into(),
        path_bytes: b"/synthetic/asset.bin".to_vec(),
        facts: None,
        seen_at_ms: 1,
        import: None,
        container: None,
    };
    let original = "prefix λ🙂 suffix".as_bytes();
    let split = 8; // Inside the UTF-8 sequence; this API delivers bytes.
    for (start, end, continues) in [(0, split, false), (split, original.len(), true)] {
        let (new_blob, file) = archive.prepare_blob(&original[start..end]).unwrap();
        let mut stored = sender
            .store_group(
                vec![crate::staging::PreparedBatch {
                    batch: Batch {
                        required_disposition: None,
                        input: input.clone(),
                        change: Change::First,
                        observation: Some(observation.clone()),
                        continues,
                        new_blob: Some(new_blob),
                        blobs: vec![],
                        length: end as u64,
                        parse: vec![],
                        acquisition: None,
                        basis: None,
                        release: Default::default(),
                    },
                    files: file.into_iter().collect(),
                }],
                Arc::clone(&store),
            )
            .unwrap();
        stored.pop().expect("one captured source chunk").unwrap();
    }
    assert!(writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&archive).unwrap();
    let asset = {
        let reader = pool.reader().unwrap();
        let source = reader.asset_source(&input.id).unwrap().unwrap();
        assert!(
            source.blob.is_none(),
            "segmented bytes must not pretend to be one stored blob"
        );
        let asset = AssetMatch {
            input: input.id.clone(),
            chain: source.chain,
            blob: source.blob,
            source_version: source.version,
            size: source.length,
            members: vec![],
            names: vec![],
            media_types: vec![],
            declared_sizes: vec![],
        };
        assert_eq!(
            reader.attachment_hash(&asset).unwrap(),
            BlobHash::of(original)
        );
        assert_eq!(
            reader.attachment_range(&asset, 2..13).unwrap(),
            &original[2..13]
        );
        assert_eq!(
            reader.attachment_range(&asset, 0..asset.size).unwrap(),
            original
        );
        assert!(
            reader
                .attachment_range(&asset, asset.size..asset.size + 1)
                .is_err()
        );
        asset
    };
    let replacement = b"replacement body";
    let (new_blob, file) = archive.prepare_blob(replacement).unwrap();
    let mut stored = sender
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: Batch {
                    required_disposition: None,
                    input: input.clone(),
                    change: Change::Rewritten,
                    observation: Some(Observation {
                        seen_at_ms: 2,
                        ..observation
                    }),
                    continues: false,
                    new_blob: Some(new_blob),
                    blobs: vec![],
                    length: replacement.len() as u64,
                    parse: vec![],
                    acquisition: None,
                    basis: None,
                    release: Default::default(),
                },
                files: file.into_iter().collect(),
            }],
            Arc::clone(&store),
        )
        .unwrap();
    stored.pop().expect("one captured rewrite").unwrap();
    assert!(writer.flush().unwrap().errors.is_empty());
    let reader = pool.reader().unwrap();
    assert!(reader.attachment_hash(&asset).is_err());
    assert!(reader.attachment_range(&asset, 0..asset.size).is_err());
}

//! A material Sinex holds as content-defined chunks: its input's extents name
//! the chunks, the cache catalogs each chunk, and the chain reads back.
use super::*;
use crate::input_rows::{Extent, InputRows, Revision};
use polylogue_model::BlobHash;
use std::io::Read;

fn varied(length: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b"abcdefgh ij\n"[(state % 12) as usize]
        })
        .collect()
}

/// Stores `bytes` as chunks the way Sinex does: a chunk already held is not
/// stored again. Returns each chunk's descriptor in order.
fn store(
    writer: &mut hashed_blobs::SegmentWriter,
    held: &mut BTreeMap<[u8; 32], hashed_blobs::Extent>,
    scratch: &std::path::Path,
    bytes: &[u8],
) -> Vec<hashed_blobs::Extent> {
    let preparation = hashed_blobs::Preparation::new(scratch).unwrap();
    hashed_blobs::Chunker::plain(hashed_blobs::ChunkSizes::default())
        .chunks(bytes)
        .map(|chunk| {
            let chunk = chunk.unwrap();
            let hash = *blake3::hash(&chunk).as_bytes();
            if let Some(extent) = held.get(&hash) {
                return extent.clone();
            }
            let prepared = preparation.plain(chunk.as_slice(), 1).unwrap();
            let group = writer
                .append_group([Ok(prepared)], |_| Ok(None))
                .unwrap()
                .wait()
                .unwrap();
            writer.acknowledge(&group).unwrap();
            held.insert(hash, group.extents[0].clone());
            group.extents[0].clone()
        })
        .collect()
}

fn step(
    position: i64,
    revision: Revision,
    offset: u64,
    bytes: &[u8],
    descriptors: &[hashed_blobs::Extent],
) -> Vec<Confirmed> {
    let mut extents = Vec::new();
    let mut at = 0;
    for descriptor in descriptors {
        extents.push(Extent {
            chain: revision.chain,
            offset: at,
            length: descriptor.decoded_len as i64,
            blob: BlobHash(descriptor.hash.0),
            blob_offset: 0,
        });
        at += descriptor.decoded_len as i64;
    }
    // A step trimmed to the confirmed prefix keeps the material's tail.
    crate::input_rows::drop_front(&mut extents, offset);
    let rows = InputRows {
        acquisition: None,
        input: "document".into(),
        format: "synthetic".into(),
        revision: Some(revision),
        extents,
        observation: None,
    };
    vec![
        Confirmed {
            position,
            event: format!("input-{position}"),
            output_index: 0,
            native: None,
            change: ConfirmedChange::Live(Box::new(Product::Input {
                rows: Box::new(rows),
                material: Some(Material {
                    id: format!("material-{position}"),
                    hash: BlobHash::of(bytes),
                    length: bytes.len() as u64,
                    extents: descriptors
                        .iter()
                        .map(hashed_blobs::Extent::encode)
                        .collect(),
                }),
            })),
        },
        barrier(position),
    ]
}

fn chain(archive: &crate::Archive, chain: i64, length: usize) -> Vec<u8> {
    let stored = crate::blobs::StoredChain::load(
        &crate::blobs::ChainSource::of(archive),
        "document",
        chain,
        length as u64,
    )
    .unwrap();
    let mut bytes = Vec::new();
    stored
        .reader(Arc::new(archive.blob_reader().unwrap()))
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

#[test]
fn chunked_materials_catalog_each_chunk_and_their_chains_read_back() {
    let root = tempfile::tempdir().unwrap();
    let mut index = cache(&root.path().join("index.db"));
    let archive = crate::Archive::on_sinex(root.path(), &root.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let mut writer = hashed_blobs::SegmentWriter::open(
        root.path().join("bytes"),
        "segments/chunks.hb",
        0,
        &[],
        0o600,
    )
    .unwrap();
    let scratch = root.path().join("scratch");
    let mut held = BTreeMap::new();

    let first = varied(400 << 10, 7);
    let mut second = first.clone();
    second[200 << 10] ^= 1;
    let mut third = second.clone();
    third.extend(varied(100 << 10, 11));

    let one = store(&mut writer, &mut held, &scratch, &first);
    assert!(one.len() > 2);
    let revision = |number, produced: &str, chain, length: &[u8]| Revision {
        number,
        produced: produced.into(),
        chain,
        length: length.len() as i64,
    };
    apply(
        &mut index,
        &resolver,
        &step(1, revision(1, "first", 1, &first), 0, &first, &one),
    )
    .unwrap();
    let two = store(&mut writer, &mut held, &scratch, &second);
    apply(
        &mut index,
        &resolver,
        &step(2, revision(2, "rewritten", 2, &second), 0, &second, &two),
    )
    .unwrap();
    // The third material is the whole grown file; its step keeps only the
    // bytes past the confirmed second revision.
    let three = store(&mut writer, &mut held, &scratch, &third);
    apply(
        &mut index,
        &resolver,
        &step(
            3,
            revision(3, "appended", 2, &third),
            second.len() as u64,
            &third,
            &three,
        ),
    )
    .unwrap();

    let catalogued: i64 = index
        .query_row("SELECT count(*) FROM source_blob", [], |row| row.get(0))
        .unwrap();
    assert_eq!(catalogued, held.len() as i64);
    let placed: Vec<(String, i64, i64, i64)> = index
        .prepare("SELECT id, offset, source_length, blob_offset FROM source_material ORDER BY id")
        .unwrap()
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let (first_len, second_len, third_len) =
        (first.len() as i64, second.len() as i64, third.len() as i64);
    assert_eq!(
        placed,
        [
            ("material-1".into(), 0, first_len, 0),
            ("material-2".into(), 0, second_len, 0),
            (
                "material-3".into(),
                second_len,
                third_len - second_len,
                second_len
            ),
        ]
    );
    assert!(chain(&archive, 1, first.len()) == first);
    assert!(chain(&archive, 2, third.len()) == third);
}

use super::*;
use crate::input_rows::{Extent, InputRows, ObservationRow, Revision};

fn material(id: &str, bytes: &[u8]) -> Material {
    let hash = polylogue_model::BlobHash::of(bytes);
    Material {
        id: id.into(),
        hash,
        length: bytes.len() as u64,
        extents: vec![
            hashed_blobs::Extent {
                segment: "segments/synthetic.hb".into(),
                offset: 0,
                len: 100,
                decoded_len: bytes.len() as u64,
                hash: hashed_blobs::Hash(hash.0),
                kind: hashed_blobs::Kind::Plain,
                index_hash: hashed_blobs::Hash([1; 32]),
                encoded_hash: hashed_blobs::Hash([2; 32]),
            }
            .encode(),
        ],
    }
}
#[test]
fn confirmed_prefix_observation_survives_source_archive_and_restore() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let full = material("full-eight", b"abcdefgh");
    let short = material("prefix-four", b"abcd");
    let source = Confirmed {
        position: 1,
        output_index: 0,
        event: "full-source".into(),
        change: ConfirmedChange::Live(Product::Input {
            rows: InputRows {
                input: "native".into(),
                format: "synthetic".into(),
                revision: Some(Revision {
                    number: 1,
                    produced: "first".into(),
                    chain: 1,
                    length: 8,
                }),
                extent: Some(Extent {
                    chain: 1,
                    offset: 0,
                    length: 8,
                    blob: full.hash,
                    blob_offset: 0,
                }),
                observation: None,
            },
            material: Some(full),
        }),
    };
    let prefix = Confirmed {
        position: 2,
        output_index: 0,
        event: "prefix-observation".into(),
        change: ConfirmedChange::Live(Product::Input {
            rows: InputRows {
                input: "native".into(),
                format: "synthetic".into(),
                revision: None,
                extent: None,
                observation: Some(ObservationRow {
                    id: "prefix-observation".into(),
                    machine: "machine".into(),
                    host: "host".into(),
                    path: "/home/op/session".into(),
                    class: "back".into(),
                    revision: None,
                    chain: Some(1),
                    length: Some(4),
                    device: None,
                    inode: None,
                    size: Some(4),
                    mtime_ns: None,
                    ctime_ns: None,
                    polylogue_version: "test".into(),
                    seen_at: 20,
                    details_json: None,
                    container: None,
                }),
            },
            material: None,
        }),
    };
    let raw = Confirmed {
        position: 3,
        output_index: 0,
        event: "prefix-raw".into(),
        change: ConfirmedChange::Live(Product::Blob(short.clone())),
    };
    apply(&mut index, &resolver, &[source, prefix, raw, barrier(3)]).unwrap();
    assert_eq!(
        crate::source::Source::reading_receipt(&index, "native", 1, 4)
            .unwrap()
            .observation
            .as_deref(),
        Some("prefix-observation")
    );
    for (position, change, revisions) in [
        (4, ConfirmedChange::Archived, 0),
        (5, ConfirmedChange::Restored, 1),
    ] {
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position,
                    output_index: 0,
                    event: "full-source".into(),
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        assert_eq!(
            index
                .query_row(
                    "SELECT count(*) FROM source_revision WHERE input='native'",
                    [],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
            revisions
        );
        assert_eq!(index.query_row("SELECT revision,chain,length FROM source_observation WHERE id='prefix-observation'",[],|row|Ok((row.get::<_,Option<i64>>(0)?,row.get::<_,i64>(1)?,row.get::<_,i64>(2)?))).unwrap(),(None,1,4));
        assert!(
            index
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1)",
                    [short.hash.0.as_slice()],
                    |row| row.get::<_, bool>(0)
                )
                .unwrap()
        );
    }
    assert_eq!(
        index
            .query_row(
                "SELECT length FROM source_revision WHERE input='native'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        8
    );
    assert_eq!(
        crate::source::Source::reading_receipt(&index, "native", 1, 4)
            .unwrap()
            .observation
            .as_deref(),
        Some("prefix-observation")
    );
}

use super::*;
use crate::{Archive, Batch, Change, InputKey, Observation, Writer};
use std::sync::Arc;

/// The writer copies a staged encoding into the segment as it is: it checks
/// the encoding's length and digest against the descriptor its preparation
/// computed, and never reads the staged bytes back to decode them. A staged
/// file replaced by same-length bytes that are no encoding at all, with a
/// descriptor naming them, is therefore stored; only a later read, which
/// decodes, finds it invalid.
#[test]
fn writer_never_decodes_staged_input() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let bytes = b"source bytes the writer stores without decoding them\n";
    let (new_blob, file) = archive.prepare_blob(bytes).unwrap();
    let mut file = file.expect("a new blob has a staged file");
    let undecodable = vec![0xa5u8; usize::try_from(file.descriptor.stored_len).unwrap()];
    std::fs::write(file.temporary.as_ref(), &undecodable).unwrap();
    file.descriptor.encoded_hash = hashed_blobs::Hash::of(&undecodable);
    let batch = Batch {
        required_disposition: None,
        input: InputKey {
            id: "staged".into(),
            format: "test-bytes".into(),
            route: polylogue_model::Route::Live,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "test-machine".into(),
            host: "test-host".into(),
            path: "/home/op/staged".into(),
            path_bytes: b"/home/op/staged".to_vec(),
            facts: None,
            seen_at_ms: 1,
            import: None,
            container: None,
            database: None,
        }),
        continues: false,
        new_blob: Some(new_blob),
        blobs: Vec::new(),
        length: bytes.len() as u64,
        parse: Vec::new(),
        acquisition: None,
        basis: None,
        release: Default::default(),
    };
    let results = writer
        .sender()
        .unwrap()
        .store_group(
            vec![PreparedBatch {
                batch,
                files: vec![file],
            }],
            Arc::new(archive.blobs().unwrap()),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].as_ref().unwrap().length, bytes.len() as u64);
    let read = archive
        .blobs()
        .unwrap()
        .read(&hashed_blobs::Hash::of(bytes))
        .and_then(|mut reader| {
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut out)?;
            Ok(out)
        });
    assert!(read.is_err(), "{read:?}");
}

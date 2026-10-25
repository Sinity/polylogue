use super::*;
use crate::schema;
use polylogue_model::BlobHash;
use std::os::unix::fs::{MetadataExt, PermissionsExt};

fn row(index: &Connection, bytes: &[u8], place: &str, data: Option<&[u8]>) -> Extent {
    let hash = BlobHash::of(bytes);
    index
        .execute(
            "INSERT OR IGNORE INTO input(id, format) VALUES ('test', 'synthetic-text')",
            [],
        )
        .unwrap();
    index
        .execute(
            "INSERT INTO blob(hash, size, stored_size, place, encoding, state, data)
        VALUES (?1, ?2, 0, ?3, 'plain', 'present', ?4)",
            rusqlite::params![&hash.0[..], bytes.len() as i64, place, data],
        )
        .unwrap();
    Extent {
        input: "test".into(),
        chain: 1,
        offset: 0,
        length: bytes.len() as u64,
        blob: hash,
        blob_offset: 0,
    }
}

#[test]
fn oversized_file_frames_decode_once_for_all_shared_and_reordered_extents() {
    let dir = tempfile::tempdir().unwrap();
    let bytes: Vec<u8> = (0..8 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600)
        .unwrap()
        .with_frame_size(bytes.len());
    store.put(bytes.as_slice(), 1).unwrap();
    let index = Connection::open_in_memory().unwrap();
    schema::create_archive(&index).unwrap();
    let mut extent = row(&index, &bytes, "file", None);
    let mut cache = PlainBytes::new().unwrap();
    let plan = Plan::new().unwrap();
    let mut calls = 0;
    for chunk in (0..128).rev().chain(0..128) {
        extent.blob_offset = chunk * READ_CHUNK as u64;
        extent.length = READ_CHUNK as u64;
        extent.offset = 57;
        cache
            .visit(&plan, &index, &store, &extent, |found, at| {
                assert!(found.len() <= READ_CHUNK);
                let from = (extent.blob_offset + at - extent.offset) as usize;
                assert_eq!(found, &bytes[from..from + found.len()]);
                calls += 1;
                Ok(())
            })
            .unwrap();
    }
    assert_eq!(calls, 256);
    assert_eq!(
        plan.placement(&extent.blob).unwrap(),
        Some((0, bytes.len() as u64))
    );
    assert_eq!(cache.length, bytes.len() as u64);
    // One shared-library integrity pass and one sequential copy, irrespective
    // of the number or order of logical visits to an oversized frame.
    assert_eq!(cache.decoded, 2 * bytes.len() as u64);
    let metadata = cache.file.metadata().unwrap();
    assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    assert_eq!(metadata.nlink(), 0);
}

#[test]
fn oversized_historical_inline_rows_use_incremental_blob_io_and_one_decode() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let bytes = vec![b'a'; 4 * 1024 * 1024];
    let compressed = zstd::bulk::compress(&bytes, 1).unwrap();
    let index = Connection::open_in_memory().unwrap();
    schema::create_archive(&index).unwrap();
    let mut extent = row(&index, &bytes, "inline", Some(&compressed));
    let mut cache = PlainBytes::new().unwrap();
    let plan = Plan::new().unwrap();
    let mut total = 0;
    for offset in [0, 3 * 1024 * 1024, 1024 * 1024, 0] {
        extent.blob_offset = offset;
        extent.length = 1024 * 1024;
        cache
            .visit(&plan, &index, &store, &extent, |found, _| {
                assert!(found.len() <= READ_CHUNK);
                assert!(found.iter().all(|b| *b == b'a'));
                total += found.len();
                Ok(())
            })
            .unwrap();
    }
    assert_eq!(total, 4 * 1024 * 1024);
    assert_eq!(cache.decoded, bytes.len() as u64);
}

#[test]
fn inline_corruption_and_extent_overflow_expose_no_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    for (stored, offset, length, corrupt) in [
        (b"public".as_slice(), 0, 6, true),
        (b"secret".as_slice(), 5, 2, false),
        (b"secret".as_slice(), u64::MAX, 2, false),
    ] {
        let index = Connection::open_in_memory().unwrap();
        schema::create_archive(&index).unwrap();
        let encoded = zstd::bulk::compress(stored, 1).unwrap();
        let mut extent = row(&index, b"secret", "inline", Some(&encoded));
        extent.blob_offset = offset;
        extent.length = length;
        let mut cache = PlainBytes::new().unwrap();
        let plan = Plan::new().unwrap();
        let mut exposed = false;
        let error = cache
            .visit(&plan, &index, &store, &extent, |_, _| {
                exposed = true;
                Ok(())
            })
            .unwrap_err();
        assert!(!exposed);
        if corrupt {
            assert!(matches!(error, StoreError::BlobCorrupt(_)));
            assert!(plan.placement(&extent.blob).unwrap().is_none());
        } else {
            assert!(matches!(error, StoreError::Invalid(_)));
        }
    }
}

#[test]
fn visitor_errors_stop_without_reading_more_logical_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let bytes = vec![b'x'; READ_CHUNK * 3];
    let compressed = zstd::bulk::compress(&bytes, 1).unwrap();
    let index = Connection::open_in_memory().unwrap();
    schema::create_archive(&index).unwrap();
    let extent = row(&index, &bytes, "inline", Some(&compressed));
    let mut cache = PlainBytes::new().unwrap();
    let plan = Plan::new().unwrap();
    let mut calls = 0;
    let error = cache
        .visit(&plan, &index, &store, &extent, |_, _| {
            calls += 1;
            Err(StoreError::Invalid("stop".into()))
        })
        .unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)));
    assert_eq!(calls, 1);
}

#[test]
fn failed_verified_copy_publishes_no_placement_and_can_retry_exact_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let index = Connection::open_in_memory().unwrap();
    schema::create_archive(&index).unwrap();
    let wrong = zstd::bulk::compress(b"public", 1).unwrap();
    let extent = row(&index, b"secret", "inline", Some(&wrong));
    let mut cache = PlainBytes::new().unwrap();
    let plan = Plan::new().unwrap();
    assert!(matches!(
        cache.visit(&plan, &index, &store, &extent, |_, _| {
            panic!("a failed copy must expose no bytes")
        }),
        Err(StoreError::BlobCorrupt(_))
    ));
    assert!(plan.placement(&extent.blob).unwrap().is_none());
    assert_eq!(cache.length, 0);
    index
        .execute(
            "UPDATE blob SET data=?1",
            [zstd::bulk::compress(b"secret", 1).unwrap()],
        )
        .unwrap();
    let mut actual = Vec::new();
    cache
        .visit(&plan, &index, &store, &extent, |bytes, _| {
            actual.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, b"secret");
    assert_eq!(plan.placement(&extent.blob).unwrap(), Some((0, 6)));
    assert_eq!(cache.length, 6);
}

#[test]
fn many_snapshot_headers_share_one_verified_decode_and_rewrite_exact_payloads() {
    use crate::excise::stored_text::framing;
    use polylogue_model::capture_log::{HEADER_LEN, Header, MAGIC};
    let payload = br#"{"v":"secret"}"#;
    let count = 5000;
    let mut original = MAGIC.to_vec();
    for _ in 0..count {
        original.extend(Header::of(payload).encode());
        original.extend(payload);
    }
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600)
        .unwrap()
        .with_frame_size(original.len());
    store.put(original.as_slice(), 1).unwrap();
    let archive = Connection::open_in_memory().unwrap();
    schema::create_archive(&archive).unwrap();
    let e = row(&archive, &original, "file", None);
    archive.execute("INSERT INTO revision(input,number,produced,chain,length,blake3,commit_seq) VALUES ('test',1,'first',1,?1,?2,1)",
        rusqlite::params![original.len() as i64,&e.blob.0[..]]).unwrap();
    archive.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES ('test',1,0,?1,?2,0)",
        rusqlite::params![original.len() as i64,&e.blob.0[..]]).unwrap();
    let plan = Plan::new().unwrap();
    let mut cache = PlainBytes::new().unwrap();
    let key = ("test".into(), 1);
    framing::scan(
        &archive,
        &store,
        &mut cache,
        &plan,
        &key,
        original.len() as u64,
        "secret",
    )
    .unwrap();
    assert_eq!(plan.counts().unwrap()["test"], count);
    assert_eq!(plan.frames(&key).len() as u64, count);
    framing::rewrite(&archive, &store, &mut cache, &plan, &key, b' ').unwrap();
    let mut reader = cache
        .reader(&plan, &archive, &store, &key, 0..original.len() as u64)
        .unwrap();
    let mut edits = plan.edits(&key, 0).unwrap();
    let mut actual = Vec::new();
    let mut buffer = [0; READ_CHUNK];
    loop {
        let at = reader.position();
        let n = reader.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        edits.rewrite(at, &mut buffer[..n], b' ').unwrap();
        actual.extend_from_slice(&buffer[..n]);
    }
    assert_eq!(cache.decoded, 2 * original.len() as u64);
    assert_eq!(&actual[..MAGIC.len()], MAGIC);
    let mut at = MAGIC.len();
    for _ in 0..count {
        let header = Header::decode(&actual[at..at + HEADER_LEN]).unwrap();
        assert_eq!(header.observed_digest, Header::of(payload).observed_digest);
        at += HEADER_LEN;
        let body = &actual[at..at + payload.len()];
        assert!(header.verifies(body));
        assert_eq!(body, br#"{"v":"      "}"#);
        at += payload.len();
    }
}

#[test]
fn whitespace_matches_never_include_structural_snapshot_headers() {
    use crate::excise::stored_text::framing;
    use polylogue_model::capture_log::{HEADER_LEN, Header, MAGIC};
    let payload = b"x y";
    let mut original = MAGIC.to_vec();
    original.extend(Header::of(payload).encode());
    original.extend(payload);
    let dir = tempfile::tempdir().unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    store.put(original.as_slice(), 1).unwrap();
    let archive = Connection::open_in_memory().unwrap();
    schema::create_archive(&archive).unwrap();
    let e = row(&archive, &original, "file", None);
    archive.execute("INSERT INTO revision(input,number,produced,chain,length,blake3,commit_seq) VALUES ('test',1,'first',1,?1,?2,1)",
        rusqlite::params![original.len() as i64,&e.blob.0[..]]).unwrap();
    archive.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES ('test',1,0,?1,?2,0)",
        rusqlite::params![original.len() as i64,&e.blob.0[..]]).unwrap();
    let plan = Plan::new().unwrap();
    let mut cache = PlainBytes::new().unwrap();
    let key = ("test".into(), 1);
    framing::scan(
        &archive,
        &store,
        &mut cache,
        &plan,
        &key,
        original.len() as u64,
        " ",
    )
    .unwrap();
    assert_eq!(plan.counts().unwrap()["test"], 1);
    framing::rewrite(&archive, &store, &mut cache, &plan, &key, b'*').unwrap();
    let mut output = original.clone();
    plan.edits(&key, 0)
        .unwrap()
        .rewrite(0, &mut output, b'*')
        .unwrap();
    assert_eq!(&output[..MAGIC.len()], MAGIC);
    let header = Header::decode(&output[MAGIC.len()..MAGIC.len() + HEADER_LEN]).unwrap();
    assert!(header.verifies(b"x*y"));
    assert_eq!(&output[MAGIC.len() + HEADER_LEN..], b"x*y");
}

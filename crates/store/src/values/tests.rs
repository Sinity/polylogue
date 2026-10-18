use super::*;
fn world() -> (tempfile::TempDir, std::path::PathBuf, Writer) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let writer = Writer::open(&root).unwrap();
    (dir, root, writer)
}
fn append(writer: &mut Writer, codec: Codec, bytes: &[u8], chunk: usize) -> Head {
    let mut h = writer.begin_value(codec, bytes.len() as u64).unwrap();
    for bytes in bytes.chunks(chunk) {
        h = writer
            .append_value_chunk(&h.id, h.appended, bytes, bytes.len())
            .unwrap();
    }
    h
}
fn validate(writer: &Writer, id: &ValueId, allowance: usize) -> Validation {
    let reader = writer.read_handle().unwrap();
    let mut proof = reader.validate_value(id).unwrap();
    while !proof.progress().complete {
        proof.advance(&reader, allowance).unwrap();
    }
    proof
}
#[test]
fn appends_survive_restart_and_hashing_restarts_at_zero() {
    let (_dir, root, mut writer) = world();
    let h = writer.begin_value(Codec::Utf8, 11).unwrap();
    let h = writer.append_value_chunk(&h.id, 0, b"hello", 5).unwrap();
    assert_eq!(h.appended, 5);
    assert!(matches!(
        writer.validate_value(&h.id),
        Err(Error::ValueNotReady)
    ));
    assert!(matches!(
        writer.read_handle().unwrap().value_range(&h.id, 0, 3),
        Err(Error::ValueNotReady)
    ));
    drop(writer);
    let mut writer = Writer::open(&root).unwrap();
    let h = writer.value_head(&h.id).unwrap();
    assert_eq!(h.appended, 5);
    let h = writer.append_value_chunk(&h.id, 5, b" world", 6).unwrap();
    let reader = writer.read_handle().unwrap();
    let mut proof = writer.validate_value(&h.id).unwrap();
    assert_eq!(proof.advance(&reader, 2).unwrap().hashed_upto, 2);
    assert!(matches!(
        writer.seal_value(&proof),
        Err(Error::ValueNotReady)
    ));
    drop(proof);
    drop(reader);
    drop(writer);
    let mut writer = Writer::open(&root).unwrap();
    let proof = writer.validate_value(&h.id).unwrap();
    assert_eq!(proof.progress().hashed_upto, 0, "hashing restarts at zero");
    let proof = validate(&writer, &h.id, 1);
    let sealed = writer.seal_value(&proof).unwrap();
    assert_eq!(sealed.ready.hash, *blake3::hash(b"hello world").as_bytes());
    let reader = writer.read_handle().unwrap();
    assert_eq!(
        reader.value_range(&sealed.ready.id, 3, 10).unwrap().bytes,
        b"lo"
    );
    assert_eq!(
        reader.value_range(&sealed.ready.id, 4, 1).unwrap().bytes,
        b"o"
    );
    assert_eq!(
        reader.value_range(&sealed.ready.id, 5, 3).unwrap().bytes,
        b" wo"
    );
    let eof = reader.value_range(&sealed.ready.id, 11, 0).unwrap();
    assert!(eof.eof && eof.bytes.is_empty());
    assert!(matches!(
        reader.value_range(&sealed.ready.id, 12, 1),
        Err(Error::Range)
    ));
    assert!(matches!(
        reader.value_range(&sealed.ready.id, 0, 0),
        Err(Error::ParserAdmission {
            resource: "normalized value read bytes",
            required: 1
        })
    ));
}
#[test]
fn empty_values_store_no_chunks_and_deduplicate() {
    let (_dir, _root, mut writer) = world();
    let h = writer.begin_value(Codec::Bytes, 0).unwrap();
    assert!(writer.append_value_chunk(&h.id, 0, b"", 0).is_err());
    let proof = writer.validate_value(&h.id).unwrap();
    assert!(proof.progress().complete);
    let sealed = writer.seal_value(&proof).unwrap();
    assert_eq!(sealed.ready.hash, *blake3::hash(b"").as_bytes());
    let duplicate = writer.begin_value(Codec::Bytes, 0).unwrap();
    let proof = writer.validate_value(&duplicate.id).unwrap();
    let result = writer.seal_value(&proof).unwrap();
    assert_eq!(result.ready.id, sealed.ready.id);
    assert_eq!(result.retired_duplicate, Some(duplicate.id.clone()));
    assert_eq!(
        writer.value_head(&duplicate.id).unwrap().state,
        State::Retiring
    );
    let count: i64 = crate::db(&writer.root.join("index.db"), || {
        writer
            ._index
            .query_row("SELECT count(*) FROM body_value_chunk", [], |r| r.get(0))
    })
    .unwrap();
    assert_eq!(count, 0);
    let different_codec = writer.begin_value(Codec::Utf8, 0).unwrap();
    let proof = writer.validate_value(&different_codec.id).unwrap();
    assert_ne!(writer.seal_value(&proof).unwrap().ready.id, sealed.ready.id);
}
#[test]
fn chunk_size_offset_and_range_limits() {
    let (_dir, _root, mut writer) = world();
    let bytes = vec![b'x'; MAX_CHUNK * 2 + 17];
    let h = writer
        .begin_value(Codec::Bytes, bytes.len() as u64)
        .unwrap();
    assert!(matches!(
        writer.append_value_chunk(&h.id, 0, &bytes[..MAX_CHUNK + 1], usize::MAX),
        Err(Error::ValueChunkSplitRequired {required,maximum}) if required==MAX_CHUNK+1 && maximum==MAX_CHUNK
    ));
    assert!(matches!(
        writer.append_value_chunk(&h.id, 0, b"xx", 1),
        Err(Error::ParserAdmission {
            resource: "normalized value append bytes",
            required: 2
        })
    ));
    assert_eq!(writer.value_head(&h.id).unwrap().appended, 0);
    writer
        .append_value_chunk(&h.id, 0, &bytes[..MAX_CHUNK], MAX_CHUNK)
        .unwrap();
    assert!(matches!(
        writer.append_value_chunk(&h.id, 0, b"x", 1),
        Err(Error::StaleValue)
    ));
    assert!(matches!(
        writer.append_value_chunk(&h.id, (MAX_CHUNK + 1) as u64, b"x", 1),
        Err(Error::StaleValue)
    ));
    writer
        .append_value_chunk(
            &h.id,
            MAX_CHUNK as u64,
            &bytes[MAX_CHUNK..2 * MAX_CHUNK],
            MAX_CHUNK,
        )
        .unwrap();
    writer
        .append_value_chunk(&h.id, (2 * MAX_CHUNK) as u64, &bytes[2 * MAX_CHUNK..], 17)
        .unwrap();
    assert!(matches!(
        writer.append_value_chunk(&h.id, bytes.len() as u64, b"x", 1),
        Err(Error::Range)
    ));
    let proof = validate(&writer, &h.id, 257);
    let ready = writer.seal_value(&proof).unwrap().ready;
    assert_eq!(ready.hash, *blake3::hash(&bytes).as_bytes());
    let reader = writer.read_handle().unwrap();
    let range = reader.value_range(&ready.id, 0, usize::MAX).unwrap();
    assert_eq!(range.bytes.len(), MAX_CHUNK);
    assert!(!range.eof);
    let range = reader
        .value_range(&ready.id, (MAX_CHUNK - 1) as u64, usize::MAX)
        .unwrap();
    assert_eq!(range.bytes.len(), 1);
}
#[test]
fn utf8_check_handles_every_split_point() {
    for codec in [Codec::Utf8, Codec::Json] {
        let (_dir, _root, mut writer) = world();
        let bytes = "aé日🦀z".as_bytes();
        let h = append(&mut writer, codec, bytes, 1);
        let proof = validate(&writer, &h.id, 1);
        writer.seal_value(&proof).unwrap();
    }
    for bytes in [&b"\xf0\x9f"[..], &b"\xed\xa0\x80"[..], &b"\xc0\x80"[..]] {
        let (_dir, _root, mut writer) = world();
        let h = append(&mut writer, Codec::Utf8, bytes, 1);
        let reader = writer.read_handle().unwrap();
        let mut proof = writer.validate_value(&h.id).unwrap();
        while !proof.failed {
            let _ = proof.advance(&reader, 1);
        }
        let offset = proof.progress().hashed_upto;
        assert!(proof.advance(&reader, 1).is_err());
        assert_eq!(offset, proof.progress().hashed_upto);
        assert!(matches!(
            writer.seal_value(&proof),
            Err(Error::ValueNotReady)
        ));
        assert_eq!(writer.value_head(&h.id).unwrap().state, State::Building);
    }
}
#[test]
fn missing_or_overlapping_chunks_fail_validation() {
    for overlap in [false, true] {
        let (_dir, _root, mut writer) = world();
        let h = append(&mut writer, Codec::Bytes, b"abcdefgh", 4);
        let path = writer.root.join("index.db");
        if overlap {
            crate::db(&path, || {
                writer._index.execute(
                    "INSERT INTO body_value_chunk(value_pk,offset,bytes) VALUES(?1,2,?2)",
                    params![h.id.pk, b"XYZW".as_slice()],
                )
            })
            .unwrap();
        } else {
            crate::db(&path, || {
                writer._index.execute(
                    "DELETE FROM body_value_chunk WHERE value_pk=?1 AND offset=0",
                    [h.id.pk],
                )
            })
            .unwrap();
        }
        let reader = writer.read_handle().unwrap();
        let mut proof = writer.validate_value(&h.id).unwrap();
        let mut failed = false;
        for _ in 0..12 {
            if proof.advance(&reader, 1).is_err() {
                failed = true;
                break;
            }
        }
        assert!(failed);
        assert!(matches!(
            writer.seal_value(&proof),
            Err(Error::ValueNotReady)
        ));
    }
}
#[test]
fn deleted_row_new_index_or_sql_error_stops_validation() {
    let (_dir, _root, mut writer) = world();
    let empty = writer.begin_value(Codec::Bytes, 0).unwrap();
    let proof = writer.validate_value(&empty.id).unwrap();
    let path = writer.root.join("index.db");
    crate::db(&path, || {
        writer
            ._index
            .execute("DELETE FROM body_value WHERE pk=?1", [empty.id.pk])
    })
    .unwrap();
    let replacement = writer.begin_value(Codec::Bytes, 0).unwrap();
    assert_ne!(empty.id.pk, replacement.id.pk);
    assert!(matches!(writer.seal_value(&proof), Err(Error::StaleValue)));
    let proof = writer.validate_value(&replacement.id).unwrap();
    crate::db(&path, || {
        writer._index.execute(
            "UPDATE meta SET value='synthetic-new-index' WHERE key='generation'",
            [],
        )
    })
    .unwrap();
    assert!(matches!(writer.seal_value(&proof), Err(Error::StaleValue)));
    let (_dir2, _root2, mut writer2) = world();
    let h = append(&mut writer2, Codec::Bytes, b"x", 1);
    let mut proof = writer2.validate_value(&h.id).unwrap();
    let reader = writer2.read_handle().unwrap();
    let path = writer2.root.join("index.db");
    crate::db(&path, || {
        writer2._index.execute_batch("DROP TABLE body_value_chunk")
    })
    .unwrap();
    assert!(matches!(proof.advance(&reader,1),Err(Error::Sqlite {file,..}) if file==path));
    assert!(proof.failed);
    assert_eq!(proof.progress().hashed_upto, 0);
}
#[test]
fn validation_rejects_another_archive_with_the_same_generation() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = Writer::open(&dir.path().join("first")).unwrap();
    let mut second = Writer::open(&dir.path().join("second")).unwrap();
    let first_value = first.begin_value(Codec::Bytes, 1).unwrap();
    first
        .append_value_chunk(&first_value.id, 0, b"a", 1)
        .unwrap();
    let second_value = second.begin_value(Codec::Bytes, 1).unwrap();
    second
        .append_value_chunk(&second_value.id, 0, b"b", 1)
        .unwrap();
    let path = second.root.join("index.db");
    crate::db(&path, || {
        second._index.execute(
            "UPDATE meta SET value=?1 WHERE key='generation'",
            [&first_value.id.index_generation],
        )
    })
    .unwrap();
    // Same generation as a copied index.db would have.
    let reader = second.read_handle().unwrap();
    let mut proof = first.validate_value(&first_value.id).unwrap();
    assert!(matches!(proof.advance(&reader, 1), Err(Error::StaleValue)));
    assert!(matches!(
        reader.value_head(&first_value.id),
        Err(Error::StaleValue)
    ));
    assert_eq!(proof.progress().hashed_upto, 0);
}
fn incoming(writer: &Writer, id: &ValueId) -> i64 {
    crate::db(&writer.root.join("index.db"), || {
        writer._index.query_row(
            "SELECT incoming FROM body_value WHERE pk=?1",
            [id.pk],
            |r| r.get(0),
        )
    })
    .unwrap()
}
#[test]
fn rolled_back_begin_leaves_no_value() {
    let (_dir, _root, mut writer) = world();
    let path = writer.root.join("index.db");
    {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        let pending = begin_in(&tx, &path, Codec::Bytes, 0, 1).unwrap();
        let count: i64 = crate::db(&path, || {
            tx.query_row(
                "SELECT incoming FROM body_value WHERE pk=?1",
                [pending.id.pk],
                |r| r.get(0),
            )
        })
        .unwrap();
        assert_eq!(count, 1);
        // SQLite may reuse a rolled-back pk, so the Head must not escape.
        drop(pending);
        crate::db(&path, || tx.rollback()).unwrap();
    }
    let count: i64 = crate::db(&path, || {
        writer
            ._index
            .query_row("SELECT count(*) FROM body_value", [], |r| r.get(0))
    })
    .unwrap();
    assert_eq!(count, 0);
    let committed = writer.begin_value(Codec::Bytes, 0).unwrap();
    assert_eq!(
        writer.value_head(&committed.id).unwrap().state,
        State::Building
    );
}
#[test]
fn dedup_moves_leases_atomically_and_rolls_back_on_error() {
    let (_dir, _root, mut writer) = world();
    let path = writer.root.join("index.db");
    let first = writer.begin_value(Codec::Bytes, 0).unwrap();
    let proof = writer.validate_value(&first.id).unwrap();
    let first = writer.seal_value(&proof).unwrap().ready;
    crate::db(&path, || {
        writer._index.execute_batch(
            "CREATE TEMP TABLE producer_lease(id INTEGER PRIMARY KEY,value_pk INTEGER NOT NULL)",
        )
    })
    .unwrap();
    let pending = {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        let h = begin_in(&tx, &path, Codec::Bytes, 0, 1).unwrap();
        crate::db(&path, || {
            tx.execute(
                "INSERT INTO producer_lease(id,value_pk) VALUES(1,?1)",
                [h.id.pk],
            )
        })
        .unwrap();
        crate::db(&path, || tx.commit()).unwrap();
        h
    };
    let proof = writer.validate_value(&pending.id).unwrap();
    {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        assert!(matches!(
            seal_in(&tx, &path, &proof, |_, _| Ok(())),
            Err(Error::StaleValue)
        ));
        crate::db(&path, || tx.rollback()).unwrap();
    }
    let transfer = |tx: &Transaction<'_>, ready: &ReadyValue| -> Result<()> {
        let changed = crate::db(&path, || {
            tx.execute(
                "UPDATE producer_lease SET value_pk=?2 WHERE id=1 AND value_pk=?1",
                params![pending.id.pk, ready.id.pk],
            )
        })?;
        if changed != 1 {
            return Err(Error::StaleValue);
        }
        let changed = crate::db(&path, || {
            tx.execute(
                "UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming=1",
                [pending.id.pk],
            )
        })?;
        if changed != 1 {
            return Err(Error::StaleValue);
        }
        crate::db(&path, || {
            tx.execute(
                "UPDATE body_value SET incoming=incoming+1 WHERE pk=?1 AND state='ready'",
                [ready.id.pk],
            )
        })?;
        Ok(())
    };
    {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        let result = seal_in(&tx, &path, &proof, |tx, ready| {
            transfer(tx, ready)?;
            Err(Error::Cancelled)
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        crate::db(&path, || tx.rollback()).unwrap();
    }
    assert_eq!(
        writer.value_head(&pending.id).unwrap().state,
        State::Building
    );
    assert_eq!(incoming(&writer, &pending.id), 1);
    assert_eq!(incoming(&writer, &first.id), 0);
    let result = {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        let result = seal_in(&tx, &path, &proof, transfer).unwrap();
        crate::db(&path, || tx.commit()).unwrap();
        result
    };
    assert_eq!(result.ready.id, first.id);
    assert_eq!(result.retired_duplicate, Some(pending.id.clone()));
    assert_eq!(
        writer.value_head(&pending.id).unwrap().state,
        State::Retiring
    );
    assert_eq!(incoming(&writer, &pending.id), 0);
    assert_eq!(incoming(&writer, &first.id), 1);
    let owner: i64 = crate::db(&path, || {
        writer
            ._index
            .query_row("SELECT value_pk FROM producer_lease WHERE id=1", [], |r| {
                r.get(0)
            })
    })
    .unwrap();
    assert_eq!(owner, first.id.pk);
}
#[test]
fn new_value_seal_keeps_lease_and_rolls_back_on_error() {
    let (_dir, _root, mut writer) = world();
    let path = writer.root.join("index.db");
    let pending = {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        let h = begin_in(&tx, &path, Codec::Utf8, 0, 1).unwrap();
        crate::db(&path, || tx.commit()).unwrap();
        h
    };
    let proof = writer.validate_value(&pending.id).unwrap();
    {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        assert!(matches!(
            seal_in(&tx, &path, &proof, |_, _| Err(Error::Cancelled)),
            Err(Error::Cancelled)
        ));
        crate::db(&path, || tx.rollback()).unwrap();
    }
    assert_eq!(
        writer.value_head(&pending.id).unwrap().state,
        State::Building
    );
    assert_eq!(incoming(&writer, &pending.id), 1);
    let result = {
        let tx = crate::db(&path, || writer._index.transaction()).unwrap();
        let result = seal_in(&tx, &path, &proof, |_, _| Ok(())).unwrap();
        crate::db(&path, || tx.commit()).unwrap();
        result
    };
    assert_eq!(result.ready.id, pending.id);
    assert!(result.retired_duplicate.is_none());
    assert_eq!(incoming(&writer, &pending.id), 1);
}

use super::*;
use crate::archive::{Archive, ArchiveLock, cache, open_read, open_write};
use std::sync::Arc;

struct World {
    _dir: tempfile::TempDir,
    archive: Archive,
    _lock: ArchiveLock,
    store: Arc<hashed_blobs::Store>,
}

impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        let store =
            Arc::new(hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap());
        Self {
            _dir: dir,
            archive,
            _lock: lock,
            store,
        }
    }

    fn seed(&self, input: &str, chain: i64, chunks: &[&[u8]]) {
        let mut a = open_write(&self.archive.archive_db(), cache::SETUP).unwrap();
        let tx = a.transaction().unwrap();
        tx.execute(
            "INSERT OR IGNORE INTO input(id, format) VALUES (?1, 'synthetic-text')",
            [input],
        )
        .unwrap();
        let bytes = chunks.concat();
        tx.execute(
            "INSERT INTO revision(input, number, produced, chain, length, blake3, commit_seq)
            VALUES (?1, ?2, 'first', ?2, ?3, ?4, 1)",
            params![
                input,
                chain,
                bytes.len() as i64,
                &BlobHash::of(&bytes).0[..]
            ],
        )
        .unwrap();
        let mut offset = 0;
        for bytes in chunks {
            let piece = Piece::prepare(&self.store, bytes).unwrap();
            let (place, stored_size, data) = match &piece.stored {
                crate::blobs::Stored::Inline(z) => ("inline", z.len() as u64, Some(z.as_slice())),
                crate::blobs::Stored::File { stored_size } => ("file", *stored_size, None),
            };
            tx.execute(
                "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state, data)
                VALUES (?1, ?2, ?3, ?4, 'plain', 'present', ?5)",
                params![
                    &piece.hash.0[..],
                    piece.size as i64,
                    stored_size as i64,
                    place,
                    data
                ],
            )
            .unwrap();
            tx.execute(
                "INSERT INTO extent(input, chain, offset, length, blob, blob_offset)
                VALUES (?1, ?2, ?3, ?4, ?5, 0)",
                params![input, chain, offset, bytes.len() as i64, &piece.hash.0[..]],
            )
            .unwrap();
            offset += bytes.len() as i64;
        }
        tx.commit().unwrap();
    }

    fn bytes(&self, input: &str, chain: i64) -> Vec<u8> {
        let a = open_read(&self.archive.archive_db()).unwrap();
        let length: i64 = a
            .query_row(
                "SELECT max(length) FROM revision WHERE input = ?1 AND chain = ?2",
                params![input, chain],
                |r| r.get(0),
            )
            .unwrap();
        crate::blobs::StoredChain::load(&a, "", input, chain, length as u64)
            .unwrap()
            .read_range(&self.store, 0, length as u64)
            .unwrap()
    }

    fn excise(&self, text: &str, dry: bool) -> Result<Excised> {
        let writer = crate::Writer::start(&self.archive).unwrap();
        writer.excise(
            ExciseRequest {
                id: if dry { "preview" } else { "execute" }.into(),
                scope: Scope::Text(text.into()),
                reason: None,
                dry_run: dry,
            },
            self.store.clone(),
        )
    }
}

#[test]
fn text_matches_across_pieces_and_old_chains_without_changing_shared_fragments() {
    let w = World::new();
    w.seed("a", 1, &[b"se", b"cret"]);
    w.seed("a", 2, &[b"latest public bytes"]);
    w.seed("b", 1, &[b"se", b"cond"]);
    let preview = w.excise("secret", true).unwrap();
    assert_eq!(preview.inputs_overwritten.get("a"), Some(&1));
    assert_eq!(w.bytes("a", 1), b"secret");
    let done = w.excise("secret", false).unwrap();
    assert_eq!(done.inputs_overwritten, preview.inputs_overwritten);
    assert_eq!(w.bytes("a", 1), b"      ");
    assert_eq!(w.bytes("a", 2), b"latest public bytes");
    assert_eq!(w.bytes("b", 1), b"second");
    let a = open_read(&w.archive.archive_db()).unwrap();
    let hashes: Vec<Vec<u8>> = a
        .prepare("SELECT blake3 FROM revision WHERE input = 'a' ORDER BY number")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(hashes[0], BlobHash::of(b"secret").0);
}

#[test]
fn mixed_json_spelling_survives_piece_splits_and_preserves_other_fields_exactly() {
    let w = World::new();
    let original = br#"{"id":"unchanged","text":"left s\u0065cr\u0065t right","other":"keep"}"#;
    w.seed(
        "json",
        1,
        &[&original[..39], &original[39..45], &original[45..]],
    );
    w.excise("secret", false).unwrap();
    let new = w.bytes("json", 1);
    assert_eq!(new.len(), original.len());
    let value: serde_json::Value = serde_json::from_slice(&new).unwrap();
    assert_eq!(value["id"], "unchanged");
    assert_eq!(value["other"], "keep");
    assert_eq!(value["text"], "left                  right");
}

#[test]
fn authored_payloads_are_redacted_without_session_targets_or_identity_changes() {
    let w = World::new();
    let a = open_write(&w.archive.archive_db(), cache::SETUP).unwrap();
    a.execute(
        "INSERT INTO note(id, root, kind, key, value, author, machine, created_at, commit_seq)
        VALUES ('note1', 'family1', 'note', 'label', 'before secret after', 'operator', 'm', 1, 1)",
        [],
    )
    .unwrap();
    a.execute(
        "INSERT INTO note_target(note_id, ref) VALUES ('note1', 'session:unrelated')",
        [],
    )
    .unwrap();
    a.execute(
        "INSERT INTO note_relation(note_id, kind, other) VALUES ('note1', 'revises', 'note:old')",
        [],
    )
    .unwrap();
    a.execute("INSERT INTO delivery(id, recipient_kind, sender, event, at, content_hash, bytes, body, refs_json, commit_seq)
        VALUES ('delivery1', 'operator', 'cli', 'recall', 1, ?1, 13, ?2, '[]', 1)",
        params![&BlobHash::of(b"use secret now").0[..], zstd::bulk::compress(b"use secret now", 1).unwrap()]).unwrap();
    a.execute("INSERT INTO kept_result(id, query, ran_at, view, rows_json) VALUES ('kept1', 'sessions where \"secret\"', 1, '{}', ?1)",
        [r#"[{"text":"s\u0065cret","id":"unchanged"}]"#]).unwrap();
    a.execute(
        "INSERT INTO variant(id, target, seen_hash, kind, coverage, body, author, machine, created_at)
        VALUES ('variant1', 'note:note1', ?1, 'summary', 'whole', 'secret summary', 'operator', 'm', 1)",
        [&[0u8; 32][..]],
    )
    .unwrap();
    let preview = w.excise("secret", true).unwrap();
    assert_eq!(
        (
            preview.notes_redacted,
            preview.deliveries_redacted,
            preview.kept_results_redacted,
            preview.variants_redacted
        ),
        (1, 1, 1, 1)
    );
    let done = w.excise("secret", false).unwrap();
    assert_eq!(done.notes_removed, 0);
    let note: (String, String, String) = a
        .query_row("SELECT id, root, value FROM note", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(
        note,
        (
            "note1".into(),
            "family1".into(),
            "before        after".into()
        )
    );
    let target: String = a
        .query_row("SELECT ref FROM note_target", [], |r| r.get(0))
        .unwrap();
    assert_eq!(target, "session:unrelated");
    let relation: String = a
        .query_row("SELECT other FROM note_relation", [], |r| r.get(0))
        .unwrap();
    assert_eq!(relation, "note:old");
    let (body, hash): (Vec<u8>, Vec<u8>) = a
        .query_row("SELECT body, content_hash FROM delivery", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    let text = unpack_text(&body).unwrap();
    assert_eq!(text, "use        now");
    assert_eq!(hash, BlobHash::of(text.as_bytes()).0);
    let rows: String = a
        .query_row("SELECT rows_json FROM kept_result", [], |r| r.get(0))
        .unwrap();
    let rows: serde_json::Value = serde_json::from_str(&rows).unwrap();
    assert_eq!(rows[0]["id"], "unchanged");
    assert!(!rows[0]["text"].as_str().unwrap().contains("secret"));
}

#[test]
fn an_unsupported_stored_format_refuses_before_any_committed_change() {
    for bytes in [
        b"PK\x03\x04secret".as_slice(),
        b"SQLite format 3\0secret",
        b"%PDF-secret",
        &[0xff, b's'],
    ] {
        let w = World::new();
        w.seed("ordinary", 1, &[b"secret"]);
        w.seed("unsupported", 1, &[bytes]);
        assert!(matches!(
            w.excise("secret", false),
            Err(StoreError::ExcisionFormat { .. })
        ));
        assert_eq!(w.bytes("ordinary", 1), b"secret");
        assert_eq!(w.bytes("unsupported", 1), bytes);
        let a = open_read(&w.archive.archive_db()).unwrap();
        let n: i64 = a
            .query_row("SELECT count(*) FROM excision", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }
}

#[test]
fn corrupt_stored_bytes_are_not_reported_as_unsupported_format() {
    let w = World::new();
    w.seed("a", 1, &[b"secret"]);
    let a = open_write(&w.archive.archive_db(), cache::SETUP).unwrap();
    a.execute(
        "UPDATE blob SET data = ?1",
        [zstd::bulk::compress(b"public", 1).unwrap()],
    )
    .unwrap();
    assert!(matches!(
        w.excise("secret", false),
        Err(StoreError::BlobCorrupt(_))
    ));
    assert_eq!(
        a.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn unsupported_signature_is_recognized_across_piece_boundaries() {
    let w = World::new();
    w.seed("zip", 1, &[b"P", b"K", b"\x03", b"\x04secret"]);
    assert!(matches!(
        w.excise("secret", false),
        Err(StoreError::ExcisionFormat { .. })
    ));
    assert_eq!(w.bytes("zip", 1), b"PK\x03\x04secret");
}

#[test]
fn unused_padding_of_a_shared_blob_is_not_kept_with_its_unrelated_slice() {
    let w = World::new();
    w.seed("a", 1, &[b"secret"]);
    w.seed("b", 1, &[b"se"]);
    let a = open_write(&w.archive.archive_db(), cache::SETUP).unwrap();
    a.execute(
        "UPDATE extent SET blob = ?1 WHERE input = 'b'",
        [&BlobHash::of(b"secret").0[..]],
    )
    .unwrap();
    w.excise("secret", false).unwrap();
    assert_eq!(w.bytes("a", 1), b"      ");
    assert_eq!(w.bytes("b", 1), b"se");
    assert_eq!(
        a.query_row(
            "SELECT count(*) FROM blob WHERE hash = ?1",
            [&BlobHash::of(b"secret").0[..]],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
}

#[test]
fn text_in_an_excision_reason_is_removed_too() {
    let w = World::new();
    let writer = crate::Writer::start(&w.archive).unwrap();
    writer
        .excise(
            ExciseRequest {
                id: "reason".into(),
                scope: Scope::Text("secret".into()),
                reason: Some("remove secret".into()),
                dry_run: false,
            },
            w.store.clone(),
        )
        .unwrap();
    let a = open_read(&w.archive.archive_db()).unwrap();
    let reason: String = a
        .query_row("SELECT reason FROM excision", [], |r| r.get(0))
        .unwrap();
    assert_eq!(reason, "remove       ");
}

#[test]
fn matching_json_keys_refuse_before_removing_any_value() {
    let embedded =
        serde_json::to_vec(&serde_json::json!({"payload": r#"{"s\u0065cret":"ordinary"}"#}))
            .unwrap();
    for key in [
        br#"{"secret":"ordinary"}"#.as_slice(),
        br#"{"s\u0065cret":"ordinary"}"#,
        &embedded,
    ] {
        let w = World::new();
        w.seed("a-value", 1, &[b"remove secret"]);
        w.seed("b-key", 1, &[key]);
        assert!(matches!(
            w.excise("secret", false),
            Err(StoreError::ExcisionFormat { .. })
        ));
        assert_eq!(w.bytes("a-value", 1), b"remove secret");
        assert_eq!(w.bytes("b-key", 1), key);
        let a = open_read(&w.archive.archive_db()).unwrap();
        assert_eq!(
            a.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}

#[test]
fn a_delta_base_refuses_text_excision_without_changing_stored_bytes() {
    let w = World::new();
    w.seed("base", 1, &[b"synthetic secret"]);
    let base = BlobHash::of(b"synthetic secret");
    let child = BlobHash::of(b"kept delta bytes");
    let a = open_write(&w.archive.archive_db(), cache::SETUP).unwrap();
    a.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state,base) VALUES (?1,16,0,'file','delta','present',?2)", params![&child.0[..],&base.0[..]]).unwrap();
    let error = w.excise("secret", false).unwrap_err();
    assert!(
        matches!(error, StoreError::ExcisionBlobReferenced(h) if h == base),
        "{error}"
    );
    assert_eq!(w.bytes("base", 1), b"synthetic secret");
    assert_eq!(
        a.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        a.query_row("SELECT base FROM blob WHERE hash=?1", [&child.0[..]], |r| r
            .get::<_, Vec<u8>>(0))
            .unwrap(),
        base.0
    );
}

#[test]
fn a_late_corrupt_capture_payload_refuses_before_installing_earlier_text_changes() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let w = World::new();
    w.seed("a-matched", 1, &[b"secret"]);
    let payload = b"public";
    let mut header = Header::of(payload);
    header.digest = blake3::hash(b"different").to_hex().to_string();
    let mut framed = MAGIC.to_vec();
    framed.extend(header.encode());
    framed.extend(payload);
    w.seed("z-corrupt", 1, &[&framed[..27], &framed[27..]]);
    assert!(
        matches!(w.excise("secret", false), Err(StoreError::Invalid(message)) if message.contains("capture payload digest mismatch"))
    );
    assert_eq!(w.bytes("a-matched", 1), b"secret");
    assert_eq!(w.bytes("z-corrupt", 1), framed);
    let archive = open_read(&w.archive.archive_db()).unwrap();
    assert_eq!(
        archive
            .query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[cfg(feature = "faults")]
#[test]
fn a_later_text_replacement_piece_failure_rolls_back_rows_and_retries() {
    let w = World::new();
    let mut original = vec![b'x'; 2 * 1024 * 1024 + 32];
    original[..6].copy_from_slice(b"secret");
    let end = original.len();
    original[end - 6..].copy_from_slice(b"secret");
    w.seed("many-pieces", 1, &[&original]);
    let writer = crate::Writer::start(&w.archive).unwrap();
    let request = ExciseRequest {
        id: "piece-retry".into(),
        scope: Scope::Text("secret".into()),
        reason: None,
        dry_run: false,
    };
    crate::fault::clear();
    crate::fault::arm(
        "excision_text.before_piece",
        "piece-retry",
        crate::fault::Action::Fail,
        1,
        1,
    );
    let error = writer.excise(request.clone(), w.store.clone()).unwrap_err();
    assert!(error.to_string().contains("injected fault"));
    crate::fault::clear();
    assert_eq!(w.bytes("many-pieces", 1), original);
    let archive = open_read(&w.archive.archive_db()).unwrap();
    assert_eq!(
        archive
            .query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        archive
            .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    writer.excise(request, w.store.clone()).unwrap();
    original[..6].fill(b' ');
    original[end - 6..].fill(b' ');
    assert_eq!(w.bytes("many-pieces", 1), original);
    assert_eq!(
        archive
            .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        3
    );
}

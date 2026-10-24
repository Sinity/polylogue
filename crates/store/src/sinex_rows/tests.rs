use super::*;

fn archive() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    c.execute_batch(
        "CREATE TABLE note (id TEXT PRIMARY KEY, kind TEXT, body BLOB, commit_seq INTEGER);
         CREATE TABLE note_target (note_id TEXT, ref TEXT, PRIMARY KEY (note_id, ref));
         CREATE TABLE note_relation (note_id TEXT, kind TEXT, other TEXT);",
    )
    .unwrap();
    c
}

#[test]
fn a_unit_round_trips_and_restoring_it_twice_changes_nothing() {
    let a = archive();
    a.execute_batch(
        "INSERT INTO note VALUES ('n1', 'lesson', x'00ff', 7);
         INSERT INTO note_target VALUES ('n1', 'session:claude-code:s1');",
    )
    .unwrap();
    let unit = Unit {
        commit_seq: 7,
        kind: "note".into(),
        key: "n1".into(),
    };
    let batch = read(&a, &unit).unwrap();
    assert_eq!(batch.rows, 2);
    let b = archive();
    assert_eq!(apply(&b, &batch.contents()).unwrap(), 2);
    assert_eq!(apply(&b, &batch.contents()).unwrap(), 0);
    let body: Vec<u8> = b
        .query_row("SELECT body FROM note WHERE id = 'n1'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(body, [0x00, 0xff]);
}

#[test]
fn malformed_blob_hex_returns_an_error_instead_of_panicking() {
    for value in ["€x", "a€", "g0", "0"] {
        let b = archive();
        let line = serde_json::to_vec(&json!({
            "table": "note",
            "row": {"id": "n1", "body": {"hex": value}}
        }))
        .unwrap();
        assert!(apply(&b, &line).is_err(), "accepted {value:?}");
        assert_eq!(
            b.query_row("SELECT COUNT(*) FROM note", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    assert_eq!(unhex("00fFaA").unwrap(), [0, 255, 170]);
}

#[test]
fn a_row_for_an_unknown_table_is_refused() {
    let b = archive();
    let line = br#"{"table":"sqlite_master","row":{"name":"x"}}"#;
    assert!(apply(&b, line).is_err());
}

#[test]
fn restored_integer_tokens_never_round_into_real_values() {
    for token in [
        "9223372036854775808",
        "18446744073709551615",
        "18446744073709551616",
        "-9223372036854775809",
        "1e999",
    ] {
        let target = archive();
        let line = format!(r#"{{"table":"note","row":{{"id":"n1","commit_seq":{token}}}}}"#);
        assert!(apply(&target, line.as_bytes()).is_err(), "accepted {token}");
        assert_eq!(
            target
                .query_row("SELECT count(*) FROM note", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    for value in [
        i64::MIN,
        -9_007_199_254_740_993,
        0,
        9_007_199_254_740_993,
        i64::MAX,
    ] {
        let target = archive();
        let line = serde_json::to_vec(&json!({
            "table": "note", "row": {"id": "n1", "commit_seq": value}
        }))
        .unwrap();
        apply(&target, &line).unwrap();
        let stored = target
            .query_row("SELECT commit_seq FROM note", [], |row| {
                row.get::<_, Sql>(0)
            })
            .unwrap();
        assert_eq!(stored, Sql::Integer(value));
    }
    for value in [-0.0_f64, 1.25, f64::MAX] {
        assert_eq!(to_sql(&json!(value)).unwrap(), Sql::Real(value));
    }
}

#[test]
fn excision_unit_updates_a_previously_restored_note_without_changing_its_identity() {
    use crate::archive::{Archive, cache, open_read, open_write};
    use crate::excise::{ExciseRequest, Scope};
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let source = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    source
        .execute(
            "INSERT INTO note(id, root, kind, value, author, machine, created_at, commit_seq)
        VALUES ('note1', 'family1', 'note', 'before secret after', 'operator', 'test', 1, 1)",
            [],
        )
        .unwrap();
    source
        .execute(
            "INSERT INTO meta(key, value) VALUES ('write_counter', '1') ON CONFLICT(key) DO UPDATE SET value = '1'",
            [],
        )
        .unwrap();
    let mut restored = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&restored).unwrap();
    crate::schema::migrate_archive(&mut restored).unwrap();
    let first = read(
        &source,
        &Unit {
            commit_seq: 1,
            kind: "note".into(),
            key: "note1".into(),
        },
    )
    .unwrap();
    apply(&restored, &first.contents()).unwrap();
    drop(source);
    let writer = crate::Writer::start(&archive).unwrap();
    writer
        .excise(
            ExciseRequest {
                id: "excise1".into(),
                scope: Scope::Text("secret".into()),
                reason: None,
                dry_run: false,
            },
            std::sync::Arc::new(
                hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
            ),
        )
        .unwrap();
    let source = open_read(&archive.archive_db()).unwrap();
    let emitted = units(&source, 1).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!(emitted[0].kind, "excision");
    let publisher = Published::open(dir.path()).unwrap();
    let originals: Vec<_> = units(&source, 0)
        .unwrap()
        .iter()
        .map(|unit| {
            let batch = publisher.read(unit).unwrap();
            RestoreUnit {
                unit: unit.clone(),
                batch: Some(format!("blake3:{}", blake3::hash(&batch.contents()))),
                bytes: None,
                rows: batch.rows,
                blobs: vec![],
                excision: publisher.publication(unit).unwrap(),
            }
        })
        .collect();
    let selected = select(&originals).unwrap();
    let current = selected_units(&selected);
    let payloads: Vec<_> = current
        .iter()
        .map(|unit| {
            unit.batch.as_ref().map(|name| {
                if unit.unit.kind == "note" {
                    file_bytes(publisher.replacement(name).unwrap())
                } else {
                    publisher.read(&unit.unit).unwrap().contents()
                }
            })
        })
        .collect();
    assert!(
        restore::install(&mut restored, &selected, None, |u| Ok(payloads
            [current.iter().position(|x| x == u).unwrap()]
        .clone()
        .map(std::io::Cursor::new)))
        .unwrap()
            > 0
    );
    assert_eq!(
        restore::install(&mut restored, &selected, None, |u| Ok(payloads
            [current.iter().position(|x| x == u).unwrap()]
        .clone()
        .map(std::io::Cursor::new)))
        .unwrap(),
        0
    );
    let fingerprint: String = restored
        .query_row(
            "SELECT value FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        fingerprint,
        blake3::hash(&serde_json::to_vec(&(None::<&str>, &current)).unwrap())
            .to_hex()
            .to_string(),
        "restore checkpoints keep their existing serialized identity"
    );
    let mut fresh = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&fresh).unwrap();
    fresh
        .execute_batch(concat!(
            include_str!("../schema/migrations/005-append-facts.sql"),
            include_str!("../schema/migrations/006-vector-deletion-policy.sql")
        ))
        .unwrap();
    restore::install(&mut fresh, &selected, None, |u| {
        Ok(payloads[current.iter().position(|x| x == u).unwrap()]
            .clone()
            .map(std::io::Cursor::new))
    })
    .unwrap();
    let mut missing = payloads.clone();
    missing[0] = None;
    let mut empty = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&empty).unwrap();
    empty
        .execute_batch(concat!(
            include_str!("../schema/migrations/005-append-facts.sql"),
            include_str!("../schema/migrations/006-vector-deletion-policy.sql")
        ))
        .unwrap();
    assert!(
        restore::install(&mut empty, &selected, None, |u| Ok(missing
            [current.iter().position(|x| x == u).unwrap()]
        .clone()
        .map(std::io::Cursor::new)))
        .is_err()
    );
    assert_eq!(
        empty
            .query_row("SELECT COUNT(*) FROM note", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let mut corrupt = payloads.clone();
    corrupt[0].as_mut().unwrap().push(b' ');
    assert!(
        restore::install(&mut empty, &selected, None, |u| Ok(corrupt
            [current.iter().position(|x| x == u).unwrap()]
        .clone()
        .map(std::io::Cursor::new)))
        .is_err()
    );
    let note = |conn: &Connection| {
        conn.query_row("SELECT id, root, value FROM note", [], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .unwrap()
    };
    assert_eq!(note(&restored), note(&source));
    assert_eq!(note(&fresh), note(&source));
    let mut altered = originals.clone();
    altered
        .last_mut()
        .unwrap()
        .excision
        .as_mut()
        .unwrap()
        .replacements[0]
        .unit
        .key = "absent".into();
    assert!(select(&altered).is_err(), "changed mapping hash refuses");
    let publication = altered.last_mut().unwrap().excision.as_mut().unwrap();
    publication.digest = publication.computed_digest().unwrap();
    assert!(
        select(&altered).is_err(),
        "a valid digest cannot authorize an absent unit"
    );
    let mut wrong_owner = current[0].clone();
    wrong_owner.unit.key = "other-note".into();
    let wrong_owner = select(&[wrong_owner]).unwrap();
    assert!(
        restore::install(&mut empty, &wrong_owner, None, |u| Ok(payloads
            [selected_units(&wrong_owner)
                .iter()
                .position(|x| x == u)
                .unwrap()]
        .clone()
        .map(std::io::Cursor::new)))
        .is_err()
    );
    let mut corrupt_repeat = payloads.clone();
    corrupt_repeat[0].as_mut().unwrap().push(b' ');
    assert!(
        restore::install(&mut restored, &selected, None, |u| Ok(corrupt_repeat
            [current.iter().position(|x| x == u).unwrap()]
        .clone()
        .map(std::io::Cursor::new)))
        .is_err()
    );
    assert_eq!(note(&restored), note(&source));
}

#[test]
fn excising_first_or_middle_units_keeps_other_unit_positions_stable() {
    use crate::archive::{Archive, cache, open_write};
    use crate::excise::{ExciseRequest, Scope};
    for remove in ["first", "middle"] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let source = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        for (seq, id) in [(1, "first"), (2, "middle"), (3, "last")] {
            source
                .execute(
                    "INSERT INTO input(id, format) VALUES (?1, 'synthetic-text')",
                    [id],
                )
                .unwrap();
            source
                .execute(
                    "INSERT INTO revision(input, number, produced, chain, length, commit_seq)
                VALUES (?1, 1, 'first', 1, 0, ?2)",
                    rusqlite::params![id, seq],
                )
                .unwrap();
        }
        source
            .execute(
                "INSERT INTO meta(key, value) VALUES ('write_counter', '3')",
                [],
            )
            .unwrap();
        let before = units(&source, 0).unwrap();
        let mut restored = Connection::open_in_memory().unwrap();
        crate::schema::create_archive(&restored).unwrap();
        restored
            .execute_batch(concat!(
                include_str!("../schema/migrations/005-append-facts.sql"),
                include_str!("../schema/migrations/006-vector-deletion-policy.sql")
            ))
            .unwrap();
        restored.execute_batch("PRAGMA foreign_keys=ON").unwrap();
        for unit in &before {
            apply(&restored, &read(&source, unit).unwrap().contents()).unwrap();
        }
        let writer = crate::Writer::start(&archive).unwrap();
        writer
            .excise(
                ExciseRequest {
                    id: "erase".into(),
                    scope: Scope::Input(remove.into()),
                    reason: None,
                    dry_run: false,
                },
                std::sync::Arc::new(
                    hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
                ),
            )
            .unwrap();
        let after = units(&source, 0).unwrap();
        assert_eq!(&after[..before.len()], before.as_slice());
        assert_eq!(after.last().unwrap().kind, "excision");
        let publication = super::excision::publication(&source, after.last().unwrap())
            .unwrap()
            .unwrap();
        publication.validate().unwrap();
        assert_eq!(publication.replacements.len(), 1);
        assert!(publication.replacements[0].batch.is_none());
        assert_eq!(publication.replacements[0].unit.key, format!("{remove}/1"));
        let originals: Vec<_> = after
            .iter()
            .map(|unit| {
                let batch = read(&source, unit).unwrap();
                RestoreUnit {
                    unit: unit.clone(),
                    batch: if batch.rows == 0 {
                        None
                    } else {
                        Some(format!("blake3:{}", blake3::hash(&batch.contents())))
                    },
                    bytes: None,
                    rows: batch.rows,
                    blobs: vec![],
                    excision: super::excision::publication(&source, unit).unwrap(),
                }
            })
            .collect();
        let selected = select(&originals).unwrap();
        restore::install(&mut restored, &selected, None, |u| {
            Ok(u.batch
                .as_ref()
                .map(|_| std::io::Cursor::new(read(&source, &u.unit).unwrap().contents())))
        })
        .unwrap();
        let remaining: Vec<String> = restored
            .prepare("SELECT input FROM revision ORDER BY input")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            remaining,
            ["first", "last", "middle"]
                .into_iter()
                .filter(|id| *id != remove)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            restored
                .query_row("SELECT COUNT(*) FROM input", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            3
        );
    }
}

#[test]
fn zero_rows_without_an_excision_cannot_silently_delete_a_unit() {
    for batch in [None, Some(format!("blake3:{}", blake3::hash(b"")))] {
        let catalog = [RestoreUnit {
            unit: Unit {
                commit_seq: 1,
                kind: "note".into(),
                key: "note1".into(),
            },
            batch,
            bytes: None,
            rows: 0,
            blobs: vec![],
            excision: None,
        }];
        assert!(select(&catalog).is_err());
    }
}

#[test]
fn retiring_a_pending_batch_preserves_identical_cataloged_file_or_inline_bytes() {
    for place in ["file", "inline"] {
        let dir = tempfile::tempdir().unwrap();
        let store = hashed_blobs::Store::open(dir.path().to_owned(), vec![], 0o600).unwrap();
        let bytes = b"synthetic retained attachment matching a pending batch";
        let hash = store.put(bytes.as_slice(), 3).unwrap();
        let conn = Connection::open_in_memory().unwrap();
        crate::schema::create_archive(&conn).unwrap();
        conn.execute_batch(concat!(
            include_str!("../schema/migrations/005-append-facts.sql"),
            include_str!("../schema/migrations/006-vector-deletion-policy.sql")
        ))
        .unwrap();
        let manifest = ExcisionPublication {
            version: 1,
            replacements: vec![],
            retired_batches: vec![format!("blake3:{}", hash.hex())],
            digest: String::new(),
        };
        let mut count = 0;
        visit_retired_local_batches(&conn, &manifest, |_| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 1, "an uncataloged superseded batch may retire");
        conn.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state,data) VALUES (?1,?2,?2,?3,'plain','present',?4)",rusqlite::params![&hash.0[..],bytes.len() as i64,place,if place=="inline" {Some(bytes.as_slice())}else{None}]).unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM extent", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        visit_retired_local_batches(&conn, &manifest, |candidate| {
            store.delete(&hashed_blobs::Hash(candidate.0))?;
            Ok(())
        })
        .unwrap();
        let mut actual = Vec::new();
        std::io::Read::read_to_end(&mut store.read(&hash).unwrap(), &mut actual).unwrap();
        assert_eq!(actual, bytes);
    }
}

#[test]
fn restore_refuses_a_different_nonempty_archive_before_fetch_or_identity_change() {
    let mut target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
    target
        .execute_batch(concat!(
            include_str!("../schema/migrations/005-append-facts.sql"),
            include_str!("../schema/migrations/006-vector-deletion-policy.sql")
        ))
        .unwrap();
    target
        .execute(
            "INSERT INTO meta(key,value) VALUES ('archive_id','unrelated-archive')",
            [],
        )
        .unwrap();
    // This table has no publication unit, so the production units-only check
    // cannot establish emptiness by itself.
    target
        .execute(
            "INSERT INTO client_request(id,content_hash,created_at) VALUES ('request1',x'0102',1)",
            [],
        )
        .unwrap();
    let source = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&source).unwrap();
    source
        .execute_batch(concat!(
            include_str!("../schema/migrations/005-append-facts.sql"),
            include_str!("../schema/migrations/006-vector-deletion-policy.sql")
        ))
        .unwrap();
    source.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('note1','note1','note','source body','operator','test',1,1)",[]).unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "note".into(),
        key: "note1".into(),
    };
    let batch = read(&source, &unit).unwrap();
    let selected = select(&[RestoreUnit {
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&batch.contents()))),
        bytes: None,
        rows: batch.rows,
        blobs: vec![],
        excision: None,
    }])
    .unwrap();
    let called = std::cell::Cell::new(false);
    let error = restore::install(&mut target, &selected, Some("source-archive"), |_| {
        called.set(true);
        Ok(Some(std::io::Cursor::new(batch.contents())))
    })
    .unwrap_err();
    assert!(error.to_string().contains("nonempty different archive"));
    assert!(!called.get());
    assert_eq!(
        target
            .query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
        "unrelated-archive"
    );
    assert_eq!(
        target
            .query_row("SELECT COUNT(*) FROM client_request", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        target
            .query_row(
                "SELECT COUNT(*) FROM meta WHERE key='sinex_restored_view'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    // Empty takeover and exact replay are valid, as is a same-archive update
    // which preserves a locally retained row outside the publication catalog.
    let mut empty = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&empty).unwrap();
    empty
        .execute_batch(concat!(
            include_str!("../schema/migrations/005-append-facts.sql"),
            include_str!("../schema/migrations/006-vector-deletion-policy.sql")
        ))
        .unwrap();
    assert_eq!(
        restore::install(&mut empty, &selected, Some("source-archive"), |_| Ok(Some(
            std::io::Cursor::new(batch.contents())
        )))
        .unwrap(),
        1
    );
    empty.execute("INSERT INTO client_request(id,content_hash,created_at) VALUES ('local-request',x'0304',2)",[]).unwrap();
    assert_eq!(
        restore::install(&mut empty, &selected, Some("source-archive"), |_| Ok(Some(
            std::io::Cursor::new(batch.contents())
        )))
        .unwrap(),
        0
    );
    assert_eq!(
        empty
            .query_row("SELECT COUNT(*) FROM client_request", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn pending_batch_placement_and_retirement_never_use_an_external_read_root() {
    use crate::archive::{Archive, cache, open_write};
    use crate::excise::{ExciseRequest, Scope};
    let dir = tempfile::tempdir().unwrap();
    let remote_dir = tempfile::tempdir().unwrap();
    let remote = hashed_blobs::Store::open(remote_dir.path().to_path_buf(), vec![], 0o600).unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('note1','note1','note','before        after','operator','test',1,1)",[]).unwrap();
    conn.execute(
        "INSERT INTO meta(key,value) VALUES ('write_counter','1')",
        [],
    )
    .unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "note".into(),
        key: "note1".into(),
    };
    let expected = read(&conn, &unit).unwrap();
    let hash = remote.put(expected.contents().as_slice(), 3).unwrap();
    conn.execute(
        "UPDATE note SET value='before secret after' WHERE id='note1'",
        [],
    )
    .unwrap();
    let writer = crate::Writer::start(&archive).unwrap();
    let with_remote = std::sync::Arc::new(
        hashed_blobs::Store::open(
            archive.root.clone(),
            vec![remote_dir.path().to_path_buf()],
            0o600,
        )
        .unwrap(),
    );
    writer
        .excise(
            ExciseRequest {
                id: "erase1".into(),
                scope: Scope::Text("secret".into()),
                reason: None,
                dry_run: false,
            },
            with_remote.clone(),
        )
        .unwrap();
    let local = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    assert!(
        local.contains(&hash),
        "placement must not deduplicate into read-only backing"
    );
    local.delete(&hash).unwrap(); // Simulate an evicted local pending copy.
    writer
        .excise(
            ExciseRequest {
                id: "erase2".into(),
                scope: Scope::Text("before".into()),
                reason: None,
                dry_run: false,
            },
            with_remote,
        )
        .unwrap();
    assert!(
        remote.contains(&hash),
        "cleanup must not fall back to an external root"
    );
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut remote.read(&hash).unwrap(), &mut bytes).unwrap();
    assert_eq!(bytes, expected.contents());
}

fn stale_catalog_cannot_restore_local_excised_text(add_remote_unit: bool) {
    use crate::archive::{Archive, cache, open_write};
    use crate::excise::{ExciseRequest, Scope};
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('note1','note1','note','before secret after','operator','test',1,1)",[]).unwrap();
    conn.execute(
        "INSERT INTO meta(key,value) VALUES ('write_counter','1')",
        [],
    )
    .unwrap();
    let id: String = conn
        .query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "note".into(),
        key: "note1".into(),
    };
    let original = read(&conn, &unit).unwrap();
    let mut catalog = vec![RestoreUnit {
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&original.contents()))),
        bytes: None,
        rows: original.rows,
        blobs: vec![],
        excision: None,
    }];
    let mut bodies = vec![original.contents()];
    if add_remote_unit {
        let remote = Connection::open_in_memory().unwrap();
        crate::schema::create_archive(&remote).unwrap();
        remote
            .execute_batch(concat!(
                include_str!("../schema/migrations/005-append-facts.sql"),
                include_str!("../schema/migrations/006-vector-deletion-policy.sql")
            ))
            .unwrap();
        remote.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('remote-note','remote-note','note','unrelated remote addition','operator','test',2,2)",[]).unwrap();
        let unit = Unit {
            commit_seq: 2,
            kind: "note".into(),
            key: "remote-note".into(),
        };
        let batch = read(&remote, &unit).unwrap();
        catalog.push(RestoreUnit {
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&batch.contents()))),
            bytes: None,
            rows: batch.rows,
            blobs: vec![],
            excision: None,
        });
        bodies.push(batch.contents());
    }
    let writer = crate::Writer::start(&archive).unwrap();
    writer
        .excise(
            ExciseRequest {
                id: "local-excision".into(),
                scope: Scope::Text("secret".into()),
                reason: None,
                dry_run: false,
            },
            std::sync::Arc::new(
                hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
            ),
        )
        .unwrap();
    drop(writer);
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    let before: String = conn
        .query_row("SELECT value FROM note WHERE id='note1'", [], |r| r.get(0))
        .unwrap();
    assert!(!before.contains("secret"));
    let selected = select(&catalog).unwrap();
    let mut restoring = Restoring::open(&archive).unwrap();
    let called = std::cell::Cell::new(false);
    let result = restoring.install(&id, &selected, |unit| {
        called.set(true);
        Ok(Some(std::io::Cursor::new(
            bodies[catalog.iter().position(|u| u.unit == unit.unit).unwrap()].clone(),
        )))
    });
    let after: String = conn
        .query_row("SELECT value FROM note WHERE id='note1'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(after, before, "stale restore returned {result:?}");
    assert!(result.is_err());
    assert!(
        !called.get(),
        "local conflict must refuse before opening old payloads"
    );
}

#[test]
fn stale_catalog_refuses_an_originating_archives_unpublished_excision() {
    stale_catalog_cannot_restore_local_excised_text(false);
}

#[test]
fn stale_catalog_with_unrelated_remote_addition_cannot_reopen_local_excised_text() {
    stale_catalog_cannot_restore_local_excised_text(true);
}

mod streams;

fn selected_units(selection: &Selection) -> Vec<RestoreUnit> {
    let mut units = Vec::new();
    selection
        .each(|unit| {
            units.push(unit.clone());
            Ok(())
        })
        .unwrap();
    units
}

mod blob_placement;

mod batch_streams;

fn file_bytes(mut file: std::fs::File) -> Vec<u8> {
    use std::io::Read;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

mod excision_recovery;

mod revision_ownership;

mod delivery_updates;

mod append_facts;

mod vector_policies;

mod document_policy;

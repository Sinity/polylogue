use super::*;
use polylogue_store::older_archive::fixture;
use rusqlite::OpenFlags;

struct Imported {
    _home: polylogue_testkit::ClaudeHome,
    engine: Engine,
    _lock: polylogue_store::ArchiveLock,
    root: PathBuf,
    older: fixture::Standard,
    _older_dir: tempfile::TempDir,
}

fn setup() -> Imported {
    let home = polylogue_testkit::ClaudeHome::new();
    let older_dir = tempfile::tempdir().unwrap();
    let older = fixture::standard(&older_dir.path().join("older"));
    let (engine, lock) = crate::import::tests::engine(&home, 64);
    let root = home.root.clone();
    Imported {
        _home: home,
        engine,
        _lock: lock,
        root,
        older,
        _older_dir: older_dir,
    }
}

fn import(w: &mut Imported, dry_run: bool) -> OlderArchiveReport {
    let report = w
        .engine
        .import_with_options(
            &[],
            &ImportOptions {
                from_archive: Some(w.older.root.clone()),
                dry_run,
                ..Default::default()
            },
        )
        .unwrap();
    report.older_archive.unwrap()
}

fn read_only(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap()
}

fn rows(conn: &rusqlite::Connection, sql: &str) -> Vec<Vec<String>> {
    let mut statement = conn.prepare(sql).unwrap();
    let columns = statement.column_count();
    statement
        .query_map([], |row| {
            (0..columns)
                .map(|i| {
                    Ok(match row.get_ref(i)? {
                        rusqlite::types::ValueRef::Null => "NULL".to_owned(),
                        rusqlite::types::ValueRef::Integer(n) => n.to_string(),
                        rusqlite::types::ValueRef::Real(n) => n.to_string(),
                        rusqlite::types::ValueRef::Text(t) => {
                            String::from_utf8_lossy(t).into_owned()
                        }
                        rusqlite::types::ValueRef::Blob(b) => {
                            b.iter().map(|b| format!("{b:02x}")).collect()
                        }
                    })
                })
                .collect()
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

/// Every revision's bytes, by input and number, assembled from its chain's extents.
fn revision_bytes(
    conn: &rusqlite::Connection,
    blob: &mut dyn FnMut(&[u8]) -> Vec<u8>,
) -> BTreeMap<(String, i64), [u8; 32]> {
    let revisions = rows(conn, "SELECT input, number, chain, length FROM revision");
    let mut out = BTreeMap::new();
    for r in revisions {
        let (input, number, chain, length) = (
            r[0].clone(),
            r[1].parse::<i64>().unwrap(),
            r[2].parse::<i64>().unwrap(),
            r[3].parse::<usize>().unwrap(),
        );
        let mut statement = conn
            .prepare("SELECT offset, length, blob, blob_offset FROM extent WHERE input = ?1 AND chain = ?2 ORDER BY offset")
            .unwrap();
        let extents: Vec<(i64, i64, Vec<u8>, i64)> = statement
            .query_map(rusqlite::params![input, chain], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let mut bytes = Vec::new();
        for (offset, len, hash, blob_offset) in extents {
            assert_eq!(offset as usize, bytes.len(), "extents join without gaps");
            let whole = blob(&hash);
            bytes.extend_from_slice(&whole[blob_offset as usize..(blob_offset + len) as usize]);
        }
        bytes.truncate(length);
        assert_eq!(bytes.len(), length);
        out.insert((input, number), *blake3::hash(&bytes).as_bytes());
    }
    out
}

/// law: older-archive-round-trip
#[test]
fn every_class_of_row_round_trips_with_its_identity_and_bytes() {
    let mut w = setup();
    let report = import(&mut w, false);
    assert!(!report.dry_run);
    let source = read_only(&w.older.root.join("archive.db"));
    let target = read_only(&w.root.join("archive.db"));

    // Inputs and every revision, extent and observation, as the older archive held them.
    let older_inputs = rows(&source, "SELECT id, format FROM input ORDER BY id");
    let rows_input = report.rows_input.clone().unwrap();
    let mut held = rows(&target, "SELECT id, format FROM input ORDER BY id");
    held.retain(|row| row[0] != rows_input);
    assert_eq!(held, older_inputs);
    for sql in [
        "SELECT input, number, produced, chain, length FROM revision WHERE input IN ({}) ORDER BY 1, 2",
        "SELECT input, chain, offset, length, blob, blob_offset FROM extent WHERE input IN ({}) ORDER BY 1, 2, 3",
        "SELECT id, input, machine, path, class, revision, length, device, inode, size, mtime_ns, ctime_ns, details_json, polylogue_version, seen_at FROM observation WHERE input IN ({}) ORDER BY input, rowid",
    ] {
        let ids = older_inputs
            .iter()
            .map(|row| format!("'{}'", row[0]))
            .collect::<Vec<_>>()
            .join(",");
        let sql = sql.replace("{}", &ids);
        assert_eq!(rows(&target, &sql), rows(&source, &sql), "{sql}");
    }
    // Exact locations, so the watched file is the same input on the next pass.
    assert_eq!(
        rows(
            &target,
            "SELECT DISTINCT CAST(path_bytes AS TEXT) FROM observation WHERE id = 'o1'"
        ),
        [[fixture::WATCHED_PATH.to_owned()]]
    );
    assert_eq!(
        rows(
            &target,
            "SELECT id, host, first_seen FROM machine WHERE host = 'workstation'"
        ),
        [[w.older.machine.clone(), "workstation".into(), "1000".into()]]
    );

    // Each revision's bytes hash the same here as in the older archive.
    let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
    let before = revision_bytes(&source, &mut |hash: &[u8]| {
        let blob = older
            .blob(&polylogue_model::BlobHash(hash.try_into().unwrap()))
            .unwrap()
            .unwrap();
        let mut out = Vec::new();
        older.copy_blob(&blob, &mut out).unwrap().unwrap();
        out
    });
    let store = w.engine.archive.blobs().unwrap();
    let mut after = revision_bytes(&target, &mut |hash: &[u8]| {
        polylogue_store::blobs::read(
            &target,
            &store,
            &polylogue_model::BlobHash(hash.try_into().unwrap()),
        )
        .unwrap()
    });
    after.retain(|(input, _), _| *input != rows_input);
    assert_eq!(after, before);
    assert_eq!(before.len(), 5);
    // A blob no extent cites is stored too.
    assert_eq!(
        polylogue_store::blobs::read(&target, &store, &w.older.loose_blob).unwrap(),
        b"bytes no extent cites\n"
    );

    // Notes and settings as assertions with their original identities.
    assert_eq!(
        rows(
            &target,
            "SELECT id, root, kind, author, via, machine FROM assertion ORDER BY id"
        ),
        [
            ["n1", "n1", "note", "operator", "import", &w.older.machine],
            ["n2", "n1", "note", "operator", "import", &w.older.machine],
            [
                "s1",
                "s1",
                "setting",
                "operator",
                "import",
                &w.older.machine
            ],
            [
                "s2",
                "s1",
                "setting",
                "operator",
                "import",
                &w.older.machine
            ],
            [
                "s3",
                "s3",
                "setting",
                "operator",
                "import",
                &w.older.machine
            ],
            ["t1", "t1", "tag", "operator", "import", &w.older.machine],
            [
                "w1",
                "w1",
                "withdrawal",
                "operator",
                "import",
                &w.older.machine
            ],
        ]
        .map(|row| row.map(str::to_owned).to_vec())
    );

    // Deliveries, their events and the kept result, with the requests that wrote them.
    assert_eq!(
        rows(
            &target,
            "SELECT id, recipient_id, sender, event, cwd, at, content_hash, bytes, refs_json FROM delivery ORDER BY id"
        ),
        rows(
            &source,
            "SELECT id, recipient_id, sender, event, cwd, at, content_hash, bytes, refs_json FROM delivery WHERE id != 'd3' ORDER BY id"
        )
    );
    assert_eq!(
        rows(
            &target,
            "SELECT delivery, what, at, detail, request FROM delivery_event ORDER BY delivery"
        ),
        [
            ["d1", "delivered", "19500", "NULL", "e1"],
            [
                "d2",
                "delivered",
                "20000",
                "written to the harness hook output",
                "d2"
            ],
        ]
        .map(|row| row.map(str::to_owned).to_vec())
    );
    assert_eq!(
        rows(&target, "SELECT id FROM client_request ORDER BY id"),
        [["d1"], ["d2"], ["e1"], ["e2"], ["k1"]].map(|row| row.map(str::to_owned).to_vec())
    );
    let imported = rows(
        &target,
        "SELECT id,name,query,ran_at,read_at,rows_json FROM kept_result",
    );
    let original = rows(
        &source,
        "SELECT id,name,query,ran_at,read_at,rows_json FROM kept_result",
    );
    assert_eq!(imported.len(), original.len());
    for (imported, original) in imported.iter().zip(&original) {
        assert_eq!(&imported[..5], &original[..5]);
        let mut body: serde_json::Value = serde_json::from_str(&imported[5]).unwrap();
        body.as_object_mut().unwrap().remove("output_metadata");
        assert_eq!(
            body,
            serde_json::from_str::<serde_json::Value>(&original[5]).unwrap()
        );
    }

    assert_eq!(
        rows(
            &target,
            "SELECT json_extract(rows_json,'$.output_metadata.version'),json_extract(rows_json,'$.output_metadata.rows.kind') FROM kept_result"
        ),
        [vec!["2".to_owned(), "unproven".to_owned()]]
    );

    // Vectors and the permanent failure, with their values and times.
    let older_vectors = read_only(&w.older.root.join("embeddings.db"));
    let vectors = read_only(&w.root.join("embeddings.db"));
    for sql in [
        "SELECT text_hash, model, chunk, dims, vector, embedded_at FROM embedded ORDER BY 1, 2, 3",
        "SELECT text_hash, model, error, failed_at FROM embed_failure ORDER BY 1, 2",
    ] {
        assert_eq!(rows(&vectors, sql), rows(&older_vectors, sql), "{sql}");
    }

    // The rows input keeps every other row verbatim.
    assert_eq!(
        rows(
            &target,
            &format!("SELECT format FROM input WHERE id = '{rows_input}'")
        ),
        [[polylogue_sources::older_archive_rows::ORIGIN.to_owned()]]
    );
}

/// law: older-archive-every-row-accounted
#[test]
fn every_row_lands_or_is_refused_with_its_reason() {
    let mut w = setup();
    let report = import(&mut w, false);
    let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
    for (table, n) in older.row_counts().unwrap() {
        let count = report.classes.get(&table).copied().unwrap_or_default();
        assert_eq!(count.found, n, "{table}: {count:?}");
        assert_eq!(
            count.found,
            count.stored + count.present + count.refused + count.kept,
            "{table}: {count:?}"
        );
    }
    assert!(
        report
            .classes
            .keys()
            .all(|table| older.row_counts().unwrap().contains_key(table))
    );
    let refused: BTreeSet<(String, String)> = report
        .refused
        .iter()
        .map(|r| (r.class.clone(), r.row.clone()))
        .collect();
    assert_eq!(
        refused,
        [
            ("note", "x1"),
            ("delivery", "d3"),
            ("chat_action", "1 rows")
        ]
        .map(|(c, r)| (c.to_owned(), r.to_owned()))
        .into_iter()
        .collect()
    );
    assert!(report.refused.iter().all(|r| !r.reason.is_empty()));
    let counts = &report.classes;
    assert_eq!(counts["input"].stored, 3);
    assert_eq!(counts["observation"].stored, 9);
    assert_eq!(counts["blob"].stored, 5);
    assert_eq!(counts["client_request"].stored, 5);
    assert_eq!(counts["client_request"].kept, 4);
    assert_eq!(counts["meta"].kept, 1);
    assert_eq!(counts["embedded"].stored, w.older.vectors);
}

/// law: older-archive-idempotent
#[test]
fn importing_the_same_older_archive_again_changes_nothing() {
    let mut w = setup();
    import(&mut w, false);
    let snapshot = |root: &Path| {
        let archive = read_only(&root.join("archive.db"));
        let embeddings = read_only(&root.join("embeddings.db"));
        let mut out = Vec::new();
        for table in [
            "input",
            "revision",
            "extent",
            "observation",
            "blob",
            "machine",
            "assertion",
            "link",
            "client_request",
            "delivery",
            "delivery_event",
            "kept_result",
            "meta",
            "segment",
        ] {
            out.push(rows(
                &archive,
                &format!("SELECT * FROM {table} ORDER BY 1, 2"),
            ));
        }
        for table in ["embedded", "embed_failure"] {
            out.push(rows(
                &embeddings,
                &format!("SELECT * FROM {table} ORDER BY 1, 2"),
            ));
        }
        out
    };
    let before = snapshot(&w.root);
    let again = import(&mut w, false);
    assert_eq!(snapshot(&w.root), before);
    for (table, count) in &again.classes {
        assert_eq!(count.stored, 0, "{table}: {count:?}");
    }
    assert_eq!(again.classes["input"].present, 3);
    assert_eq!(again.classes["note"].present, 4);
    assert_eq!(again.classes["delivery"].present, 2);
    assert_eq!(again.classes["embedded"].present, w.older.vectors);
}

/// law: older-archive-read-only
#[test]
fn neither_an_import_nor_a_dry_run_writes_in_the_older_folder() {
    let mut w = setup();
    let before = fixture::file_hashes(&w.older.root);
    import(&mut w, true);
    import(&mut w, false);
    import(&mut w, false);
    assert_eq!(fixture::file_hashes(&w.older.root), before);
}

/// law: older-archive-dry-run
#[test]
fn a_dry_run_counts_every_class_and_writes_nothing_here() {
    let mut w = setup();
    let empty = |root: &Path| {
        let archive = read_only(&root.join("archive.db"));
        let vectors = read_only(&root.join("embeddings.db"));
        [
            rows(&archive, "SELECT COUNT(*) FROM input")[0][0].clone(),
            rows(&archive, "SELECT COUNT(*) FROM blob")[0][0].clone(),
            rows(&archive, "SELECT COUNT(*) FROM assertion")[0][0].clone(),
            rows(&archive, "SELECT COUNT(*) FROM client_request")[0][0].clone(),
            rows(&vectors, "SELECT COUNT(*) FROM embedded")[0][0].clone(),
        ]
    };
    let before = empty(&w.root);
    let report = import(&mut w, true);
    assert!(report.dry_run);
    assert!(
        report
            .dry_run_limits
            .as_deref()
            .is_some_and(|limits| limits.contains("writer-side refusals are only known on import"))
    );
    assert_eq!(empty(&w.root), before);
    assert_eq!(before, ["0", "0", "0", "0", "0"]);
    let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
    for (table, n) in older.row_counts().unwrap() {
        let count = report.classes.get(&table).copied().unwrap_or_default();
        assert_eq!(count.found, n, "{table}");
        assert_eq!(count.present, 0, "{table}");
    }
    assert_eq!(report.classes["input"].stored, 3);
    assert!(
        report
            .refused
            .iter()
            .any(|r| r.class == "note" && r.row == "x1")
    );
}

/// law: older-archive-refusals
#[test]
fn bytes_that_do_not_verify_refuse_the_inputs_that_cite_them_and_nothing_else() {
    let mut w = setup();
    // Same length, other bytes: the file decodes but no longer hashes to its name.
    let path = w
        .older
        .root
        .join("blobs")
        .join(&w.older.file_blob.to_string()[..2])
        .join(format!("{}.zst", w.older.file_blob));
    let original_len = {
        let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
        older.blob(&w.older.file_blob).unwrap().unwrap().size as usize
    };
    std::fs::write(&path, fixture::seekable(&vec![b'x'; original_len], 16)).unwrap();
    let report = import(&mut w, false);
    let refused = |class: &str| {
        report
            .refused
            .iter()
            .filter(|r| r.class == class)
            .map(|r| (r.row.clone(), r.reason.clone()))
            .collect::<Vec<_>>()
    };
    let inputs = refused("input");
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].0, w.older.watched);
    assert!(
        inputs[0].1.contains(&w.older.file_blob.to_string()),
        "{}",
        inputs[0].1
    );
    let blobs = refused("blob");
    assert_eq!(blobs.len(), 1);
    assert!(
        blobs[0].1.contains("do not hash to its name"),
        "{}",
        blobs[0].1
    );
    assert_eq!(report.classes["input"].stored, 2);
    assert_eq!(report.classes["observation"].refused, 6);
    let target = read_only(&w.root.join("archive.db"));
    assert_eq!(
        rows(
            &target,
            &format!(
                "SELECT COUNT(*) FROM revision WHERE input = '{}'",
                w.older.watched
            )
        ),
        [["0".to_owned()]]
    );
    assert_eq!(
        rows(
            &target,
            &format!(
                "SELECT COUNT(*) FROM revision WHERE input = '{}'",
                w.older.deleted
            )
        ),
        [["1".to_owned()]]
    );
}

/// law: older-archive-refusals
#[test]
fn a_location_this_archive_already_watches_as_another_input_is_refused() {
    let mut w = setup();
    import(&mut w, false);
    // A second older archive holding another input at the same location.
    let second = tempfile::tempdir().unwrap();
    let b = fixture::Builder::create(second.path(), 4);
    b.machine(&w.older.machine, "workstation", 1_000);
    let blob = b.inline_blob(b"other bytes\n");
    b.archive
        .execute_batch(&format!(
            "INSERT INTO input(id, format) VALUES ('0190f000-0000-7000-8000-000000000009', 'claude-code');
             INSERT INTO revision VALUES ('0190f000-0000-7000-8000-000000000009', 1, 'first', 1, 12, 1);
             INSERT INTO extent VALUES ('0190f000-0000-7000-8000-000000000009', 1, 0, 12, X'{blob}', 0);
             INSERT INTO observation(id, input, machine, path, class, revision, length, polylogue_version, seen_at)
               VALUES ('p1', '0190f000-0000-7000-8000-000000000009', '{machine}', '{path}', 'first', 1, 12, '0.9.0', 1);",
            machine = w.older.machine,
            path = fixture::WATCHED_PATH,
        ))
        .unwrap();
    let root = b.finish();
    let report = w.engine.import_older_archive(&root, false, None).unwrap();
    let input = report.refused.iter().find(|r| r.class == "input").unwrap();
    assert_eq!(input.row, "0190f000-0000-7000-8000-000000000009");
    assert!(input.reason.contains("already watches"), "{}", input.reason);
}

/// law: older-archive-every-row-accounted
#[test]
fn a_delivery_its_own_request_could_not_store_but_an_event_request_did_counts_as_stored() {
    let mut w = setup();
    // This archive already holds another request under the delivery's
    // request ID, so restoring that request conflicts; the request that
    // reported the delivery's event carries the whole delivery and stores it.
    w.engine
        .writer()
        .unwrap()
        .user(polylogue_model::user::UserRequest {
            id: "d1".into(),
            machine: "test-machine".into(),
            authorship: polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
            at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            write: polylogue_model::user::UserWrite::SetSetting {
                key: "fetch-links".into(),
                value: "on".into(),
                machine: None,
            },
        })
        .unwrap();
    let report = import(&mut w, false);
    let request = report
        .refused
        .iter()
        .find(|r| r.class == "client_request" && r.row == "d1")
        .unwrap();
    assert!(!request.reason.is_empty());
    assert!(
        !report
            .refused
            .iter()
            .any(|r| r.class == "delivery" && r.row == "d1")
    );
    assert!(!report.refused.iter().any(|r| r.class == "delivery_event"));
    assert_eq!(report.classes["delivery"].stored, 2);
    assert_eq!(report.classes["delivery_event"].stored, 2);
    let target = read_only(&w.root.join("archive.db"));
    assert_eq!(
        rows(&target, "SELECT id FROM delivery ORDER BY id"),
        [["d1"], ["d2"]].map(|row| row.map(str::to_owned).to_vec())
    );
    let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
    for (table, n) in older.row_counts().unwrap() {
        let count = report.classes.get(&table).copied().unwrap_or_default();
        assert_eq!(count.found, n, "{table}");
        assert_eq!(
            count.found,
            count.stored + count.present + count.refused + count.kept,
            "{table}"
        );
    }
}

/// law: older-archive-refusals
#[test]
fn a_refused_input_is_set_aside_with_every_row_and_its_stored_bytes() {
    let mut w = setup();
    let path = w
        .older
        .root
        .join("blobs")
        .join(&w.older.file_blob.to_string()[..2])
        .join(format!("{}.zst", w.older.file_blob));
    let size = {
        let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
        older.blob(&w.older.file_blob).unwrap().unwrap().size as usize
    };
    let damaged = fixture::seekable(&vec![b'x'; size], 16);
    std::fs::write(&path, &damaged).unwrap();
    let report = import(&mut w, false);
    let set_aside = report.refused_inputs.clone().unwrap();
    let target = read_only(&w.root.join("archive.db"));
    let extents = rows(
        &target,
        &format!("SELECT blob FROM extent WHERE input = '{set_aside}' ORDER BY chain, offset"),
    );
    assert_eq!(extents.len(), 1);
    let store = w.engine.archive.blobs().unwrap();
    let hash = polylogue_model::BlobHash::parse(&extents[0][0]).unwrap();
    let bytes = polylogue_store::blobs::read(&target, &store, &hash).unwrap();
    let lines: Vec<serde_json::Value> = bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    let reason = &report
        .refused
        .iter()
        .find(|r| r.class == "input")
        .unwrap()
        .reason;
    assert!(lines.iter().all(|line| line["reason"] == reason.as_str()));
    let source = read_only(&w.older.root.join("archive.db"));
    let watched = &w.older.watched;
    // Every row of the refused input, column for column.
    for (table, filter) in [
        ("input", format!("id = '{watched}'")),
        ("revision", format!("input = '{watched}'")),
        ("extent", format!("input = '{watched}'")),
        ("observation", format!("input = '{watched}'")),
    ] {
        let expected = rows(&source, &format!("SELECT * FROM {table} WHERE {filter}"));
        let carried: Vec<&serde_json::Value> =
            lines.iter().filter(|line| line["table"] == table).collect();
        assert_eq!(carried.len(), expected.len(), "{table}");
        assert!(!expected.is_empty(), "{table}");
    }
    let ids: Vec<&str> = lines
        .iter()
        .filter(|line| line["table"] == "observation")
        .map(|line| line["row"]["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["o1", "o2", "o3", "o4", "o5", "o0"]);
    // Every blob it cites, the damaged file with its bytes as they were found.
    let blobs: Vec<&serde_json::Value> = lines
        .iter()
        .filter(|line| line["table"] == "blob")
        .collect();
    assert_eq!(blobs.len(), 3);
    let file = blobs
        .iter()
        .find(|line| line["row"]["place"] == "file")
        .unwrap();
    let hex: String = damaged.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(file["file_hex"], hex.as_str());
    assert!(
        blobs
            .iter()
            .filter(|line| line["row"]["place"] == "inline")
            .all(|line| line["row"]["data"]["sqlite_blob_hex"].is_string())
    );
}

/// Original Notes queries read the preserved rows, including kinds that the
/// assertion importer cannot express. The predecessor SQL is an independent oracle.
#[test]
fn imported_notes_match_the_original_sql_and_replay_prior_kept_imports() {
    let mut w = setup();
    let source = rusqlite::Connection::open(w.older.root.join("archive.db")).unwrap();
    for id in [
        "CaseA",
        "caseB",
        "éclair",
        "Éclair",
        "literal%one",
        "literal_one",
        "literal\\one",
        "nul\0tail",
    ] {
        source.execute("INSERT INTO note(id,first_note,kind,key,value,author,machine,created_at,valid_until,commit_seq) SELECT ?1,?1,'note',NULL,?2,'operator',machine,15000,NULL,99 FROM note LIMIT 1",rusqlite::params![id," { \"x\": 1 } "]).unwrap();
    }
    drop(source);
    let older = OlderArchive::open(&w.older.root, &w.root.join("copies")).unwrap();
    let (bytes, _) = w
        .engine
        .archive
        .prepare_write(|out| older.write_rows(out))
        .unwrap();
    let input = bytes.hash().to_string();
    let path = w.older.root.join("archive.db!").join("rows.jsonl");
    let member = polylogue_store::ImportMember {
        version: 1,
        import_id: "older-row-test".into(),
        member: "rows.jsonl".into(),
        authorship: None,
        format: None,
    };
    // This is the exact retained-file route used before the rows had a parser.
    w.engine
        .store_import_group(
            vec![(
                path,
                polylogue_sources::origin::by_name(KEPT).unwrap(),
                None,
                Captured {
                    prepared: bytes,
                    head: Vec::new(),
                    member: Some(member),
                    container: None,
                },
            )],
            &mut ImportReport::default(),
        )
        .unwrap();
    let pass = w.engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let original = read_only(&w.older.root.join("archive.db"));
    let expected = rows(
        &original,
        "SELECT n.id,n.kind,n.key,n.value,n.author,n.created_at,n.first_note,(SELECT group_concat(ref,' ') FROM (SELECT t.ref FROM note_target t WHERE t.note_id=n.id ORDER BY t.ref)) AS target FROM note n WHERE n.kind<>'withdrawal' AND NOT EXISTS(SELECT 1 FROM note later WHERE later.first_note=n.first_note AND later.commit_seq>n.commit_seq) AND NOT EXISTS(SELECT 1 FROM note_relation w WHERE w.kind='withdraws' AND w.other='note:'||n.first_note) ORDER BY n.created_at DESC,n.id DESC",
    );
    let columns = [
        "id",
        "kind",
        "key",
        "value",
        "author",
        "created_at",
        "first_note",
        "target",
    ];
    let selected=w.engine.reads.reader().unwrap().query("SELECT id,kind,key,value,author,created_at,first_note,target,input FROM imported_note WHERE kind<>'withdrawal' AND current AND NOT withdrawn ORDER BY created_at DESC,id DESC",&[]).unwrap();
    let got = selected
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|name| match &row[*name] {
                    serde_json::Value::Null => "NULL".to_owned(),
                    serde_json::Value::String(s) => s.clone(),
                    v => v.to_string(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(got, expected);
    assert!(selected.iter().all(|row| row["input"] == input));
    assert!(selected.iter().any(|row| row["kind"] == "distillate"));
    let options = polylogue_store::query::Options {
        clock: polylogue_store::query::time::Clock::new(
            jiff::Timestamp::from_millisecond(20_000).unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("UTC").unwrap(),
                jiff::civil::Time::constant(0, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: Vec::new(),
        page_size: None,
        random_seed: None,
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: Vec::new(),
        kept: Default::default(),
        chat_pages: Vec::new(),
    };
    let default_program = polylogue_store::query::compile("imported_notes", &options).unwrap();
    assert_eq!(
        default_program
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        columns
    );
    let mut default_rows = w
        .engine
        .reads
        .reader()
        .unwrap()
        .query_rows(&default_program)
        .unwrap();
    for row in &mut default_rows {
        polylogue_store::query::execute::present(&default_program, row).unwrap();
    }
    assert_eq!(
        default_rows
            .iter()
            .map(|row| row["id"].clone())
            .collect::<Vec<_>>(),
        selected
            .iter()
            .map(|row| row["id"].clone())
            .collect::<Vec<_>>()
    );
    for prefix in ["c", "é", "É", "literal%", "literal_", "literal\\", "nul\0"] {
        let pattern = format!(
            "{}%",
            prefix
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_")
        );
        let expected = original
            .prepare("SELECT id FROM note WHERE id LIKE ?1 ESCAPE '\\' ORDER BY id")
            .unwrap()
            .query_map([pattern], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let text = format!(
            "imported_notes where id:{} | fields id | sort by id",
            polylogue_store::query::quoted(prefix)
        );
        let program = polylogue_store::query::compile(&text, &options).unwrap();
        let mut actual = w
            .engine
            .reads
            .reader()
            .unwrap()
            .query_rows(&program)
            .unwrap();
        for row in &mut actual {
            polylogue_store::query::execute::present(&program, row).unwrap();
        }
        assert_eq!(
            actual
                .iter()
                .map(|row| row["id"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>(),
            expected,
            "prefix {prefix:?}"
        );
    }

    // A new pass uses the same stored chain and emits no duplicate source rows.
    assert!(w.engine.run_pass().unwrap().errors.is_empty());
    let repeated=w.engine.reads.reader().unwrap().query("SELECT id,kind,key,value,author,created_at,first_note,target,input FROM imported_note WHERE kind<>'withdrawal' AND current AND NOT withdrawn ORDER BY created_at DESC,id DESC",&[]).unwrap();
    assert_eq!(selected, repeated);
}

use super::*;
use std::io::Cursor;

fn source() -> Connection {
    let source = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&source).unwrap();
    source
        .execute_batch(concat!(
            include_str!("../../schema/migrations/005-append-facts.sql"),
            include_str!("../../schema/migrations/009-machine-order.sql")
        ))
        .unwrap();
    source.execute_batch("PRAGMA foreign_keys=ON;
        INSERT INTO input VALUES ('input1','synthetic-text');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input1',1,'first',1,3,1);
        INSERT INTO machine(id,host,first_seen) VALUES ('machine1','host-a',1),('machine1','host-b',1);
        INSERT INTO observation(id,input,machine,path,class,revision,polylogue_version,seen_at) VALUES ('seen1','input1','machine1','synthetic','first',1,'test',1);").unwrap();
    let hash = blake3::hash(b"abc");
    source
        .execute(
            "INSERT INTO blob VALUES (?1,3,3,'file','plain',NULL,'present',NULL)",
            [hash.as_bytes().as_slice()],
        )
        .unwrap();
    source
        .execute(
            "INSERT INTO extent VALUES ('input1',1,0,3,?1,0)",
            [hash.as_bytes().as_slice()],
        )
        .unwrap();
    source
}

fn rows() -> Vec<Value> {
    let source = source();
    let batch = read(
        &source,
        &Unit {
            commit_seq: 1,
            kind: "revision".into(),
            key: "input1/1".into(),
        },
    )
    .unwrap();
    batch
        .contents()
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| serde_json::from_slice(s).unwrap())
        .collect()
}

fn selection(rows: &[Value], blobs: Vec<String>) -> crate::Result<(Selection, Vec<u8>)> {
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, row).unwrap();
        bytes.push(b'\n');
    }
    let unit = RestoreUnit {
        unit: Unit {
            commit_seq: 1,
            kind: "revision".into(),
            key: "input1/1".into(),
        },
        batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
        bytes: Some(bytes.len() as u64),
        rows: rows.len() as u64,
        blobs,
        excision: None,
    };
    Ok((select(&[unit])?, bytes))
}

fn install(rows: &[Value], blobs: Vec<String>, target: &mut Connection) -> crate::Result<u64> {
    let (selected, bytes) = selection(rows, blobs)?;
    selected.verify_source_blobs(|name| {
        if name == format!("blake3:{}", blake3::hash(b"abc")) {
            Ok(3)
        } else if name == format!("blake3:{}", blake3::hash(b"other")) {
            Ok(5)
        } else {
            Err(crate::StoreError::Invalid(
                "unknown synthetic source".into(),
            ))
        }
    })?;
    restore::install_fixture(target, &selected, None, |_| {
        Ok(Some(Cursor::new(bytes.clone())))
    })
}

fn names() -> Vec<String> {
    vec![format!("blake3:{}", blake3::hash(b"abc"))]
}
fn target() -> Connection {
    let target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
    target
        .execute_batch(concat!(
            include_str!("../../schema/migrations/005-append-facts.sql"),
            include_str!("../../schema/migrations/009-machine-order.sql")
        ))
        .unwrap();
    target.execute_batch("PRAGMA foreign_keys=ON; INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('kept','kept','note','unchanged','operator','test',1,2);").unwrap();
    target
}

#[test]
fn uncited_blob_or_machine_rows_refuse_even_with_correct_payload_hashes() {
    for extra in [
        json!({"table":"machine","row":{"id":"unrelated","host":"other","first_seen":1}}),
        json!({"table":"blob","row":{"hash":{"hex":hex(blake3::hash(b"other").as_bytes())},"size":5,"stored_size":5,"place":"file","encoding":"plain","base":null,"state":"present","data":null}}),
    ] {
        let mut rows = rows();
        rows.push(extra);
        let mut target = target();
        assert!(
            install(&rows, names(), &mut target).is_err(),
            "uncited row was installed"
        );
        assert_eq!(
            target
                .query_row("SELECT COUNT(*) FROM revision", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            target
                .query_row("SELECT value FROM note WHERE id='kept'", [], |r| r
                    .get::<_, String>(0))
                .unwrap(),
            "unchanged"
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
    }
}

#[test]
fn actual_revision_rows_restore_in_any_order_with_shared_machine_hosts() {
    let mut rows = rows();
    for reverse in [false, true] {
        if reverse {
            rows.reverse();
        }
        let mut target = target();
        assert_eq!(
            install(&rows, names(), &mut target).unwrap(),
            rows.len() as u64
        );
        assert_eq!(install(&rows, names(), &mut target).unwrap(), 0);
        assert_eq!(
            target
                .query_row(
                    "SELECT COUNT(*) FROM machine WHERE id='machine1'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            2
        );
    }
}

fn refuses(rows: &[Value], names: Vec<String>) {
    let mut target = target();
    target.execute_batch("INSERT INTO input VALUES ('input1','synthetic-text');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input1',1,'first',1,99,1);").unwrap();
    assert!(install(rows, names, &mut target).is_err());
    assert_eq!(
        target
            .query_row(
                "SELECT length FROM revision WHERE input='input1' AND number=1",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        99
    );
    for table in ["blob", "machine", "observation", "extent"] {
        assert_eq!(
            target
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
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
}

#[test]
fn descriptor_and_supplied_blobs_must_match_the_extent_references() {
    let rows = rows();
    refuses(&rows, vec![]);
    refuses(&rows, vec![format!("blake3:{}", blake3::hash(b"other"))]);
    let mut extra = names();
    extra.push(format!("blake3:{}", blake3::hash(b"other")));
    refuses(&rows, extra);
    let missing: Vec<_> = rows
        .iter()
        .filter(|v| v["table"] != "blob")
        .cloned()
        .collect();
    refuses(&missing, names());
    let mut duplicates = names();
    duplicates.extend(names());
    assert!(
        install(&rows, duplicates, &mut target()).is_ok(),
        "reference sets allow repeats"
    );
}

#[test]
fn wrong_chain_and_foreign_observations_cannot_authorize_extra_rows() {
    for (table, field, value) in [
        ("extent", "chain", json!(2)),
        ("extent", "offset", json!(-1)),
        ("extent", "offset", json!(3)),
        ("observation", "input", json!("unrelated")),
        ("observation", "revision", json!(2)),
    ] {
        let mut rows = rows();
        rows.iter_mut().find(|v| v["table"] == table).unwrap()["row"][field] = value;
        refuses(&rows, names());
    }
}

#[test]
fn source_catalog_rows_use_the_plain_file_representation() {
    for (field, value) in [
        ("place", json!("inline")),
        ("encoding", json!("delta")),
        ("state", json!("writing")),
        ("data", json!({"hex":"00"})),
        ("base", json!({"hex":"00".repeat(32)})),
        ("stored_size", json!(1)),
        ("size", json!(-1)),
    ] {
        let mut rows = rows();
        rows.iter_mut().find(|v| v["table"] == "blob").unwrap()["row"][field] = value;
        refuses(&rows, names());
    }
    let mut rows = rows();
    for row in &mut rows {
        let field = match row["table"].as_str() {
            Some("blob") => "hash",
            Some("extent") => "blob",
            _ => continue,
        };
        let hex = row["row"][field]["hex"].as_str().unwrap().to_uppercase();
        row["row"][field]["hex"] = json!(hex);
    }
    assert!(install(&rows, names(), &mut target()).is_ok());
}

#[test]
fn extent_slices_must_fit_their_supplied_blob() {
    for (field, value) in [
        ("length", json!(-1)),
        ("length", json!(0)),
        ("length", json!(4)),
        ("blob_offset", json!(-1)),
        ("blob_offset", json!(1)),
        ("blob_offset", json!(i64::MAX)),
    ] {
        let mut rows = rows();
        rows.iter_mut().find(|v| v["table"] == "extent").unwrap()["row"][field] = value;
        refuses(&rows, names());
    }
}

#[test]
fn verified_source_bytes_must_match_the_catalog_size() {
    let mut rows = rows();
    for value in &mut rows {
        if value["table"] == "blob" {
            value["row"]["size"] = json!(100);
            value["row"]["stored_size"] = json!(100);
        }
        if value["table"] == "revision" {
            value["row"]["length"] = json!(10);
        }
        if value["table"] == "extent" {
            value["row"]["blob_offset"] = json!(50);
            value["row"]["length"] = json!(10);
        }
    }
    let (selected, bytes) = selection(&rows, names()).unwrap();
    selected
        .verify_source_blobs(|name| {
            assert_eq!(name, format!("blake3:{}", blake3::hash(b"abc")));
            // This is the entire verified source, despite the catalog's size100.
            assert_eq!(b"abc".len(), 3);
            Ok(3)
        })
        .unwrap();
    let mut target = target();
    assert!(
        restore::install_fixture(&mut target, &selected, None, |_| Ok(Some(Cursor::new(
            bytes.clone()
        ))))
        .is_err(),
        "installed a slice beyond the verified source"
    );
    assert_eq!(
        target
            .query_row("SELECT COUNT(*) FROM revision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn each_install_requires_fresh_completed_source_verification() {
    let (selected, bytes) = selection(&rows(), names()).unwrap();
    let checkpoint = serde_json::to_vec(&selected).unwrap();
    let mut target = target();
    assert!(
        restore::install_fixture::<Cursor<Vec<u8>>>(&mut target, &selected, None, |_| panic!(
            "unverified payload opened"
        ))
        .is_err()
    );
    selected.verify_source_blobs(|_| Ok(3)).unwrap();
    assert!(
        restore::install_fixture(&mut target, &selected, None, |_| Ok(Some(Cursor::new(
            bytes.clone()
        ))))
        .unwrap()
            > 0
    );
    assert!(
        restore::install_fixture::<Cursor<Vec<u8>>>(&mut target, &selected, None, |_| panic!(
            "verification reused"
        ))
        .is_err()
    );
    selected.verify_source_blobs(|_| Ok(3)).unwrap();
    assert!(
        selected
            .verify_source_blobs(|_| Err(crate::StoreError::Invalid("source disappeared".into())))
            .is_err()
    );
    assert!(
        restore::install_fixture::<Cursor<Vec<u8>>>(&mut target, &selected, None, |_| panic!(
            "failed verification reused"
        ))
        .is_err()
    );
    selected.verify_source_blobs(|_| Ok(3)).unwrap();
    assert_eq!(
        restore::install_fixture(&mut target, &selected, None, |_| Ok(Some(Cursor::new(
            bytes.clone()
        ))))
        .unwrap(),
        0
    );
    assert_eq!(
        serde_json::to_vec(&selected).unwrap(),
        checkpoint,
        "verification must not change checkpoint identity"
    );
}

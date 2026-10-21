use super::*;
use std::io::Cursor;

fn source() -> Connection {
    let source = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&source).unwrap();
    source.execute_batch("PRAGMA foreign_keys=ON;
        INSERT INTO input VALUES ('input1','synthetic-text');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input1',1,'first',1,3,1);
        INSERT INTO machine VALUES ('machine1','host-a',1),('machine1','host-b',1);
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

fn install(rows: &[Value], blobs: Vec<String>, target: &mut Connection) -> crate::Result<u64> {
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
    let selected = select(&[unit])?;
    restore::install(target, &selected, None, |_| {
        Ok(Some(Cursor::new(bytes.clone())))
    })
}

fn names() -> Vec<String> {
    vec![format!("blake3:{}", blake3::hash(b"abc"))]
}
fn target() -> Connection {
    let target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
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

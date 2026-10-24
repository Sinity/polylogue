//! Private native reproduction of pre9 held-frame machine dependency drift.
use super::*;
use polylogue_store::sinex_rows::Published;

fn bytes(source: &Published, unit: &polylogue_store::sinex_rows::Unit) -> Vec<u8> {
    let mut batch = source.read(unit).unwrap();
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut batch.file, &mut out).unwrap();
    out
}

// Match the archive8 three-table observation wire format without migrating it.
fn archive8_observation(db: &rusqlite::Connection, key: &str) -> Vec<u8> {
    use rusqlite::types::ValueRef;
    use std::collections::BTreeMap;
    let mut out = Vec::new();
    for (table, sql) in [
        (
            "input",
            "SELECT * FROM input WHERE id IN (SELECT input FROM observation WHERE id=?1)",
        ),
        (
            "machine",
            "SELECT * FROM machine WHERE id IN (SELECT machine FROM observation WHERE id=?1)",
        ),
        ("observation", "SELECT * FROM observation WHERE id=?1"),
    ] {
        let mut statement = db.prepare(sql).unwrap();
        let names = statement
            .column_names()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>();
        let mut rows = statement.query([key]).unwrap();
        while let Some(row) = rows.next().unwrap() {
            let mut object = BTreeMap::new();
            for (at, name) in names.iter().enumerate() {
                let value = match row.get_ref(at).unwrap() {
                    ValueRef::Null => serde_json::Value::Null,
                    ValueRef::Integer(n) => serde_json::json!(n),
                    ValueRef::Real(n) => serde_json::json!(n),
                    ValueRef::Text(t) => serde_json::json!(String::from_utf8_lossy(t)),
                    ValueRef::Blob(b) => {
                        serde_json::json!({"hex":polylogue_store::sinex_rows::hex(b)})
                    }
                };
                object.insert(name, value);
            }
            serde_json::to_writer(&mut out, &serde_json::json!({"row":object,"table":table}))
                .unwrap();
            out.push(b'\n');
        }
    }
    out
}

pub(super) async fn proof(sinex: &mut Sinex, root: &Path) {
    let root = std::env::var_os("POLYLOGUE_HELD_PROOF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("historical-machine"));
    fs::create_dir_all(&root).unwrap();
    let source = root.join("source8");
    let scratch = root.join("scratch");
    fs::create_dir_all(&scratch).unwrap();
    assert!(!source.exists());
    let home = root.join("home");
    let cfg = root.join("config");
    let old = std::env::var_os("POLYLOGUE_OLD8")
        .expect("POLYLOGUE_OLD8 names the actual archive8 frontend");
    let input = home.join(".claude/projects/synthetic/held-machine.jsonl");
    fs::create_dir_all(input.parent().unwrap()).unwrap();
    let line = |id: &str| {
        serde_json::json!({"type":"user","uuid":id,"sessionId":"held-machine","message":{"role":"user","content":format!("Synthetic {id}")}}).to_string()+"\n"
    };
    fs::write(&input, line("first")).unwrap();
    fs::create_dir_all(cfg.join("polylogue")).unwrap();
    let spawn = |host: &str| {
        let mut command = Command::new("unshare");
        command
            .args([
                "--user",
                "--map-root-user",
                "--uts",
                "sh",
                "-c",
                "hostname \"$1\" || exit; shift; exec \"$@\"",
                "held-machine",
                host,
            ])
            .arg(&old)
            .arg("daemon")
            .env("POLYLOGUE_ARCHIVE", &source)
            .env("POLYLOGUE_HOME", &home)
            .env("XDG_CONFIG_HOME", &cfg)
            .env("TMPDIR", &scratch);
        Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap())
    };
    let first = spawn("host-a");
    let start = Instant::now();
    loop {
        if source.join("archive.db").exists() {
            let db = polylogue_store::archive::open_read(&source.join("archive.db")).unwrap();
            if db
                .query_row("SELECT COUNT(*) FROM observation", [], |r| {
                    r.get::<_, i64>(0)
                })
                .unwrap_or(0)
                > 0
            {
                break;
            }
        }
        assert!(start.elapsed() < Duration::from_secs(15));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop(first);
    // Capture the actual old8 JSONL closure before migration. This small
    // synthetic input is inline, so SQLite backup owns every required byte.
    let db = polylogue_store::archive::open_read(&source.join("archive.db")).unwrap();
    assert_eq!(polylogue_store::schema::version(&db).unwrap(), 8);
    let unit = db
        .query_row(
            "SELECT commit_seq,id FROM observation ORDER BY commit_seq LIMIT 1",
            [],
            |r| {
                Ok(polylogue_store::sinex_rows::Unit {
                    commit_seq: r.get(0)?,
                    kind: "observation".into(),
                    key: r.get(1)?,
                })
            },
        )
        .unwrap();
    let original = archive8_observation(&db, &unit.key);
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM blob WHERE place!='inline'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let snapshot = root.join("publication-snapshot");
    fs::create_dir_all(&snapshot).unwrap();
    db.backup(rusqlite::MAIN_DB, snapshot.join("archive.db"), None)
        .unwrap();
    let copy = polylogue_store::archive::open_read(&snapshot.join("archive.db")).unwrap();
    assert_eq!(archive8_observation(&copy, &unit.key), original);
    drop(copy);
    drop(db);
    let (migrated, snapshot_lock) = Archive::open_or_create(&snapshot).unwrap();
    let rows = Published::open(&migrated.root).unwrap();
    assert_eq!(bytes(&rows, &unit), original);
    assert_eq!(blake3::hash(&bytes(&rows, &unit)), blake3::hash(&original));
    let id = rows.archive_id().unwrap();
    drop(rows);
    drop(snapshot_lock);
    let published = sinex.publish(&snapshot).await.unwrap();
    assert_eq!((published.units, published.held), (2, 0));
    let (prefix, _lock) = Archive::open_or_create(&root.join("verified-prefix")).unwrap();
    let start = Instant::now();
    loop {
        if sinex.restore(&prefix, &id).await.unwrap().units == 2 {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let prefix = Published::open(&prefix.root).unwrap();
    assert_eq!(bytes(&prefix, &unit), original);
    fs::write(root.join("original-observation.jsonl"), &original).unwrap();
    let stream = jetstream::new(sinex.client.clone())
        .get_stream(sinex.subjects.confirmed_stream())
        .await
        .unwrap();
    let last = stream
        .get_last_raw_message_by_subject(&sinex.subjects.confirmed_event("polylogue.rows"))
        .await
        .unwrap()
        .sequence;
    let mut descriptor = None;
    for position in 1..=last {
        let raw = stream.get_raw_message(position).await.unwrap();
        if let sinex_protocol::Confirmed::Event(event) =
            serde_json::from_slice(&raw.payload).unwrap()
            && let Some(payload) = event.payload
            && payload["archive"] == id
            && payload["key"] == unit.key
        {
            descriptor = Some((payload, event.id));
            break;
        }
    }
    let (descriptor, event_id) = descriptor.expect("original held observation descriptor");
    assert_eq!(
        descriptor["batch"],
        format!("blake3:{}", blake3::hash(&original))
    );
    fs::write(
        root.join("original-descriptor.json"),
        serde_json::to_vec_pretty(&descriptor).unwrap(),
    )
    .unwrap();
    let old_ack = sinex
        .have(&[format!("rows:{id}:1"), format!("rows:{id}:2")])
        .await
        .unwrap();
    let frame = format!("rows:{id}:{}", descriptor["position"].as_u64().unwrap());
    let ack = old_ack
        .get(&frame)
        .expect("actual acknowledged observation frame");
    assert_eq!(ack.frame, frame);
    assert!(ack.admitted.contains(&event_id));
    fs::write(
        root.join("original-acknowledged-evidence.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"ack":ack,"confirmed_event":event_id,"descriptor":descriptor}),
        )
        .unwrap(),
    )
    .unwrap();
    fs::write(&input, line("first") + &line("second")).unwrap();
    let second = spawn("host-b");
    let start = Instant::now();
    loop {
        let db = polylogue_store::archive::open_read(&source.join("archive.db")).unwrap();
        let complete: bool = db.query_row(
            "SELECT (SELECT COUNT(*) FROM machine)=2 AND (SELECT COUNT(*) FROM observation)+(SELECT COUNT(*) FROM revision)>2 AND NOT EXISTS(SELECT 1 FROM blob WHERE state!='present')",
            [], |r|r.get(0),
        ).unwrap();
        if complete {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(15));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop(second);
    assert_eq!(
        sinex
            .have(&old_ack.keys().cloned().collect::<Vec<_>>())
            .await
            .unwrap(),
        old_ack
    );
    let db = polylogue_store::archive::open_read(&source.join("archive.db")).unwrap();
    assert_eq!(polylogue_store::schema::version(&db).unwrap(), 8);
    let drifted = archive8_observation(&db, &unit.key);
    assert_ne!(drifted, original);
    drop(db);
    let (archive, _lock) = Archive::open_or_create(&source).unwrap();
    let current = Published::open(&archive.root).unwrap();
    let changed = bytes(&current, &unit);
    assert_eq!(changed, drifted);
    fs::write(root.join("current-observation.jsonl"), &changed).unwrap();
    assert_ne!(original, changed);
    let positions = (1..=current.inventory().unwrap().len())
        .map(|p| format!("rows:{id}:{p}"))
        .collect::<Vec<_>>();
    let held_before = sinex.have(&positions).await.unwrap();
    let report = sinex.publish(&source).await;
    let receipt = serde_json::json!({"original":blake3::hash(&original).to_hex().to_string(),"current":blake3::hash(&changed).to_hex().to_string(),"result":format!("{report:?}"),"source_version":9,"writer_version":8,"publisher":"current reserved transport","snapshot":"SQLite consistent backup"});
    fs::write(
        root.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    eprintln!("pre9 actual held-frame drift: {receipt}");
    assert!(
        matches!(report,Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message))) if message.contains("machine dependencies conflict"))
    );
    assert_eq!(sinex.have(&positions).await.unwrap(), held_before);
    let read_root = sinex.store_root.clone();
    sinex.store_root = root.join("unavailable-original-evidence");
    let unavailable = sinex.publish(&source).await;
    sinex.store_root = read_root;
    assert!(
        matches!(unavailable,Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message))) if message.contains("evidence") && message.contains("unavailable"))
    );
    assert_eq!(
        sinex
            .have(&old_ack.keys().cloned().collect::<Vec<_>>())
            .await
            .unwrap(),
        old_ack
    );
}

pub(super) async fn cached_comparison(sinex: &mut Sinex, root: &Path) {
    use polylogue_store::writer::{Batch, InFlight, InputKey, Observation};
    use polylogue_store::{Change, UserRequest, UserWrite, Writer};
    let root = root.join("valid-legacy-machine-history");
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    let observe = |writer: &Writer, host: &str, change| {
        writer
            .send(Batch {
                input: InputKey {
                    id: "synthetic-legacy-machine".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change,
                observation: Some(Observation {
                    machine: "synthetic-machine".into(),
                    host: host.into(),
                    path: "synthetic".into(),
                    facts: None,
                    seen_at_ms: 1,
                    import: None,
                }),
                continues: false,
                piece: None,
                blobs: vec![],
                length: 0,
                parse: vec![],
                release: InFlight::default(),
            })
            .unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
    };
    let writer = Writer::start(&archive).unwrap();
    observe(&writer, "host-a", Change::First);
    observe(&writer, "host-b", Change::Unchanged);
    drop(writer);
    // Both aliases existed before the first publication. Baseline0 is the
    // exact migration representation and changes no previously held frame.
    let db = polylogue_store::archive::open_write(
        &archive.archive_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    db.execute("UPDATE machine SET first_commit_seq=0", [])
        .unwrap();
    assert_eq!(sinex.publish(&root).await.unwrap().units, 3);
    let start = Instant::now();
    loop {
        match sinex.publish(&root).await {
            Ok(done) if done.held == 3 && done.units == 0 => break,
            Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message)))
                if message.contains("no confirmed descriptor")
                    && start.elapsed() < Duration::from_secs(10) =>
            {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            other => panic!("valid legacy comparison: {other:?}"),
        }
    }
    assert!(sinex.machine_memo.is_some());
    // Reusing a historical verdict does not require or assert current payload
    // availability. Move only this test reader's root; retain all actual bytes.
    let read_root = sinex.store_root.clone();
    sinex.store_root = root.join("unavailable-reader-root");
    assert_eq!(sinex.publish(&root).await.unwrap().units, 0);
    let writer = Writer::start(&archive).unwrap();
    writer
        .user(UserRequest::now(
            "unrelated-setting".into(),
            UserWrite::SetSetting {
                key: "synthetic-setting".into(),
                value: "value".into(),
                author: polylogue_model::Author::Operator,
                machine: None,
            },
        ))
        .unwrap();
    drop(writer);
    assert_eq!(sinex.publish(&root).await.unwrap().units, 1);
    assert_eq!(sinex.publish(&root).await.unwrap().units, 0);
    let writer = Writer::start(&archive).unwrap();
    observe(&writer, "host-c", Change::Unchanged);
    drop(writer);
    let unavailable = sinex.publish(&root).await;
    assert!(
        matches!(unavailable,Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message))) if message.contains("unavailable"))
    );
    sinex.store_root = read_root;
    assert_eq!(sinex.publish(&root).await.unwrap().units, 1);
    // The new candidate acknowledgment requires a fresh binding even though
    // the alias state itself did not change after the preceding publication.
    let start = Instant::now();
    loop {
        match sinex.publish(&root).await {
            Ok(done) if done.units == 0 => break,
            Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message)))
                if message.contains("no confirmed descriptor")
                    && start.elapsed() < Duration::from_secs(10) =>
            {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            other => panic!("new candidate acknowledgment comparison: {other:?}"),
        }
    }
}

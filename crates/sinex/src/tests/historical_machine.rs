//! Private native reproduction of pre9 held-frame machine dependency drift.
use super::*;
use polylogue_store::sinex_rows::Published;

fn bytes(source: &Published, unit: &polylogue_store::sinex_rows::Unit) -> Vec<u8> {
    let mut batch = source.read(unit).unwrap();
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut batch.file, &mut out).unwrap();
    out
}

pub(super) async fn proof(sinex: &mut Sinex, config: &crate::Config, root: &Path) {
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
    fs::write(
        cfg.join("polylogue/sinex.toml"),
        format!(
            "mode='primary'\nnats={:?}\nsubjects={:?}\nproducer={:?}\nstore={:?}\n",
            config.nats,
            config.subjects,
            config.producer,
            fs::canonicalize(&config.store).unwrap()
        ),
    )
    .unwrap();
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
    // The ordinary publisher polls once a minute. Restart after the canonical
    // write so its immediate first pass sees the completed observation.
    let first = spawn("host-a");
    let start = Instant::now();
    let id = loop {
        if source.join("archive.db").exists() {
            let db = polylogue_store::archive::open_read(&source.join("archive.db")).unwrap();
            if let Ok(id) = db.query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
                r.get::<_, String>(0)
            }) {
                let held = sinex
                    .have(&[format!("rows:{id}:1"), format!("rows:{id}:2")])
                    .await
                    .unwrap();
                if held.len() == 2 {
                    break id;
                }
            }
        }
        assert!(start.elapsed() < Duration::from_secs(30));
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    drop(first);
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
    let unit = inventory_units(&prefix)
        .into_iter()
        .find(|u| u.kind == "observation")
        .unwrap();
    let original = bytes(&prefix, &unit);
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
        {
            if let Some(payload) = event.payload {
                if payload["archive"] == id && payload["key"] == unit.key {
                    descriptor = Some((payload, event.id));
                    break;
                }
            }
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
        if db
            .query_row("SELECT COUNT(*) FROM machine", [], |r| r.get::<_, i64>(0))
            .unwrap()
            == 2
        {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(15));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop(second);
    let second = spawn("host-b");
    let start = Instant::now();
    loop {
        let db = polylogue_store::archive::open_read(&source.join("archive.db")).unwrap();
        let aliases: i64 = db
            .query_row("SELECT COUNT(*) FROM machine", [], |r| r.get(0))
            .unwrap();
        let units: i64 = db
            .query_row(
                "SELECT (SELECT COUNT(*) FROM observation)+(SELECT COUNT(*) FROM revision)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        if aliases == 2
            && units > 2
            && sinex
                .have(&[format!("rows:{id}:{units}")])
                .await
                .unwrap()
                .len()
                == 1
        {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(30));
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
    let (archive, _lock) = Archive::open_or_create(&source).unwrap();
    let current = Published::open(&archive.root).unwrap();
    let changed = bytes(&current, &unit);
    fs::write(root.join("current-observation.jsonl"), &changed).unwrap();
    assert_ne!(original, changed);
    let report = sinex.publish(&source).await;
    let receipt = serde_json::json!({"original":blake3::hash(&original).to_hex().to_string(),"current":blake3::hash(&changed).to_hex().to_string(),"result":format!("{report:?}"),"source_version":9,"producer_version":8});
    fs::write(
        root.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    eprintln!("pre9 actual held-frame drift: {receipt}");
    assert!(
        report.is_err(),
        "held original machine payload changed but publication reported success"
    );
}

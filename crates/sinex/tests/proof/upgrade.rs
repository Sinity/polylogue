//! An actual archive7 producer, followed by source-free archive8/index33 replay.
use super::*;
use polylogue_store::sinex_rows::Published;
use rusqlite::{Connection, types::Value};
use std::io::Read;

fn command(bin: &Path, root: &Path, home: &Path, config: &Path) -> Command {
    let mut command = Command::new(bin);
    command
        .env("POLYLOGUE_ARCHIVE", root)
        .env("POLYLOGUE_HOME", home)
        .env("XDG_CONFIG_HOME", config);
    command
}
fn cli(bin: &Path, root: &Path, home: &Path, config: &Path, args: &[&str]) -> serde_json::Value {
    let output = command(bin, root, home, config)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "old CLI {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn rows(path: &Path, sql: &str) -> Vec<Vec<Value>> {
    let db = Connection::open(path).unwrap();
    let mut statement = db.prepare(sql).unwrap();
    let count = statement.column_count();
    statement
        .query_map([], |row| {
            (0..count)
                .map(|i| row.get(i))
                .collect::<rusqlite::Result<Vec<Value>>>()
        })
        .unwrap()
        .map(Result::unwrap)
        .collect()
}
fn versions(root: &Path) -> Vec<Vec<Value>> {
    rows(
        &root.join("index.db"),
        "SELECT r.id,s.chain,s.source_start,s.source_end,s.parser_version,hex(s.version_hash) FROM message_source s JOIN input_ref r ON r.pk=s.input_pk ORDER BY r.id,s.chain,s.source_start,hex(s.version_hash)",
    )
}
fn notes(root: &Path) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    (
        rows(&root.join("archive.db"), "SELECT * FROM note ORDER BY id"),
        rows(
            &root.join("archive.db"),
            "SELECT * FROM note_target ORDER BY note_id,ref",
        ),
    )
}
fn record(word: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"user","uuid":"native-upgrade-message","sessionId":"native-upgrade-session","message":{"role":"user","content":[{"type":"text","text":text},{"type":"image","source":{"type":"base64","media_type":"image/png","data":word.repeat(45000)}}]}})
    )
}
fn wait_versions(root: &Path, count: i64) {
    let start = Instant::now();
    loop {
        if root.join("index.db").exists() {
            let db = Connection::open(root.join("index.db")).unwrap();
            if db
                .query_row("SELECT count(*) FROM message_source", [], |r| {
                    r.get::<_, i64>(0)
                })
                .ok()
                == Some(count)
            {
                return;
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "old producer did not retain {count} versions"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}
fn replay(root: &Path, home: &Path, machine: &str, host: &str, roots: Vec<PathBuf>) {
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let mut engine = Engine::new(
        Config {
            home: home.into(),
            tool_dirs: Default::default(),
            workers: 2,
            reader_memory_bytes: 128 << 20,
            bytes_in_flight: 16 << 20,
            machine: machine.into(),
            host: host.into(),
            chunk: 64 << 20,
            blob_roots: roots,
        },
        archive,
    )
    .unwrap();
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
}
fn fingerprint(root: &Path) -> Vec<(polylogue_store::sinex_rows::Unit, String)> {
    let published = Published::open(root).unwrap();
    let inventory = published.inventory().unwrap();
    (1..=inventory.len())
        .map(|position| {
            let unit = inventory.unit(position).unwrap();
            let mut batch = published.read(&unit).unwrap();
            let mut hash = blake3::Hasher::new();
            std::io::copy(&mut batch.file, &mut hash).unwrap();
            (unit, hash.finalize().to_hex().to_string())
        })
        .collect()
}
fn assert_refused(mut command: Command, needle: &str, dir: &Path) {
    let error = dir.join(format!("old-refusal-{}.log", needle.len()));
    command
        .stdout(std::process::Stdio::null())
        .stderr(fs::File::create(&error).unwrap());
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(15) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "old writer/decoder did not refuse: {}",
                fs::read_to_string(&error).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let error = fs::read_to_string(error).unwrap();
    assert!(
        !status.success() && error.contains(needle),
        "old refusal: {status}: {error}"
    );
}

pub(super) async fn verify(sinex: &mut Sinex, config: &polylogue_sinex::Config, dir: &Path) {
    let old = PathBuf::from(
        std::env::var_os("POLYLOGUE_OLD7")
            .expect("POLYLOGUE_OLD7 names the retained archive7 producer"),
    );
    let root = dir.join("old7-source");
    let home = dir.join("old7-home");
    let cfg = dir.join("old7-config");
    let path = home.join(".claude/projects/synthetic/native-upgrade-session.jsonl");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, record("QUFB", "Image version one")).unwrap();
    {
        let daemon = Daemon(
            polylogue_testkit::spawn_daemon(command(&old, &root, &home, &cfg).arg("daemon"))
                .unwrap(),
        );
        wait_versions(&root, 1);
        fs::write(&path, record("QkJC", "Image version two")).unwrap();
        wait_versions(&root, 2);
        let read = cli(
            &old,
            &root,
            &home,
            &cfg,
            &["--json", "read", "claude-code:native-upgrade-session"],
        );
        assert_eq!(
            read["messages"][0]["blocks"][0]["text"],
            "Image version two"
        );
        cli(
            &old,
            &root,
            &home,
            &cfg,
            &[
                "--json",
                "note",
                "add",
                "--body",
                "Synthetic retained image note",
                "message:claude-code:native-upgrade-message",
            ],
        );
        drop(daemon);
    }
    assert_eq!(
        rows(&root.join("archive.db"), "PRAGMA user_version"),
        vec![vec![Value::Integer(7)]]
    );
    assert_eq!(
        rows(&root.join("index.db"), "PRAGMA user_version"),
        vec![vec![Value::Integer(32)]]
    );
    let source_versions = versions(&root);
    assert_eq!(source_versions.len(), 2);
    assert_ne!(
        source_versions[0][1], source_versions[1][1],
        "the old producer retained different raw chains"
    );
    let db = Connection::open(root.join("archive.db")).unwrap();
    db.execute("UPDATE note_target SET seen_hash=zeroblob(32),anchor_input=(SELECT input FROM revision ORDER BY number LIMIT 1),anchor_chain=1,anchor_start=0,anchor_end=1,anchor_parser=1",[]).unwrap();
    let machine: String = db
        .query_row("SELECT machine FROM observation LIMIT 1", [], |r| r.get(0))
        .unwrap();
    let host: String = db
        .query_row(
            "SELECT host FROM machine WHERE id=?1 ORDER BY host LIMIT 1",
            [&machine],
            |r| r.get(0),
        )
        .unwrap();
    drop(db);
    let embeddings = Connection::open(root.join("embeddings.db")).unwrap();
    embeddings.execute("INSERT INTO embedded(id,text_hash,model,dims,vector,embedded_at) VALUES (7,?1,'synthetic/embedding/1/document/unit',1,x'0000803e',1)",[blake3::hash(b"Image version two").as_bytes().as_slice()]).unwrap();
    drop(embeddings);
    let authored = notes(&root);
    let paid = rows(
        &root.join("embeddings.db"),
        "SELECT * FROM embedded ORDER BY id",
    );
    fs::remove_file(&path).unwrap();
    let empty = dir.join("upgrade-empty-home");
    fs::create_dir_all(&empty).unwrap();

    // Migration creates the derived generation; raw units can be acknowledged
    // before retained-source replay adds any extraction association.
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    drop(lock);
    assert_eq!(
        rows(&root.join("archive.db"), "PRAGMA user_version"),
        vec![vec![Value::Integer(8)]]
    );
    assert_eq!(
        rows(&root.join("index.db"), "PRAGMA user_version"),
        vec![vec![Value::Integer(33)]]
    );
    let original = fingerprint(&root);
    let first = sinex.publish(&root).await.unwrap();
    assert!(first.units > 0);
    let start = Instant::now();
    let id = Published::open(&root).unwrap().archive_id().unwrap();
    loop {
        let restored = dir.join("old7-prefix-restore");
        let (target, lock) = Archive::open_or_create(&restored).unwrap();
        let report = sinex.restore(&target, &id).await.unwrap();
        drop(lock);
        if report.units == first.units {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    replay(&root, &empty, &machine, &host, Vec::new());
    assert_eq!(versions(&root), source_versions);
    assert_eq!(notes(&root), authored);
    assert_eq!(
        rows(
            &root.join("embeddings.db"),
            "SELECT * FROM embedded ORDER BY id"
        ),
        paid
    );
    assert_eq!(
        rows(
            &root.join("archive.db"),
            "SELECT count(*) FROM extracted_blob WHERE state='present'"
        ),
        vec![vec![Value::Integer(2)]]
    );
    let after = fingerprint(&root);
    let added: Vec<_> = after
        .iter()
        .filter(|(unit, _)| !original.iter().any(|(old, _)| old == unit))
        .collect();
    assert_eq!(
        added
            .iter()
            .filter(|(unit, _)| unit.kind == "extracted_blob")
            .count(),
        2
    );
    assert!(
        added
            .iter()
            .all(|(unit, _)| matches!(unit.kind.as_str(), "extracted_blob" | "observation"))
    );
    let added_count = added.len() as u64;
    for (unit, hash) in original {
        assert_eq!(
            after.iter().find(|(u, _)| u == &unit).unwrap().1,
            hash,
            "late extraction changed an acknowledged unit {unit:?}"
        );
    }
    let late = sinex.publish(&root).await.unwrap();
    assert_eq!(late.units, added_count);
    let expected = answers(&root);
    let total = Published::open(&root).unwrap().inventory().unwrap().len();
    let restored = dir.join("upgraded-source-restore");
    let start = Instant::now();
    loop {
        let (target, lock) = Archive::open_or_create(&restored).unwrap();
        let report = sinex.restore(&target, &id).await.unwrap();
        drop(lock);
        if report.units == total {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    replay(
        &restored,
        &empty,
        &machine,
        &host,
        vec![config.store.clone()],
    );
    assert_eq!(answers(&restored), expected);
    assert_eq!(versions(&restored), source_versions);
    assert_eq!(notes(&restored), authored);
    let reader = hashed_blobs::Reader::new(config.store.clone(), Vec::new());
    for bytes in [b"AAA".repeat(45000), b"BBB".repeat(45000)] {
        let mut source = reader
            .read_plain(
                &hashed_blobs::Hash::of(&bytes),
                Some(bytes.len() as u64),
                || Ok(()),
            )
            .unwrap();
        let mut actual = Vec::new();
        source.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, bytes);
    }

    let mut old_writer = command(&old, &root, &empty, &cfg);
    old_writer.arg("daemon");
    assert_refused(old_writer, "archive format 8", dir);
    let decoder_cfg = dir.join("old-decoder-config");
    fs::create_dir_all(decoder_cfg.join("polylogue")).unwrap();
    fs::write(
        decoder_cfg.join("polylogue/sinex.toml"),
        format!(
            "mode='primary'\nnats={:?}\nsubjects={:?}\nproducer={:?}\nstore={:?}\narchive={:?}\n",
            config.nats,
            config.subjects,
            config.producer,
            config.store.display().to_string(),
            id
        ),
    )
    .unwrap();
    let decoder_root = dir.join("old-decoder-target");
    let mut old_decoder = command(&old, &decoder_root, &empty, &decoder_cfg);
    old_decoder.arg("daemon");
    let daemon = Daemon(polylogue_testkit::spawn_daemon(&mut old_decoder).unwrap());
    let status = cli(
        &old,
        &decoder_root,
        &empty,
        &decoder_cfg,
        &["--json", "status"],
    );
    assert_eq!(status["backing"]["state"], "blocked");
    assert!(
        status["backing"]["error"]
            .as_str()
            .unwrap()
            .contains("confirmed unit identities are not in strict archive order"),
        "{status}"
    );
    assert_eq!(
        rows(
            &decoder_root.join("archive.db"),
            "SELECT count(*) FROM revision"
        ),
        vec![vec![Value::Integer(0)]]
    );
    drop(daemon);
    drop(archive);
}

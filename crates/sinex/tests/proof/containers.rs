//! Genuine archive9 member-only capture, then exact original custody and archive10 restore.
use super::*;
use polylogue_model::{BlobHash, MessageId, SessionId};
use polylogue_store::assets::AttachmentState;
use polylogue_store::sinex_rows::Published;
use rusqlite::{Connection, types::Value};
use std::io::Write;

const SESSION: &str = "chatgpt:zip-proof-session";
const MESSAGE: &str = "chatgpt:zip-proof-message";
const ASSET: &[u8] = b"synthetic ZIP member bytes\0\xff";

fn make_zip(path: &Path) -> Vec<u8> {
    let conversation=serde_json::to_vec(&serde_json::json!([{"id":"zip-proof-session","title":"Native ZIP","mapping":{"node":{"id":"node","parent":null,"children":[],"message":{"id":"zip-proof-message","author":{"role":"user"},"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Zip"}]},"metadata":{}}}},"current_node":"node"}])).unwrap();
    let mut output = zip::ZipWriter::new(fs::File::create(path).unwrap());
    output.set_comment("synthetic original ZIP metadata λ🙂");
    for (name, bytes) in [
        ("conversations.json", conversation.as_slice()),
        ("file-Zip.dat", ASSET),
        (
            "conversation_asset_file_names.json",
            br#"{"file-Zip.dat":"native-original.png"}"#.as_slice(),
        ),
    ] {
        output
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        output.write_all(bytes).unwrap();
    }
    output.finish().unwrap();
    fs::read(path).unwrap()
}

fn check(root: &Path, roots: &[PathBuf], original: &[u8]) {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let reader = pool.reader().unwrap();
    let id = BlobHash::of(original).to_string();
    let input = reader.retained_input(&id).unwrap().unwrap();
    let store = hashed_blobs::Store::open(root, roots.to_vec(), 0o600).unwrap();
    assert_eq!(
        reader
            .stored_range(&id, input.chain, 0, input.length, &store)
            .unwrap(),
        original
    );
    let descriptor = reader
        .attachment(&MessageId::parse(MESSAGE).unwrap(), 0, 0)
        .unwrap();
    assert_eq!(descriptor.state, AttachmentState::Available);
    assert_eq!(descriptor.candidates[0].names, ["native-original.png"]);
    assert_eq!(
        reader
            .attachment_range(&descriptor.candidates[0], roots, 0..ASSET.len() as u64)
            .unwrap(),
        ASSET
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM a.observation_container", &[])
            .unwrap()[0]["n"],
        3
    );
}

fn plan_removal(root: &Path, home: &Path, roots: Vec<PathBuf>, original: &[u8]) {
    converge(root, home, roots, |engine| {
        for (index, scope) in [
            polylogue_store::excise::Scope::Input(BlobHash::of(ASSET).to_string()),
            polylogue_store::excise::Scope::Input(BlobHash::of(original).to_string()),
            polylogue_store::excise::Scope::Session(SessionId::parse(SESSION).unwrap()),
        ]
        .into_iter()
        .enumerate()
        {
            let planned = engine
                .excise(polylogue_store::excise::ExciseRequest {
                    id: format!("zip-removal-plan-{index}"),
                    scope,
                    reason: None,
                    dry_run: true,
                })
                .unwrap();
            assert!(planned.dry_run);
            assert_eq!(
                planned.containers_retired,
                [BlobHash::of(original).to_string()]
            );
        }
    });
}

pub(super) async fn verify(sinex: &mut Sinex, config: &polylogue_sinex::Config, dir: &Path) {
    let old = PathBuf::from(
        std::env::var_os("POLYLOGUE_OLD9")
            .expect("POLYLOGUE_OLD9 names a retained archive9 frontend"),
    );
    let root = dir.join("original-zip-upgrade");
    let home = dir.join("zip-empty-home");
    let cfg = dir.join("zip-old-config");
    fs::create_dir_all(&home).unwrap();
    fs::create_dir_all(&cfg).unwrap();
    let path = dir.join("native-original.zip");
    let original = make_zip(&path);
    let mut command = upgrade::command(&old, &root, &home, &cfg);
    command.arg("daemon");
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let started = Instant::now();
    while !root.join("daemon.sock").exists() {
        assert!(daemon.0.try_wait().unwrap().is_none());
        assert!(started.elapsed() < Duration::from_secs(15));
        std::thread::sleep(Duration::from_millis(20));
    }
    let imported = upgrade::cli(
        &old,
        &root,
        &home,
        &cfg,
        &["--json", "import", path.to_str().unwrap()],
    );
    assert!(
        imported["pass"]["errors"].as_array().unwrap().is_empty(),
        "{imported}"
    );
    let message = upgrade::cli(&old, &root, &home, &cfg, &["--json", "read", SESSION]);
    assert_eq!(message["messages"].as_array().unwrap().len(), 1);
    upgrade::cli(
        &old,
        &root,
        &home,
        &cfg,
        &[
            "--json",
            "note",
            "add",
            "message:chatgpt:zip-proof-message",
            "--body",
            "Synthetic preserved ZIP note",
        ],
    );
    drop(daemon);
    assert_eq!(
        upgrade::rows(&root.join("archive.db"), "PRAGMA user_version"),
        vec![vec![Value::Integer(9)]]
    );
    let units = upgrade::fingerprint(&root);
    let versions = upgrade::versions(&root);
    let notes = upgrade::notes(&root);
    let embeddings = Connection::open(root.join("embeddings.db")).unwrap();
    embeddings.execute("INSERT INTO embedded(id,text_hash,model,dims,vector,embedded_at) VALUES (91,?1,'synthetic/embedding/1/document/unit',1,x'0000803e',1)",[blake3::hash(b"synthetic ZIP source").as_bytes().as_slice()]).unwrap();
    drop(embeddings);
    let paid = upgrade::rows(
        &root.join("embeddings.db"),
        "SELECT * FROM embedded ORDER BY id",
    );
    converge(&root, &home, vec![], |engine| {
        assert!(
            ReadPool::open(&Archive::at(&root))
                .unwrap()
                .reader()
                .unwrap()
                .retained_input(&BlobHash::of(&original).to_string())
                .unwrap()
                .is_none(),
            "old member-only capture cannot fabricate original ZIP bytes"
        );
        let imported = engine.import(std::slice::from_ref(&path)).unwrap();
        assert!(
            imported.pass.errors.is_empty(),
            "{:?}",
            imported.pass.errors
        );
    });
    fs::remove_file(path).unwrap();
    polylogue_testkit::delete_index(&root);
    converge(&root, &home, vec![], |_| {});
    assert_eq!(
        upgrade::rows(&root.join("archive.db"), "PRAGMA user_version"),
        vec![vec![Value::Integer(10)]]
    );
    assert_eq!(upgrade::versions(&root), versions);
    assert_eq!(upgrade::notes(&root), notes);
    assert_eq!(
        upgrade::rows(
            &root.join("embeddings.db"),
            "SELECT * FROM embedded ORDER BY id"
        ),
        paid
    );
    let after = upgrade::fingerprint(&root);
    for (unit, hash) in units {
        assert_eq!(
            after
                .iter()
                .find(|(candidate, _)| candidate == &unit)
                .unwrap()
                .1,
            hash,
            "old acknowledged unit changed"
        );
    }
    check(&root, &[], &original);
    plan_removal(&root, &home, vec![], &original);
    let expected = answers(&root);
    let id = Published::open(&root).unwrap().archive_id().unwrap();
    let first = sinex.publish(&root).await.unwrap();
    assert!(first.units > 0);
    let physical = hashed_blobs::Store::open(config.store.clone(), vec![], 0o600).unwrap();
    for body in [original.as_slice(), ASSET] {
        assert!(physical.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
    }
    let before = blake3::hash(&fs::read(root.join("archive.db")).unwrap());
    let mut old_writer = upgrade::command(&old, &root, &home, &cfg);
    old_writer.arg("daemon");
    upgrade::assert_refused(old_writer, "archive format 10", dir);
    assert_eq!(
        blake3::hash(&fs::read(root.join("archive.db")).unwrap()),
        before,
        "old writer changed archive10"
    );
    let target = dir.join("original-zip-restored");
    let total = Published::open(&root).unwrap().inventory().unwrap().len();
    let started = Instant::now();
    loop {
        let (archive, lock) = Archive::open_or_create(&target).unwrap();
        let restored = sinex.restore(&archive, &id).await.unwrap();
        drop(lock);
        if restored.units == total {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    converge(&target, &home, vec![config.store.clone()], |_| {});
    assert_eq!(answers(&target), expected);
    check(&target, std::slice::from_ref(&config.store), &original);
    plan_removal(&target, &home, vec![config.store.clone()], &original);
    old_decoder(&old, config, &id, &home, dir);
    converge(&target, &home, vec![config.store.clone()], |engine| {
        let removed = engine
            .excise(polylogue_store::excise::ExciseRequest {
                id: "restored-zip-member".into(),
                scope: polylogue_store::excise::Scope::Input(BlobHash::of(ASSET).to_string()),
                reason: None,
                dry_run: false,
            })
            .unwrap();
        assert_eq!(
            removed.containers_retired,
            [BlobHash::of(&original).to_string()]
        );
        let pool = ReadPool::open(&engine.archive).unwrap();
        let reader = pool.reader().unwrap();
        assert!(
            reader
                .retained_input(&BlobHash::of(&original).to_string())
                .unwrap()
                .is_none()
        );
        assert!(
            reader
                .retained_input(&BlobHash::of(ASSET).to_string())
                .unwrap()
                .is_none()
        );
    });
    let published = sinex.publish(&target).await.unwrap();
    assert!(published.units > 0);
    assert_eq!(
        published.physical_retirement,
        polylogue_sinex::PhysicalRetirement::Completed
    );
    for body in [original.as_slice(), ASSET] {
        assert!(!physical.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
    }
    let total = Published::open(&target).unwrap().inventory().unwrap().len();
    let after = dir.join("zip-retirement-restored");
    let started = Instant::now();
    loop {
        let (archive, lock) = Archive::open_or_create(&after).unwrap();
        let restored = sinex.restore(&archive, &id).await.unwrap();
        drop(lock);
        if restored.units == total {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    converge(&after, &home, vec![config.store.clone()], |_| {});
    let pool = ReadPool::open(&Archive::at(&after)).unwrap();
    let reader = pool.reader().unwrap();
    for body in [original.as_slice(), ASSET] {
        assert!(
            reader
                .retained_input(&BlobHash::of(body).to_string())
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM a.observation_container", &[])
            .unwrap()[0]["n"],
        0
    );
    let descriptor = reader
        .attachment(&MessageId::parse(MESSAGE).unwrap(), 0, 0)
        .unwrap();
    assert_eq!(descriptor.state, AttachmentState::Unavailable);
    assert!(descriptor.candidates.is_empty());
}

fn old_decoder(old: &Path, config: &polylogue_sinex::Config, id: &str, home: &Path, dir: &Path) {
    let cfg = dir.join("zip-old-decoder-config");
    fs::create_dir_all(cfg.join("polylogue")).unwrap();
    fs::write(
        cfg.join("polylogue/sinex.toml"),
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
    let root = dir.join("zip-old-decoder-target");
    let mut command = upgrade::command(old, &root, home, &cfg);
    command.arg("daemon");
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let started = Instant::now();
    while !root.join("daemon.sock").exists() {
        assert!(daemon.0.try_wait().unwrap().is_none());
        assert!(started.elapsed() < Duration::from_secs(15));
        std::thread::sleep(Duration::from_millis(20));
    }
    let status = upgrade::cli(old, &root, home, &cfg, &["--json", "status"]);
    assert_eq!(status["backing"]["state"], "blocked", "{status}");
    let error = status["backing"]["error"].as_str().unwrap();
    assert!(
        error.contains("outside its unit") || error.contains("known table"),
        "{status}"
    );
    assert_eq!(
        upgrade::rows(&root.join("archive.db"), "SELECT count(*) FROM revision"),
        vec![vec![Value::Integer(0)]]
    );
    drop(daemon);
}

use super::*;
use polylogue_store::sinex_rows::{Published, Unit};
use polylogue_store::writer::{Batch, InFlight, InputKey, Observation};
use polylogue_store::{Change, Writer};

fn payload(source: &Published, unit: &Unit) -> Vec<u8> {
    let mut batch = source.read(unit).unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut batch.file, &mut bytes).unwrap();
    bytes
}

pub(super) async fn rename(sinex: &mut Sinex, root: &Path) {
    let path = root.join("machine-alias-source");
    let (source, lock) = Archive::open_or_create(&path).unwrap();
    let writer = Writer::start(&source).unwrap();
    let observe = |host: &str, at, change| {
        writer
            .send(Batch {
                input: InputKey {
                    id: "synthetic-machine-input".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change,
                observation: Some(Observation {
                    machine: "synthetic-machine".into(),
                    host: host.into(),
                    path: "synthetic".into(),
                    facts: None,
                    seen_at_ms: at,
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
    observe("host-a", 900, Change::First);
    let published = Published::open(&path).unwrap();
    let id = published.archive_id().unwrap();
    let unit = inventory_units(&published)
        .into_iter()
        .find(|u| u.kind == "observation")
        .unwrap();
    let before = payload(&published, &unit);
    let hash = blake3::hash(&before);
    drop(published);
    let first = sinex.publish(&path).await.unwrap();
    assert_eq!((first.units, first.held), (2, 0));
    observe("host-b", 1, Change::Unchanged);
    let published = Published::open(&path).unwrap();
    assert_eq!(payload(&published, &unit), before);
    assert_eq!(blake3::hash(&payload(&published, &unit)), hash);
    drop(published);
    let next = sinex.publish(&path).await.unwrap();
    assert_eq!((next.units, next.held), (1, 2));
    drop(writer);
    drop(lock);
    let target = root.join("machine-alias-restored");
    let (restored, lock) = Archive::open_or_create(&target).unwrap();
    let deadline = Instant::now();
    loop {
        if sinex.restore(&restored, &id).await.unwrap().units == 3 {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(lock);
    let (restored, _lock) = Archive::open_or_create(&target).unwrap();
    let published = Published::open(&target).unwrap();
    assert_eq!(payload(&published, &unit), before);
    assert_eq!(blake3::hash(&payload(&published, &unit)), hash);
    let db = polylogue_store::archive::open_read(&restored.archive_db()).unwrap();
    let hosts: Vec<String> = db
        .prepare("SELECT host FROM machine WHERE id='synthetic-machine' ORDER BY host")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(hosts, ["host-a", "host-b"]);
    assert_eq!(sinex.publish(&target).await.unwrap().units, 0);
}

#[test]
#[ignore = "needs the retained actual archive8 frontend in POLYLOGUE_OLD8"]
fn an_actual_archive8_writer_migrates_and_then_refuses_archive9() {
    let old = std::env::var_os("POLYLOGUE_OLD8")
        .expect("POLYLOGUE_OLD8 names the retained archive8 frontend");
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let home = dir.path().join("home");
    let config = dir.path().join("config");
    let input = home.join(".claude/projects/synthetic/machine-upgrade.jsonl");
    fs::create_dir_all(input.parent().unwrap()).unwrap();
    fs::write(&input,"{\"type\":\"user\",\"uuid\":\"machine-upgrade\",\"sessionId\":\"machine-upgrade\",\"message\":{\"role\":\"user\",\"content\":\"Synthetic machine migration\"}}\n").unwrap();
    let command = || {
        let mut command = Command::new(&old);
        command
            .env("POLYLOGUE_ARCHIVE", &root)
            .env("POLYLOGUE_HOME", &home)
            .env("XDG_CONFIG_HOME", &config);
        command
    };
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(command().arg("daemon")).unwrap());
    let started = Instant::now();
    loop {
        if root.join("archive.db").exists() {
            let db = polylogue_store::archive::open_read(&root.join("archive.db")).unwrap();
            let count = db
                .query_row("SELECT count(*) FROM observation", [], |r| {
                    r.get::<_, i64>(0)
                })
                .unwrap_or(0);
            if count > 0 {
                break;
            }
        }
        assert!(
            daemon.0.try_wait().unwrap().is_none(),
            "old8 frontend exited during collection"
        );
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "old8 frontend did not commit its synthetic observation"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(daemon);
    let machine_rows = |archive: &Archive| {
        let db = polylogue_store::archive::open_read(&archive.archive_db()).unwrap();
        db.prepare("SELECT id,host,first_seen FROM machine ORDER BY id,host")
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    let archive = Archive::at(&root);
    let before = machine_rows(&archive);
    assert_eq!(before.len(), 1);
    let db = polylogue_store::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(polylogue_store::schema::version(&db).unwrap(), 8);
    drop(db);
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    assert_eq!(machine_rows(&archive), before);
    let db = polylogue_store::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(polylogue_store::schema::version(&db).unwrap(), 9);
    let published = Published::open(&root).unwrap();
    let units = inventory_units(&published);
    let batches: Vec<_> = units.iter().map(|unit| payload(&published, unit)).collect();
    drop(published);
    drop(db);
    drop(lock);
    let refused = command().arg("daemon").output().unwrap();
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr)
            .contains("archive format 9 is newer than this Polylogue (8)")
    );
    assert_eq!(machine_rows(&archive), before);
    let db = polylogue_store::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(polylogue_store::schema::version(&db).unwrap(), 9);
    let published = Published::open(&root).unwrap();
    assert_eq!(inventory_units(&published), units);
    for (unit, bytes) in units.iter().zip(batches) {
        assert_eq!(payload(&published, unit), bytes);
    }
}

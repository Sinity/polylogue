//! The rebuild proof (joint #7, architecture 7.1): an archive converged in
//! `standalone`, published to Sinex, restored from Sinex into an empty
//! archive and converged there with no provider files present gives the same
//! answers. It runs a real `sinexd` over a throwaway PostgreSQL and NATS, so
//! it needs Sinex's binaries: `SINEXD` and `SINEX_SCHEMA` name them
//! (`cargo build -p sinexd -p sinex-schema` in a Sinex checkout).
use async_nats::jetstream;
use polylogue_engine::{Config, Engine};
use polylogue_sinex::{Mode, Sinex};
use polylogue_store::{Archive, ReadPool};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

#[cfg(feature = "faults")]
#[path = "proof/retention.rs"]
mod retention;
#[cfg(feature = "faults")]
#[path = "proof/upgrade.rs"]
mod upgrade;

#[path = "proof/assets.rs"]
mod assets;
#[cfg(feature = "faults")]
#[path = "proof/containers.rs"]
mod containers;

const EXAMPLES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/sources/claude-code"
);

/// The Claude Code examples placed where Claude Code writes them.
fn home_with_examples(home: &Path) {
    for case in fs::read_dir(EXAMPLES).unwrap() {
        let case = case.unwrap().path();
        let Ok(rel) = fs::read_to_string(case.join("path")) else {
            continue;
        };
        let to = home.join(rel.trim());
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(case.join("input.jsonl"), to).unwrap();
    }
}

fn converge(root: &Path, home: &Path, blob_roots: Vec<PathBuf>, prepare: impl FnOnce(&mut Engine)) {
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let config = Config {
        home: home.to_path_buf(),
        tool_dirs: Default::default(),
        workers: 2,
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 16 << 20,
        machine: "proof-machine".into(),
        host: "proof".into(),
        chunk: 64 << 20,
        blob_roots,
    };
    let mut engine = Engine::new(config, archive).unwrap();
    prepare(&mut engine);
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
}

/// Every session and its messages, as the reader answers them.
fn answers(root: &Path) -> String {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let reader = pool.reader().unwrap();
    let mut out = String::new();
    let sessions = reader
        .query(
            "SELECT id, kind, coalesce(title, derived_title) AS title, started_at, last_activity_at, cwd \
             FROM session ORDER BY id",
            &[],
        )
        .unwrap();
    for s in sessions {
        out.push_str(&format!("session {s}\n"));
        let id = polylogue_model::SessionId::parse(s["id"].as_str().unwrap()).unwrap();
        for m in reader.messages(&id, false).unwrap() {
            out.push_str(&format!("  {m:?}\n"));
        }
    }
    for row in reader
        .query("SELECT id FROM current_document ORDER BY id", &[])
        .unwrap()
    {
        let id = polylogue_model::DocumentId::parse(row["id"].as_str().unwrap()).unwrap();
        let document = reader.document(&id).unwrap().unwrap();
        out.push_str(&format!("document {document:?}\n"));
    }
    out
}

fn document_sources(root: &Path) -> Vec<serde_json::Value> {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let reader = pool.reader().unwrap();
    reader.query("SELECT d.id,r.id AS input,s.chain,s.source_start,s.source_end,s.parser_version,hex(s.version_hash) AS version FROM document_source s JOIN document d ON d.pk=s.document_pk JOIN input_ref r ON r.pk=s.input_pk ORDER BY d.id,r.id,s.chain,s.source_start",&[]).unwrap()
}

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
#[ignore = "needs Sinex's binaries; set SINEXD and SINEX_SCHEMA"]
async fn an_archive_restored_from_sinex_gives_the_same_answers() {
    let sinexd = std::env::var("SINEXD").expect("SINEXD names a built sinexd");
    let schema = std::env::var("SINEX_SCHEMA").expect("SINEX_SCHEMA names a built sinex-schema");
    let dir = tempfile::tempdir().unwrap();

    // 1. A standalone archive.
    let home = dir.path().join("home");
    home_with_examples(&home);
    let a = dir.path().join("a");
    let source = dir.path().join("projects/native-project.json");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    let project = serde_json::json!({"uuid":"native-project","updated_at":"2026-01-01T00:00:00Z","prompt_template":"Supplied instructions","docs":[{"uuid":"gone","content":"Remove this supplied text"},{"uuid":"kept","filename":"manual.txt","content":"Exact supplied document λ🙂\nsecond line"}]});
    fs::write(&source, serde_json::to_vec(&project).unwrap()).unwrap();
    converge(&a, &home, vec![], |engine| {
        assets::seed(engine, dir.path());
        for revision in [0, 1, 2] {
            let mut value = project.clone();
            if revision > 0 {
                value["updated_at"] =
                    serde_json::json!(format!("2026-01-0{}T00:00:00Z", revision + 1));
                value["prompt_template"] =
                    serde_json::json!(format!("Instruction revision {revision}"));
                value["docs"] = if revision == 1 {
                    serde_json::Value::Null
                } else {
                    serde_json::json!([])
                };
            }
            fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(
                engine
                    .import(std::slice::from_ref(&source))
                    .unwrap()
                    .pass
                    .errors
                    .is_empty()
            );
        }
        engine
            .excise(polylogue_store::excise::ExciseRequest {
                id: "remove-document".into(),
                scope: polylogue_store::excise::Scope::parse(
                    "document:claude-ai:project:native-project:doc:gone",
                )
                .unwrap(),
                reason: None,
                dry_run: false,
            })
            .unwrap();
    });
    fs::remove_file(&source).unwrap();
    let expected = answers(&a);
    let source_basis = document_sources(&a);
    assert!(expected.contains("Exact supplied document λ🙂"));
    assert!(!expected.contains("Remove this supplied text"));
    assert!(
        expected.contains("session {"),
        "the examples produced sessions"
    );

    // 2. A Sinex: PostgreSQL with its schema, NATS, sinexd.
    let pg = devkit::pg::Server::start(&devkit::pg::ClusterOptions {
        preload: vec!["timescaledb".into()],
        settings: vec![("timescaledb.enable_chunk_skipping".into(), "on".into())],
    })
    .unwrap();
    pg.sql("postgres", "CREATE DATABASE sinex").unwrap();
    let url = pg.url("sinex", None);
    let applied = Command::new(&schema)
        .args(["apply", &url])
        .status()
        .unwrap();
    assert!(applied.success(), "sinex-schema apply");
    let nats = devkit::nats::Server::start().unwrap();
    let store = dir.path().join("sinex-store");
    let tokens = dir.path().join("tokens");
    fs::create_dir_all(&tokens).unwrap();
    for t in ["read", "agent", "operator"] {
        fs::write(
            tokens.join(format!("{t}.token")),
            format!("{t}-proof-token"),
        )
        .unwrap();
    }
    let _daemon = Daemon(
        polylogue_testkit::spawn_daemon(
            Command::new(&sinexd)
                .arg("run")
                .args(["--database", &url])
                .arg("--store")
                .arg(&store)
                .args(["--nats", nats.url(), "--subjects", "proof"])
                .arg("--tokens")
                .arg(&tokens)
                .arg("--socket")
                .arg(dir.path().join("sinexd.sock")),
        )
        .unwrap(),
    );
    let client = async_nats::connect(nats.url()).await.unwrap();
    let js = jetstream::new(client);
    let started = Instant::now();
    while js.get_stream("PROOF_INGRESS").await.is_err() {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "sinexd did not start"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // 3. Publish; a second pass sends nothing.
    let config = polylogue_sinex::Config {
        mode: Mode::Primary,
        nats: nats.url().into(),
        subjects: "proof".into(),
        producer: "polylogue".into(),
        store: store.clone(),
        archive: None,
    };
    let mut sinex = Sinex::connect(&config).await.unwrap();
    let deadline = Instant::now();
    loop {
        match sinex.have(&["readiness-probe".into()]).await {
            Ok(_) => break,
            Err(polylogue_sinex::Error::Unreachable(_))
                if deadline.elapsed() < Duration::from_secs(10) =>
            {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            other => panic!("producer have endpoint did not become ready: {other:?}"),
        }
    }
    #[cfg(feature = "faults")]
    upgrade::verify(&mut sinex, &config, dir.path()).await;
    #[cfg(feature = "faults")]
    containers::verify(&mut sinex, &config, dir.path()).await;
    let first = sinex.publish(&a).await.unwrap();
    assert!(first.units > 0 && first.held == 0, "{first:?}");
    let second = sinex.publish(&a).await.unwrap();
    assert_eq!((second.units, second.held), (0, first.units), "{second:?}");

    #[cfg(feature = "faults")]
    retention::verify(&mut sinex, &a, &home, dir.path()).await;
    assets::remove(&a, &home);
    assert!(sinex.publish(&a).await.unwrap().units > 0);
    let asset_basis = assets::verify(&a, &[]);
    let expected = answers(&a);
    let total = polylogue_store::sinex_rows::Published::open(&a)
        .unwrap()
        .inventory()
        .unwrap()
        .len();

    // 4. Restore into an empty archive once every unit is confirmed.
    let id = polylogue_store::sinex_rows::Published::open(&a)
        .unwrap()
        .archive_id()
        .unwrap();
    let b = dir.path().join("b");
    let started = Instant::now();
    let restored = loop {
        let (archive, lock) = Archive::open_or_create(&b).unwrap();
        let r = sinex.restore(&archive, &id).await.unwrap();
        drop(lock);
        if r.units == total {
            break r;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "only {r:?} of {first:?} confirmed"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    };
    assert_eq!(restored.units, total);

    // 5. Converge with no provider files: every byte comes from Sinex's store.
    let empty = dir.path().join("empty-home");
    fs::create_dir_all(&empty).unwrap();
    converge(&b, &empty, vec![store.clone()], |_| {});
    assert_eq!(answers(&b), expected);
    assert_eq!(document_sources(&b), source_basis);
    assert_eq!(
        assets::verify(&b, std::slice::from_ref(&store)),
        asset_basis
    );
    assets::refuse_reimport(&b, &empty, vec![store.clone()], dir.path());
    assert_eq!(
        assets::verify(&b, std::slice::from_ref(&store)),
        asset_basis
    );
    // A restored standing document selection refuses an original export again.
    let mut reimport = project.clone();
    reimport["docs"][0]["content"] = serde_json::json!("Attempt from a different immutable input");
    fs::write(&source, serde_json::to_vec(&reimport).unwrap()).unwrap();
    converge(&b, &empty, vec![store], |engine| {
        assert!(
            engine
                .import(std::slice::from_ref(&source))
                .unwrap()
                .pass
                .errors
                .is_empty()
        );
    });
    assert_eq!(answers(&b), expected);
}

#[path = "proof/cross_uid.rs"]
mod cross_uid;

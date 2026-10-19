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

fn converge(root: &Path, home: &Path, blob_roots: Vec<PathBuf>) {
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let config = Config {
        home: home.to_path_buf(),
        tool_dirs: Default::default(),
        workers: 2,
        bytes_in_flight: 16 << 20,
        machine: "proof-machine".into(),
        host: "proof".into(),
        chunk: 64 << 20,
        blob_roots,
    };
    let mut engine = Engine::new(config, archive).unwrap();
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
}

/// Every session and its messages, as the reader answers them.
fn answers(root: &Path) -> String {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let reader = pool.reader().unwrap();
    let mut out = String::new();
    let mut sessions = reader.list(u64::MAX, None).unwrap();
    sessions.sort_by_key(|a| a.id.to_string());
    for s in sessions {
        out.push_str(&format!("{s:?}\n"));
        for m in reader.messages(&s.id, false).unwrap() {
            out.push_str(&format!("  {m:?}\n"));
        }
    }
    out
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
    converge(&a, &home, vec![]);
    let expected = answers(&a);
    assert!(
        expected.contains("SessionSummary"),
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
        Command::new(&sinexd)
            .arg("run")
            .args(["--database", &url])
            .arg("--store")
            .arg(&store)
            .args(["--nats", nats.url(), "--subjects", "proof"])
            .arg("--tokens")
            .arg(&tokens)
            .arg("--socket")
            .arg(dir.path().join("sinexd.sock"))
            .spawn()
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
    let first = sinex.publish(&a).await.unwrap();
    assert!(first.units > 0 && first.held == 0, "{first:?}");
    let second = sinex.publish(&a).await.unwrap();
    assert_eq!((second.units, second.held), (0, first.units), "{second:?}");

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
        if r.units == first.units {
            break r;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "only {r:?} of {first:?} confirmed"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    };
    assert_eq!(restored.units, first.units);

    // 5. Converge with no provider files: every byte comes from Sinex's store.
    let empty = dir.path().join("empty-home");
    fs::create_dir_all(&empty).unwrap();
    converge(&b, &empty, vec![store]);
    assert_eq!(answers(&b), expected);
}

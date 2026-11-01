//! Polylogue on a real Sinex: the Morrow example built in primary mode
//! gives the answers standalone gives; deleting the local copy and the index
//! and rebuilding them from Sinex, with no provider file left, gives the same
//! answers again; and Sinex answers its own query language about the
//! sessions. It runs a real `sinexd` over a throwaway PostgreSQL and NATS, so
//! it needs Sinex's binaries: `SINEXD` and `SINEX_SCHEMA` name them
//! (`cargo build -p sinexd -p sinex-schema` in a Sinex checkout).
use polylogue_engine::Engine;
use polylogue_model::Reference;
use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
use polylogue_model::user::UserWrite;
use polylogue_sinex::{Authentication, Config, Mode, Sinex};
use polylogue_store::{Archive, ReadPool};
use polylogue_testkit::{Morrow, Set};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A Sinex: PostgreSQL with its schema, NATS and `sinexd`.
struct Running {
    config: Config,
    socket: PathBuf,
    _daemon: Daemon,
    _nats: devkit::nats::Server,
    _pg: devkit::pg::Server,
}

fn sinex(dir: &Path) -> Running {
    let sinexd = std::env::var("SINEXD").expect("SINEXD names a built sinexd");
    let schema = std::env::var("SINEX_SCHEMA").expect("SINEX_SCHEMA names a built sinex-schema");
    let pg = devkit::pg::Server::start(&devkit::pg::ClusterOptions {
        required_extensions: vec!["timescaledb".into(), "vector".into()],
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
    let store = dir.join("sinex-store");
    let tokens = dir.join("tokens");
    fs::create_dir_all(&tokens).unwrap();
    for t in ["read", "agent", "operator"] {
        fs::write(
            tokens.join(format!("{t}.token")),
            format!("{t}-rebuild-token"),
        )
        .unwrap();
    }
    let socket = dir.join("sinexd.sock");
    let daemon = Daemon(
        polylogue_testkit::spawn_daemon(
            Command::new(&sinexd)
                .arg("run")
                .args(["--database", &url])
                .arg("--store")
                .arg(&store)
                .args(["--nats", nats.url(), "--subject-root", "rebuild"])
                .arg("--tokens")
                .arg(&tokens)
                .arg("--socket")
                .arg(&socket),
        )
        .unwrap(),
    );
    let config = Config {
        mode: Mode::Primary,
        nats: nats.url().into(),
        password_file: None,
        subject_root: "rebuild".into(),
        producer: "polylogue".into(),
        store,
    };
    Running {
        config,
        socket,
        _daemon: daemon,
        _nats: nats,
        _pg: pg,
    }
}

/// A connection once `sinexd` answers.
fn connect(config: &Config) -> Arc<Sinex> {
    let started = Instant::now();
    loop {
        if let Ok(sinex) = Sinex::connect(config, &Authentication::Anonymous, "fixture-host")
            && sinex.reachable().is_ok()
        {
            return Arc::new(sinex);
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "sinexd did not start"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Builds the Morrow example `example` with its home at `home`, on Sinex.
fn build_on_sinex(example: &Morrow, home: &Path, sinex: &Arc<Sinex>, config: &Config) {
    let (archive, _lock) = Archive::open_primary(&example.root, &config.store).unwrap();
    let mut settings = example.config();
    settings.home = home.to_path_buf();
    let mut engine = Engine::on_sinex(settings, archive.clone(), sinex.clone()).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    // The idle writer follows the confirmed stream until it reaches Sinex's head.
    let started = Instant::now();
    loop {
        let applied = ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .confirmed_checkpoint()
            .unwrap();
        let head = sinex.status().unwrap().owner_position;
        if head.is_some() && applied == head {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "the index did not reach Sinex's head: {applied:?} of {head:?}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Every session as the export format holds it, and every note.
fn answers(archive: &Archive) -> (BTreeMap<String, serde_json::Value>, Vec<serde_json::Value>) {
    let pool = ReadPool::open(archive).unwrap();
    let reader = pool.reader().unwrap();
    let sessions = reader
        .query("SELECT id FROM session ORDER BY id", &[])
        .unwrap()
        .into_iter()
        .map(|row| {
            let id = polylogue_model::SessionId::parse(row["id"].as_str().unwrap()).unwrap();
            let export = reader.session_export(&id).unwrap().unwrap();
            (id.to_string(), serde_json::to_value(export).unwrap())
        })
        .collect();
    let notes = reader
        .query(
            "SELECT n.id,n.kind,n.value,t.ref FROM source_assertion n JOIN source_assertion_link t ON t.assertion=n.id ORDER BY n.id,t.role,t.ref",
            &[],
        )
        .unwrap();
    (sessions, notes)
}

/// Rows a Sinex query answers, asked over its API socket as a reader.
fn sinex_rows(socket: &Path, query: &str) -> usize {
    let body = serde_json::json!({ "method": "query", "query": query, "page": 1000 }).to_string();
    let mut stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
    write!(
        stream,
        "POST /v1 HTTP/1.1\r\nHost: sinexd\r\nContent-Type: application/json\r\nAuthorization: Bearer read-rebuild-token\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    let json = reply
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or_default();
    // A chunked body: the JSON object starts at its first brace.
    let start = json.find('{').expect("a JSON reply");
    let end = json.rfind('}').expect("a JSON reply");
    let value: serde_json::Value =
        serde_json::from_str(&json[start..=end]).unwrap_or_else(|e| panic!("{e}: {reply}"));
    value["result"]["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("no rows in {value}"))
        .len()
}

#[test]
#[ignore = "needs Sinex's binaries; set SINEXD and SINEX_SCHEMA"]
fn the_morrow_example_rebuilt_from_sinex_gives_the_same_answers() {
    let started = Instant::now();
    let phase = |name: &str| eprintln!("{:>6.1}s {name}", started.elapsed().as_secs_f64());
    let standalone = Morrow::build(Set::Main);
    phase("standalone built");
    let expected = answers(&Archive::at(&standalone.root)).0;
    assert!(expected.len() >= 8, "the example produced its sessions");

    let example = Morrow::lay_out(Set::Main);
    let running = sinex(example.dir.path());
    let sinex = connect(&running.config);
    phase("sinexd answers");
    build_on_sinex(&example, &example.home, &sinex, &running.config);
    phase("built on Sinex");
    {
        let (archive, _lock) = Archive::open_primary(&example.root, &running.config.store).unwrap();
        let mut engine = Engine::on_sinex(example.config(), archive, sinex.clone()).unwrap();
        example.import_exports(&mut engine);
        engine
            .writer()
            .unwrap()
            .user(polylogue_store::fixture::authored::operator_request(
                "0199d8a0-0000-7000-8000-00000000a001".into(),
                UserWrite::Assert {
                    assertions: vec![NewAssertion {
                        kind: AssertionKind::Note,
                        key: None,
                        value: Some(serde_json::json!("Synthetic note kept in Sinex")),
                        links: vec![NewAssertionLink {
                            role: LinkRole::About,
                            reference: Reference::Session(polylogue_testkit::session(1)),
                        }],
                        valid_until: None,
                    }],
                },
            ))
            .unwrap();
    }
    phase("exports imported");
    let on_sinex = Archive::on_sinex(&example.root, &running.config.store);
    let (sessions, notes) = answers(&on_sinex);
    assert_eq!(
        sessions, expected,
        "primary and standalone give the same answers"
    );
    assert_eq!(notes.len(), 1);
    assert!(
        !example.root.join("archive.db").exists(),
        "primary keeps no archive.db"
    );
    assert!(
        !example.root.join("blobs").exists(),
        "primary keeps no local blobs"
    );

    // Sinex answers its own query language about the sessions.
    assert!(
        sinex_rows(
            &running.socket,
            "events where event_type = polylogue.session"
        ) >= 8
    );
    assert!(
        sinex_rows(
            &running.socket,
            "events where event_type = polylogue.message"
        ) > 0
    );
    assert_eq!(
        sinex_rows(&running.socket, "events where event_type = polylogue.assertion"),
        1
    );

    // Delete every local file Polylogue derived (the archive directory with
    // its index, and the producer spool), then rebuild from Sinex alone with
    // no provider file.
    drop(sinex);
    fs::remove_dir_all(&example.root).unwrap();
    fs::create_dir_all(&example.root).unwrap();
    fs::remove_dir_all(running.config.store.join("producer-spool")).unwrap();
    let sinex = connect(&running.config);
    let empty = example.dir.path().join("empty-home");
    fs::create_dir_all(&empty).unwrap();
    build_on_sinex(&example, &empty, &sinex, &running.config);
    phase("rebuilt from Sinex");
    assert_eq!(answers(&on_sinex), (sessions, notes), "rebuilt from Sinex");
}

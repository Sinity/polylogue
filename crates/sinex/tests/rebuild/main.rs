//! Polylogue on a real Sinex: the Morrow example built in primary mode
//! gives the answers standalone gives; deleting the local copy and the index
//! and rebuilding them from Sinex, with no provider file left, gives the same
//! answers again; and Sinex answers its own query language about the
//! sessions. It runs a real `sinexd` over a throwaway PostgreSQL and NATS, so
//! it needs Sinex's binaries: `SINEXD` and `SINEX_SCHEMA` name them
//! (`cargo build -p sinexd -p sinex-schema` in a Sinex checkout).
use polylogue_engine::Engine;
use polylogue_model::user::UserWrite;
use polylogue_model::{Author, NewNote, NoteKind, Reference};
use polylogue_sinex::{Config, Mode, Sinex};
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
fn connect(config: &Config, root: &Path) -> Arc<Sinex> {
    let started = Instant::now();
    loop {
        if let Ok(sinex) = Sinex::connect(
            config,
            &polylogue_sinex::Authentication::Anonymous,
            root,
            "synthetic-host",
        ) && sinex.reachable().is_ok()
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
    let (archive, _lock) =
        Archive::open_primary(&example.root, &config.store, sinex.as_ref()).unwrap();
    let mut settings = example.config();
    settings.home = home.to_path_buf();
    let mut engine = Engine::on_sinex(settings, archive, sinex.clone()).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
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
            "SELECT n.id, n.kind, n.value, t.ref FROM a.note n JOIN a.note_target t ON t.note_id = n.id ORDER BY n.id",
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
    let sinex = connect(&running.config, &example.root);
    phase("sinexd answers");
    build_on_sinex(&example, &example.home, &sinex, &running.config);
    phase("built on Sinex");
    {
        let (archive, _lock) =
            Archive::open_primary(&example.root, &running.config.store, sinex.as_ref()).unwrap();
        let mut engine = Engine::on_sinex(example.config(), archive, sinex.clone()).unwrap();
        example.import_exports(&mut engine);
        engine
            .writer()
            .unwrap()
            .user(polylogue_store::user_writes::request_now(
                "0199d8a0-0000-7000-8000-00000000a001".into(),
                UserWrite::AddNote {
                    note: NewNote {
                        kind: NoteKind::Note,
                        key: None,
                        value: Some("Synthetic note kept in Sinex".into()),
                        author: Author::Operator,
                        targets: vec![Reference::Session(polylogue_testkit::session(1))],
                        relations: vec![],
                        revises: None,
                        valid_until: None,
                    },
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
        sinex_rows(&running.socket, "events where event_type = polylogue.note"),
        1
    );

    // Re-read the complete note material and replace its repeat-policy reading.
    let saved = running._pg.sql("sinex", "SELECT json_build_object('id',e.id,'event',json_build_object('kind',json_build_object('name',e.event_type,'version',e.kind_version),'payload',e.payload,'location',json_build_object('material',e.material_id,'anchor',e.anchor),'output_index',e.output_index,'parents','[]'::json,'reading_order',0,'class',e.class))::text FROM core.events e WHERE event_type='polylogue.note' AND NOT EXISTS(SELECT 1 FROM audit.archived a WHERE a.event_id=e.id)").unwrap();
    let saved: serde_json::Value = serde_json::from_str(saved.trim()).unwrap();
    let event: sinex_protocol::ProducerEvent =
        serde_json::from_value(saved["event"].clone()).unwrap();
    let material = event.location.unwrap().material;
    let scope = sinex_protocol::ReplayScopeQuery {
        materials: vec![material],
        event_types: Some(vec!["polylogue.note".into()]),
        from: None,
        until: None,
    };
    let mut replay = sinex.replay().unwrap();
    let selection = replay.scope(&scope).unwrap().into_iter().next().unwrap();
    assert_eq!(selection.rows, 1);
    let operation = sinex_model::OperationId::new();
    let batches = vec![sinex_protocol::ReplayBatch {
        events: vec![event],
        failures: vec![],
    }];
    let replaced = replay
        .sequence(operation, selection.clone(), scope.clone(), batches.clone())
        .unwrap();
    assert_eq!(replaced.admitted.len(), 1);
    assert_ne!(
        replaced.admitted[0].to_string(),
        saved["id"].as_str().unwrap()
    );
    assert_eq!(
        replaced,
        replay
            .sequence(operation, selection, scope, batches)
            .unwrap()
    );
    drop(replay);
    assert_eq!(
        sinex_rows(&running.socket, "events where event_type = polylogue.note"),
        1
    );
    assert_eq!(answers(&on_sinex), (sessions.clone(), notes.clone()));
    phase("note replay acknowledged");

    // Delete the local copy and the index; rebuild with no provider file.
    for name in [
        "sinex.db",
        "sinex.db-wal",
        "sinex.db-shm",
        "index.db",
        "index.db-wal",
        "index.db-shm",
        "sinex-sent.json",
    ] {
        let _ = fs::remove_file(example.root.join(name));
    }
    let empty = example.dir.path().join("empty-home");
    fs::create_dir_all(&empty).unwrap();
    build_on_sinex(&example, &empty, &sinex, &running.config);
    phase("rebuilt from Sinex");
    assert_eq!(answers(&on_sinex), (sessions, notes), "rebuilt from Sinex");
}

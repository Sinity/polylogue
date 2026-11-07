//! Polylogue on a real Sinex: the Morrow example built in primary mode
//! gives the answers standalone gives; deleting the local copy and the index
//! and rebuilding them from Sinex, with no provider file left, gives the same
//! answers again; and Sinex answers its own query language about the
//! sessions. It runs a real `sinexd` over a throwaway PostgreSQL and NATS, so
//! `SINEXD` names the matching built daemon, including its admin schema command.
use polylogue_engine::Engine;
use polylogue_model::Reference;
use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
use polylogue_model::user::UserWrite;
use polylogue_sinex::{Authentication, Config, Mode, Sinex};
use polylogue_store::{Archive, ReadPool};
use polylogue_testkit::{Morrow, Set};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// This shared fixture exposes restore phases exercised by the unit-level backup test.
#[allow(dead_code)]
#[path = "../support/native.rs"]
mod native;

struct Running {
    config: Config,
    socket: PathBuf,
    _native: native::Native,
}

fn sinex(dir: &Path) -> Running {
    let native = native::Native::start(
        dir,
        serde_json::to_value(polylogue_sinex::kinds::source_declaration()).unwrap(),
    );
    Running {
        config: Config {
            mode: Mode::Primary,
            nats: native.nats.url().into(),
            password_file: None,
            subject_root: "rebuild".into(),
            producer: "polylogue".into(),
            store: native.store.clone(),
        },
        socket: native.socket.clone(),
        _native: native,
    }
}

/// A connection once `sinexd` answers.
fn connect(config: &Config) -> Arc<Sinex> {
    let started = Instant::now();
    loop {
        let error = match Sinex::connect(config, &Authentication::Anonymous, "fixture-host") {
            Ok(sinex) => match sinex.reachable() {
                Ok(()) => return Arc::new(sinex),
                Err(error @ polylogue_sinex::Error::Unreachable(_)) => {
                    format!("producer reachability: {error}")
                }
                Err(error) => panic!("producer reachability failed: {error}"),
            },
            Err(error @ polylogue_sinex::Error::Unreachable(_)) => {
                format!("producer connection: {error}")
            }
            Err(error) => panic!("producer connection failed: {error}"),
        };
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "Sinex producer did not become reachable: {error}"
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
            "SELECT n.id,n.kind,n.value,t.ref FROM source_assertion n JOIN source_link t ON t.record=n.id ORDER BY n.id,t.role,t.ref",
            &[],
        )
        .unwrap();
    (sessions, notes)
}

/// Rows a Sinex query answers, asked over its API socket as a reader.
fn sinex_rows(socket: &Path, query: &str) -> usize {
    let value = native::request(
        socket,
        "read",
        serde_json::json!({ "method": "query", "query": query, "page": 1000 }),
    );
    value["result"]["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("no rows in {value}"))
        .len()
}

#[test]
#[ignore = "needs the matching Sinex daemon; set SINEXD"]
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
        sinex_rows(
            &running.socket,
            "events where event_type = polylogue.assertion"
        ),
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

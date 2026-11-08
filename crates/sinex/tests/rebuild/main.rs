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
    native: native::Native,
}

impl Running {
    fn evidence(&self, sinex: &Sinex, archive: &Archive) -> String {
        let local = ReadPool::open(archive).and_then(|pool| {
            let reader = pool.reader()?;
            reader.query(
                "SELECT 'meta.'||key AS key,value FROM meta WHERE key IN ('confirmed_position','confirmed_scope','confirmed_replay','generation') UNION ALL SELECT 'source_meta.'||key,value FROM source_meta WHERE key='write_counter' ORDER BY key",
                &[],
            )
        });
        let pending = polylogue_store::primary::Sinex::pending_captures(sinex).map(|captures| {
            let mut groups = BTreeMap::<String, usize>::new();
            for capture in &captures {
                if let polylogue_store::primary::Capture::Input(input) = &capture.captured
                    && let Some(group) = &input.group
                {
                    *groups.entry(group.identity.clone()).or_default() += 1;
                }
            }
            captures
                .into_iter()
                .map(|capture| match capture.captured {
                    polylogue_store::primary::Capture::Input(input) => {
                        let current = (|| -> polylogue_store::Result<serde_json::Value> {
                            let source = input.basis.as_ref().map(|basis| basis.source.clone()).or_else(|| {
                                let observation = input.rows.observation.as_ref()?;
                                Some(polylogue_store::primary::SourceSelector {
                                    origin: input.rows.format.clone(), machine: observation.machine.clone(),
                                    path_bytes: observation.path_bytes.clone()?,
                                })
                            });
                            let Some(source) = source else { return Ok(serde_json::json!({"source_selector":"missing"})); };
                            let pool = ReadPool::open(archive)?;
                            let reader = pool.reader()?;
                            let boundary = reader.source_boundary()?;
                            let (status, confirmed) = reader.source_observation(&source)?;
                            let subject = boundary.as_ref().map(|cut| reader.source_subject_at(cut, &input.rows.input)).transpose()?.flatten();
                            Ok(serde_json::json!({
                                "status":status,"boundary":boundary,"subject":subject,
                                "confirmed_input":confirmed.as_ref().map(|value| &value.input),
                                "confirmed_observation":confirmed.as_ref().map(|value| &value.observation.id),
                            }))
                        })();
                        let current = current.unwrap_or_else(|error| serde_json::json!({"error":error.to_string()}));
                        serde_json::json!({
                        "current_source":current,
                        "members_present":input.group.as_ref().map(|group| groups.get(&group.identity).copied().unwrap_or(0)),
                        "identity":capture.identity,
                        "anchor":capture.anchor,
                        "material":capture.material.as_ref().map(|material| (&material.id, material.hash, material.length)),
                        "input":input.rows.input,"format":input.rows.format,"group":input.group,
                        "basis":input.basis,"raw_range":input.raw_range,"generation":input.generation,
                        "revision":input.rows.revision,"extent":input.rows.extent,
                        "observation":input.rows.observation.as_ref().map(|observation| serde_json::json!({
                            "id":observation.id,"machine":observation.machine,"path":observation.path,
                            "path_bytes":observation.path_bytes,"class":observation.class,
                            "revision":observation.revision,"chain":observation.chain,"length":observation.length
                        }))
                    })
                    },
                    polylogue_store::primary::Capture::Blob { hash, length } => serde_json::json!({
                        "identity":capture.identity,"anchor":capture.anchor,"material":capture.material,
                        "blob":hash,"length":length
                    }),
                })
                .collect::<Vec<_>>()
        });
        let sources = ReadPool::open(archive).and_then(|pool| {
            pool.reader()?.query(
                "SELECT i.id AS input,i.format,o.id AS observation,o.machine,o.path,hex(o.path_bytes) AS path_bytes,o.class,o.revision AS observed_revision,o.chain AS observed_chain,o.length AS observed_length,r.number,r.chain,r.length FROM source_input i LEFT JOIN source_observation o ON o.rowid=(SELECT max(x.rowid) FROM source_observation x WHERE x.input=i.id) LEFT JOIN source_revision r ON r.input=i.id AND r.number=(SELECT max(x.number) FROM source_revision x WHERE x.input=i.id) ORDER BY i.id", &[],
            )
        });
        let staged = ReadPool::open(archive).and_then(|pool| {
            pool.reader()?.query(
                "SELECT 'pending' AS stage,position,change,count(*) AS records FROM primary_pending GROUP BY position,change UNION ALL SELECT 'complete',position,NULL,1 FROM primary_complete ORDER BY position,stage LIMIT 32", &[],
            )
        });
        let refused = ReadPool::open(archive).and_then(|pool| {
            pool.reader()?.query(
                "SELECT r.event,r.kind,r.reason,r.unreadable_input,f.state,f.product IS NULL AS unreadable FROM primary_refused r LEFT JOIN primary_fact f ON f.event=r.event ORDER BY r.event", &[],
            )
        });
        format!(
            "producer: {:?}; local source heads: {sources:?}; staged positions: {staged:?}; local refused facts: {refused:?}; local replay: {local:?}; pending capture metadata: {pending:?}; first native refusals: {}; latest native receipts: {}; native event counts: {}",
            sinex.status(),
            self.native.sql("SELECT frame,refusal FROM core.ingress_frames WHERE producer='polylogue' AND refusal IS NOT NULL ORDER BY number LIMIT 8"),
            self.native.sql("SELECT number,frame,commit_seq,admitted,outcome,refusal FROM core.ingress_frames WHERE producer='polylogue' ORDER BY number DESC LIMIT 8"),
            self.native.sql("SELECT event_type,kind_version,source,source_version,source_fingerprint,count(*) FROM core.events WHERE producer_family='polylogue' GROUP BY event_type,kind_version,source,source_version,source_fingerprint ORDER BY event_type,kind_version"),
        )
    }
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
        native,
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
fn build_on_sinex(example: &Morrow, home: &Path, sinex: &Arc<Sinex>, running: &Running) {
    let (archive, _lock) = Archive::open_primary(&example.root, &running.config.store).unwrap();
    let mut settings = example.config();
    settings.home = home.to_path_buf();
    let mut engine = Engine::on_sinex(
        settings,
        archive.clone(),
        sinex.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    let commits = engine.writer().unwrap().completion_commits();
    let started = Instant::now();
    loop {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "capture did not reach confirmed application within its budget; {}",
            running.evidence(sinex, &archive)
        );
        let pass = engine.run_pass().unwrap_or_else(|error| {
            panic!(
                "native capture pass failed: {error}; {}",
                running.evidence(sinex, &archive)
            )
        });
        assert!(
            pass.errors.is_empty(),
            "{:?}; {}",
            pass.errors,
            running.evidence(sinex, &archive)
        );
        if !pass.capture_pending {
            break;
        }
        let remaining = Duration::from_secs(120).saturating_sub(started.elapsed());
        commits.recv_timeout(remaining).unwrap_or_else(|error| {
            panic!(
                "capture did not reach confirmed application: {error}; {}",
                running.evidence(sinex, &archive)
            )
        });
    }
    // The idle writer follows the confirmed stream until it reaches Sinex's head.
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
            "the index did not reach Sinex's head: {applied:?} of {head:?}; {}",
            running.evidence(sinex, &archive)
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let acquisitions = running.native.sql("SELECT frame,material_id FROM core.ingress_frames WHERE producer='polylogue' AND frame LIKE 'acquisition:%' ORDER BY frame");
    let repeated = engine.run_pass().unwrap();
    assert!(
        repeated.errors.is_empty() && !repeated.capture_pending,
        "an unchanged capture must already be confirmed: {repeated:?}; {}",
        running.evidence(sinex, &archive)
    );
    assert_eq!(
        running.native.sql("SELECT frame,material_id FROM core.ingress_frames WHERE producer='polylogue' AND frame LIKE 'acquisition:%' ORDER BY frame"),
        acquisitions,
        "an unchanged capture keeps its original acquisition and material identities"
    );
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

/// Structural comparison of synthetic answers, without logging their text bodies.
fn answer_difference(
    expected: &BTreeMap<String, serde_json::Value>,
    actual: &BTreeMap<String, serde_json::Value>,
) -> String {
    use serde_json::Value;
    fn summary(value: Option<&Value>) -> String {
        match value {
            None => "absent".into(),
            Some(Value::Null) => "null".into(),
            Some(Value::Bool(value)) => format!("bool({value})"),
            Some(Value::Number(value)) => format!("number({value})"),
            Some(Value::String(value)) => format!(
                "string(bytes={},blake3={})",
                value.len(),
                blake3::hash(value.as_bytes()).to_hex()
            ),
            Some(Value::Array(value)) => format!("array(length={})", value.len()),
            Some(Value::Object(value)) => format!("object(fields={})", value.len()),
        }
    }
    fn differences(
        expected: Option<&Value>,
        actual: Option<&Value>,
        path: String,
        changed: &mut Vec<String>,
    ) {
        if expected == actual {
            return;
        }
        match (expected, actual) {
            (Some(Value::Object(left)), Some(Value::Object(right))) => {
                let keys: std::collections::BTreeSet<_> = left.keys().chain(right.keys()).collect();
                for key in keys {
                    differences(
                        left.get(key),
                        right.get(key),
                        format!("{path}/{}", key.replace('~', "~0").replace('/', "~1")),
                        changed,
                    );
                }
            }
            (Some(Value::Array(left)), Some(Value::Array(right))) => {
                for index in 0..left.len().max(right.len()) {
                    differences(
                        left.get(index),
                        right.get(index),
                        format!("{path}/{index}"),
                        changed,
                    );
                }
            }
            _ => changed.push(format!(
                "{path}: expected {}, actual {}",
                summary(expected),
                summary(actual)
            )),
        }
    }
    let missing: Vec<_> = expected
        .keys()
        .filter(|id| !actual.contains_key(*id))
        .take(8)
        .collect();
    let extra: Vec<_> = actual
        .keys()
        .filter(|id| !expected.contains_key(*id))
        .take(8)
        .collect();
    let mut changed = Vec::new();
    for (id, value) in expected {
        if let Some(actual) = actual.get(id) {
            differences(Some(value), Some(actual), id.clone(), &mut changed);
        }
    }
    format!(
        "sessions expected={}, actual={}; missing(first8)={missing:?}; extra(first8)={extra:?}; changed={changed:?}",
        expected.len(),
        actual.len()
    )
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
    build_on_sinex(&example, &example.home, &sinex, &running);
    phase("built on Sinex");
    {
        let (archive, _lock) = Archive::open_primary(&example.root, &running.config.store).unwrap();
        let mut engine = Engine::on_sinex(
            example.config(),
            archive.clone(),
            sinex.clone(),
            polylogue_testkit::stopped_attempt(),
        )
        .unwrap();
        let commits = engine.writer().unwrap().completion_commits();
        let before_import = answer_difference(&expected, &answers(&archive).0);
        let imported = example.import_exports(&mut engine).unwrap();
        assert!(
            imported.pass.errors.is_empty() && imported.refused.is_empty(),
            "{imported:?}; {}",
            running.evidence(&sinex, &archive)
        );
        let imported_at = Instant::now();
        loop {
            assert!(
                imported_at.elapsed() < Duration::from_secs(120),
                "imported sessions did not reach the standalone answer; before import: {before_import}; current: {}; {}",
                answer_difference(&expected, &answers(&archive).0),
                running.evidence(&sinex, &archive)
            );
            let pass = engine.run_pass().unwrap();
            assert!(
                pass.errors.is_empty(),
                "{pass:?}; {}",
                running.evidence(&sinex, &archive)
            );
            let applied = ReadPool::open(&archive)
                .unwrap()
                .reader()
                .unwrap()
                .confirmed_checkpoint()
                .unwrap();
            let status = sinex.status().unwrap();
            let actual = answers(&archive).0;
            // Session presence retains the declared-source discovery barrier.
            // Once all work is confirmed, semantics are an assertion, not a wait.
            if !pass.capture_pending
                && expected.keys().all(|id| actual.contains_key(id))
                && status.prepared_materials == 0
                && status.prepared_writes == 0
                && status.owner_position.is_some()
                && applied == status.owner_position
            {
                assert!(
                    actual == expected,
                    "confirmed imported answers differ; before import: {before_import}; current: {}; {}",
                    answer_difference(&expected, &actual),
                    running.evidence(&sinex, &archive)
                );
                break;
            }
            commits
                .recv_timeout(Duration::from_secs(120).saturating_sub(imported_at.elapsed()))
                .unwrap_or_else(|error| {
                    panic!(
                        "import answer wait: {error}; before import: {before_import}; current: {}; {}",
                        answer_difference(&expected, &answers(&archive).0),
                        running.evidence(&sinex, &archive)
                    )
                });
        }
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
        let authored_at = Instant::now();
        loop {
            assert!(
                authored_at.elapsed() < Duration::from_secs(120),
                "the authored note did not reach confirmed application; {}",
                running.evidence(&sinex, &archive)
            );
            if answers(&archive).1.len() == 1 {
                break;
            }
            commits
                .recv_timeout(Duration::from_secs(120).saturating_sub(authored_at.elapsed()))
                .unwrap_or_else(|error| {
                    panic!(
                        "note confirmation: {error}; {}",
                        running.evidence(&sinex, &archive)
                    )
                });
        }
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
    build_on_sinex(&example, &empty, &sinex, &running);
    phase("rebuilt from Sinex");
    assert_eq!(answers(&on_sinex), (sessions, notes), "rebuilt from Sinex");
}

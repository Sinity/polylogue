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
                        "revision":input.rows.revision,"extents":input.rows.extents,
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
            upload_root: None,
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

/// The same outer turn boundary as the daemon: a scope interruption discards
/// that read and waits for real reset/application progress before a fresh turn.
fn confirm_capture(
    engine: &mut Engine,
    commits: &polylogue_store::completion::CommitSubscription,
    started: Instant,
    context: (&Archive, &Sinex, &Running),
    mut complete: impl FnMut(&polylogue_engine::Pass) -> polylogue_store::Result<bool>,
    failure: impl Fn() -> String,
) -> polylogue_engine::Pass {
    let (archive, sinex, running) = context;
    loop {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "native capture/replay did not reach confirmed application: {}; {}",
            failure(),
            running.evidence(sinex, archive)
        );
        let result =
            (|| -> Result<(bool, polylogue_engine::Pass), polylogue_engine::EngineError> {
                let pass = engine.run_pass()?;
                let scope = archive.scope_token()?;
                assert!(
                    pass.errors.is_empty(),
                    "{pass:?}; {}",
                    running.evidence(sinex, archive)
                );
                let done = complete(&pass)?;
                scope.check()?;
                Ok((done, pass))
            })();
        let interrupted = match result {
            Ok((true, pass)) => return pass,
            Ok((false, _)) => false,
            Err(polylogue_engine::EngineError::Store(
                polylogue_store::StoreError::ScopeChanged,
            )) => true,
            Err(error) => panic!(
                "native capture/replay failed: {error}; {}",
                running.evidence(sinex, archive)
            ),
        };
        let remaining = Duration::from_secs(120).saturating_sub(started.elapsed());
        // A failed reset intentionally emits no commit cue. Retain the old
        // replay-readiness tick while fenced, so its queued error is observed.
        let wait = if interrupted {
            remaining.min(Duration::from_millis(200))
        } else {
            remaining
        };
        match commits.recv_timeout(wait) {
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if interrupted => {}
            Err(error) => panic!(
                "native capture/replay wait: {error}; {}; {}",
                failure(),
                running.evidence(sinex, archive)
            ),
        }
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
    confirm_capture(
        &mut engine,
        &commits,
        started,
        (&archive, sinex, running),
        |pass| {
            let applied = ReadPool::open(&archive)?.reader()?.confirmed_checkpoint()?;
            let head = sinex.status().unwrap().owner_position;
            Ok(!pass.capture_pending && head.is_some() && applied == head)
        },
        String::new,
    );
    let acquisitions = running.native.sql("SELECT frame,material_id FROM core.ingress_frames WHERE producer='polylogue' AND frame LIKE 'acquisition:%' ORDER BY frame");
    let repeated = confirm_capture(
        &mut engine,
        &commits,
        started,
        (&archive, sinex, running),
        |pass| Ok(!pass.capture_pending),
        String::new,
    );
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
    read_answers(archive).unwrap()
}

fn read_answers(
    archive: &Archive,
) -> polylogue_store::Result<(BTreeMap<String, serde_json::Value>, Vec<serde_json::Value>)> {
    let pool = ReadPool::open(archive)?;
    let reader = pool.reader()?;
    let sessions = reader
        .query("SELECT id FROM session ORDER BY id", &[])?
        .into_iter()
        .map(|row| {
            let id = polylogue_model::SessionId::parse(row["id"].as_str().unwrap()).unwrap();
            let export = reader.session_export(&id)?.unwrap();
            Ok((id.to_string(), serde_json::to_value(export).unwrap()))
        })
        .collect::<polylogue_store::Result<_>>()?;
    let notes = reader
        .query(
            "SELECT n.id,n.kind,n.value,t.ref FROM source_assertion n JOIN source_link t ON t.record=n.id ORDER BY n.id,t.role,t.ref",
            &[],
        )?;
    Ok((sessions, notes))
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

/// Admit captured evidence through the public producer transport. Every event
/// cites its exact record in the original synthetic JSONL material.
fn captured_evidence(
    running: &Running,
    frame: &str,
    rows: &[(sinex_model::KindRef, serde_json::Value)],
) {
    let root = tempfile::tempdir().unwrap();
    let mut bytes = Vec::new();
    let material = sinex_model::MaterialId::new();
    let events = rows
        .iter()
        .map(|(kind, payload)| {
            let anchor = bytes.len() as u64;
            serde_json::to_writer(&mut bytes, payload).unwrap();
            bytes.push(b'\n');
            sinex_protocol::ProducerEvent {
                kind: kind.clone(),
                payload: payload.clone(),
                location: Some(sinex_model::RecordLocation { material, anchor }),
                output_index: 0,
                parents: vec![],
                reading_order: None,
                class: sinex_model::Class::Open,
            }
        })
        .collect();
    let blob = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let source = sinex_model::SourceRef {
        name: "synthetic-evidence".into(),
        version: 1,
    };
    let raw = sinex_protocol::Ingress::Material {
        frame: format!("{frame}/material"),
        material: Box::new(sinex_protocol::MaterialFrame {
            source: source.clone(),
            id: material,
            input: sinex_model::InputRef {
                kind: "synthetic-evidence".into(),
                locator: frame.into(),
                host: "fixture-host".into(),
            },
            blob: blob.clone(),
            length: bytes.len() as u64,
            chunks: vec![],
            parser: "synthetic-evidence".into(),
            parser_version: 1,
            part_index: 0,
            previous_part: None,
            manifest: Default::default(),
        }),
    };
    sinex_transport::frame_bytes::write(
        root.path(),
        "evidence",
        raw.frame(),
        &blob,
        bytes.len() as u64,
        bytes.as_slice(),
    )
    .unwrap();
    let interpreted = sinex_protocol::Ingress::Events {
        frame: format!("{frame}/events"),
        source: Some(source),
        coverage: None,
        events,
        failures: vec![],
    };
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let client = async_nats::connect(running.native.nats.url())
            .await
            .unwrap();
        let subjects = sinex_protocol::Subjects::new("rebuild");
        let mut acks = client.subscribe(subjects.ack("evidence")).await.unwrap();
        client.flush().await.unwrap();
        sinex_transport::producer::transfer_frame(
            &client,
            &subjects,
            "evidence",
            &mut acks,
            root.path(),
            &raw,
            Duration::from_secs(30),
        )
        .await
        .unwrap();
        for frame in [&raw, &interpreted] {
            let receipt = sinex_transport::producer::submit_frame(
                &client,
                &subjects,
                "evidence",
                &mut acks,
                frame,
                Duration::from_secs(30),
            )
            .await
            .unwrap();
            assert_eq!(receipt.refusal(), None, "{receipt:?}");
        }
    });
}

fn native_rows(running: &Running, query: &str) -> Vec<serde_json::Value> {
    let answer = native::request(
        &running.socket,
        "read",
        serde_json::json!({"method":"query","query":query}),
    );
    assert_eq!(answer["outcome"], "ok", "{answer}");
    answer["result"]["rows"].as_array().unwrap().clone()
}

fn exact_crossdomain_evidence(running: &Running) {
    use serde_json::json;
    use sinex_model::EventPayload;
    use sinex_model::event_types::{machine::FileChanged, session::TerminalReport, work::WorkAct};
    let at = "2026-03-01T10:00:01Z";
    let terminal = |execution: &str| {
        json!({
            "received_at":at,"terminal":"same-terminal","execution":execution,"event":"cmd_end",
            "window_id":1,"cwd":"/home/op/project","cmdline":"cat /home/op/project/draft.txt",
            "exit_status":0,"title":null,"output":"draft content"
        })
    };
    let file = || {
        json!({"received_at":at,"change":"written","path":"/home/op/project/draft.txt",
        "path_hex":null,"to":null,"is_dir":false,"note":null})
    };
    captured_evidence(
        running,
        "external-originals",
        &[
            (
                TerminalReport::declaration().kind,
                terminal("chosen-execution"),
            ),
            (FileChanged::declaration().kind, file()),
            (
                TerminalReport::declaration().kind,
                terminal("unrelated-execution"),
            ),
            (FileChanged::declaration().kind, file()),
        ],
    );
    let tools = native_rows(
        running,
        "events where event_type = polylogue.tool_call | fields id, key, source, event_type, material, offset, output, payload",
    );
    assert_eq!(tools.len(), 2);
    let chosen = tools
        .iter()
        .find(|row| row["payload"]["call_id"] == "call-read-draft")
        .unwrap();
    let other = tools
        .iter()
        .find(|row| row["payload"]["call_id"] == "call-unrelated")
        .unwrap();
    assert_eq!(chosen["payload"]["session"], other["payload"]["session"]);
    assert_eq!(chosen["payload"]["message"], other["payload"]["message"]);
    let mut external = native_rows(
        running,
        "events where event_type = terminal.reported OR event_type = file.changed | fields id, key, source, event_type, material, offset, output, payload",
    );
    external.sort_by_key(|row| row["offset"].as_u64().unwrap());
    assert_eq!(external.len(), 4);
    let evidence = |row: &serde_json::Value| {
        let record = row["key"].as_str().map(str::to_owned).unwrap_or_else(|| {
            format!(
                "{}:{}:{}",
                row["material"].as_str().unwrap(),
                row["offset"].as_u64().unwrap(),
                row["output"].as_u64().unwrap()
            )
        });
        json!({"event":row["id"],"record":serde_json::to_string(&(&row["source"],&row["event_type"],record)).unwrap(),
            "event_type":row["event_type"],"source":row["source"]})
    };
    // The association itself is explicitly captured. Ordinary terminal/file
    // watchers supply no causal edge from a provider's tool-call identity.
    let acts: Vec<_> = external.iter().enumerate().map(|(index, row)| {
        let tool = if index < 2 { chosen } else { other };
        (WorkAct::declaration().kind, json!({
            "act":format!("explicit-{index}"),"session":"claude-code:original-session",
            "episode":null,"objective":null,"label":null,"label_source":evidence(tool),
            "machine":"fixture-host","at":at,"at_evidence":{"quality":"captured","precision":"instant","until":null},
            "ended_at":null,"evidence":evidence(row),"agency":"delegated",
            "action":"inspect","path":"/home/op/project/draft.txt"
        }))
    }).collect();
    captured_evidence(running, "explicit-associations", &acts);
    let linked = native_rows(
        running,
        "events where event_type = polylogue.tool_call AND payload.call_id = call-read-draft | join (events where event_type = work.act) on id = payload.label_source.event | fields payload.call_id as call_id, match.payload.evidence.event as evidence | join (events where event_type = terminal.reported OR event_type = file.changed) on evidence = id | fields call_id, evidence, match.id as id, match.event_type as event_type",
    );
    assert_eq!(linked.len(), 2, "{linked:?}");
    let actual: std::collections::BTreeSet<_> = linked
        .iter()
        .map(|row| {
            assert_eq!(row["call_id"], "call-read-draft");
            assert_eq!(row["id"], row["evidence"]);
            row["id"].as_str().unwrap().to_owned()
        })
        .collect();
    let expected: std::collections::BTreeSet<_> = external[..2]
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(actual, expected);
    for unrelated in &external[2..] {
        assert!(!actual.contains(unrelated["id"].as_str().unwrap()));
    }
}

#[test]
#[ignore = "needs the matching Sinex daemon; set SINEXD"]
fn confirmed_original_reconstructs_a_claude_session_and_its_tool_citation() {
    use polylogue_model::SessionId;
    use polylogue_testkit::{ClaudeHome, claude_user};
    let home = ClaudeHome::new();
    let user = claude_user("user-1", "original-session", "Inspect the draft");
    let action = format!(
        "{}\n",
        serde_json::json!({
            "type":"assistant","uuid":"assistant-1","sessionId":"original-session",
            "timestamp":"2026-03-01T10:00:01Z",
            "message":{"id":"msg_action","role":"assistant","model":"synthetic-model",
                "stop_reason":"tool_use","content":[
                    {"type":"text","text":"I will inspect the draft."},
                    {"type":"tool_use","id":"call-read-draft","name":"Bash",
                        "input":{"command":"cat /home/op/project/draft.txt"}},
                    {"type":"tool_use","id":"call-unrelated","name":"Bash",
                        "input":{"command":"cat /home/op/project/draft.txt"}}
                ]}
        })
    );
    let original = format!("{user}{action}");
    home.append("original-session.jsonl", &original);
    let standalone = ClaudeHome::new();
    standalone.append("original-session.jsonl", &original);
    let (mut ordinary, lock) = standalone.engine(1 << 20);
    assert!(ordinary.run_pass().unwrap().errors.is_empty());
    drop(ordinary);
    drop(lock);
    let session = SessionId::parse("claude-code:original-session").unwrap();
    let expected_messages = standalone
        .reads()
        .reader()
        .unwrap()
        .messages(&session, false)
        .unwrap();
    assert_eq!(expected_messages.len(), 2);

    let running = sinex(home.dir.path());
    let sdk = connect(&running.config);
    let build = |source_home: &Path, wanted: &[polylogue_model::Message]| {
        let (archive, lock) = Archive::open_primary(&home.root, &running.config.store).unwrap();
        let mut config = home.config(1 << 20);
        config.home = source_home.into();
        let mut engine = Engine::on_sinex(
            config,
            archive.clone(),
            sdk.clone(),
            polylogue_testkit::stopped_attempt(),
        )
        .unwrap();
        let commits = engine.writer().unwrap().completion_commits();
        let ready = Instant::now();
        loop {
            match ReadPool::open(&archive)
                .unwrap()
                .reader()
                .unwrap()
                .source_boundary()
            {
                Ok(Some(_)) => break,
                Ok(None) | Err(polylogue_store::StoreError::ScopeChanged) => {}
                Err(error) => panic!("initial source snapshot: {error}"),
            }
            assert!(
                ready.elapsed() < Duration::from_secs(30),
                "initial source snapshot did not complete"
            );
            let _ = commits.recv_timeout(Duration::from_millis(200));
        }
        confirm_capture(
            &mut engine,
            &commits,
            Instant::now(),
            (&archive, &sdk, &running),
            |_| {
                let pool = ReadPool::open(&archive)?;
                let reader = pool.reader()?;
                Ok(reader.messages(&session, false)? == wanted)
            },
            || {
                let actual = ReadPool::open(&archive)
                    .and_then(|pool| pool.reader()?.messages(&session, false));
                format!("expected {wanted:?}; actual {actual:?}")
            },
        );
        engine.writer().unwrap().derive().unwrap();
        drop(engine);
        drop(lock);
        archive
    };
    let archive = build(&home.home, &expected_messages);
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let coordinates = reader.query(
        "SELECT m.source_start AS start,m.source_end AS end,s.id AS material FROM message m JOIN source_material s ON s.input=(SELECT id FROM input_ref WHERE pk=m.input_pk) AND s.chain=m.source_chain AND s.offset=0 WHERE m.id='claude-code:assistant-1'",
        &[],
    ).unwrap();
    assert_eq!(coordinates.len(), 1);
    let citation = &coordinates[0];
    assert_eq!(citation["start"].as_u64(), Some(user.len() as u64));
    // JSON record citations exclude their newline delimiter.
    let end = original.len() as u64 - 1;
    assert_eq!(citation["end"].as_u64(), Some(end));
    let material = sinex_model::MaterialId(citation["material"].as_str().unwrap().parse().unwrap());
    let mut offset = user.len() as u64;
    let mut quoted = Vec::new();
    loop {
        let page = sdk
            .fetch_material_bytes(material, offset, Some(end), 37)
            .unwrap();
        quoted.extend(page.bytes);
        match page.next_offset {
            Some(next) => offset = next,
            None => break,
        }
    }
    assert_eq!(quoted, action.trim_end_matches('\n').as_bytes());
    assert_eq!(
        sinex_rows(
            &running.socket,
            "events where event_type = polylogue.tool_call AND payload.call_id = call-read-draft"
        ),
        1
    );
    exact_crossdomain_evidence(&running);
    let before = reader.session_export(&session).unwrap().unwrap();
    drop(reader);
    drop(pool);
    // Every removed path belongs to this fixture; the native originals remain.
    fs::remove_file(home.file("original-session.jsonl")).unwrap();
    fs::remove_dir_all(&home.root).unwrap();
    let empty = home.dir.path().join("empty-home");
    fs::create_dir(&empty).unwrap();
    let rebuilt = build(&empty, &expected_messages);
    let after = ReadPool::open(&rebuilt)
        .unwrap()
        .reader()
        .unwrap()
        .session_export(&session)
        .unwrap()
        .unwrap();
    assert_eq!(after, before);

    let selected = native::request(
        &running.socket,
        "read",
        serde_json::json!({
            "method":"query", "query":"events where event_type = polylogue.message AND payload.role = assistant"
        }),
    );
    assert_eq!(selected["outcome"], "ok", "{selected}");
    let rows = selected["result"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{selected}");
    assert_eq!(rows[0]["payload"]["message"], "claude-code:assistant-1");
    let id = rows[0]["id"].as_str().unwrap();
    let target = format!("sinex:ev/{id}");
    let archived = native::request(
        &running.socket,
        "agent",
        serde_json::json!({"method":"archive", "target":target}),
    );
    assert_eq!(archived["outcome"], "ok", "{archived}");
    let hidden = build(&empty, &expected_messages[..1]);
    assert_eq!(
        ReadPool::open(&hidden)
            .unwrap()
            .reader()
            .unwrap()
            .messages(&session, false)
            .unwrap(),
        expected_messages[..1]
    );
    let restored = native::request(
        &running.socket,
        "agent",
        serde_json::json!({"method":"restore", "target":target}),
    );
    assert_eq!(restored["outcome"], "ok", "{restored}");
    let restored = build(&empty, &expected_messages);
    assert_eq!(
        ReadPool::open(&restored)
            .unwrap()
            .reader()
            .unwrap()
            .session_export(&session)
            .unwrap()
            .unwrap(),
        before
    );
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
        confirm_capture(
            &mut engine,
            &commits,
            imported_at,
            (&archive, &sinex, &running),
            |pass| {
                let applied = ReadPool::open(&archive)?.reader()?.confirmed_checkpoint()?;
                let status = sinex.status().unwrap();
                let actual = read_answers(&archive)?.0;
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
                    return Ok(true);
                }
                Ok(false)
            },
            || {
                format!(
                    "before import: {before_import}; current: {}",
                    read_answers(&archive)
                        .map(|answers| answer_difference(&expected, &answers.0))
                        .unwrap_or_else(|error| error.to_string())
                )
            },
        );
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
        confirm_capture(
            &mut engine,
            &commits,
            authored_at,
            (&archive, &sinex, &running),
            |pass| Ok(!pass.capture_pending && read_answers(&archive)?.1.len() == 1),
            String::new,
        );
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
